//! A branch that yields a value.
//!
//! [`Lanes::if_uniform`] runs a body for its effects and nothing survives the merge. That is the
//! honest shape for a store, and the wrong one for everything else: the moment a kernel wants
//! *this sum or that one*, it needs a value that arrives from one arm or the other.
//!
//! SPIR-V spells that as `OpPhi`, which names the block each value came through:
//!
//! ```text
//!   %entry:    OpSelectionMerge %merge None
//!              OpBranchConditional %cond %then %else
//!   %then:     …                                  ← ends somewhere; call it %from_then
//!              OpBranch %merge
//!   %else:     …                                  ← ends somewhere; call it %from_else
//!              OpBranch %merge
//!   %merge:    %result = OpPhi %type %a %from_then %b %from_else
//! ```
//!
//! The trap is `%from_then`. It is *not* `%then` whenever the arm branched again — a nested
//! selection, a loop — and a phi naming the wrong predecessor is a module that validates and
//! computes the wrong thing. So the arm's last open block is read back from
//! [`crate::module::Module::current_block`] rather than assumed, and an arm that left no open
//! block at all is refused by name.
//!
//! # Both arms run in the SPIR-V sense, neither runs on the machine
//!
//! The condition is a [`Uniform`], so the whole subgroup takes the same edge — DR-0003, same as
//! everywhere else here. Unlike [`Lanes::select`], which evaluates both sides and picks, only one
//! arm's instructions execute. That is the reason to reach for this: an arm that is expensive, or
//! that contains a reduction, is genuinely skipped.

use super::{LaneError, Lanes, Uniform};
use crate::module::Id;
use crate::spec::SelectionControl;

impl Lanes<'_> {
    /// Take one arm or the other, and yield the value the taken arm produced.
    ///
    /// Both closures return an [`Id`] of type `value_type`; the result is whichever one ran. Use
    /// [`Lanes::type_of`] to name the type, or [`crate::module::Module::type_bool`] and friends
    /// for something that is not an element type.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoOpenBlock`] if an arm ends its own block — a `return`, or a branch it did
    /// not close — because then its value has no edge into the merge. [`LaneError::Build`] if an
    /// instruction cannot be emitted, or whatever an arm returns.
    pub fn choose_uniform<T, E>(
        &mut self,
        condition: Uniform,
        value_type: Id,
        when_true: T,
        when_false: E,
    ) -> Result<Id, LaneError>
    where
        T: FnOnce(&mut Self) -> Result<Id, LaneError>,
        E: FnOnce(&mut Self) -> Result<Id, LaneError>,
    {
        let then_block = self.module().alloc_id()?;
        let else_block = self.module().alloc_id()?;
        let merge_block = self.module().alloc_id()?;

        self.module()
            .selection_merge(merge_block, SelectionControl::None)?;
        self.module()
            .branch_conditional(condition.id(), then_block, else_block)?;

        let (taken, from_then) = self.arm(then_block, merge_block, "then", when_true)?;
        let (untaken, from_else) = self.arm(else_block, merge_block, "else", when_false)?;

        self.module().label_at(merge_block)?;
        Ok(self
            .module()
            .phi(value_type, &[(taken, from_then), (untaken, from_else)])?)
    }

    /// Take the arm when `condition` holds, and fall back to `otherwise` when it does not.
    ///
    /// The one-armed form, which is what most callers mean. `otherwise` is a value that already
    /// exists before the branch — a running total, an identity element — so the false edge needs
    /// no block of its own.
    ///
    /// # Errors
    ///
    /// As [`Lanes::choose_uniform`].
    pub fn if_uniform_value<F>(
        &mut self,
        condition: Uniform,
        value_type: Id,
        otherwise: Id,
        body: F,
    ) -> Result<Id, LaneError>
    where
        F: FnOnce(&mut Self) -> Result<Id, LaneError>,
    {
        self.choose_uniform(condition, value_type, body, |_| Ok(otherwise))
    }

    /// Emit one arm: open its block, build it, close it into the merge, and report which block it
    /// actually finished in.
    fn arm<F>(
        &mut self,
        block: Id,
        merge: Id,
        name: &'static str,
        body: F,
    ) -> Result<(Id, Id), LaneError>
    where
        F: FnOnce(&mut Self) -> Result<Id, LaneError>,
    {
        self.module().label_at(block)?;
        let value = body(self)?;
        // Read back *before* the branch closes it. Nesting is exactly why this cannot be `block`.
        let finished = self
            .module()
            .current_block()
            .ok_or(LaneError::NoOpenBlock { arm: name })?;
        self.module().branch(merge)?;
        Ok((value, finished))
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{F32, Uniform};
    use crate::module::{Module, Version, op};

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    /// A uniform condition, built the only way there is.
    fn condition(lanes: &mut Lanes<'_>) -> Uniform {
        let zero = lanes
            .splat_bits::<F32, 32>(0.0_f32.to_bits())
            .expect("zero");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");
        let over = lanes.greater_than(one, zero).expect("compared");
        lanes.any_uniform(over).expect("voted")
    }

    /// The operands of the sole `OpPhi` in `words`.
    fn phi_operands(words: &[u32]) -> Vec<u32> {
        decode::body(words)
            .find(|instruction| instruction.opcode() == op::PHI)
            .expect("a phi was emitted")
            .operands()
            .to_vec()
    }

    /// The label ids in the order they were opened.
    fn labels(words: &[u32]) -> Vec<u32> {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == op::LABEL)
            .filter_map(|instruction| instruction.operands().first().copied())
            .collect()
    }

    #[test]
    fn a_two_armed_choice_emits_three_blocks_and_one_phi() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes
            .choose_uniform(
                when,
                float,
                |lanes| Ok(lanes.add(one, one)?.id()),
                |lanes| Ok(lanes.mul(one, one)?.id()),
            )
            .expect("chosen");

        let words = module.finish();
        assert_eq!(count(&words, op::LABEL), 3, "then, else, merge");
        assert_eq!(count(&words, op::PHI), 1);
        assert_eq!(count(&words, op::SELECTION_MERGE), 1);
        assert_eq!(
            count(&words, op::BRANCH),
            2,
            "each arm falls into the merge"
        );
    }

    #[test]
    fn the_phi_names_the_block_each_arm_finished_in() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes
            .choose_uniform(when, float, |_| Ok(one.id()), |_| Ok(one.id()))
            .expect("chosen");

        let words = module.finish();
        let opened = labels(&words);
        let operands = phi_operands(&words);

        // result type, result id, then (value, from), else (value, from).
        assert_eq!(operands.len(), 6);
        assert_eq!(operands[3], opened[0], "the then arm");
        assert_eq!(operands[5], opened[1], "the else arm");
    }

    #[test]
    fn a_nested_choice_is_named_by_its_own_merge_and_not_the_arm_it_opened() {
        // The whole reason `current_block` exists. An arm that branches again finishes somewhere
        // other than the block it opened, and a phi naming the block it opened would be a module
        // that validates and reads a value from an edge that does not carry it.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes
            .choose_uniform(
                when,
                float,
                |lanes| lanes.choose_uniform(when, float, |_| Ok(one.id()), |_| Ok(one.id())),
                |_| Ok(one.id()),
            )
            .expect("chosen");

        let words = module.finish();
        let opened = labels(&words);
        // outer then, inner then, inner else, inner merge, outer else, outer merge.
        assert_eq!(opened.len(), 6);

        let outer = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::PHI)
            .last()
            .expect("two phis")
            .operands()
            .to_vec();

        assert_eq!(
            outer[3], opened[3],
            "the outer then arm finished in the inner merge block, not the block it opened"
        );
        assert_ne!(outer[3], opened[0]);
    }

    #[test]
    fn the_one_armed_form_takes_its_fallback_from_before_the_branch() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes
            .if_uniform_value(when, float, one.id(), |lanes| Ok(lanes.add(one, one)?.id()))
            .expect("chosen");

        let words = module.finish();
        let operands = phi_operands(&words);

        assert_eq!(
            operands[4],
            one.id().word(),
            "the fallback is the value that already existed"
        );
        assert_eq!(count(&words, op::F_ADD), 1, "only the arm computes");
    }

    #[test]
    fn an_arm_that_ends_its_own_block_is_refused_rather_than_mis_attributed() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        let refused = lanes.choose_uniform(
            when,
            float,
            |lanes| {
                lanes.module().return_void()?;
                Ok(one.id())
            },
            |_| Ok(one.id()),
        );

        assert_eq!(refused.err(), Some(LaneError::NoOpenBlock { arm: "then" }));
    }

    #[test]
    fn an_arm_that_fails_carries_its_error_out() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        let refused = lanes.choose_uniform(
            when,
            float,
            |_| Ok(one.id()),
            |lanes| {
                // 12 lanes has no mapping onto a 32-wide subgroup.
                lanes.splat_bits::<F32, 12>(0)?;
                Ok(one.id())
            },
        );

        assert!(matches!(refused, Err(LaneError::NoMapping { .. })));
    }

    #[test]
    fn the_merge_is_declared_immediately_before_the_branch_that_splits() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes
            .choose_uniform(when, float, |_| Ok(one.id()), |_| Ok(one.id()))
            .expect("chosen");

        let words = module.finish();
        let seen: Vec<u16> = decode::body(&words)
            .map(|instruction| instruction.opcode())
            .collect();
        let merge = seen
            .iter()
            .position(|opcode| *opcode == op::SELECTION_MERGE)
            .expect("declared");

        assert_eq!(
            seen.get(merge + 1).copied(),
            Some(op::BRANCH_CONDITIONAL),
            "OpSelectionMerge must be the second-to-last instruction in its block"
        );
    }

    #[test]
    fn the_phi_is_the_first_instruction_of_the_merge_block() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes
            .choose_uniform(when, float, |_| Ok(one.id()), |_| Ok(one.id()))
            .expect("chosen");

        let words = module.finish();
        let seen: Vec<u16> = decode::body(&words)
            .map(|instruction| instruction.opcode())
            .collect();
        let phi = seen
            .iter()
            .position(|opcode| *opcode == op::PHI)
            .expect("emitted");

        assert_eq!(
            seen.get(phi.wrapping_sub(1)).copied(),
            Some(op::LABEL),
            "SPIR-V requires every OpPhi at the very start of its block"
        );
    }
}
