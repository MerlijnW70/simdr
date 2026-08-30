//! ```text
//!   %entry:    OpSelectionMerge %merge None
//!              OpBranchConditional %cond %then %else
//!   %then:     …                                  ← ends somewhere; call it %from_then
//!              OpBranch %merge
//!   %else:     …                                  ← ends somewhere; call it %from_else
//!              OpBranch %merge
//!   %merge:    %result = OpPhi %type %a %from_then %b %from_else
//! ```

use super::{LaneError, Lanes, Uniform};
use crate::module::Id;
use crate::spec::SelectionControl;

impl Lanes<'_> {
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

    fn condition(lanes: &mut Lanes<'_>) -> Uniform {
        let zero = lanes
            .splat_bits::<F32, 32>(0.0_f32.to_bits())
            .expect("zero");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");
        let over = lanes.greater_than(one, zero).expect("compared");
        lanes.any_uniform(over).expect("voted")
    }

    fn phi_operands(words: &[u32]) -> Vec<u32> {
        decode::body(words)
            .find(|instruction| instruction.opcode() == op::PHI)
            .expect("a phi was emitted")
            .operands()
            .to_vec()
    }

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

        assert_eq!(operands.len(), 6);
        assert_eq!(operands[3], opened[0], "the then arm");
        assert_eq!(operands[5], opened[1], "the else arm");
    }

    #[test]
    fn a_nested_choice_is_named_by_its_own_merge_and_not_the_arm_it_opened() {
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
