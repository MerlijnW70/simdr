//! Loops, and the values that survive them.
//!
//! A loop is four blocks in SPIR-V, and they have to be arranged exactly:
//!
//! ```text
//!   …            OpBranch %header
//!   %header:     OpLoopMerge %merge %continue   ← declared before anything else in the block
//!                OpBranch %body                    (or a conditional, for a while)
//!   %body:       …                              ← the work
//!                OpBranch %continue
//!   %continue:   …                              ← the step
//!                OpBranch %header               ← the back edge
//!   %merge:      …                              ← where the loop leaves off
//! ```
//!
//! Getting that wrong produces a module the validator rejects for reasons that read like riddles,
//! which is why the shape lives here once rather than at each call site.
//!
//! # The trip count is uniform
//!
//! Same rule as a branch, same reason: a loop whose condition varies per lane leaves some lanes
//! going round again while others have left, and a subgroup instruction inside it answers for
//! whoever is still there. `decisions/DR-0003`. What is offered is a loop of a *fixed* number of
//! iterations, decided when the kernel is built — which is what a strip-mined `Simd` needs
//! anyway, and what an unrolled reduction is.

use super::{LaneError, Lanes};
use crate::module::Id;
use crate::spec::LoopControl;

impl Lanes<'_> {
    /// Repeat `body` a fixed number of times, threading one value through it.
    ///
    /// `body` receives the value carried from the previous iteration and returns the value for
    /// the next; the loop yields whatever the last iteration produced. That threading is the
    /// whole difficulty — SPIR-V has no mutable locals in the logical addressing model, so the
    /// carried value is an `OpPhi` at the loop header, and its incoming edges cannot be written
    /// until the body has been built.
    ///
    /// The count is a Rust `u32` rather than an [`Id`], and deliberately: a trip count that
    /// varied per lane would diverge, and one that varied uniformly at runtime would still need a
    /// condition this does not take. A fixed count covers the strip-mined and tree-reduction
    /// shapes that lane code actually wants.
    ///
    /// # Errors
    ///
    /// [`LaneError::Build`] if any instruction cannot be emitted, or whatever `body` returns.
    pub fn repeat<F>(&mut self, times: u32, initial: Id, mut body: F) -> Result<Id, LaneError>
    where
        F: FnMut(&mut Self, Id, u32) -> Result<Id, LaneError>,
    {
        // Unrolled, and it takes no `carried_type` because it needs none: with no phi there is
        // nothing whose type has to be declared. A `times` known at build time makes the loop
        // machinery pure cost — no phi, no back edge, no counter — and the driver was going to
        // unroll a small fixed loop anyway. [`Lanes::repeat_rolled`] is the one that emits a real
        // loop; this is what nearly every caller means, and it produces better SPIR-V.
        let mut carried = initial;
        for iteration in 0..times {
            carried = body(self, carried, iteration)?;
        }
        Ok(carried)
    }

    /// The same, as a real loop rather than an unrolled one.
    ///
    /// Emits the four-block shape above with an `OpPhi` carrying both the counter and the value.
    /// Use it when `times` is large enough that unrolling would bloat the module.
    ///
    /// `body` receives the carried value and the *iteration number* — the counter phi, a `u32`
    /// id that is 0 on the first trip and `times - 1` on the last. It is one value rather than one
    /// per iteration, because the body is built once; a body that wants `data[i]` indexes with it,
    /// and a body that wants to unroll by iteration wants [`Lanes::repeat`] instead.
    ///
    /// # Errors
    ///
    /// [`LaneError::Build`] if any instruction cannot be emitted, or whatever `body` returns.
    pub fn repeat_rolled<F>(
        &mut self,
        times: u32,
        carried_type: Id,
        initial: Id,
        body: F,
    ) -> Result<Id, LaneError>
    where
        F: FnOnce(&mut Self, Id, Id) -> Result<Id, LaneError>,
    {
        if times == 0 {
            return Ok(initial);
        }

        let uint = self.module().type_int(32, false)?;
        let boolean = self.module().type_bool()?;
        let zero = self.module().constant_u32(0)?;
        let one = self.module().constant_u32(1)?;
        let limit = self.module().constant_u32(times)?;

        let header = self.module().alloc_id()?;
        let body_block = self.module().alloc_id()?;
        let continue_block = self.module().alloc_id()?;
        let merge_block = self.module().alloc_id()?;

        // The block the loop is entered from, which the phis name as an incoming edge.
        let entry = self.module().alloc_id()?;
        self.module().branch(entry)?;
        self.module().label_at(entry)?;
        self.module().branch(header)?;

        self.module().label_at(header)?;
        // The phis come first in the header, before the merge declaration — SPIR-V requires every
        // `OpPhi` at the very start of its block.
        let counter = self.module().alloc_id()?;
        let carried = self.module().alloc_id()?;
        let stepped = self.module().alloc_id()?;
        let produced = self.module().alloc_id()?;
        self.module()
            .phi_at(counter, uint, &[(zero, entry), (stepped, continue_block)])?;
        self.module().phi_at(
            carried,
            carried_type,
            &[(initial, entry), (produced, continue_block)],
        )?;

        // The comparison comes *before* the merge declaration. `OpLoopMerge` has to be the
        // second-to-last instruction in its block, immediately preceding the branch — putting the
        // comparison between them is a module the validator rejects, and it did.
        let carry_on =
            self.module()
                .binary(crate::module::op::U_LESS_THAN, boolean, counter, limit)?;
        self.module()
            .loop_merge(merge_block, continue_block, LoopControl::None)?;
        self.module()
            .branch_conditional(carry_on, body_block, merge_block)?;

        self.module().label_at(body_block)?;
        let result = body(self, carried, counter)?;
        // `produced` was promised to the phi above, so the body's result has to arrive under that
        // name. A copy is the honest way to say so without a mutable local.
        self.module()
            .copy_object_at(produced, carried_type, result)?;
        self.module().branch(continue_block)?;

        self.module().label_at(continue_block)?;
        self.module().i_add_at(stepped, uint, counter, one)?;
        self.module().branch(header)?;

        self.module().label_at(merge_block)?;
        Ok(carried)
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::F32;
    use crate::module::{Module, Version, op};

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    #[test]
    fn an_unrolled_repeat_emits_the_body_once_per_iteration() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes
            .repeat(3, one.id(), |lanes, carried, _| {
                let value = lanes.from_lane_value::<F32, 32>(carried)?;
                Ok(lanes.add(value, one)?.id())
            })
            .expect("repeated");

        let words = module.finish();
        assert_eq!(count(&words, op::F_ADD), 3);
        assert_eq!(count(&words, op::LOOP_MERGE), 0, "nothing to merge");
    }

    #[test]
    fn an_unrolled_repeat_of_zero_yields_what_it_was_given() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        let out = lanes
            .repeat(0, one.id(), |_, _, _| unreachable!("never runs"))
            .expect("repeated");

        assert_eq!(out, one.id());
        assert_eq!(count(&module.finish(), op::F_ADD), 0);
    }

    #[test]
    fn the_body_is_told_which_iteration_it_is() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let start = lanes
            .splat_bits::<F32, 32>(0.0_f32.to_bits())
            .expect("zero");

        let mut seen = Vec::new();
        lanes
            .repeat(4, start.id(), |lanes, carried, iteration| {
                seen.push(iteration);
                let value = lanes.from_lane_value::<F32, 32>(carried)?;
                Ok(lanes.add(value, value)?.id())
            })
            .expect("repeated");

        assert_eq!(seen, vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_rolled_loop_emits_the_four_blocks_and_two_phis() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes
            .repeat_rolled(8, float, one.id(), |lanes, carried, _| {
                let value = lanes.from_lane_value::<F32, 32>(carried)?;
                Ok(lanes.add(value, one)?.id())
            })
            .expect("looped");

        let words = module.finish();
        assert_eq!(count(&words, op::LOOP_MERGE), 1);
        assert_eq!(count(&words, op::PHI), 2, "the counter and the value");
        assert_eq!(count(&words, op::F_ADD), 1, "the body is built once");
        // entry, header, body, continue, merge.
        assert_eq!(count(&words, op::LABEL), 5);
    }

    #[test]
    fn a_rolled_loop_declares_its_merge_before_the_branch_that_leaves_it() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes
            .repeat_rolled(4, float, one.id(), |_, carried, _| Ok(carried))
            .expect("looped");

        let words = module.finish();
        let seen: Vec<u16> = decode::body(&words)
            .map(|instruction| instruction.opcode())
            .collect();

        let merge = seen
            .iter()
            .position(|opcode| *opcode == op::LOOP_MERGE)
            .expect("declared");

        // *Immediately* before, not merely somewhere before. The looser check passed while the
        // comparison sat between them, and only `spirv-val` noticed.
        assert_eq!(
            seen.get(merge + 1).copied(),
            Some(op::BRANCH_CONDITIONAL),
            "OpLoopMerge must be the second-to-last instruction in its block"
        );
    }

    #[test]
    fn the_phis_come_before_the_merge_declaration_in_the_header() {
        // SPIR-V requires every `OpPhi` at the very start of its block, and the header's merge
        // instruction is not a phi.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes
            .repeat_rolled(4, float, one.id(), |_, carried, _| Ok(carried))
            .expect("looped");

        let words = module.finish();
        let seen: Vec<u16> = decode::body(&words)
            .map(|instruction| instruction.opcode())
            .collect();

        let last_phi = seen
            .iter()
            .rposition(|opcode| *opcode == op::PHI)
            .expect("emitted");
        let merge = seen
            .iter()
            .position(|opcode| *opcode == op::LOOP_MERGE)
            .expect("declared");

        assert!(last_phi < merge);
    }

    #[test]
    fn a_rolled_body_is_handed_the_counter_phi_itself() {
        // Not a copy of it and not a fresh id: the value the body indexes with has to be the same
        // one the continue block steps, or every iteration would read the same element.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let float = lanes.type_of::<F32>().expect("f32");
        let uint = lanes.module().type_int(32, false).expect("u32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        let mut seen = None;
        lanes
            .repeat_rolled(8, float, one.id(), |_, carried, iteration| {
                seen = Some(iteration);
                Ok(carried)
            })
            .expect("looped");

        let words = module.finish();
        let counter = decode::body(&words)
            .find(|instruction| {
                instruction.opcode() == op::PHI
                    && instruction.operands().first() == Some(&uint.word())
            })
            .and_then(|instruction| instruction.operands().get(1).copied())
            .expect("the counter is the u32 phi");

        assert_eq!(seen.map(|id| id.word()), Some(counter));
    }

    #[test]
    fn a_rolled_loop_of_zero_emits_nothing_at_all() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let float = lanes.type_of::<F32>().expect("f32");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        let out = lanes
            .repeat_rolled(0, float, one.id(), |_, _, _| unreachable!("never runs"))
            .expect("looped");

        assert_eq!(out, one.id());
        assert_eq!(count(&module.finish(), op::LOOP_MERGE), 0);
    }
}
