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

use super::{LaneError, Lanes};
use crate::module::{Id, Module};
use crate::spec::LoopControl;

pub(crate) trait Emits {
    fn module(&mut self) -> &mut Module;
}

impl Emits for Lanes<'_> {
    fn module(&mut self) -> &mut Module {
        Self::module(self)
    }
}

pub(crate) fn rolled<H, F>(
    host: &mut H,
    times: u32,
    carried_type: Id,
    initial: Id,
    body: F,
) -> Result<Id, LaneError>
where
    H: Emits + ?Sized,
    F: FnOnce(&mut H, Id, Id) -> Result<Id, LaneError>,
{
    let held = rolled_many(
        host,
        times,
        carried_type,
        &[initial],
        |host, carried, counter| {
            let one = carried.first().copied().ok_or(LaneError::BadCarry {
                given: carried.len(),
                wanted: 1,
            })?;
            body(host, one, counter).map(|one| vec![one])
        },
    )?;
    held.first().copied().ok_or(LaneError::BadCarry {
        given: held.len(),
        wanted: 1,
    })
}

pub(crate) fn rolled_many<H, F>(
    host: &mut H,
    times: u32,
    carried_type: Id,
    initial: &[Id],
    body: F,
) -> Result<Vec<Id>, LaneError>
where
    H: Emits + ?Sized,
    F: FnOnce(&mut H, &[Id], Id) -> Result<Vec<Id>, LaneError>,
{
    if times == 0 || initial.is_empty() {
        return Ok(initial.to_vec());
    }

    let uint = host.module().type_int(32, false)?;
    let boolean = host.module().type_bool()?;
    let zero = host.module().constant_u32(0)?;
    let one = host.module().constant_u32(1)?;
    let limit = host.module().constant_u32(times)?;

    let header = host.module().alloc_id()?;
    let body_block = host.module().alloc_id()?;
    let continue_block = host.module().alloc_id()?;
    let merge_block = host.module().alloc_id()?;

    let entry = host.module().alloc_id()?;
    host.module().branch(entry)?;
    host.module().label_at(entry)?;
    host.module().branch(header)?;

    host.module().label_at(header)?;
    let counter = host.module().alloc_id()?;
    let stepped = host.module().alloc_id()?;
    let mut carried = Vec::with_capacity(initial.len());
    let mut produced = Vec::with_capacity(initial.len());
    for _ in initial {
        carried.push(host.module().alloc_id()?);
        produced.push(host.module().alloc_id()?);
    }
    host.module()
        .phi_at(counter, uint, &[(zero, entry), (stepped, continue_block)])?;
    for ((&name, &was), &will) in carried.iter().zip(initial).zip(&produced) {
        host.module()
            .phi_at(name, carried_type, &[(was, entry), (will, continue_block)])?;
    }

    let carry_on = host
        .module()
        .binary(crate::module::op::U_LESS_THAN, boolean, counter, limit)?;
    host.module()
        .loop_merge(merge_block, continue_block, LoopControl::None)?;
    host.module()
        .branch_conditional(carry_on, body_block, merge_block)?;

    host.module().label_at(body_block)?;
    let result = body(host, &carried, counter)?;
    if result.len() != produced.len() {
        return Err(LaneError::BadCarry {
            given: result.len(),
            wanted: produced.len(),
        });
    }
    for (&will, &one) in produced.iter().zip(&result) {
        host.module().copy_object_at(will, carried_type, one)?;
    }
    host.module().branch(continue_block)?;

    host.module().label_at(continue_block)?;
    host.module().i_add_at(stepped, uint, counter, one)?;
    host.module().branch(header)?;

    host.module().label_at(merge_block)?;
    Ok(carried)
}

impl Lanes<'_> {
    pub fn repeat<F>(&mut self, times: u32, initial: Id, mut body: F) -> Result<Id, LaneError>
    where
        F: FnMut(&mut Self, Id, u32) -> Result<Id, LaneError>,
    {
        let mut carried = initial;
        for iteration in 0..times {
            carried = body(self, carried, iteration)?;
        }
        Ok(carried)
    }

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
        rolled(self, times, carried_type, initial, body)
    }
}

#[cfg(test)]
mod tests {
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

        assert_eq!(
            seen.get(merge + 1).copied(),
            Some(op::BRANCH_CONDITIONAL),
            "OpLoopMerge must be the second-to-last instruction in its block"
        );
    }

    #[test]
    fn the_phis_come_before_the_merge_declaration_in_the_header() {
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
