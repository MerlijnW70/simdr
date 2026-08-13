//! Prefix sums: every element replaced by the total of everything up to and including it.
//!
//! A reduction throws away all but one number. A scan keeps them all, and that makes it the harder
//! of the two — every output depends on a different amount of the input, so there is no arrangement
//! of the work in which every invocation does the same thing to the same amount of data.
//!
//! # Why this is worth having beyond "a second algorithm"
//!
//! Everything else in `kernels/` reduces, maps, or shuffles. A scan is the first thing here that
//! needs a *partial* result from a neighbour rather than a total from everyone, which exercises
//! three parts of the emitter together that nothing else did: the subgroup scan instruction, the
//! workgroup handover through shared memory, and a per-lane `OpSelect` on a value that differs by
//! lane. If the lane mapping were wrong in a way a reduction hides — a reduction sums the same set
//! whatever order the lanes are in — a scan gets a different answer.
//!
//! # The shape, and why there is no divergence in it
//!
//! ```text
//!   running = prefix_sum(value)     inclusive, within this invocation's subgroup
//!   total   = reduce_sum(value)     this subgroup's whole total, in every one of its lanes
//!   shared[local_index] = total     every invocation writes its own slot
//!   barrier
//!   offset  = sum of the totals of the subgroups before mine
//!   out[i]  = running + offset
//! ```
//!
//! That last sum is the interesting line. Which subgroups come "before mine" differs per lane, and
//! the obvious way to write it is a loop bounded by this invocation's subgroup index — a loop that
//! runs a different number of times per lane, which is the divergence `decisions/DR-0003` refuses.
//!
//! So it is written as a fixed number of steps instead, one per subgroup in the workgroup, each of
//! which adds that subgroup's total **or not**:
//!
//! ```text
//!   for each earlier subgroup k:
//!       offset = local_index > (k+1)*width - 1  ?  offset + shared[k*width]  :  offset
//! ```
//!
//! Every invocation executes all of them; the `OpSelect` is what makes the answer differ. The step
//! count is `workgroup / subgroup`, which is fixed when the module is built — 1 on a 64-wide
//! device, 2 on a 32-wide one, 16 on a four-wide one — so this is straight-line code whose length
//! the device's width decides and whose *shape* it does not.
//!
//! # What it does not do
//!
//! **One workgroup.** This scans [`super::WORKGROUP_SIZE`] elements and no more; a longer input
//! needs the block totals scanned and added back, which is a second and third dispatch and is not
//! built. The limit is in the name of the function rather than hidden in its behaviour, because a
//! scan that silently restarted at every block boundary would return plausible numbers.

use super::{shape, whole_subgroup_of};
use simdr::kernel::Kernel;
use simdr::lanes::{Element, LaneError};
use simdr::module::op;

/// `out[i] = in[0] + in[1] + … + in[i]`, within one workgroup.
///
/// Inclusive: element `i` of the output includes element `i` of the input. The exclusive form is
/// this shifted by one and is not built — a caller who wants it can subtract its own element,
/// which costs one instruction and no second kernel.
///
/// # Errors
///
/// [`LaneError::BadWidth`] if `subgroup` is not a width this can build for, [`LaneError::BadShape`]
/// if the workgroup is not a whole number of subgroups, otherwise if the module cannot be built.
pub fn scan_workgroup<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, scan_workgroup_at)
}

/// The builder, at a lane count that has to equal the subgroup width.
///
/// `prefix_sum` refuses any mapping but the whole subgroup — a strip-mined scan would have to
/// carry a running total between strips, which is not built — so this is only ever instantiated
/// with `LANES` equal to `subgroup`, which is what [`whole_subgroup_of`] arranges.
fn scan_workgroup_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let workgroup = super::WORKGROUP_SIZE;
    if subgroup == 0 || !workgroup.is_multiple_of(subgroup) {
        return Err(LaneError::BadShape {
            workgroup,
            buffers: subgroup,
        });
    }
    let subgroups = workgroup / subgroup;

    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;

    // The two subgroup instructions, from the same input. `prefix_sum` gives each lane its running
    // total within the subgroup; `reduce_sum` gives every lane of that subgroup the whole of it,
    // which is what the *next* subgroup needs and what goes into shared memory.
    let running = kernel.lanes()?.prefix_sum(value)?;
    let total = kernel.lanes()?.reduce_sum(value)?;

    let shared = kernel.shared(workgroup)?;
    let slot = kernel.local_index();
    kernel.store_shared(shared, slot, total)?;
    kernel.barrier()?;

    let element = kernel.element();
    // Zero of whatever `T` is, by bit pattern: `0` is `0.0` as an `f32` and zero as either
    // integer. The additive identity is the right starting offset for the first subgroup, which
    // has nothing before it.
    let mut offset = kernel.module().constant_scalar(element, 0)?;

    for earlier in 0..subgroups.saturating_sub(1) {
        // Slot `k * width` is where subgroup `k` wrote its total. Every lane of that subgroup
        // wrote the same value to a different slot, so any one of them will do and this takes the
        // first — a constant index, which is what makes the read the same instruction in every
        // invocation.
        let theirs = kernel.load_shared(shared, earlier * subgroup)?;

        // The last lane of subgroup `k`. An invocation past it belongs to a later subgroup and
        // owes this total; one at or before it does not. Written as `>` against the last index
        // rather than `>=` against the first so that it is one comparison either way.
        let boundary = kernel.module().constant_u32((earlier + 1) * subgroup - 1)?;
        let boolean = kernel.module().type_bool()?;
        let after = kernel
            .module()
            .binary(op::U_GREATER_THAN, boolean, slot, boundary)?;

        // Both arms are computed, and one is thrown away. That is the point: `OpSelect` is not a
        // branch, so every lane runs the same instructions and no subgroup operation below it can
        // find itself in non-uniform control flow.
        let with = kernel.module().binary(T::ADD, element, offset, theirs)?;
        offset = kernel.module().select(element, after, with, offset)?;
    }

    let scanned = kernel
        .module()
        .binary(T::ADD, element, running.id(), offset)?;
    kernel.store_scalar(1, scanned)?;
    kernel.finish()
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used)]

    use super::{scan_workgroup, scan_workgroup_at};
    use simdr::decode;
    use simdr::lanes::{F32, LaneError};
    use simdr::module::op;
    use simdr::spec::StorageClass;
    use std::collections::HashMap;

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    /// Which slots of shared memory this module reads, in the order it reads them.
    ///
    /// Decoded rather than assumed. The arithmetic that picks a slot — `k * width` — is the one
    /// line here whose mistakes are invisible at 32 lanes: the loop runs once, `k` is zero, and
    /// zero times anything is zero times anything else. It takes a narrow device, or this, to tell
    /// a multiply from a divide.
    fn shared_slots_read(words: &[u32]) -> Vec<u32> {
        // Constants first: an access chain names its index by id, not by value.
        let values: HashMap<u32, u32> = decode::body(words)
            .filter(|instruction| instruction.opcode() == op::CONSTANT)
            .filter_map(|instruction| match instruction.operands() {
                [_type, id, literal] => Some((*id, *literal)),
                _ => None,
            })
            .collect();

        // The one variable in workgroup storage is the shared array.
        let Some(shared) = decode::body(words)
            .filter(|instruction| instruction.opcode() == op::VARIABLE)
            .find_map(|instruction| match instruction.operands() {
                [_type, id, class] if *class == StorageClass::Workgroup.word() => Some(*id),
                _ => None,
            })
        else {
            return Vec::new();
        };

        decode::body(words)
            .filter(|instruction| instruction.opcode() == op::ACCESS_CHAIN)
            .filter_map(|instruction| match instruction.operands() {
                [_type, _id, base, index] if *base == shared => values.get(index).copied(),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_width_that_is_no_devices_subgroup_is_refused() {
        for width in [0_u32, 24, 48, 128] {
            assert!(
                matches!(
                    scan_workgroup::<F32>(width),
                    Err(LaneError::BadWidth { .. })
                ),
                "a subgroup of {width} was accepted"
            );
        }
    }

    #[test]
    fn the_widths_a_device_reports_are_accepted() {
        for width in [4_u32, 8, 16, 32, 64] {
            assert!(
                scan_workgroup::<F32>(width).is_ok(),
                "a subgroup of {width} was refused"
            );
        }
    }

    #[test]
    fn one_subgroup_per_workgroup_needs_no_select_at_all() {
        // At 64 lanes the workgroup *is* one subgroup: there are no earlier subgroups, the loop
        // runs zero times, and the whole shared-memory combine costs nothing but the barrier.
        let words = scan_workgroup::<F32>(64).expect("built");
        assert_eq!(count(&words, op::SELECT), 0);
    }

    #[test]
    fn the_select_count_is_one_fewer_than_the_subgroups() {
        // Straight-line and fixed at build time: 64/width subgroups, each of which owes every
        // later one its total, so 64/width - 1 steps. A loop bounded by the *lane's* subgroup
        // would emit one step and diverge instead, which is what this rules out.
        for (width, subgroups) in [(64_u32, 1_u32), (32, 2), (16, 4), (8, 8), (4, 16)] {
            let words = scan_workgroup::<F32>(width).expect("built");
            assert_eq!(
                count(&words, op::SELECT),
                subgroups as usize - 1,
                "at width {width}"
            );
        }
    }

    #[test]
    fn each_step_reads_the_slot_its_own_subgroup_wrote() {
        // Subgroup `k` writes its total at slot `k * width`, because every lane stored at its own
        // local index and the first lane of subgroup `k` sits there. Reading anywhere else adds
        // the wrong subgroup's total — or, if the multiply became a divide, adds subgroup zero's
        // total over and over and reads nothing else at all.
        //
        // At width 4 there are sixteen subgroups and fifteen reads, so the sequence is long enough
        // to be wrong in a visible way.
        let words = scan_workgroup::<F32>(4).expect("built");
        let expected: Vec<u32> = (0..15).map(|step| step * 4).collect();

        assert_eq!(shared_slots_read(&words), expected);
    }

    #[test]
    fn the_slots_are_spaced_by_the_width_at_every_width() {
        // The same claim across the range, which is what stops the test above from being a fact
        // about the number four.
        for width in [4_u32, 8, 16, 32, 64] {
            let words = scan_workgroup::<F32>(width).expect("built");
            let subgroups = super::super::WORKGROUP_SIZE / width;
            let expected: Vec<u32> = (0..subgroups - 1).map(|step| step * width).collect();

            assert_eq!(shared_slots_read(&words), expected, "at width {width}");
        }
    }

    #[test]
    fn a_subgroup_the_workgroup_does_not_divide_by_is_refused() {
        // Asked of the builder directly. The public wrapper refuses any width it has no lane count
        // for *first* — the lane count is a const generic, so only the widths `whole_subgroup_of!`
        // lists can be instantiated at all — and going through it would test the dispatcher while
        // leaving this guard unreached, which is how it came to survive a mutation run.
        for width in [24_u32, 48, 63] {
            assert!(
                matches!(
                    scan_workgroup_at::<F32, 32>(width),
                    Err(LaneError::BadShape { .. })
                ),
                "a subgroup of {width} was accepted"
            );
        }
    }

    #[test]
    fn a_subgroup_of_zero_is_refused_before_it_divides() {
        // The other half of the same condition, and the one that would divide by zero if the two
        // were reordered — which is why they are an `||` in that order rather than either alone.
        assert!(matches!(
            scan_workgroup_at::<F32, 32>(0),
            Err(LaneError::BadShape { .. })
        ));
    }

    #[test]
    fn the_scan_instruction_is_emitted_once_and_so_is_the_reduction() {
        // Both, from the same input. A version that scanned twice, or reduced instead of scanning,
        // would still produce a plausible-looking module.
        let words = scan_workgroup::<F32>(32).expect("built");
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_ADD), 2);
    }

    #[test]
    fn there_is_exactly_one_barrier_and_it_is_not_inside_anything() {
        // Every invocation must reach it. One barrier, emitted at the top level of the function
        // rather than once per loop step, is the only shape in which that is obvious.
        let words = scan_workgroup::<F32>(4).expect("built");
        assert_eq!(count(&words, op::CONTROL_BARRIER), 1);
        assert_eq!(
            count(&words, op::BRANCH_CONDITIONAL),
            0,
            "a conditional branch would put the barrier in some lanes' path and not others"
        );
    }
}
