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
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, LaneError};
use simdr::module::{Id, op};

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

/// `out[i] = in[0] + … + in[i]` within each block, **and** each block's total to binding 2.
///
/// The same scan as [`scan_workgroup`] with one instruction more: the last thing every invocation
/// holds is its subgroup's running total plus the offset of the subgroups before it, so the *last*
/// subgroup's lanes are holding the block's whole total. That value goes to
/// [`simdr::kernel::Kernel::workgroup_index`] of binding 2.
///
/// **This is what a scan longer than one workgroup needs.** With block totals in a buffer of their
/// own, scanning *those* and adding the result back to each block turns 64 elements into 64 × 64,
/// and again for as many levels as the length needs. Nothing here does that yet; this is the pass
/// that makes it possible, and it is useful on its own to anyone who wants per-block sums.
///
/// # Every invocation writes the block total, and they all write the same one
///
/// The whole workgroup runs the store, so binding 2's slot is written 64 times. That is the case
/// [`simdr::kernel::Kernel::store_at`] documents as its own: identical values to one address, where
/// the order they land in cannot change the answer. Electing one lane to write instead would need a
/// branch that some invocations do not take, which `decisions/DR-0003` refuses.
///
/// # Errors
///
/// As [`scan_workgroup`].
pub fn scan_blocks<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, scan_blocks_at)
}

/// The builder for [`scan_blocks`].
fn scan_blocks_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    // Three buffers rather than two: the block totals need one of their own.
    let mut kernel = Kernel::<T>::new(Shape::new(subgroup, super::WORKGROUP_SIZE, 3))?;
    let (scanned, total) = scanned_at::<T, LANES>(&mut kernel, subgroup, true)?;

    kernel.store_scalar(1, scanned)?;

    let Some(total) = total else {
        return Err(LaneError::BadShape {
            workgroup: super::WORKGROUP_SIZE,
            buffers: subgroup,
        });
    };
    let block = kernel.workgroup_index();
    kernel.store_at(2, block, total)?;

    kernel.finish()
}

/// The builder, at a lane count that has to equal the subgroup width.
///
/// `prefix_sum` refuses any mapping but the whole subgroup — a strip-mined scan would have to
/// carry a running total between strips, which is not built — so this is only ever instantiated
/// with `LANES` equal to `subgroup`, which is what [`whole_subgroup_of`] arranges.
fn scan_workgroup_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let (scanned, _) = scanned_at::<T, LANES>(&mut kernel, subgroup, false)?;

    kernel.store_scalar(1, scanned)?;
    kernel.finish()
}

/// The scan itself: this invocation's running total, and the block's whole total if asked for.
///
/// **One copy of the arithmetic, used by both kernels.** Writing it twice would be two things to
/// keep in step, and the one that got less attention would be the one nobody had run at width 4 —
/// where the cross-subgroup combine is fifteen steps rather than none.
///
/// `want_total` rather than always computing it, because the block total costs one addition per
/// subgroup and [`scan_workgroup`] has nowhere to put it. Emitting instructions whose result is
/// discarded would make the module say something the kernel does not do; a driver would remove
/// them, and the module is what this project checks.
fn scanned_at<T: Element, const LANES: u32>(
    kernel: &mut Kernel<T>,
    subgroup: u32,
    want_total: bool,
) -> Result<(Id, Option<Id>), LaneError> {
    let workgroup = super::WORKGROUP_SIZE;
    if subgroup == 0 || !workgroup.is_multiple_of(subgroup) {
        return Err(LaneError::BadShape {
            workgroup,
            buffers: subgroup,
        });
    }
    let subgroups = workgroup / subgroup;

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
    // has nothing before it, and the right start for a sum.
    let zero = kernel.module().constant_scalar(element, 0)?;
    let mut offset = zero;
    let mut block = want_total.then_some(zero);

    // **The last subgroup is read only if the block total wants it.** It is nobody's predecessor,
    // so it contributes no offset to anyone; loading it regardless left `scan_workgroup` with one
    // shared read whose result went nowhere, which the tests below caught by counting the slots.
    let steps = if want_total {
        subgroups
    } else {
        subgroups.saturating_sub(1)
    };

    for earlier in 0..steps {
        // Slot `k * width` is where subgroup `k` wrote its total. Every lane of that subgroup
        // wrote the same value to a different slot, so any one of them will do and this takes the
        // first — a constant index, which is what makes the read the same instruction in every
        // invocation.
        let theirs = kernel.load_shared(shared, earlier * subgroup)?;

        // **The block total takes every subgroup, the offset only the earlier ones.** That is the
        // whole difference between the two numbers, and it is why the block total is not simply
        // the last lane's `offset`: no lane's offset includes its own subgroup.
        if let Some(sum) = block {
            block = Some(kernel.module().binary(T::ADD, element, sum, theirs)?);
        }

        if earlier + 1 == subgroups {
            continue;
        }

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
    Ok((scanned, block))
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used)]

    use super::{scan_blocks, scan_workgroup, scan_workgroup_at};
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
    fn the_block_scan_reads_every_subgroups_total_and_selects_on_all_but_the_last() {
        // Two counts that pull in opposite directions, which is what makes the loop's shape
        // testable at all. The block total needs **every** subgroup's slot; the offset needs a
        // select for every subgroup **but the last**, because the last one is nobody's
        // predecessor.
        //
        // The gate found this: skipping the offset work on the final iteration is invisible in the
        // *answer* — the boundary would be `workgroup - 1` and no lane's index exceeds it, so the
        // select would pick the unchanged offset every time — but it is one comparison and one
        // select the module should not contain. A behavioural test cannot see it. This can.
        for width in [4_u32, 8, 16, 32, 64] {
            let words = scan_blocks::<F32>(width).expect("built");
            let subgroups = super::super::WORKGROUP_SIZE / width;

            let expected: Vec<u32> = (0..subgroups).map(|step| step * width).collect();
            assert_eq!(
                shared_slots_read(&words),
                expected,
                "the block total needs every subgroup, at width {width}"
            );
            assert_eq!(
                count(&words, op::SELECT),
                subgroups as usize - 1,
                "the last subgroup is nobody's predecessor, at width {width}"
            );
        }
    }

    #[test]
    fn the_plain_scan_reads_one_slot_fewer_than_the_block_scan() {
        // The other half of the same claim. `scan_workgroup` has nowhere to put a block total, so
        // it must not compute one — and the difference between the two kernels is exactly the one
        // shared read the total needs.
        for width in [4_u32, 8, 16, 32] {
            let plain = shared_slots_read(&scan_workgroup::<F32>(width).expect("built"));
            let blocks = shared_slots_read(&scan_blocks::<F32>(width).expect("built"));

            assert_eq!(
                plain.len() + 1,
                blocks.len(),
                "at width {width}: {plain:?} against {blocks:?}"
            );
        }
    }

    #[test]
    fn the_block_scan_writes_one_more_time_than_the_plain_one() {
        // The store at a runtime index is the point of the pass. A kernel that dropped it would
        // still scan correctly and leave the totals buffer holding whatever was in it.
        //
        // Two and three rather than one and two: the handover into shared memory is an `OpStore`
        // as well, and it is the first of them in both kernels.
        for width in [4_u32, 32, 64] {
            assert_eq!(
                count(&scan_workgroup::<F32>(width).expect("built"), op::STORE),
                2,
                "shared, then the scanned value"
            );
            assert_eq!(
                count(&scan_blocks::<F32>(width).expect("built"), op::STORE),
                3,
                "shared, the scanned value, then the block total"
            );
        }
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
