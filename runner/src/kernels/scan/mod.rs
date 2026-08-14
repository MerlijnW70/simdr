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
//! # What is here and what is next door
//!
//! This file is the scan of **one workgroup** and the arithmetic every scan shares. `blocks.rs` is
//! what a longer input needs — the per-block kernels and the offset addition that pays each block
//! what it owes. They were one file until it reached 639 lines holding two jobs at two scales.
//!
//! # What it does not do
//!
//! **One workgroup.** This scans [`super::WORKGROUP_SIZE`] elements and no more; a longer input
//! needs the block totals scanned and added back, which is a second and third dispatch and is not
//! built. The limit is in the name of the function rather than hidden in its behaviour, because a
//! scan that silently restarted at every block boundary would return plausible numbers.

mod blocks;

pub use blocks::{add_offsets, scan_blocks, scan_blocks_exclusive};

use super::{shape, whole_subgroup, whole_subgroup_of};
use simdr::kernel::Kernel;
use simdr::lanes::{Element, LaneError};
use simdr::module::{Id, op};

/// Which of the two scans a kernel is built for.
///
/// They differ in one literal — the group operation — and in nothing else, which is why they share
/// a builder. What they are *for* differs completely: the inclusive form is the answer a caller
/// asked for, and the exclusive form is what a block owes the blocks before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Scan {
    /// Element `i` includes element `i`.
    Inclusive,
    /// Element `i` includes everything before `i` and not `i` itself, so element 0 is zero.
    Exclusive,
}

/// `out[i] = in[0] + in[1] + … + in[i]`, within one workgroup.
///
/// Inclusive: element `i` of the output includes element `i` of the input.
/// [`scan_workgroup_exclusive`] is the other direction.
///
/// This used to say the exclusive form was not built and that a caller could subtract their own
/// element instead. **Both halves were wrong.** It is built, and the subtraction is the thing it
/// exists to avoid: over floats it takes a large running total back off itself and loses precisely
/// the low bits the scan had just accumulated, which is why SPIR-V has a separate group operation
/// for it rather than leaving it to arithmetic.
///
/// # Errors
///
/// [`LaneError::BadWidth`] if `subgroup` is not a width this can build for, [`LaneError::BadShape`]
/// if the workgroup is not a whole number of subgroups, otherwise if the module cannot be built.
pub fn scan_workgroup<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, scan_workgroup_at)
}

/// A prefix sum **within each invocation's own vector**, for a vector wider than the subgroup.
///
/// The strip-mined mapping, which `Lanes::prefix_sum` refused until it could carry a running total
/// between strips. `LANES` elements per subgroup rather than one each: lane `l` holds the elements
/// at `l`, `l + width`, `l + 2·width`, and the answer at vector position `j` is the sum of
/// positions `0..=j` of *that subgroup's* vector.
///
/// **Not the same thing as [`scan_workgroup`].** This scans each subgroup's vector on its own and
/// does not cross between subgroups; it is the lane mapping under test rather than a whole
/// algorithm. A workgroup-wide scan of a strip-mined load would need both, and nothing wants that
/// yet.
///
/// # Errors
///
/// As [`scan_workgroup`], and [`LaneError::NoSuchForm`] if `LANES` is *narrower* than the subgroup
/// — SPIR-V has no clustered scan.
pub fn scan_strips<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<simdr::lanes::F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let scanned = kernel.lanes()?.prefix_sum(value)?;
    kernel.store(1, scanned)?;
    kernel.finish()
}

/// A prefix sum **within each cluster** of `cluster` adjacent lanes.
///
/// The third mapping, and the one SPIR-V has no single instruction for: there is a
/// `ClusteredReduce` and no clustered scan. `Simd<f32, 8>` on a 32-wide subgroup is four
/// independent vectors packed into it, and a scan has to stop at each one's edge.
///
/// Built as a Hillis-Steele ladder — `log2(cluster)` steps, each adding the element `d` lanes
/// below and masking off the lanes whose neighbour `d` below belongs to a different cluster:
///
/// ```text
///   for d in 1, 2, 4, … < cluster:
///       value += (local_index % cluster) > d - 1  ?  the value d lanes below  :  nothing
/// ```
///
/// **Exact, and that is why it is a ladder rather than a subtraction.** The three-instruction
/// alternative is a subgroup-wide scan minus each cluster's starting offset, which in floating
/// point takes a large running total back off itself and loses precisely the low bits the scan just
/// accumulated — the same reason [`scan_workgroup_exclusive`] exists rather than a subtraction.
///
/// # Why the vector is the whole subgroup
///
/// The value is loaded as `Simd<f32, width>` and not `Simd<f32, cluster>`, even though what is
/// computed is the narrower vector's scan. `Lanes::shift_up` refuses a clustered vector on purpose
/// — for every other caller, a shuffle reaching into the lanes of a *different* packed vector is a
/// bug, and that refusal is worth more than this one caller's convenience. Here crossing the
/// boundary is deliberate and the mask is what undoes it, so the ladder runs on the subgroup the
/// hardware actually has and the clusters exist only in the comparison.
///
/// That is also why this is a kernel rather than `Lanes::prefix_sum` learning a third mapping: to
/// move it there, `Lanes` would need the invocation's index within its subgroup, and it has no way
/// to reach one — see `notes/NEXT.md`.
///
/// # Errors
///
/// [`LaneError::BadWidth`] if `subgroup` is not a width this can build for, otherwise if the module
/// cannot be built. `cluster` is not checked against the width: a cluster as wide as the subgroup
/// is a whole-subgroup scan and a wider one simply never masks, both of which are answers.
pub fn scan_clusters(subgroup: u32, cluster: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, scan_clusters_at, cluster)
}

/// The builder for [`scan_clusters`], at the device's own width.
fn scan_clusters_at<const LANES: u32>(subgroup: u32, cluster: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::{F32, U32};

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let mut value = kernel.load::<LANES>(0)?;

    // Where this invocation sits inside its cluster. `cluster` is a power of two, so the remainder
    // is a mask — and it is the *cluster* position rather than the subgroup lane, because that is
    // what decides whether the neighbour `d` below is a neighbour at all.
    let index = kernel.index_type();
    let local = kernel.local_index();
    let wrap = kernel.module().constant_u32(cluster.saturating_sub(1))?;
    let within = kernel
        .module()
        .binary(op::BITWISE_AND, index, local, wrap)?;

    let mut distance = 1;
    while distance < cluster {
        let mut lanes = kernel.lanes()?;

        let below = lanes.shift_up(value, distance)?;
        let raised = lanes.add(value, below)?;

        // `> distance - 1` rather than `>= distance`: unsigned greater-than is the comparison the
        // lane API has, and over integers the two say the same thing.
        let position = lanes.from_lane_value::<U32, LANES>(within)?;
        let edge = lanes.splat_bits::<U32, LANES>(distance - 1)?;
        let inside = lanes.greater_than(position, edge)?;

        // Both arms are computed and one is discarded. That is what makes this safe as well as
        // branch-free: `shift_up` leaves the bottom `distance` lanes of the *subgroup* undefined,
        // and the mask is what stops either that or a neighbouring cluster reaching the answer.
        value = lanes.select(inside, raised, value)?;
        distance *= 2;
    }

    kernel.store(1, value)?;
    kernel.finish()
}

/// The exclusive scan of one workgroup — `out[i] = in[0] + … + in[i-1]`, and `out[0] = 0`.
///
/// The top of a long scan. Once the block totals have been reduced to no more than
/// [`super::WORKGROUP_SIZE`] of them, one workgroup scans them and the recursion stops; what comes
/// out is the offset each block at the level below owes.
///
/// # Errors
///
/// As [`scan_workgroup`].
pub fn scan_workgroup_exclusive<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, scan_workgroup_exclusive_at)
}

/// The builder for [`scan_workgroup_exclusive`].
fn scan_workgroup_exclusive_at<T: Element, const LANES: u32>(
    subgroup: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let (scanned, _) = scanned_at::<T, LANES>(&mut kernel, subgroup, Scan::Exclusive, false)?;

    kernel.store_scalar(1, scanned)?;
    kernel.finish()
}

/// The builder, at a lane count that has to equal the subgroup width.
///
/// `LANES` equals `subgroup` here because that is what [`whole_subgroup_of`] arranges, not because
/// the other mappings are unavailable — this line used to say a strip-mined scan "is not built",
/// and `Lanes::prefix_sum` has carried a running total between strips since. `scan_strips` is the
/// kernel that uses it, and `scan_clusters` is the third mapping.
///
/// What is still refused is a *clustered* vector through `Lanes::prefix_sum`, and that is a
/// question of where the ladder lives rather than whether it works — see `notes/NEXT.md`.
fn scan_workgroup_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let (scanned, _) = scanned_at::<T, LANES>(&mut kernel, subgroup, Scan::Inclusive, false)?;

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
pub(super) fn scanned_at<T: Element, const LANES: u32>(
    kernel: &mut Kernel<T>,
    subgroup: u32,
    kind: Scan,
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
    let running = match kind {
        Scan::Inclusive => kernel.lanes()?.prefix_sum(value)?,
        Scan::Exclusive => kernel.lanes()?.prefix_sum_exclusive(value)?,
    };
    // **The subgroup's total comes from a reduce either way.** An exclusive scan does not hand any
    // lane the whole subgroup's sum — that is the one value it leaves out — so it cannot be read
    // off the scan, and taking the last lane's exclusive result would be short by that lane.
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

    use super::{
        scan_blocks, scan_blocks_exclusive, scan_clusters, scan_workgroup, scan_workgroup_at,
        scan_workgroup_exclusive,
    };
    use simdr::decode;
    use simdr::lanes::{F32, LaneError};
    use simdr::module::op;
    use simdr::spec::GroupOperation;
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
    fn a_scan_with_nowhere_to_put_a_block_total_does_not_compute_one() {
        // The other half of the same claim, for **both** kernels that have no totals binding. A
        // block total costs one shared read and one addition per subgroup, and a kernel that
        // computed one and stored it nowhere would return the right answer from a module saying
        // it did more work than it does. The gate finds exactly that by flipping the flag, so the
        // difference has to be visible here.
        for width in [4_u32, 8, 16, 32] {
            let with_totals = shared_slots_read(&scan_blocks::<F32>(width).expect("built"));

            for (name, words) in [
                ("inclusive", scan_workgroup::<F32>(width).expect("built")),
                (
                    "exclusive",
                    scan_workgroup_exclusive::<F32>(width).expect("built"),
                ),
            ] {
                let plain = shared_slots_read(&words);
                assert_eq!(
                    plain.len() + 1,
                    with_totals.len(),
                    "the {name} scan at width {width}: {plain:?} against {with_totals:?}"
                );
            }
        }
    }

    #[test]
    fn the_exclusive_kernels_name_the_exclusive_group_operation() {
        // What distinguishes the pairs. Both members of each pair emit the same instructions in
        // the same order, so the only thing saying which scan a module runs is this literal — and
        // a builder that ignored its `Scan` argument would pass every other test in this file.
        for width in [4_u32, 32, 64] {
            for (words, wanted) in [
                (
                    scan_workgroup::<F32>(width).expect("built"),
                    GroupOperation::InclusiveScan,
                ),
                (
                    scan_workgroup_exclusive::<F32>(width).expect("built"),
                    GroupOperation::ExclusiveScan,
                ),
                (
                    scan_blocks::<F32>(width).expect("built"),
                    GroupOperation::InclusiveScan,
                ),
                (
                    scan_blocks_exclusive::<F32>(width).expect("built"),
                    GroupOperation::ExclusiveScan,
                ),
            ] {
                // The scan is the first of the two group instructions; the second is the reduce
                // that produces the subgroup's total, and it is `Reduce` in every one of them.
                let operations: Vec<u32> = decode::body(&words)
                    .filter(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
                    .filter_map(|instruction| instruction.operands().get(3).copied())
                    .collect();

                assert_eq!(
                    operations.first().copied(),
                    Some(wanted.word()),
                    "at width {width}"
                );
                assert!(
                    operations
                        .iter()
                        .skip(1)
                        .all(|&operation| operation == GroupOperation::Reduce.word()),
                    "the total comes from a reduce, at width {width}"
                );
            }
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
    fn the_clustered_ladder_is_one_step_per_doubling_and_no_more() {
        // `log2(cluster)` steps, each a shuffle, a comparison and a select. The gate found the
        // loop bound flipped to `<=`, which is invisible in the *answer* — the extra step's mask
        // asks whether a lane's position exceeds `cluster - 1`, and no position does — and is one
        // shuffle, one compare and one select the module should not contain.
        for (cluster, steps) in [(1_u32, 0_usize), (2, 1), (4, 2), (8, 3), (16, 4)] {
            let words = scan_clusters(32, cluster).expect("built");

            assert_eq!(
                count(&words, op::SELECT),
                steps,
                "a cluster of {cluster} doubles {steps} times"
            );
            assert_eq!(
                count(&words, op::GROUP_NON_UNIFORM_SHUFFLE_UP),
                steps,
                "one shuffle per step, at cluster {cluster}"
            );
        }
    }

    #[test]
    fn a_cluster_of_one_scans_nothing_and_emits_no_ladder() {
        // The degenerate end, which the loop has to reach rather than divide by. A single-lane
        // cluster's prefix sum is the element itself, so the right number of steps is none.
        let words = scan_clusters(32, 1).expect("built");

        assert_eq!(count(&words, op::SELECT), 0);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE_UP), 0);
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
