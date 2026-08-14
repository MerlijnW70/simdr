//! The clustered scan: a prefix within each of the vectors packed into one subgroup.
//!
//! `Simd<f32, 8>` on a 32-wide subgroup is four independent vectors sharing the hardware, and a
//! scan has to stop at each one's edge. SPIR-V has a `ClusteredReduce` and **no clustered scan**,
//! so this is the one mapping that costs a ladder rather than an instruction.
//!
//! Split from [`super`] because it shares nothing with what is there. The workgroup scan and the
//! block composition both build on [`super::scanned_at`]; this builds on a shuffle and a mask, and
//! it was what put that file back over 650 lines the day after it was split.

use crate::kernels::{shape, whole_subgroup};
use simdr::kernel::Kernel;
use simdr::lanes::LaneError;
use simdr::module::op;

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

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used)]

    use super::scan_clusters;
    use simdr::decode;
    use simdr::module::op;

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
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
}
