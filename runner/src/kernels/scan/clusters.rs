//! The clustered scan: a prefix within each of the vectors packed into one subgroup.
//!
//! `Simd<f32, 8>` on a 32-wide subgroup is four independent vectors sharing the hardware, and a
//! scan has to stop at each one's edge. SPIR-V has a `ClusteredReduce` and **no clustered scan**,
//! so this is the one mapping that costs a ladder rather than an instruction.
//!
//! **The ladder is not here any more.** It was written twice over — once as a kernel, because
//! `Lanes::prefix_sum` refused a clustered vector, and the refusal was the thing to fix rather than
//! to work around. What is left in this file is the kernel that *runs* the lane API's third
//! mapping: a load, a scan and a store, with the cluster width in the type. The ladder itself, and
//! the argument for why it is a ladder rather than a subtraction, are in `simdr`'s `lanes::reduce`.
//!
//! Split from [`super`] because it shares nothing with what is there: the workgroup scan and the
//! block composition both build on [`super::scanned_at`], and this builds on the mapping.

use super::Scan;
use crate::kernels::shape;
use simdr::kernel::Kernel;
use simdr::lanes::LaneError;

/// A prefix sum **within each cluster** of `cluster` adjacent lanes.
///
/// The third mapping. `Simd<f32, 8>` on a 32-wide subgroup is four independent vectors packed into
/// it, and each one is scanned on its own — which is what `Lanes::prefix_sum` emits a
/// `log2(cluster)`-step ladder for, there being no clustered scan instruction to ask for.
///
/// **The cluster is the vector's own width, and it is in the type.** That is the whole of what this
/// kernel says: `load::<8>` on a 32-wide device *is* the clustered mapping, and the same source at
/// width 8 is a whole-subgroup scan and one instruction. Which of the three a caller gets is the
/// mapping's decision rather than this kernel's.
///
/// # Errors
///
/// [`LaneError::NoMapping`] if `cluster` is not a power of two this can build for, or if it does
/// not divide the device's width — a cluster *wider* than the subgroup is a strip-mined vector,
/// which reads more elements per invocation than a caller of this kernel has bound. Otherwise
/// whatever stopped the module being built.
pub fn scan_clusters(subgroup: u32, cluster: u32) -> Result<Vec<u32>, LaneError> {
    clustered(subgroup, cluster, Scan::Inclusive)
}

/// The same, with each lane's own element left out: the first lane of each cluster gets zero.
///
/// The form that cannot be recovered by subtraction, and the reason the ladder ends in a shuffle
/// rather than in arithmetic. Worth running on a device rather than reasoning about, because the
/// inclusive and exclusive answers agree at exactly one element of each cluster — the last one —
/// and disagree everywhere else.
///
/// # Errors
///
/// As [`scan_clusters`].
pub fn scan_clusters_exclusive(subgroup: u32, cluster: u32) -> Result<Vec<u32>, LaneError> {
    clustered(subgroup, cluster, Scan::Exclusive)
}

/// Both forms, at a cluster width that has to be a constant.
///
/// The list is the powers of two a subgroup can be cut into, and it stops at the largest width any
/// implementation reports. A cluster that is not on it has no mapping, which is what
/// `Lanes::mapping` would say about it anyway — said here, where the number is still a number.
fn clustered(subgroup: u32, cluster: u32, kind: Scan) -> Result<Vec<u32>, LaneError> {
    match cluster {
        1 => scan_clusters_at::<1>(subgroup, kind),
        2 => scan_clusters_at::<2>(subgroup, kind),
        4 => scan_clusters_at::<4>(subgroup, kind),
        8 => scan_clusters_at::<8>(subgroup, kind),
        16 => scan_clusters_at::<16>(subgroup, kind),
        32 => scan_clusters_at::<32>(subgroup, kind),
        64 => scan_clusters_at::<64>(subgroup, kind),
        lanes => Err(LaneError::NoMapping {
            lanes,
            width: subgroup,
        }),
    }
}

/// The builder for both, at a known cluster width.
fn scan_clusters_at<const CLUSTER: u32>(subgroup: u32, kind: Scan) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    // A cluster wider than the subgroup is the strip-mined mapping, and it would read `CLUSTER /
    // width` elements per invocation from a buffer bound for one each — the `kernels::scale` bug,
    // which cost a day of green tests at width 8 and an access violation at 4.
    if CLUSTER > subgroup {
        return Err(LaneError::NoMapping {
            lanes: CLUSTER,
            width: subgroup,
        });
    }

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<CLUSTER>(0)?;
    let scanned = match kind {
        Scan::Inclusive => kernel.lanes()?.prefix_sum(value)?,
        Scan::Exclusive => kernel.lanes()?.prefix_sum_exclusive(value)?,
    };
    kernel.store(1, scanned)?;
    kernel.finish()
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used)]

    use super::{scan_clusters, scan_clusters_exclusive};
    use simdr::decode;
    use simdr::lanes::LaneError;
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
    fn a_cluster_as_wide_as_the_subgroup_is_the_instruction_and_not_the_ladder() {
        // The mapping decides, and this is the boundary it decides at: 32 lanes of a 32-wide
        // subgroup is a whole-subgroup scan, which is one `InclusiveScan` and no ladder at all.
        let words = scan_clusters(32, 32).expect("built");

        assert_eq!(count(&words, op::SELECT), 0);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_ADD), 1);
    }

    #[test]
    fn a_cluster_wider_than_the_subgroup_is_refused_rather_than_strip_mined() {
        // It would build — a wider vector is the strip-mined mapping and `prefix_sum` has carried
        // a running total between strips for a week. It would also read four elements per
        // invocation out of a buffer bound for one each.
        assert_eq!(
            scan_clusters(8, 32).err(),
            Some(LaneError::NoMapping {
                lanes: 32,
                width: 8
            })
        );
    }

    #[test]
    fn a_cluster_that_is_not_a_power_of_two_has_no_mapping() {
        assert_eq!(
            scan_clusters(32, 12).err(),
            Some(LaneError::NoMapping {
                lanes: 12,
                width: 32
            })
        );
    }

    #[test]
    fn the_exclusive_form_shifts_the_ladders_answer_by_one_lane() {
        // One shuffle and one select more than the inclusive form, and nothing else: the exclusive
        // answer is the inclusive one moved a lane up with the identity where the cluster starts.
        let inclusive = scan_clusters(32, 8).expect("built");
        let exclusive = scan_clusters_exclusive(32, 8).expect("built");

        assert_eq!(
            count(&exclusive, op::GROUP_NON_UNIFORM_SHUFFLE_UP),
            count(&inclusive, op::GROUP_NON_UNIFORM_SHUFFLE_UP) + 1
        );
        assert_eq!(
            count(&exclusive, op::SELECT),
            count(&inclusive, op::SELECT) + 1
        );
        assert_eq!(
            count(&exclusive, op::F_ADD),
            count(&inclusive, op::F_ADD),
            "and the same arithmetic, because nothing is subtracted back off"
        );
    }
}
