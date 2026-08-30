use super::Scan;
use crate::kernels::shape;
use simdr::kernel::Kernel;
use simdr::lanes::LaneError;

pub fn scan_clusters(subgroup: u32, cluster: u32) -> Result<Vec<u32>, LaneError> {
    clustered(subgroup, cluster, Scan::Inclusive)
}

pub fn scan_clusters_exclusive(subgroup: u32, cluster: u32) -> Result<Vec<u32>, LaneError> {
    clustered(subgroup, cluster, Scan::Exclusive)
}

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

fn scan_clusters_at<const CLUSTER: u32>(subgroup: u32, kind: Scan) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

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
        let words = scan_clusters(32, 32).expect("built");

        assert_eq!(count(&words, op::SELECT), 0);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_ADD), 1);
    }

    #[test]
    fn a_cluster_wider_than_the_subgroup_is_refused_rather_than_strip_mined() {
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
