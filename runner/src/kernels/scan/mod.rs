//! ```text
//!   running = prefix_sum(value)     inclusive, within this invocation's subgroup
//!   total   = reduce_sum(value)     this subgroup's whole total, in every one of its lanes
//!   shared[local_index] = total     every invocation writes its own slot
//!   barrier
//!   offset  = sum of the totals of the subgroups before mine
//!   out[i]  = running + offset
//! ```
//! ```text
//!   for each earlier subgroup k:
//!       offset = local_index > (k+1)*width - 1  ?  offset + shared[k*width]  :  offset
//! ```

mod blocks;
mod clusters;

pub use blocks::{add_offsets, scan_blocks, scan_blocks_exclusive};
pub use clusters::{scan_clusters, scan_clusters_exclusive};

use super::{shape, whole_subgroup_of};
use simdr::kernel::Kernel;
use simdr::lanes::{Element, LaneError};
use simdr::module::{Id, op};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Scan {
    Inclusive,
    Exclusive,
}

pub fn scan_workgroup<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, scan_workgroup_at)
}

pub fn scan_strips<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<simdr::lanes::F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let scanned = kernel.lanes()?.prefix_sum(value)?;
    kernel.store(1, scanned)?;
    kernel.finish()
}

pub fn scan_workgroup_exclusive<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, scan_workgroup_exclusive_at)
}

fn scan_workgroup_exclusive_at<T: Element, const LANES: u32>(
    subgroup: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let scanned = scanned_at::<T, LANES>(&mut kernel, Scan::Exclusive, None)?;

    kernel.store_scalar(1, scanned)?;
    kernel.finish()
}

fn scan_workgroup_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let scanned = scanned_at::<T, LANES>(&mut kernel, Scan::Inclusive, None)?;

    kernel.store_scalar(1, scanned)?;
    kernel.finish()
}

pub(super) fn scanned_at<T: Element, const LANES: u32>(
    kernel: &mut Kernel<T>,
    kind: Scan,
    total_to: Option<u32>,
) -> Result<Id, LaneError> {
    let workgroup = super::WORKGROUP_SIZE;
    let subgroup = kernel.shape().subgroup;

    if !workgroup.is_multiple_of(subgroup) {
        return Err(LaneError::NoSuchForm {
            operation: "scan_workgroup",
            because: "the cross-subgroup combine is one step per subgroup, and a workgroup that \
                      is not a whole number of subgroups has no such count",
        });
    }
    let subgroups = workgroup / subgroup;

    let value = kernel.load::<LANES>(0)?;

    let running = match kind {
        Scan::Inclusive => kernel.lanes()?.prefix_sum(value)?,
        Scan::Exclusive => kernel.lanes()?.prefix_sum_exclusive(value)?,
    };
    let total = kernel.lanes()?.reduce_sum(value)?;

    let shared = kernel.shared(workgroup)?;
    let slot = kernel.local_index();
    kernel.store_shared(shared, slot, total)?;
    kernel.barrier()?;

    let element = kernel.element();
    let zero = kernel.module().constant_scalar(element, 0)?;
    let mut offset = zero;
    let mut block: Option<(u32, Id)> = total_to.map(|binding| (binding, zero));

    let steps = if block.is_some() {
        subgroups
    } else {
        subgroups.saturating_sub(1)
    };

    for earlier in 0..steps {
        let theirs = kernel.load_shared(shared, earlier * subgroup)?;

        if let Some((binding, sum)) = block {
            let raised = kernel.module().binary(T::ADD, element, sum, theirs)?;
            block = Some((binding, raised));
        }

        if earlier + 1 == subgroups {
            continue;
        }

        let boundary = kernel.module().constant_u32((earlier + 1) * subgroup - 1)?;
        let boolean = kernel.module().type_bool()?;
        let after = kernel
            .module()
            .binary(op::U_GREATER_THAN, boolean, slot, boundary)?;

        let with = kernel.module().binary(T::ADD, element, offset, theirs)?;
        offset = kernel.module().select(element, after, with, offset)?;
    }

    if let Some((binding, total)) = block {
        let at = kernel.workgroup_index();
        kernel.store_at(binding, at, total)?;
    }

    Ok(kernel
        .module()
        .binary(T::ADD, element, running.id(), offset)?)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::{
        scan_blocks, scan_blocks_exclusive, scan_workgroup, scan_workgroup_at,
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

    fn shared_slots_read(words: &[u32]) -> Vec<u32> {
        let values: HashMap<u32, u32> = decode::body(words)
            .filter(|instruction| instruction.opcode() == op::CONSTANT)
            .filter_map(|instruction| match instruction.operands() {
                [_type, id, literal] => Some((*id, *literal)),
                _ => None,
            })
            .collect();

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
        let words = scan_workgroup::<F32>(64).expect("built");
        assert_eq!(count(&words, op::SELECT), 0);
    }

    #[test]
    fn the_select_count_is_one_fewer_than_the_subgroups() {
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
        let words = scan_workgroup::<F32>(4).expect("built");
        let expected: Vec<u32> = (0..15).map(|step| step * 4).collect();

        assert_eq!(shared_slots_read(&words), expected);
    }

    #[test]
    fn the_slots_are_spaced_by_the_width_at_every_width() {
        for width in [4_u32, 8, 16, 32, 64] {
            let words = scan_workgroup::<F32>(width).expect("built");
            let subgroups = super::super::WORKGROUP_SIZE / width;
            let expected: Vec<u32> = (0..subgroups - 1).map(|step| step * width).collect();

            assert_eq!(shared_slots_read(&words), expected, "at width {width}");
        }
    }

    #[test]
    fn a_width_that_is_not_a_power_of_two_never_reaches_the_scan() {
        for width in [0_u32, 24, 48, 63] {
            assert!(
                matches!(
                    scan_workgroup_at::<F32, 32>(width),
                    Err(LaneError::BadWidth { width: refused }) if refused == width
                ),
                "a subgroup of {width} was accepted"
            );
        }
    }

    #[test]
    fn a_workgroup_that_is_not_a_whole_number_of_subgroups_is_refused() {
        for width in [128_u32, 256] {
            assert!(
                matches!(
                    scan_workgroup_at::<F32, 32>(width),
                    Err(LaneError::NoSuchForm {
                        operation: "scan_workgroup",
                        ..
                    })
                ),
                "a subgroup of {width} against a workgroup of {} was accepted",
                super::super::WORKGROUP_SIZE
            );
        }
    }

    #[test]
    fn the_block_scan_reads_every_subgroups_total_and_selects_on_all_but_the_last() {
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
        let words = scan_workgroup::<F32>(32).expect("built");
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_ADD), 2);
    }

    #[test]
    fn there_is_exactly_one_barrier_and_it_is_not_inside_anything() {
        let words = scan_workgroup::<F32>(4).expect("built");
        assert_eq!(count(&words, op::CONTROL_BARRIER), 1);
        assert_eq!(
            count(&words, op::BRANCH_CONDITIONAL),
            0,
            "a conditional branch would put the barrier in some lanes' path and not others"
        );
    }
}
