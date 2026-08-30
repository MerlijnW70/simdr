use super::{shape, whole_subgroup, whole_subgroup_of};
use simdr::kernel::Kernel;
use simdr::lanes::{Element, Integer, LaneError, Mapping};

pub fn lane_sum<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let total = kernel.lanes()?.reduce_sum(value)?;
    kernel.store_scalar(1, total)?;
    kernel.finish()
}

pub fn lane_product<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let product = kernel.lanes()?.reduce_product(value)?;
    kernel.store_scalar(1, product)?;
    kernel.finish()
}

pub fn lane_and<T: Integer, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let all = kernel.lanes()?.reduce_and(value)?;
    kernel.store_scalar(1, all)?;
    kernel.finish()
}

pub fn lane_or<T: Integer, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let any = kernel.lanes()?.reduce_or(value)?;
    kernel.store_scalar(1, any)?;
    kernel.finish()
}

pub fn lane_xor<T: Integer, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let parity = kernel.lanes()?.reduce_xor(value)?;
    kernel.store_scalar(1, parity)?;
    kernel.finish()
}

pub fn lane_max<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let largest = kernel.lanes()?.reduce_max(value)?;
    kernel.store_scalar(1, largest)?;
    kernel.finish()
}

fn butterfly_pair_sum_at<const LANES: u32>(
    subgroup: u32,
    mask: u32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let paired = {
        let mut lanes = kernel.lanes()?;
        let neighbour = lanes.butterfly(value, mask)?;
        lanes.add(value, neighbour)?
    };
    kernel.store(1, paired)?;
    kernel.finish()
}

fn fold_halves_at<const LANES: u32>(subgroup: u32, half: u32) -> Result<Vec<u32>, LaneError> {
    fold_by_at::<LANES>(subgroup, 2, half)
}

pub fn fold_by(subgroup: u32, factor: u32, stride: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, fold_by_at, factor, stride)
}

fn fold_by_at<const LANES: u32>(
    subgroup: u32,
    factor: u32,
    stride: u32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;

    let mut total = kernel.load::<LANES>(0)?;
    for step in 1..factor {
        let next = kernel.load_offset::<LANES>(0, step.saturating_mul(stride))?;
        total = kernel.lanes()?.add(total, next)?;
    }

    kernel.store(1, total)?;
    kernel.finish()
}

pub fn fold_halves_open(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, fold_halves_open_at)
}

fn fold_halves_open_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let element = kernel.index_type();
    let half = kernel
        .module()
        .spec_constant(element, 0, FOLD_HALF_SPEC_ID)?;

    let near = kernel.load::<LANES>(0)?;
    let far = kernel.load_offset_by::<LANES>(0, half)?;
    let folded = kernel.lanes()?.add(near, far)?;
    kernel.store(1, folded)?;
    kernel.finish()
}

pub const FOLD_HALF_SPEC_ID: u32 = 0;

pub fn dot_product<T: Element, const LANES: u32>(
    subgroup: u32,
    offset: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let left = kernel.load::<LANES>(0)?;
    let right = kernel.load_offset::<LANES>(0, offset)?;
    let total = {
        let mut lanes = kernel.lanes()?;
        let products = lanes.mul(left, right)?;
        lanes.reduce_sum(products)?
    };
    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// ```text
///   total = reduce_sum(value)     each lane holds its own subgroup's total
///   shared[local] = total         every invocation writes a different slot
///   barrier                       reached by all of them, so it is well defined
///   answer = shared[0] + shared[w] + …    constant indices, one per subgroup
/// ```
fn workgroup_sum_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
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
    let mine = kernel.lanes()?.reduce_sum(value)?;

    let shared = kernel.shared(workgroup)?;
    let slot = kernel.local_index();
    kernel.store_shared(shared, slot, mine)?;
    kernel.barrier()?;

    let mut total = kernel.load_shared(shared, 0)?;
    for index in 1..subgroups {
        let next = kernel.load_shared(shared, index * subgroup)?;
        let element = kernel.element();
        total = kernel.module().binary(T::ADD, element, total, next)?;
    }

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

fn butterfly_tree_sum_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let steps = subgroup.trailing_zeros();
    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;

    let total = {
        let mut lanes = kernel.lanes()?;
        lanes.repeat(steps, value.id(), |lanes, carried, step| {
            let held = lanes.from_lane_value::<F32, LANES>(carried)?;
            let partner = lanes.butterfly(held, 1 << step)?;
            Ok(lanes.add(held, partner)?.id())
        })?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

pub fn butterfly_cluster_sum(subgroup: u32, cluster: u32) -> Result<Vec<u32>, LaneError> {
    match cluster {
        1 => butterfly_cluster_sum_at::<1>(subgroup),
        2 => butterfly_cluster_sum_at::<2>(subgroup),
        4 => butterfly_cluster_sum_at::<4>(subgroup),
        8 => butterfly_cluster_sum_at::<8>(subgroup),
        16 => butterfly_cluster_sum_at::<16>(subgroup),
        32 => butterfly_cluster_sum_at::<32>(subgroup),
        lanes => Err(LaneError::NoMapping {
            lanes,
            width: subgroup,
        }),
    }
}

fn butterfly_cluster_sum_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    if !matches!(
        Mapping::of(LANES, subgroup),
        Ok(Mapping::WholeSubgroup | Mapping::Clusters { .. })
    ) {
        return Err(LaneError::NoMapping {
            lanes: LANES,
            width: subgroup,
        });
    }

    let steps = LANES.trailing_zeros();
    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;

    let total = {
        let mut lanes = kernel.lanes()?;
        lanes.repeat(steps, value.id(), |lanes, carried, step| {
            let held = lanes.from_lane_value::<F32, LANES>(carried)?;
            let partner = lanes.butterfly(held, 1 << step)?;
            Ok(lanes.add(held, partner)?.id())
        })?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

pub fn lane_sum_whole<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, lane_sum)
}

pub fn lane_product_whole<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, lane_product)
}

pub fn lane_and_whole<T: Integer>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, lane_and)
}

pub fn lane_or_whole<T: Integer>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, lane_or)
}

pub fn lane_xor_whole<T: Integer>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, lane_xor)
}

pub fn lane_max_whole<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, lane_max)
}

pub fn dot_product_whole<T: Element>(subgroup: u32, offset: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, dot_product, offset)
}

pub fn butterfly_pair_sum(subgroup: u32, mask: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, butterfly_pair_sum_at, mask)
}

pub fn fold_halves(subgroup: u32, half: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, fold_halves_at, half)
}

pub fn butterfly_tree_sum(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, butterfly_tree_sum_at)
}

pub fn workgroup_sum<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, workgroup_sum_at)
}

#[cfg(test)]
mod tests {
    use super::{butterfly_cluster_sum, workgroup_sum, workgroup_sum_at};
    use simdr::lanes::{F32, LaneError};

    #[test]
    fn a_cluster_the_subgroups_width_builds_and_a_wider_one_is_refused_by_name() {
        assert!(
            butterfly_cluster_sum(32, 32).is_ok(),
            "a cluster exactly the subgroup's width is a whole-subgroup vector, not a missing \
             mapping"
        );

        assert_eq!(
            butterfly_cluster_sum(16, 32).err(),
            Some(LaneError::NoMapping {
                lanes: 32,
                width: 16
            }),
            "a cluster wider than the subgroup is refused here and names both numbers"
        );
    }

    #[test]
    fn a_subgroup_the_workgroup_does_not_divide_by_is_refused() {
        for width in [24_u32, 48, 63] {
            assert!(
                matches!(
                    workgroup_sum_at::<F32, 32>(width),
                    Err(LaneError::BadShape { .. })
                ),
                "a subgroup of {width} was accepted"
            );
        }
    }

    #[test]
    fn a_subgroup_of_zero_is_refused_before_it_divides() {
        assert!(matches!(
            workgroup_sum_at::<F32, 32>(0),
            Err(LaneError::BadShape { .. })
        ));
    }

    #[test]
    fn a_width_that_is_no_devices_subgroup_is_refused_by_the_dispatcher() {
        for width in [0_u32, 24, 48, 128] {
            assert!(
                matches!(workgroup_sum::<F32>(width), Err(LaneError::BadWidth { .. })),
                "a subgroup of {width} was accepted"
            );
        }
    }

    #[test]
    fn the_widths_a_device_reports_are_accepted() {
        for width in [32_u32, 64] {
            assert!(
                workgroup_sum::<F32>(width).is_ok(),
                "a subgroup of {width} was refused"
            );
        }
    }
}
