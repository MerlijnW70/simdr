use super::{shape, whole_subgroup, whole_subgroup_of};
use simdr::kernel::Kernel;
use simdr::lanes::{Element, LaneError};

fn prefix_sum_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let running = kernel.lanes()?.prefix_sum(value)?;
    kernel.store(1, running)?;
    kernel.finish()
}

fn broadcast_at<T: Element, const LANES: u32>(
    subgroup: u32,
    source: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let shared = kernel.lanes()?.broadcast(value, source)?;
    kernel.store(1, shared)?;
    kernel.finish()
}

fn shift_down_at<T: Element, const LANES: u32>(
    subgroup: u32,
    delta: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let shifted = kernel.lanes()?.shift_down(value, delta)?;
    kernel.store(1, shifted)?;
    kernel.finish()
}

fn shift_up_at<T: Element, const LANES: u32>(
    subgroup: u32,
    delta: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let shifted = kernel.lanes()?.shift_up(value, delta)?;
    kernel.store(1, shifted)?;
    kernel.finish()
}

pub fn lane_min<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let smallest = kernel.lanes()?.reduce_min(value)?;
    kernel.store_scalar(1, smallest)?;
    kernel.finish()
}

fn all_above_at<const LANES: u32>(subgroup: u32, threshold: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let answer = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<U32, LANES>(threshold)?;
        let above = lanes.greater_than(value, limit)?;
        let verdict = lanes.all(above)?;

        let yes = lanes.splat_bits::<U32, LANES>(1)?;
        let no = lanes.splat_bits::<U32, LANES>(0)?;
        let element = lanes.type_of::<U32>()?;
        lanes.module().select(element, verdict, yes.id(), no.id())?
    };
    kernel.store_scalar(1, answer)?;
    kernel.finish()
}

fn subgroup_agrees_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;

    let zero = kernel.module().constant_u32(0)?;
    let one = kernel.module().constant_u32(1)?;
    let slot = kernel.local_index();
    let out = kernel.element_pointer_to(1, slot)?;

    kernel.module().store(out, zero)?;

    let agreed = kernel.lanes()?.all_equal_uniform(value)?;
    kernel
        .lanes()?
        .if_uniform(agreed, |lanes| Ok(lanes.module().store(out, one)?))?;

    kernel.finish()
}

pub fn subgroup_agrees_wide<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;

    let zero = kernel.module().constant_u32(0)?;
    let one = kernel.module().constant_u32(1)?;
    let slot = kernel.local_index();
    let out = kernel.element_pointer_to(1, slot)?;
    kernel.module().store(out, zero)?;

    let agreed = kernel.lanes()?.all_equal_uniform(value)?;
    kernel
        .lanes()?
        .if_uniform(agreed, |lanes| Ok(lanes.module().store(out, one)?))?;

    kernel.finish()
}

pub fn rotate_in_cluster(subgroup: u32, cluster: u32, delta: u32) -> Result<Vec<u32>, LaneError> {
    fn build<const LANES: u32>(subgroup: u32, delta: u32) -> Result<Vec<u32>, LaneError> {
        use simdr::lanes::U32;

        let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
        let value = kernel.load::<LANES>(0)?;
        let rotated = kernel.lanes()?.rotate_up(value, delta)?;
        kernel.store(1, rotated)?;
        kernel.finish()
    }

    match cluster {
        2 => build::<2>(subgroup, delta),
        4 => build::<4>(subgroup, delta),
        8 => build::<8>(subgroup, delta),
        16 => build::<16>(subgroup, delta),
        32 => build::<32>(subgroup, delta),
        64 => build::<64>(subgroup, delta),
        lanes => Err(LaneError::NoMapping {
            lanes,
            width: subgroup,
        }),
    }
}

fn equals_at<const LANES: u32>(subgroup: u32, wanted: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let flag = {
        let mut lanes = kernel.lanes()?;
        let target = lanes.splat_bits::<U32, LANES>(wanted)?;
        let same = lanes.equal(value, target)?;
        let yes = lanes.splat_bits::<U32, LANES>(1)?;
        let no = lanes.splat_bits::<U32, LANES>(0)?;
        lanes.select(same, yes, no)?
    };
    kernel.store(1, flag)?;
    kernel.finish()
}

pub fn equals(subgroup: u32, wanted: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, equals_at, wanted)
}

pub fn subgroup_agrees(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, subgroup_agrees_at)
}

fn ballot_above_at<const LANES: u32>(subgroup: u32, threshold: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let low = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<U32, LANES>(threshold)?;
        let above = lanes.greater_than(value, limit)?;
        let mask = lanes.ballot(above)?;

        let uint = lanes.type_of::<U32>()?;
        lanes.module().composite_extract(uint, mask, &[0])?
    };
    kernel.store_scalar(1, low)?;
    kernel.finish()
}

pub fn all_above(subgroup: u32, threshold: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, all_above_at, threshold)
}

pub fn ballot_above(subgroup: u32, threshold: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, ballot_above_at, threshold)
}

pub fn prefix_sum<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, prefix_sum_at)
}

pub fn broadcast<T: Element>(subgroup: u32, source: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, broadcast_at, source)
}

pub fn broadcast_in_cluster<T: Element>(
    subgroup: u32,
    cluster: u32,
    source: u32,
) -> Result<Vec<u32>, LaneError> {
    match cluster {
        1 => broadcast_at::<T, 1>(subgroup, source),
        2 => broadcast_at::<T, 2>(subgroup, source),
        4 => broadcast_at::<T, 4>(subgroup, source),
        8 => broadcast_at::<T, 8>(subgroup, source),
        16 => broadcast_at::<T, 16>(subgroup, source),
        32 => broadcast_at::<T, 32>(subgroup, source),
        lanes => Err(LaneError::NoMapping {
            lanes,
            width: subgroup,
        }),
    }
}

pub fn shift_down<T: Element>(subgroup: u32, delta: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, shift_down_at, delta)
}

pub fn shift_up<T: Element>(subgroup: u32, delta: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, shift_up_at, delta)
}

fn centre_and_scale_at<const LANES: u32>(
    subgroup: u32,
    centre: f32,
    scale: f32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let element = kernel.element();
    let value = kernel.load::<LANES>(0)?.id();

    let middle = F32::constant_from_bits(kernel.module(), centre.to_bits())?;
    let divisor = F32::constant_from_bits(kernel.module(), scale.to_bits())?;

    let centred = kernel.module().f_sub(element, value, middle)?;
    let scaled = kernel.module().f_div(element, centred, divisor)?;
    let flipped = kernel.module().f_negate(element, scaled)?;

    kernel.store_scalar(1, flipped)?;
    kernel.finish()
}

fn remainder_at<const LANES: u32>(subgroup: u32, divisor: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let element = kernel.element();
    let value = kernel.load::<LANES>(0)?.id();

    let by = kernel.module().constant_u32(divisor)?;
    let whole = kernel.module().u_div(element, value, by)?;
    let back = kernel.module().i_mul(element, whole, by)?;
    let left = kernel.module().i_sub(element, value, back)?;

    kernel.store_scalar(1, left)?;
    kernel.finish()
}

fn rolled_block_sum_at<const LANES: u32>(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let element = kernel.element();
    let uint = kernel.index_type();
    let span = kernel.module().constant_u32(super::WORKGROUP_SIZE)?;
    let nought = kernel.module().constant_u32(0)?;

    let total = kernel.repeat_rolled(times, element, nought, |kernel, carried, counter| {
        let offset = kernel.module().i_mul(uint, counter, span)?;
        let block = kernel.load_offset_by::<LANES>(0, offset)?.id();
        Ok(kernel.module().i_add(element, carried, block)?)
    })?;

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

fn rolled_weighted_totals_at<const LANES: u32>(
    subgroup: u32,
    times: u32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let element = kernel.element();
    let uint = kernel.index_type();
    let span = kernel.module().constant_u32(super::WORKGROUP_SIZE)?;
    let one = kernel.module().constant_u32(1)?;
    let nought = kernel.module().constant_u32(0)?;

    let totals = kernel.repeat_rolled_many(
        times,
        element,
        &[nought, nought],
        |kernel, carried, counter| {
            let [plain, weighted] = *carried else {
                return Err(LaneError::BadCarry {
                    given: carried.len(),
                    wanted: 2,
                });
            };

            let offset = kernel.module().i_mul(uint, counter, span)?;
            let block = kernel.load_offset_by::<LANES>(0, offset)?.id();

            let weight = kernel.module().i_add(uint, counter, one)?;
            let scaled = kernel.module().i_mul(element, block, weight)?;

            Ok(vec![
                kernel.module().i_add(element, plain, block)?,
                kernel.module().i_add(element, weighted, scaled)?,
            ])
        },
    )?;

    let [plain, weighted] = totals[..] else {
        return Err(LaneError::BadCarry {
            given: totals.len(),
            wanted: 2,
        });
    };
    let answer = kernel.module().i_sub(element, weighted, plain)?;

    kernel.store_scalar(1, answer)?;
    kernel.finish()
}

pub fn centre_and_scale(subgroup: u32, centre: f32, scale: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, centre_and_scale_at, centre, scale)
}

pub fn remainder(subgroup: u32, divisor: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, remainder_at, divisor)
}

pub fn rolled_block_sum(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, rolled_block_sum_at, times)
}

pub fn rolled_weighted_totals(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, rolled_weighted_totals_at, times)
}
