use super::{shape, whole_subgroup};
use simdr::kernel::Kernel;
use simdr::lanes::LaneError;
use simdr::spec::SelectionControl;

fn any_above_at<const LANES: u32>(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let answer = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let over = lanes.greater_than(value, limit)?;
        let verdict = lanes.any(over)?;

        let yes = lanes.splat_bits::<F32, LANES>(1.0_f32.to_bits())?;
        let no = lanes.splat_bits::<F32, LANES>(0.0_f32.to_bits())?;
        let element = lanes.type_of::<F32>()?;
        lanes.module().select(element, verdict, yes.id(), no.id())?
    };
    kernel.store_scalar(1, answer)?;
    kernel.finish()
}

fn scale_if_any_above_at<const LANES: u32>(
    subgroup: u32,
    threshold: f32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;

    kernel.store(1, value)?;

    let over = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let above = lanes.greater_than(value, limit)?;
        lanes.any_uniform(above)?
    };

    let scaled = {
        let mut lanes = kernel.lanes()?;
        let ten = lanes.splat_bits::<F32, LANES>(10.0_f32.to_bits())?;
        lanes.mul(value, ten)?
    };

    let then_block = kernel.module().alloc_id()?;
    let merge_block = kernel.module().alloc_id()?;
    kernel
        .module()
        .selection_merge(merge_block, SelectionControl::None)?;
    kernel
        .module()
        .branch_conditional(over.id(), then_block, merge_block)?;
    kernel.module().label_at(then_block)?;
    kernel.store(1, scaled)?;
    kernel.module().branch(merge_block)?;
    kernel.module().label_at(merge_block)?;

    kernel.finish()
}

fn branch_only_at<const LANES: u32>(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    kernel.store(1, value)?;

    {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let above = lanes.greater_than(value, limit)?;
        let over = lanes.any_uniform(above)?;

        lanes.if_uniform(over, |lanes| {
            let two = lanes.splat_bits::<F32, LANES>(2.0_f32.to_bits())?;
            lanes.mul(value, two)?;
            Ok(())
        })?;
    }

    kernel.finish()
}

fn sum_or_max_at<const LANES: u32>(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let answer = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let above = lanes.greater_than(value, limit)?;
        let over = lanes.any_uniform(above)?;

        lanes.choose_uniform(
            over,
            element,
            |lanes| lanes.reduce_sum(value),
            |lanes| lanes.reduce_max(value),
        )?
    };

    kernel.store_scalar(1, answer)?;
    kernel.finish()
}

fn rolled_counter_sum_at<const LANES: u32>(
    subgroup: u32,
    times: u32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes()?;
        lanes.repeat_rolled(times, element, value.id(), |lanes, carried, iteration| {
            let held = lanes.from_lane_value::<U32, LANES>(carried)?;
            let step = lanes.from_lane_value::<U32, LANES>(iteration)?;
            Ok(lanes.add(held, step)?.id())
        })?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

fn branch_in_loop_at<const LANES: u32>(
    subgroup: u32,
    times: u32,
    threshold: f32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let above = lanes.greater_than(value, limit)?;
        let over = lanes.any_uniform(above)?;
        let one = lanes.splat_bits::<F32, LANES>(1.0_f32.to_bits())?;

        lanes.repeat_rolled(times, element, value.id(), |lanes, carried, _| {
            let held = lanes.from_lane_value::<F32, LANES>(carried)?;
            lanes.choose_uniform(
                over,
                element,
                |lanes| Ok(lanes.add(held, held)?.id()),
                |lanes| Ok(lanes.add(held, one)?.id()),
            )
        })?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

fn loop_in_branch_at<const LANES: u32>(
    subgroup: u32,
    times: u32,
    threshold: f32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let above = lanes.greater_than(value, limit)?;
        let over = lanes.any_uniform(above)?;

        lanes.choose_uniform(
            over,
            element,
            |lanes| {
                lanes.repeat_rolled(times, element, value.id(), |lanes, carried, _| {
                    let held = lanes.from_lane_value::<F32, LANES>(carried)?;
                    Ok(lanes.add(held, held)?.id())
                })
            },
            |_| Ok(value.id()),
        )?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

fn rolled_doubling_at<const LANES: u32>(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes()?;
        lanes.repeat_rolled(times, element, value.id(), |lanes, carried, _| {
            let held = lanes.from_lane_value::<F32, LANES>(carried)?;
            Ok(lanes.add(held, held)?.id())
        })?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

pub fn any_above(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, any_above_at, threshold)
}

pub fn scale_if_any_above(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, scale_if_any_above_at, threshold)
}

pub fn branch_only(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, branch_only_at, threshold)
}

pub fn sum_or_max(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, sum_or_max_at, threshold)
}

pub fn rolled_counter_sum(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, rolled_counter_sum_at, times)
}

pub fn branch_in_loop(subgroup: u32, times: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, branch_in_loop_at, times, threshold)
}

pub fn loop_in_branch(subgroup: u32, times: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, loop_in_branch_at, times, threshold)
}

pub fn rolled_doubling(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, rolled_doubling_at, times)
}
