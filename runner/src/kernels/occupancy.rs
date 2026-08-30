use super::whole_subgroup;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{LaneError, U32};

pub const LIMIT: u32 = 0x00FF_FFFF;

pub fn sized_repeated_scale(
    subgroup: u32,
    workgroup: u32,
    times: u32,
    factor: u32,
) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, sized_repeated_scale_at, workgroup, times, factor)
}

fn sized_repeated_scale_at<const LANES: u32>(
    subgroup: u32,
    workgroup: u32,
    times: u32,
    factor: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, workgroup, 2))?;
    let value = kernel.load::<LANES>(0)?;

    let result = {
        let mut lanes = kernel.lanes()?;
        let factor = lanes.splat_bits::<U32, LANES>(factor)?;
        let limit = lanes.splat_bits::<U32, LANES>(LIMIT)?;

        let mut running = value;
        for step in 0..times {
            let salt = lanes.splat_bits::<U32, LANES>(step)?;
            let scaled = lanes.mul(running, factor)?;
            let shifted = lanes.add(scaled, salt)?;
            running = lanes.min(shifted, limit)?;
        }
        running
    };

    kernel.store(1, result)?;
    kernel.finish()
}

pub fn sized_lane_sum(subgroup: u32, workgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, sized_lane_sum_at, workgroup)
}

fn sized_lane_sum_at<const LANES: u32>(
    subgroup: u32,
    workgroup: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, workgroup, 2))?;
    let value = kernel.load::<LANES>(0)?;
    let total = kernel.lanes()?.reduce_sum(value)?;
    kernel.store_scalar(1, total)?;
    kernel.finish()
}
