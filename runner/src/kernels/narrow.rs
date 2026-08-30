use super::{shape, whole_subgroup_of};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, LaneError};

pub fn narrow_add<T: Element, const LANES: u32>(
    subgroup: u32,
    addend: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let raised = {
        let mut lanes = kernel.lanes()?;
        let addend = lanes.splat_bits::<T, LANES>(addend)?;
        lanes.add(value, addend)?
    };
    kernel.store(1, raised)?;
    kernel.finish()
}

pub fn narrow_sum<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let total = kernel.lanes()?.reduce_sum(value)?;
    kernel.store_scalar(1, total)?;
    kernel.finish()
}

pub fn narrow_sum_whole<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, narrow_sum)
}

pub fn narrow_clamp<T: Element, const LANES: u32>(
    subgroup: u32,
    workgroup: u32,
    low: u32,
    high: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(Shape::new(subgroup, workgroup, 2))?;
    let value = kernel.load::<LANES>(0)?;
    let bounded = {
        let mut lanes = kernel.lanes()?;
        let low = lanes.splat_bits::<T, LANES>(low)?;
        let high = lanes.splat_bits::<T, LANES>(high)?;
        lanes.clamp(value, low, high)?
    };
    kernel.store(1, bounded)?;
    kernel.finish()
}
