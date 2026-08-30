use super::shape;
use simdr::kernel::Kernel;
use simdr::lanes::{Element, F32, LaneError, Signed};

pub fn clamped<T: Element, const LANES: u32>(
    subgroup: u32,
    low: u32,
    high: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
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

pub fn magnitude<T: Signed, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let magnitude = kernel.lanes()?.abs(value)?;
    kernel.store(1, magnitude)?;
    kernel.finish()
}

pub fn larger<T: Element, const LANES: u32>(
    subgroup: u32,
    other: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let largest = {
        let mut lanes = kernel.lanes()?;
        let other = lanes.splat_bits::<T, LANES>(other)?;
        lanes.max(value, other)?
    };
    kernel.store(1, largest)?;
    kernel.finish()
}

pub fn smaller<T: Element, const LANES: u32>(
    subgroup: u32,
    other: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let smallest = {
        let mut lanes = kernel.lanes()?;
        let other = lanes.splat_bits::<T, LANES>(other)?;
        lanes.min(value, other)?
    };
    kernel.store(1, smallest)?;
    kernel.finish()
}

pub fn root<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let root = kernel.lanes()?.sqrt(value)?;
    kernel.store(1, root)?;
    kernel.finish()
}

pub fn fused_square<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let fused = kernel.lanes()?.fma(value, value, value)?;
    kernel.store(1, fused)?;
    kernel.finish()
}
