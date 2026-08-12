//! Kernels over elements narrower than a lane.
//!
//! Nothing here is a new *shape* — an elementwise add and a subgroup sum, which the 32-bit kernels
//! already have. That is the claim being made: `decisions/DR-0004` says a narrow element changes
//! the type's width and the buffer's stride and nothing else, and these exist so a device can be
//! asked whether that is true.
//!
//! The two that matter are separate on purpose. An elementwise kernel needs `shaderInt8` and
//! `storageBuffer8BitAccess`; a reduction needs `shaderSubgroupExtendedTypes` on top, and that one
//! leaves no trace in the module at all.

use super::shape;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, LaneError};

/// `out[i] = in[i] + addend`, in whatever `T` is.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
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

/// `out[i] = sum of in[i]'s subgroup`, in whatever `T` is.
///
/// Needs `shaderSubgroupExtendedTypes` when `T` is narrow. The module says nothing about that, so
/// a device without it refuses the *pipeline* rather than the module — which is a failure that
/// arrives much later than it looks like it should.
///
/// # Errors
///
/// As [`narrow_add`].
pub fn narrow_sum<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let total = kernel.lanes()?.reduce_sum(value)?;
    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// `out[i] = clamp(in[i], low, high)`, over a narrow type and a chosen workgroup size.
///
/// The workgroup size is a parameter here and fixed in [`super::shape`] elsewhere, because the
/// bandwidth comparison this exists for dispatches the same *element* count at two different
/// widths — and an element count is invocations times strips.
///
/// # Errors
///
/// As [`narrow_add`].
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
