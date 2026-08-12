//! Kernels that reach the GLSL.std.450 set, for running on a device.
//!
//! Every one of these is elementwise, so no lane talks to any other and the expected answer is a
//! `map` over the input. That is deliberate: what is under test is whether `OpExtInst` computes
//! what its name says on real hardware, and a reduction on top would put the subgroup mapping
//! between the instruction and the answer.
//!
//! The bounds and operands arrive as **bits** rather than as a number, for the reason
//! `Lanes::splat_bits` does: one signature serves `f32`, `i32` and `u32`, and the standard library
//! has no numeric trait that would cover the three.

use super::shape;
use simdr::kernel::Kernel;
use simdr::lanes::{Element, F32, LaneError, Signed};

/// `out[i] = clamp(in[i], low, high)`.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
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

/// `out[i] = abs(in[i])`.
///
/// # Errors
///
/// As [`clamped`].
pub fn magnitude<T: Signed, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let magnitude = kernel.lanes()?.abs(value)?;
    kernel.store(1, magnitude)?;
    kernel.finish()
}

/// `out[i] = max(in[i], other)`, elementwise.
///
/// The operand order is the input first, which is what the NaN observation in
/// `runner/tests/extended.rs` needs to be able to say: `FMax(NaN, x)` and `FMax(x, NaN)` are
/// different calls and the specification pins neither.
///
/// # Errors
///
/// As [`clamped`].
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

/// `out[i] = min(in[i], other)`, elementwise.
///
/// # Errors
///
/// As [`clamped`].
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

/// `out[i] = sqrt(in[i])`.
///
/// # Errors
///
/// As [`clamped`].
pub fn root<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let root = kernel.lanes()?.sqrt(value)?;
    kernel.store(1, root)?;
    kernel.finish()
}

/// `out[i] = in[i] * in[i] + in[i]`, through one `Fma` rather than a multiply and an add.
///
/// Built to be *compared against* the two-instruction spelling rather than checked against a
/// number: `Fma` rounds once and `OpFMul` then `OpFAdd` rounds twice, so the two agree only where
/// the intermediate product is exact.
///
/// # Errors
///
/// As [`clamped`].
pub fn fused_square<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let fused = kernel.lanes()?.fma(value, value, value)?;
    kernel.store(1, fused)?;
    kernel.finish()
}
