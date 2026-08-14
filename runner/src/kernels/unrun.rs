//! The lane operations that had unit tests and had never run.
//!
//! A coverage sweep on 2026-08-11 found five of them — `prefix_sum`, `ballot`, `shift_down`,
//! `broadcast`, `all_uniform` — with unit tests only, plus `reduce_min` with a fuzz case and no
//! kernel. That is the weakest evidence this project has: a unit test decodes the module and
//! agrees that the emitter emitted what the test expected, which says nothing whatever about what
//! the hardware then does with it.
//!
//! It is also not hypothetical. `reduce_min` folded its strips with a maximum for weeks under
//! seven passing unit tests, and only a differential run against a CPU reference noticed.
//!
//! Everything here exists to be executed and compared.

use super::{shape, whole_subgroup, whole_subgroup_of};
use simdr::kernel::Kernel;
use simdr::lanes::{Element, LaneError};

/// `out[i] = Σ in[first..=i]` within the subgroup — an *inclusive* scan.
///
/// Inclusive is the whole question. An exclusive scan is the same instruction with a different
/// `GroupOperation`, the two differ by exactly one element, and every unit test that counted
/// opcodes would pass for either. Only running it says which one came out.
///
/// Whole-subgroup vectors only, which is this kernel's choice rather than the lane API's limit:
/// the other two mappings scan as well and are the business of `kernels::scan`, which runs them
/// against a CPU reference. Here `LANES` equals the width so that what is under test is the one
/// instruction.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn prefix_sum_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let running = kernel.lanes()?.prefix_sum(value)?;
    kernel.store(1, running)?;
    kernel.finish()
}

/// `out[i] = in[source]` — one lane's value, delivered to all of them.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
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

/// `out[i] = in[i + delta]` within the subgroup, where that lane exists.
///
/// The lanes it does not exist for are **undefined** by the specification, so the test that runs
/// this checks only the lanes that have a source and reports what the rest happened to hold rather
/// than asserting it. Observing an undefined value is fine; pinning one is how a test starts
/// depending on a driver.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
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

/// The same upward: `out[i] = in[i - delta]`.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
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

/// `out[i] = Simd::<T, LANES>::reduce_min(in[…])`.
///
/// The operation whose strip fold was wrong, and which had no kernel at all until it was.
///
/// # Errors
///
/// [`LaneError`] if the lane count has no mapping onto this subgroup.
pub fn lane_min<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let smallest = kernel.lanes()?.reduce_min(value)?;
    kernel.store_scalar(1, smallest)?;
    kernel.finish()
}

/// `out[i] = 1` where *every* element of the subgroup exceeds `threshold`, else `0`.
///
/// The other vote. `any` has been run since the first week; `all` reaches a different instruction
/// and had never been dispatched.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
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

/// `out[i] = ` the low 32 bits of the ballot of `in[i] > threshold`.
///
/// One bit per lane, set where that lane's predicate held. Every lane of the subgroup receives the
/// same mask, which is what makes it checkable: the host knows which lanes qualify and can build
/// the same word.
///
/// A ballot is a `uvec4` in SPIR-V and only the first component is read here — a 32-wide subgroup
/// fits in it, and a 64-wide one would need two.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
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

/// `all_above_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn all_above(subgroup: u32, threshold: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, all_above_at, threshold)
}

/// `ballot_above_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn ballot_above(subgroup: u32, threshold: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, ballot_above_at, threshold)
}

/// `prefix_sum_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn prefix_sum<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, prefix_sum_at)
}

/// `broadcast_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn broadcast<T: Element>(subgroup: u32, source: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, broadcast_at, source)
}

/// `shift_down_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn shift_down<T: Element>(subgroup: u32, delta: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, shift_down_at, delta)
}

/// `shift_up_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn shift_up<T: Element>(subgroup: u32, delta: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, shift_up_at, delta)
}
