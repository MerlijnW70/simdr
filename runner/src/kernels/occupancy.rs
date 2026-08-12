//! Kernels whose workgroup size is an argument.
//!
//! Everything else here is built from [`super::shape`], which hard-codes [`super::WORKGROUP_SIZE`]
//! at 64. That number has been 64 since the first kernel and was chosen once — and on the three
//! devices this runs on, 64 invocations is eight subgroups, two, or one. Whatever a workgroup size
//! is worth, it is not the same thing on all three.
//!
//! These exist so `runner/examples/occupancy.rs` can vary it while holding everything else still:
//! same element type, same total invocations, same total work, same buffers. The lane count stays
//! the subgroup width, so the *mapping* is `WholeSubgroup` at every size and one strip per access.
//!
//! # Three shapes, because one would answer for itself
//!
//! A memory-bound kernel, an arithmetic-bound one, and a reduction. `notes/FINDINGS.md` already
//! carries one measurement that generalised from a single elementwise kernel and had to be
//! qualified afterwards; three shapes is the cheapest way not to do it again.
//!
//! # What is deliberately absent
//!
//! [`super::reduce::workgroup_sum`], which combines subgroups through shared memory. Its
//! *algorithm* changes with the workgroup size — more subgroups means more slots to fold — so a
//! row of it in the same table would be comparing two different amounts of work and reading like a
//! comparison of one.

use super::whole_subgroup;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{LaneError, U32};

/// What [`sized_repeated_scale`] clamps its running value to.
///
/// Low enough that it clamps almost immediately, which is the point: a minimum that never fires is
/// one a compiler deletes. High enough that the multiply below it does not wrap, so the values are
/// still ordinary arithmetic rather than a modular curiosity.
pub const LIMIT: u32 = 0x00FF_FFFF;

/// `out[i] = min(in[i] × factor + step, limit)`, repeated `times` over — the arithmetic-bound
/// shape.
///
/// **The `min` is what makes the loop survive, and it has to be a `min` that bites.** The first
/// version multiplied and added, and `times` iterations of `x × f + s` compose into a single
/// `x × f^times + c` — which a driver is entitled to fold and this one did: sixty-four iterations
/// cost exactly what one did, to the hundredth of a microsecond, on all three kernel shapes at
/// once. The second version added `min(_, u32::MAX)`, which is the identity function and was
/// removed just as fast, leaving the same affine chain behind. Five hundred and twelve iterations
/// still cost 2.14 µs, which is how it was caught.
///
/// [`LIMIT`] is well below `u32::MAX`, so the minimum actually clamps and no closed form exists.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
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

/// `out[i] = Σ in[subgroup of i]`, at a chosen workgroup size — the reduction shape.
///
/// One subgroup instruction and one store per invocation. The reduction is over the subgroup, so
/// what it computes does not change with the workgroup size; only how many subgroups share a
/// group does.
///
/// # Errors
///
/// As [`sized_repeated_scale`].
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
