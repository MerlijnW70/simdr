//! Kernels that address their buffers as rows and columns.
//!
//! Every one of these is built from [`simdr::kernel::Shape::grid`] and dispatched with a
//! [`crate::Grid`]. What they are here to catch is the arithmetic between those two: a row index
//! the kernel computes one way and the dispatch lays out another agrees on a square grid, on one
//! workgroup, and on a device whose subgroup happens to be the number that was hard-coded.
//!
//! # Why the workgroup is exactly one subgroup wide
//!
//! `whole_subgroup!` picks `LANES` equal to the device's width, and the shape below makes the
//! workgroup that wide too. So there is exactly one strip, one subgroup per invocation row, and
//! `pitch / width` workgroups across. A row-wise reduction then means what it says: the subgroup
//! that reduces is the row it is reducing.
//!
//! Widen the workgroup past the subgroup and a row-wise `reduce_sum` returns per-*subgroup*
//! totals, which is a different and equally plausible number — the same trap the 64-wide device
//! found in ten of these tests a month ago.

use super::whole_subgroup;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{LaneError, U32};

/// The shape a grid kernel here is built to: one subgroup across, `rows` deep, two buffers.
fn grid_shape(subgroup: u32, rows: u32) -> Shape {
    Shape::grid(subgroup, subgroup, rows, 2)
}

/// `out[row][column] = in[row][column] × factor`, over a buffer `pitch` elements to the row.
///
/// The control, and the same job [`super::scale`] does on one axis: no lane talks to any other, so
/// a wrong answer is a wrong *address* rather than a wrong reduction. Run it first, and run it at a
/// pitch wider than one workgroup — that is the only thing here that exercises the dispatch's x
/// and y at once.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
pub fn row_scale(subgroup: u32, pitch: u32, rows: u32, factor: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, row_scale_at, pitch, rows, factor)
}

fn row_scale_at<const LANES: u32>(
    subgroup: u32,
    pitch: u32,
    rows: u32,
    factor: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(grid_shape(subgroup, rows))?;
    let value = kernel.load_row::<LANES>(0, pitch)?;
    let scaled = {
        let mut lanes = kernel.lanes()?;
        let factor = lanes.splat_bits::<U32, LANES>(factor)?;
        lanes.mul(value, factor)?
    };
    kernel.store_row(1, pitch, scaled)?;
    kernel.finish()
}

/// `out[i] = in[i] × factor`, on one axis — the twin [`row_scale`] is measured against.
///
/// Deliberately not [`super::scale`]: that one is `f32`, a workgroup of 64 and a fixed 32 lanes,
/// so three things differ between it and a grid kernel at once. This differs in exactly one — the
/// address — which is the only way the difference between the two can be attributed to anything.
///
/// `workgroup` is open for the same reason. A grid `rows` deep has `subgroup × rows` invocations
/// per group, so comparing it against a one-axis kernel of `subgroup` invocations varies the
/// occupancy and the address at once, and the occupancy turns out to be the larger of the two.
///
/// # Errors
///
/// As [`row_scale`].
pub fn flat_scale(subgroup: u32, workgroup: u32, factor: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, flat_scale_at, workgroup, factor)
}

fn flat_scale_at<const LANES: u32>(
    subgroup: u32,
    workgroup: u32,
    factor: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, workgroup, 2))?;
    let value = kernel.load::<LANES>(0)?;
    let scaled = {
        let mut lanes = kernel.lanes()?;
        let factor = lanes.splat_bits::<U32, LANES>(factor)?;
        lanes.mul(value, factor)?
    };
    kernel.store(1, scaled)?;
    kernel.finish()
}

/// `out[row][column] = Σ in[row][*]`, one subgroup reduction per row.
///
/// Every column of an output row holds that row's total, because every lane of the reduction does.
/// Checking all of them rather than the first is the stronger assertion: a reduction that summed
/// the wrong lanes would still fill column zero with *a* number.
///
/// The pitch is the subgroup width, and cannot be anything else — a row wider than the subgroup
/// needs the workgroup handover in [`super::reduce::workgroup_sum`], which is a different kernel.
///
/// # Errors
///
/// As [`row_scale`].
pub fn row_sum(subgroup: u32, rows: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, row_sum_at, rows)
}

fn row_sum_at<const LANES: u32>(subgroup: u32, rows: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(grid_shape(subgroup, rows))?;
    let value = kernel.load_row::<LANES>(0, subgroup)?;
    let total = kernel.lanes()?.reduce_sum(value)?;
    kernel.store_row_scalar(1, subgroup, total)?;
    kernel.finish()
}

/// `out[row][column] = in[row][column] + in[0][column]`.
///
/// A bias row added to every row of a matrix, and the only kernel here that reads a row other than
/// its own. Row zero is a constant rather than a computed index, which keeps it in range without a
/// branch — and every row reads it, including row zero, which adds itself.
///
/// What this catches: a [`simdr::kernel::Kernel::load_row_at`] that ignored its argument and read
/// this invocation's row would double every row, which is a wrong answer that looks like a
/// working kernel until row zero is compared with the rest.
///
/// # Errors
///
/// As [`row_scale`].
pub fn row_bias(subgroup: u32, pitch: u32, rows: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, row_bias_at, pitch, rows)
}

fn row_bias_at<const LANES: u32>(
    subgroup: u32,
    pitch: u32,
    rows: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(grid_shape(subgroup, rows))?;
    let mine = kernel.load_row::<LANES>(0, pitch)?;

    let first = kernel.module().constant_u32(0)?;
    let bias = kernel.load_row_at::<LANES>(0, pitch, first)?;

    let sum = kernel.lanes()?.add(mine, bias)?;
    kernel.store_row(1, pitch, sum)?;
    kernel.finish()
}

/// `out[row][column] = row`, which is the row index and nothing else.
///
/// The narrowest test there is of where an invocation thinks it is. Every other kernel here reads
/// its own address and writes it back, so an address that is wrong in the same way twice cancels
/// out; this one writes the *index* and can only agree by being right.
///
/// # Errors
///
/// As [`row_scale`].
pub fn row_index(subgroup: u32, pitch: u32, rows: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, row_index_at, pitch, rows)
}

fn row_index_at<const LANES: u32>(
    subgroup: u32,
    pitch: u32,
    rows: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(grid_shape(subgroup, rows))?;
    // Loaded and dropped, so the input buffer is read by something and the dispatch is shaped like
    // the others. What is stored has nothing to do with it.
    let _ = kernel.load_row::<LANES>(0, pitch)?;

    let row = kernel.row()?;
    kernel.store_row_scalar(1, pitch, row)?;
    kernel.finish()
}
