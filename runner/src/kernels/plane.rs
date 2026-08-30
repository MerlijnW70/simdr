use super::whole_subgroup;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{LaneError, U32};

fn grid_shape(subgroup: u32, rows: u32) -> Shape {
    Shape::grid(subgroup, subgroup, rows, 2)
}

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

pub fn row_index(subgroup: u32, pitch: u32, rows: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, row_index_at, pitch, rows)
}

fn row_index_at<const LANES: u32>(
    subgroup: u32,
    pitch: u32,
    rows: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(grid_shape(subgroup, rows))?;
    let _ = kernel.load_row::<LANES>(0, pitch)?;

    let row = kernel.row()?;
    kernel.store_row_scalar(1, pitch, row)?;
    kernel.finish()
}
