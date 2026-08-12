//! Two-axis kernels, validated.
//!
//! Split from `kernels.rs`. A second axis changes the interface — `LocalSize` gains a y and two
//! more components come out of the built-ins — and the address gains a multiply. Both are places a
//! module can stop being valid without computing anything different, which is exactly what a
//! validator is for and what counting instructions in a unit test is not.
//!
//! `decisions/DR-0006` is the design these check.

mod common;

use common::{VULKAN_1_1, expect_valid};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::U32;
// ---------------------------------------------------------------------------------------------
// Grids
//
// A second axis changes the interface — `LocalSize` gains a y and two more components come out of
// the built-ins — and the address gains a multiply. Both are places a module can stop being valid
// without computing anything different.
// ---------------------------------------------------------------------------------------------

/// One subgroup across, `rows` deep, two buffers.
fn grid(rows: u32) -> Shape {
    Shape::grid(32, 32, rows, 2)
}

#[test]
fn a_grid_kernel_one_row_deep_is_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(grid(1)).expect("built");
    let value = kernel.load_row::<32>(0, 1024).expect("loaded");
    let scaled = {
        let mut lanes = kernel.lanes().expect("lanes");
        let three = lanes.splat_bits::<U32, 32>(3).expect("three");
        lanes.mul(value, three).expect("scaled")
    };
    kernel.store_row(1, 1024, scaled).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-grid-flat",
        VULKAN_1_1,
    );
}

#[test]
fn a_workgroup_several_rows_deep_is_valid_spirv() {
    // The only shape that reads `LocalInvocationId.y`, and the only one whose row is arithmetic
    // rather than a built-in component straight through.
    let mut kernel = Kernel::<U32>::new(grid(4)).expect("built");
    let value = kernel.load_row::<32>(0, 256).expect("loaded");
    kernel.store_row(1, 256, value).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-grid-deep",
        VULKAN_1_1,
    );
}

#[test]
fn a_row_wise_reduction_is_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(grid(2)).expect("built");
    let value = kernel.load_row::<32>(0, 32).expect("loaded");
    let total = kernel
        .lanes()
        .expect("lanes")
        .reduce_sum(value)
        .expect("sum");
    kernel.store_row_scalar(1, 32, total).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-grid-reduce",
        VULKAN_1_1,
    );
}

#[test]
fn reading_a_second_row_is_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(grid(1)).expect("built");
    let mine = kernel.load_row::<32>(0, 512).expect("loaded");

    let first = kernel.module().constant_u32(0).expect("0");
    let bias = kernel.load_row_at::<32>(0, 512, first).expect("loaded");

    let sum = kernel.lanes().expect("lanes").add(mine, bias).expect("sum");
    kernel.store_row(1, 512, sum).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-grid-bias",
        VULKAN_1_1,
    );
}

#[test]
fn a_strip_mined_grid_kernel_is_valid_spirv() {
    // Four elements per lane on each axis at once: the column arithmetic strips and the row
    // arithmetic multiplies, and the two have to compose into one index.
    let mut kernel = Kernel::<U32>::new(grid(2)).expect("built");
    let value = kernel.load_row::<128>(0, 4096).expect("loaded");
    kernel.store_row(1, 4096, value).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-grid-strips",
        VULKAN_1_1,
    );
}
