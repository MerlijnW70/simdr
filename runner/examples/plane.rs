//! What does a second axis cost?
//!
//! A grid kernel's address is `row × pitch + column` where a linear one's is just the column, so
//! every access pays one multiply and one add more, and the dispatch launches workgroups on two
//! axes instead of one. The question is whether either of those is visible.
//!
//! # It is a two-by-two, and the first version of this file was not
//!
//! A grid `rows` deep has `subgroup × rows` invocations per workgroup. Comparing it against a
//! one-axis kernel of `subgroup` invocations changes the address *and* the occupancy, and the
//! first run of this example showed the eight-deep grid at 2× — which reads as "the second axis is
//! faster" and is nothing of the kind.
//!
//! So both variables move independently below. Each column pair differs in exactly one thing:
//!
//! | | workgroup of `width` | workgroup of `width × 8` |
//! | --- | --- | --- |
//! | **one axis** | `flat wg=w` | `flat wg=8w` |
//! | **two axes** | `grid 1 deep` | `grid 8 deep` |
//!
//! Down a column is the cost of the address. Across a row is the cost of the workgroup size.
//!
//! # What this cannot say
//!
//! One device per run, and an elementwise kernel that is memory-bound at every size below. If two
//! cells come out equal, the arithmetic between them hid behind the loads — not that it is free in
//! a kernel where it would not.

mod common;

use runner::{Gpu, Grid, Timing, kernels};
use std::time::Duration;

/// How many rows of the matrix to run, and how many timed iterations at that size.
const SIZES: [(u32, u32); 3] = [(8, 400), (256, 200), (4_096, 50)];

/// How many invocation rows the deep shapes hold, and the factor the wide workgroup is wider by.
const DEEP: u32 = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    let width = limits.subgroup_size;
    println!(
        "{} — subgroup {width}, so the two workgroup sizes below are {width} and {}\n",
        limits.name,
        width * DEEP
    );

    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>13} {:>13} {:>12}",
        "rows", "invocations", "flat wg=w", "flat wg=8w", "grid 1 deep", "grid 8 deep", "wide rows"
    );

    for (height, iterations) in SIZES {
        // Every shape covers `height × width` elements with one element per invocation, so the
        // invocation count and the memory traffic are the same in all five.
        let invocations = (height * width) as usize;
        let input: Vec<u32> = (0..invocations).map(|index| index as u32).collect();

        //   flat wg=w    — one axis, one subgroup per workgroup
        //   flat wg=8w   — one axis, eight subgroups per workgroup: the occupancy control
        //   grid 1 deep  — two axes, one invocation row per workgroup, row is `group.y` outright
        //   grid 8 deep  — two axes, eight rows per workgroup: the only shape reading `local.y`
        //   wide rows    — two axes, rows eight workgroups across, so a row spans x as well as y
        let flat = kernels::flat_scale(width, width, 3)?;
        let flat_wide = kernels::flat_scale(width, width * DEEP, 3)?;
        let shallow = kernels::row_scale(width, width, 1, 3)?;
        let deep = kernels::row_scale(width, width, DEEP, 3)?;
        let wide = kernels::row_scale(width, width * DEEP, 1, 3)?;

        let cases = [
            (&flat, Grid::linear(height)),
            (&flat_wide, Grid::linear(height / DEEP)),
            (&shallow, Grid::new(1, height)),
            (&deep, Grid::new(1, height / DEEP)),
            (&wide, Grid::new(DEEP, height / DEEP)),
        ];

        let mut timings = Vec::with_capacity(cases.len());
        for (spirv, grid) in cases {
            if grid.x == 0 || grid.y == 0 {
                timings.push(None);
                continue;
            }
            // One untimed pass, so the driver's lazy pipeline work stays out of the measurement.
            gpu.time_grid(spirv, &input, grid, 1)?;
            timings.push(Some(common::samples(common::SAMPLES, || {
                gpu.time_grid(spirv, &input, grid, iterations)
            })?));
        }

        println!(
            "{:>8} {:>12} {:>12} {:>12} {:>13} {:>13} {:>12}",
            height,
            thousands(invocations),
            cell(timings.first().copied().flatten(), iterations),
            cell(timings.get(1).copied().flatten(), iterations),
            cell(timings.get(2).copied().flatten(), iterations),
            cell(timings.get(3).copied().flatten(), iterations),
            cell(timings.get(4).copied().flatten(), iterations),
        );
    }

    println!(
        "\nAll five compute the same answer over the same elements. `runner/tests/plane.rs`\n\
         checks the grid ones against a host reference; `kernels::flat_scale` differs from them\n\
         in the address and in nothing else.
{}",
        common::LEGEND
    );

    Ok(())
}

/// One cell of the table, or a dash where the shape did not divide.
///
/// The median of the repeats, marked when they disagreed. Five cells across a row are compared by
/// eye here, and a single unrepeated sample in any one of them makes the whole comparison a guess.
fn cell(timing: Option<Timing>, iterations: u32) -> String {
    timing.map_or_else(
        || "-".to_owned(),
        |timing| {
            format!(
                "{}{}",
                micros(timing.median / iterations),
                common::mark(timing)
            )
        },
    )
}

/// Microseconds, which is the scale these land on.
fn micros(duration: Duration) -> String {
    format!("{:.2} us", duration.as_secs_f64() * 1e6)
}

/// A count with separators.
fn thousands(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(digit);
    }
    out
}
