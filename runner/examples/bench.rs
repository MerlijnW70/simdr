//! What each mapping actually costs on this device.
//!
//! Everything else is correctness. This asks the other question: given that a `Simd<f32, 8>` on a
//! 32-lane subgroup *can* run four reductions at once instead of idling twenty-four lanes, does
//! it — and what does a strip-mined vector cost now that the layout blocks by workgroup?
//!
//! Read the last column. `elem/s` counts the elements actually reduced, so a strip-mined kernel
//! is not flattered for doing more work per dispatch.
//!
//! Caveats, because a benchmark without them is a claim rather than a measurement. A barrier
//! separates the iterations, so this is the sum of the kernels' own times and not peak overlapped
//! throughput. The buffers are host-visible, which is the slow choice everywhere and equally slow
//! for every row. And one run on one device says nothing about another.

mod common;

use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;
use std::time::Duration;

/// Dispatch sizes to measure at, and how many passes to time together at each.
///
/// Two sizes rather than one, because a single size cannot tell work from overhead: if the small
/// and the large dispatch take the same time, the number being measured is the launch.
const SIZES: [(u32, u32); 2] = [(4_096, 200), (65_536, 40)];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    let width = limits.subgroup_size;
    println!("{} — subgroup {width}", limits.name);

    if width != 32 {
        println!("the lane counts below are written for a 32-wide subgroup; skipping");
        return Ok(());
    }

    for (workgroups, iterations) in SIZES {
        measure(&gpu, width, workgroups, iterations)?;
    }

    Ok(())
}

/// Time every mapping at one dispatch size.
fn measure(
    gpu: &Gpu,
    width: u32,
    workgroups: u32,
    iterations: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let invocations = (workgroups * WORKGROUP_SIZE) as usize;
    // Long enough for the widest strip-mined kernel below, which reads four per invocation.
    let input = vec![1_u32; invocations * 4];

    println!(
        "\n{workgroups} workgroups — {} invocations",
        thousands(invocations)
    );
    println!(
        "{:<22} {:>10} {:>12} {:>14}",
        "kernel", "per pass", "elem/pass", "elem/s"
    );

    // `strips` is how many elements each invocation folds in, which is what makes the last
    // column comparable across rows.
    for (label, spirv, strips) in [
        ("empty (dispatch only)", kernels::empty(width)?, 1),
        ("scale (no reduce)", kernels::scale(width, 2.0)?, 1),
        (
            "Simd<f32,4> cluster",
            kernels::lane_sum::<F32, 4>(width)?,
            1,
        ),
        (
            "Simd<f32,8> cluster",
            kernels::lane_sum::<F32, 8>(width)?,
            1,
        ),
        (
            "Simd<f32,32> whole",
            kernels::lane_sum::<F32, 32>(width)?,
            1,
        ),
        (
            "Simd<f32,64> 2 strips",
            kernels::lane_sum::<F32, 64>(width)?,
            2,
        ),
        (
            "Simd<f32,128> 4 strips",
            kernels::lane_sum::<F32, 128>(width)?,
            4,
        ),
    ] {
        // One untimed pass so the driver's lazy pipeline work does not land in the measurement.
        gpu.time(&spirv, &input, workgroups, 1)?;

        let timing = gpu.time_repeated(&spirv, &input, workgroups, iterations, common::SAMPLES)?;
        let per_pass = timing.median / iterations;
        let elements = invocations * strips;
        let per_second = elements as f64 / per_pass.as_secs_f64();
        let mark = common::mark(timing);

        println!(
            "{label:<22} {:>10} {:>12} {:>14}",
            format!("{}{mark}", format_duration(per_pass)),
            thousands(elements),
            format!("{:.1} G{mark}", per_second / 1e9)
        );
    }
    println!("{}", common::LEGEND);

    Ok(())
}

/// Microseconds, which is the scale everything here lands on.
fn format_duration(duration: Duration) -> String {
    format!("{:.0} us", duration.as_secs_f64() * 1e6)
}

/// A count with separators, because eight-digit numbers are unreadable without them.
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
