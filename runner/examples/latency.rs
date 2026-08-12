//! What one answer costs when you need it *now*.
//!
//! Every other benchmark here amortises: hundreds of dispatches in one submission, divided out.
//! That is the right measurement for throughput and the wrong one for a caller that has a single
//! question and cannot continue until it is answered — a game-tree search asking for one
//! evaluation, say.
//!
//! So this measures the whole round trip from the host's point of view: submit, wait, read back.
//! It is deliberately the least flattering number this project can produce, and it is the one that
//! decides whether a latency-bound workload belongs on a GPU at all.

use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;
use std::time::Instant;

/// How many round trips to time, after a warm-up.
const ROUND_TRIPS: u32 = 200;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    let width = limits.subgroup_size;
    println!("{} — subgroup {width}\n", limits.name);

    let input = vec![1.0_f32; WORKGROUP_SIZE as usize];

    println!("{:<26} {:>12} {:>14}", "what", "per answer", "answers/s");
    for (label, spirv) in [
        ("empty kernel", kernels::empty(width)?),
        ("one subgroup sum", kernels::lane_sum::<F32, 32>(width)?),
    ] {
        // Warm up: the first call of a module pays for pipeline creation.
        gpu.run(&spirv, &input, 1)?;

        let started = Instant::now();
        for _ in 0..ROUND_TRIPS {
            gpu.run(&spirv, &input, 1)?;
        }
        let each = started.elapsed() / ROUND_TRIPS;

        println!(
            "{label:<26} {:>12} {:>14}",
            format!("{:.1} us", each.as_secs_f64() * 1e6),
            format!("{:.0}", 1.0 / each.as_secs_f64())
        );
    }

    // And the same device, asked the same question a million times at once. The gap between these
    // two blocks is the entire argument about which workloads belong here.
    println!("\nthe same work, batched:");
    let wide = vec![1.0_f32; 65_536];
    let spirv = kernels::lane_sum::<F32, 32>(width)?;
    let workgroups = 65_536 / WORKGROUP_SIZE;

    gpu.time(&spirv, &to_words(&wide), workgroups, 1)?;
    let timing = gpu.time_repeated(&spirv, &to_words(&wide), workgroups, 100, 5)?;
    let each = timing.best / 100;
    let subgroup_sums = 65_536 / width;

    println!(
        "{:<26} {:>12} {:>14}",
        format!("{subgroup_sums} sums per dispatch"),
        format!("{:.1} us", each.as_secs_f64() * 1e6),
        format!("{:.0}", f64::from(subgroup_sums) / each.as_secs_f64())
    );

    Ok(())
}

/// Floats as the words a buffer holds.
fn to_words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}
