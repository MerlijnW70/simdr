use runner::kernels::{self, WORKGROUP_SIZE};
use runner::{Gpu, Timing};
use simdr::lanes::F32;

const ITERATIONS: u32 = 20;
const REPEATS: u32 = 5;

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

    const SETS: [usize; 9] = [8, 16, 32, 48, 56, 64, 72, 80, 96];

    let largest = SETS.iter().copied().max().unwrap_or(96);
    let placement = gpu.probe_memory((largest * 1024 * 1024) as u64)?;
    println!(
        "device-local heap {:.1} GB; a {largest} MB request landed {}",
        placement.largest_device_heap as f64 / 1e9,
        if placement.device_local {
            "device-local"
        } else {
            "HOST-VISIBLE — every number below is measuring the bus"
        }
    );
    println!(
        "the harness allocates three buffers per run, so the largest point holds {} MB",
        largest * 3
    );
    println!(
        "timing: {}",
        if limits.timestamp_period_ns > 0.0 {
            format!(
                "the device's own clock, {:.1} ns per tick",
                limits.timestamp_period_ns
            )
        } else {
            String::from("the HOST clock — includes scheduling this harness cannot see")
        }
    );

    sweep(&gpu, "Simd<f32,128> — 4 strips", 4, &SETS, |width| {
        kernels::lane_sum::<F32, 128>(width)
    })?;
    sweep(&gpu, "Simd<f32,64> — 2 strips", 2, &SETS, |width| {
        kernels::lane_sum::<F32, 64>(width)
    })?;

    println!("\n`!` marks a row whose repeats disagreed by more than a fifth: not evidence.");
    Ok(())
}

fn sweep(
    gpu: &Gpu,
    label: &str,
    strips: usize,
    megabytes: &[usize],
    build: impl Fn(u32) -> Result<Vec<u32>, simdr::lanes::LaneError>,
) -> Result<(), Box<dyn std::error::Error>> {
    let width = gpu.limits().subgroup_size;
    let spirv = build(width)?;

    println!("\n{label}");
    println!(
        "{:>11} {:>10} {:>10} {:>10} {:>8} {:>10}",
        "working set", "workgroups", "best", "median", "spread", "GB/s"
    );

    for &target in megabytes {
        let per_workgroup = WORKGROUP_SIZE as usize * strips * 4;
        let workgroups = (target * 1024 * 1024 / per_workgroup) as u32;
        let elements = workgroups as usize * WORKGROUP_SIZE as usize * strips;
        let input = vec![1_u32; elements];

        gpu.time(&spirv, &input, workgroups, 1)?;

        let timing: Timing = gpu.time_repeated(&spirv, &input, workgroups, ITERATIONS, REPEATS)?;
        let per_pass = timing.best / ITERATIONS;
        let bytes = (elements * 4) as f64;
        let rate = bytes / per_pass.as_secs_f64() / 1e9;

        println!(
            "{:>8} MB {:>10} {:>7.0} us {:>7.0} us {:>7.1}x {:>10.0}{}",
            target,
            workgroups,
            timing.best.as_secs_f64() * 1e6 / f64::from(ITERATIONS),
            timing.median.as_secs_f64() * 1e6 / f64::from(ITERATIONS),
            timing.spread(),
            rate,
            if timing.is_steady() { "" } else { "  !" }
        );
    }

    Ok(())
}
