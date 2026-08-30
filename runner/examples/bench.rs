use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;
use std::time::Duration;

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

fn measure(
    gpu: &Gpu,
    width: u32,
    workgroups: u32,
    iterations: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let invocations = (workgroups * WORKGROUP_SIZE) as usize;
    let input = vec![1_u32; invocations * 4];

    println!(
        "\n{workgroups} workgroups — {} invocations",
        thousands(invocations)
    );
    println!(
        "{:<22} {:>10} {:>12} {:>14}",
        "kernel", "per pass", "elem/pass", "elem/s"
    );

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
        gpu.time(&spirv, &input, workgroups, 1)?;

        let elapsed = gpu.time(&spirv, &input, workgroups, iterations)?;
        let per_pass = elapsed / iterations;
        let elements = invocations * strips;
        let per_second = elements as f64 / per_pass.as_secs_f64();

        println!(
            "{label:<22} {:>10} {:>12} {:>14}",
            format_duration(per_pass),
            thousands(elements),
            format!("{:.1} G", per_second / 1e9)
        );
    }

    Ok(())
}

fn format_duration(duration: Duration) -> String {
    format!("{:.0} us", duration.as_secs_f64() * 1e6)
}

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
