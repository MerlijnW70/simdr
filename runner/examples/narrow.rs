use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{Element, I8, I16, I32};
use std::time::Duration;

const SIZES: [(usize, u32); 2] = [(1 << 20, 50), (1 << 24, 10)];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    println!("{} — subgroup {}", limits.name, limits.subgroup_size);

    if limits.subgroup_size < 32 {
        println!(
            "SKIPPED narrow: four i8 strips need a subgroup of at least 32 and this device              reports {}. The comparison is about strip mining, and sixteen strips is a refusal              rather than a smaller number.",
            limits.subgroup_size
        );
        return Ok(());
    }

    let narrow = limits.narrow;
    println!(
        "  shaderInt8 {}   storageBuffer8BitAccess {}   shaderInt16 {}   storageBuffer16BitAccess {}",
        narrow.int8, narrow.storage8, narrow.int16, narrow.storage16
    );
    if !narrow.byte_kernel() || !narrow.short_kernel() {
        println!("\nthis device cannot run all three widths, so there is nothing to compare");
        return Ok(());
    }

    for (elements, iterations) in SIZES {
        measure(&gpu, elements, iterations)?;
    }

    Ok(())
}

fn measure(gpu: &Gpu, elements: usize, iterations: u32) -> Result<(), Box<dyn std::error::Error>> {
    let width = gpu.limits().subgroup_size;

    println!("\n{} elements", thousands(elements));
    println!(
        "{:<16} {:>10} {:>12} {:>12} {:>10}",
        "element", "per pass", "bytes in", "GB/s", "vs i32"
    );

    let mut rows: Vec<(&str, Duration, usize)> = Vec::new();
    for (label, spirv, stride, strips) in [
        (
            "i8",
            kernels::narrow_clamp::<I8, 32>(width, WORKGROUP_SIZE, 0, 100)?,
            I8::STRIDE,
            1,
        ),
        (
            "i8 x4 strips",
            kernels::narrow_clamp::<I8, 128>(width, WORKGROUP_SIZE, 0, 100)?,
            I8::STRIDE,
            4,
        ),
        (
            "i16",
            kernels::narrow_clamp::<I16, 32>(width, WORKGROUP_SIZE, 0, 100)?,
            I16::STRIDE,
            1,
        ),
        (
            "i16 x2 strips",
            kernels::narrow_clamp::<I16, 64>(width, WORKGROUP_SIZE, 0, 100)?,
            I16::STRIDE,
            2,
        ),
        (
            "i32",
            kernels::narrow_clamp::<I32, 32>(width, WORKGROUP_SIZE, 0, 100)?,
            I32::STRIDE,
            1,
        ),
    ] {
        let bytes = elements * stride as usize;
        let input = vec![0x0102_0304_u32; bytes.div_ceil(4)];
        let workgroups = u32::try_from(elements / (WORKGROUP_SIZE as usize * strips))?;

        gpu.time(&spirv, &input, workgroups, 1)?;
        let per_pass = gpu.time(&spirv, &input, workgroups, iterations)? / iterations;
        rows.push((label, per_pass, bytes));
    }

    let widest = rows.last().map(|row| row.1);
    for (label, per_pass, bytes) in &rows {
        let rate = (bytes * 2) as f64 / per_pass.as_secs_f64() / 1e9;
        let against = widest.map_or_else(
            || String::from("—"),
            |slowest| format!("{:.2}x", slowest.as_secs_f64() / per_pass.as_secs_f64()),
        );

        println!(
            "{label:<16} {:>10} {:>12} {:>12} {:>10}",
            format_duration(*per_pass),
            thousands(*bytes),
            format!("{rate:.1}"),
            against
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
