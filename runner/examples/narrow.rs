//! Does a narrow element type actually move fewer bytes per second of wall clock?
//!
//! `notes/NEXT.md` argued for `i8` and `i16` on the grounds that a bandwidth-bound kernel moving a
//! quarter of the bytes should take a quarter of the time. That is an argument, not a measurement,
//! and this is the measurement.
//!
//! The kernel is a clamp — one instruction per element, so the arithmetic is as close to free as
//! this crate can make it and whatever is left is memory. The same **element count** runs at each
//! width, which is the comparison that means something: a run that held the *byte* count fixed
//! would be four times the work at `i8` and would prove nothing.
//!
//! Caveats, because a benchmark without them is a claim. One device, one run. The buffers are
//! device-local and the timing is the device's own timestamps, so the host copies are outside it.
//! And a clamp over 8-bit elements is not what a real kernel does with them — it is what isolates
//! the thing being asked about.

mod common;

use runner::kernels::{self, WORKGROUP_SIZE};
use runner::{Gpu, Timing};
use simdr::lanes::{Element, I8, I16, I32};
use std::time::Duration;

/// How many elements to sweep, and how many timed passes at each size.
///
/// Two sizes because one cannot tell work from overhead: if the small and the large dispatch cost
/// the same, what is being measured is the launch.
const SIZES: [(usize, u32); 2] = [(1 << 20, 50), (1 << 24, 10)];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    println!("{} — subgroup {}", limits.name, limits.subgroup_size);

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

/// Time the same clamp over the same element count at three widths.
fn measure(gpu: &Gpu, elements: usize, iterations: u32) -> Result<(), Box<dyn std::error::Error>> {
    let width = gpu.limits().subgroup_size;

    println!("\n{} elements", thousands(elements));
    println!(
        "{:<16} {:>10} {:>12} {:>12} {:>10}",
        "element", "per pass", "bytes in", "GB/s", "vs i32"
    );

    // Timed first and printed second, so the last column can be a ratio against `i32` rather
    // than against whichever row happened to come first.
    // The strip count is a free parameter and it turns out to matter more than the width does: an
    // `i8` invocation holding one element loads a single byte, and one holding four loads a word.
    // Both rows read the same buffer and compute the same answer.
    let mut rows: Vec<(&str, Timing, usize)> = Vec::new();
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
        // The buffer is words whatever the element width is; the kernel reads it at its own
        // stride, and this is how many words that many elements occupy.
        let input = vec![0x0102_0304_u32; bytes.div_ceil(4)];
        // Fewer invocations when each holds more elements, so every row covers the same elements.
        let workgroups = u32::try_from(elements / (WORKGROUP_SIZE as usize * strips))?;

        // One untimed pass, so the driver's lazy pipeline work stays out of the measurement.
        gpu.time(&spirv, &input, workgroups, 1)?;
        let timing = gpu.time_repeated(&spirv, &input, workgroups, iterations, common::SAMPLES)?;
        rows.push((label, timing, bytes));
    }

    let widest = rows.last().map(|row| row.1);
    for (label, timing, bytes) in &rows {
        let per_pass = timing.median / iterations;
        // Read once and written once, so twice the buffer.
        let rate = (bytes * 2) as f64 / per_pass.as_secs_f64() / 1e9;
        let mark = common::mark(*timing);
        let against = widest.map_or_else(
            || String::from("-"),
            |slowest| common::ratio(slowest, *timing),
        );

        println!(
            "{label:<16} {:>10} {:>12} {:>12} {:>10}",
            format!("{}{mark}", format_duration(per_pass)),
            thousands(*bytes),
            format!("{rate:.1}{mark}"),
            against
        );
    }
    println!("{}", common::LEGEND);

    Ok(())
}

/// Microseconds, which is the scale these land on.
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
