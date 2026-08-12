//! Is one dot-product instruction faster than the eleven it replaces?
//!
//! `OpSDot` computes four 8-bit products and their sum in one instruction. Written out, that is
//! four shifts up, four bitcasts, four shifts down, four multiplies and three adds. Both devices
//! here report `integerDotProduct4x8BitPackedSignedAccelerated`, which says the hardware does it
//! in one go rather than lowering it back to those.
//!
//! So there should be a difference, and this is where it is or is not.
//!
//! # Two shapes, because the first one answers the wrong question
//!
//! One dot product per element loaded is **memory-bound** on a fast device: the arithmetic hides
//! behind the load and ten instructions cost the same as one. That is the first table, and it is
//! worth having because it is what a real elementwise kernel looks like.
//!
//! The second repeats the dot product thirty-two times per element, so the loads are amortised and
//! what is left is the arithmetic. That is where an instruction that replaces eleven others has
//! somewhere to show up.
//!
//! Caveats: one device per run, and a synthetic loop rather than a real layer. What this can say is
//! whether the instruction does anything and when; what it cannot say is what a given kernel gains.

use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::pack;
use std::time::Duration;

/// Dispatch sizes, and how many timed iterations at each.
const SIZES: [(u32, u32); 2] = [(64, 400), (4_096, 100)];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    let width = limits.subgroup_size;
    println!("{} — subgroup {width}", limits.name);
    println!(
        "  shaderIntegerDotProduct {}   4x8-bit packed signed accelerated {}\n",
        limits.narrow.integer_dot_product, limits.narrow.packed_dot_accelerated
    );

    if !limits.narrow.integer_dot_product {
        println!("this device has no integer dot product, so there is nothing to compare");
        return Ok(());
    }

    let packed = kernels::packed_dot(width)?;
    let unpacked = kernels::unpacked_dot(width)?;

    println!("one dot product per element — the load is most of the work");
    compare(&gpu, &packed, &unpacked)?;

    // Thirty-two per element, so the loads are amortised and the arithmetic is what is left.
    let repeats = 32;
    let repeated_packed = kernels::repeated_packed_dot(width, repeats)?;
    let repeated_unpacked = kernels::repeated_unpacked_dot(width, repeats)?;

    println!("\n{repeats} dot products per element — the arithmetic is most of the work");
    compare(&gpu, &repeated_packed, &repeated_unpacked)?;

    println!(
        "\nBoth kernels of each pair read the same buffer and write the same answer;\n\
         `runner/tests/dot_product.rs` checks that they agree, and against a host reference so\n\
         that agreeing with each other is not enough."
    );

    Ok(())
}

/// Time the two spellings against each other at every size.
fn compare(gpu: &Gpu, packed: &[u32], unpacked: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{:>12} {:>14} {:>16} {:>10}",
        "invocations", "OpSDot", "written out", "faster"
    );

    for (workgroups, iterations) in SIZES {
        let invocations = (workgroups * WORKGROUP_SIZE) as usize;
        let input: Vec<u32> = (0..invocations)
            .map(|index| pack([index as i32 % 100 - 50, 2, -3, 4]))
            .collect();

        // One untimed pass each, so the driver's lazy pipeline work stays out of the measurement.
        gpu.time(packed, &input, workgroups, 1)?;
        gpu.time(unpacked, &input, workgroups, 1)?;

        let one = gpu.time(packed, &input, workgroups, iterations)? / iterations;
        let many = gpu.time(unpacked, &input, workgroups, iterations)? / iterations;

        println!(
            "{:>12} {:>14} {:>16} {:>10}",
            thousands(invocations),
            micros(one),
            micros(many),
            format!("{:.2}x", many.as_secs_f64() / one.as_secs_f64())
        );
    }

    Ok(())
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
