//! What a binding costs, for a caller deciding between one buffer and several.
//!
//! [`Gpu::run_bound`] uploads each input through the shared staging buffer in turn, and every one
//! of those is a `Gpu::copy` — a whole command buffer, submission and fence of its own. So `k`
//! inputs cost `k + 2` submissions where a chain costs one, which is the shape
//! `notes/FINDINGS.md` records being worth 116 µs when it was fixed for the reduction.
//!
//! Whether it is worth fixing here is what this measures rather than assumes.
//!
//! # The comparison
//!
//! `kernels::network::clipped_dot` reads its activations and its weights from **one** buffer with
//! the join passed as an offset. `clipped_dot_split` reads them from **two**. `runner/tests/
//! network.rs` asserts the two give the same answer, so the difference between them is the second
//! binding and nothing else — one extra upload submission, against one fewer offset in the
//! addressing.
//!
//! Both are one-shot calls that build a pipeline per invocation, so both pay that equally and it
//! cancels out of the difference.

mod common;

use runner::kernels::{
    WORKGROUP_SIZE,
    network::{Layer, clipped_dot, clipped_dot_split},
};
use runner::{Error, Gpu};
use std::time::Duration;

/// How many times each side runs. Enough that the median is not one scheduling accident.
const REPEATS: u32 = 40;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        eprintln!("no Vulkan device");
        return Ok(());
    };
    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}\n", gpu.limits().name);

    if width != 32 {
        println!("the network kernels here are written for a 32-wide subgroup; skipping");
        return Ok(());
    }

    println!(
        "{:>12} {:>14} {:>14} {:>12}",
        "operands", "one buffer", "two buffers", "difference"
    );

    for operands in [WORKGROUP_SIZE as usize * 8, WORKGROUP_SIZE as usize * 64] {
        let activations: Vec<i32> = (0..operands).map(|index| (index % 127) as i32).collect();
        let weights: Vec<i32> = (0..operands)
            .map(|index| (index % 31) as i32 - 15)
            .collect();

        let joined_words: Vec<u32> = activations
            .iter()
            .chain(weights.iter())
            .map(|value| bits(*value))
            .collect();
        let left: Vec<u32> = activations.iter().map(|value| bits(*value)).collect();
        let right: Vec<u32> = weights.iter().map(|value| bits(*value)).collect();

        let joined = clipped_dot::<256>(width, operands as u32, Layer::QA).map_err(Error::Emit)?;
        let split = clipped_dot_split::<256>(width, Layer::QA).map_err(Error::Emit)?;

        // Once each before timing, so neither side pays for a cold driver the other does not.
        gpu.run_u32(&joined, &joined_words, 1)?;
        gpu.run_bound(&split, &[&left, &right], operands, 1)?;

        let one = common::host(REPEATS, || {
            gpu.run_u32(&joined, &joined_words, 1).map(|_| ())
        })?;
        let two = common::host(REPEATS, || {
            gpu.run_bound(&split, &[&left, &right], operands, 1)
                .map(|_| ())
        })?;

        // The difference is the whole point of the table and it is a *subtraction*, so it inherits
        // the instability of both sides rather than either. Marked when either wandered.
        let steady = one.is_steady() && two.is_steady();
        println!(
            "{operands:>12} {:>14} {:>14} {:>12}",
            common::marked(one, 1),
            common::marked(two, 1),
            format!(
                "{}{}",
                signed_micros(two.median, one.median),
                if steady { "" } else { "!" }
            ),
        );
    }
    println!("\n{}", common::LEGEND);

    println!(
        "\n  Two buffers is one more upload submission — a command buffer, a submission and a\n\
         fence, which `runner/examples/reducer.rs` prices at 50-80 us on this device. One buffer\n\
         is one upload and an offset in the addressing instead.\n\n\
       \x20 If the difference is around that price, the submission is what it costs and\n\
         `run_bound` recording its uploads inside one command buffer would recover it. If it is\n\
         not, the submission is hiding behind something larger and the change is not worth making\n\
         — which is a result, and `notes/NEXT.md` has two other items that ended that way."
    );

    Ok(())
}

/// `value` as the bits a `u32` buffer carries it in.
fn bits(value: i32) -> u32 {
    u32::from_ne_bytes(value.to_ne_bytes())
}

/// The difference between two of them, with its sign.
fn signed_micros(left: Duration, right: Duration) -> String {
    let difference = left.as_secs_f64() - right.as_secs_f64();
    format!("{:+.1} us", difference * 1e6)
}
