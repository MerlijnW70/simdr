//! What holding a reduction's pipelines is worth.
//!
//! `runner/examples/specialize.rs` measured a pipeline at ~485 µs against a dispatch at ~0.8 µs,
//! and `Gpu::sum` builds one per fold on every call. This is the other side of that number: the
//! same reduction, asked repeatedly, with the pipelines built once instead of every time.
//!
//! Both columns do identical work on the device — the same modules, the same dispatch counts, the
//! same host copies. The only difference is what is rebuilt between calls.
//!
//! Caveats, because a benchmark without them is a claim. One device, one run. The host copy is
//! included in both and is the same in both. And a reduction is a chain of a dozen dispatches, so
//! this is a larger saving than a single-kernel caller would see — `Session` is the number for
//! that, and it is in the README.

use runner::Gpu;
use runner::reduction::dispatches_for;
use std::time::{Duration, Instant};

/// Element counts to measure at, and how many reductions to time at each.
///
/// Two sizes because one cannot separate setup from work: if the small and the large case save the
/// same absolute time, what was removed is per-call rather than per-element — which is exactly the
/// claim being made, so it should be visible.
const SIZES: [(usize, u32); 2] = [(1 << 13, 40), (1 << 20, 10)];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    println!(
        "{} — subgroup {}\n",
        gpu.limits().name,
        gpu.limits().subgroup_size
    );
    println!(
        "{:>12} {:>10} {:>14} {:>14} {:>10}",
        "elements", "folds", "Gpu::sum", "Reducer::sum", "faster"
    );

    for (elements, repeats) in SIZES {
        let input: Vec<f32> = (0..elements).map(|index| (index % 16) as f32).collect();
        let expected: f32 = input.iter().sum();

        // Warm both paths, so neither pays for the driver's first compile of these modules.
        gpu.sum(&input)?;
        let mut reducer = gpu.reducer(elements)?;
        reducer.sum(&input)?;

        let started = Instant::now();
        for _ in 0..repeats {
            let reduction = gpu.sum(&input)?;
            assert_eq!(reduction.total, expected, "the one-shot sum is wrong");
        }
        let fresh = started.elapsed() / repeats;

        let started = Instant::now();
        for _ in 0..repeats {
            let reduction = reducer.sum(&input)?;
            assert_eq!(reduction.total, expected, "the held sum is wrong");
        }
        let held = started.elapsed() / repeats;

        println!(
            "{:>12} {:>10} {:>14} {:>14} {:>10}",
            thousands(elements),
            dispatches_for(elements),
            micros(fresh),
            micros(held),
            format!("{:.1}x", fresh.as_secs_f64() / held.as_secs_f64())
        );
    }

    println!(
        "\nBoth columns run the same dispatches over the same data. What the right one does not do\n\
         is allocate three buffers and build one pipeline per fold, every time."
    );

    Ok(())
}

/// Microseconds, which is the scale these land on.
fn micros(duration: Duration) -> String {
    format!("{:.1} us", duration.as_secs_f64() * 1e6)
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
