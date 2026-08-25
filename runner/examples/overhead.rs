//! Where the time goes when you ask for one answer.
//!
//! `examples/latency.rs` reported ~940 us per round trip against a chess engine's 199 ns, and said
//! the number "overstates the floor" because `Gpu::run` allocates buffers and builds a pipeline on
//! every call. That was a reasonable thing to say and it was not measured. This measures it.
//!
//! The method is subtraction, twice over:
//!
//! - The **device clock** reports what the dispatch itself cost. `Gpu::time` reads it.
//! - The **host clock** around `Gpu::run` reports everything: allocation, pipeline creation, the
//!   uploads, the submit, the fence, the readback.
//! - Varying the buffer size while holding the kernel empty separates the fixed per-call cost from
//!   the per-byte one.
//!
//! What is left after both subtractions is the part a persistent-resource API could remove, and
//! that is the only honest way to say how much is on the table.

mod common;

use runner::Gpu;
use runner::kernels;
use std::time::Duration;

/// Buffer sizes in words. The kernel is empty at every one, so any difference is not the kernel.
const SIZES: [(usize, &str); 5] = [
    (64, "256 B"),
    (16_384, "64 KB"),
    (262_144, "1 MB"),
    (4_194_304, "16 MB"),
    (16_777_216, "64 MB"),
];

/// How many round trips to average.
const TRIPS: u32 = 40;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}\n", gpu.limits().name);

    let empty = kernels::empty(width)?;

    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>10}",
        "buffer", "round trip", "dispatch", "overhead", "MB/s"
    );

    let mut fixed = Duration::ZERO;
    let mut fixed_steady = true;
    for (words, label) in SIZES {
        let input = vec![1_u32; words];

        // Warm: the first call of a module pays for pipeline compilation.
        gpu.run_u32(&empty, &input, 1)?;

        let round = common::host(common::SAMPLES, || {
            for _ in 0..TRIPS {
                gpu.run_u32(&empty, &input, 1)?;
            }
            Ok::<(), runner::Error>(())
        })?;
        let trip = round.median / TRIPS;

        // The device's own clock, for the dispatch alone. Everything else in `run` is untimed by
        // it: the allocations, both copies, and the fence.
        let timed = gpu.time_repeated(&empty, &input, 1, 1, common::SAMPLES)?;
        let dispatch = timed.median;
        let overhead = trip.saturating_sub(dispatch);

        if words == 64 {
            fixed = overhead;
            fixed_steady = round.is_steady();
        }
        let bytes = (words * size_of::<u32>()) as f64;
        let mark = common::mark(round);

        println!(
            "{label:>8} {:>12} {:>12} {:>12} {:>10}",
            format!("{}{mark}", micros(trip)),
            format!("{}{}", micros(dispatch), common::mark(timed)),
            format!("{}{mark}", micros(overhead)),
            format!("{:.0}{mark}", bytes / overhead.as_secs_f64() / 1e6)
        );
    }

    // Split the fixed cost in two. `probe_resident` allocates a device-local buffer and frees it
    // and does nothing else — no command buffer, no submit — so it isolates `vkAllocateMemory`
    // from the three submit-and-fence round trips `run` performs.
    println!(
        "\n{:>8} {:>16} {:>18}",
        "buffer", "allocate + free", "per run (3 of them)"
    );
    for (words, label) in SIZES {
        let bytes = (words * size_of::<u32>()) as u64;

        gpu.probe_resident(bytes, 1)?;
        let allocation = common::host(common::SAMPLES, || {
            for _ in 0..TRIPS {
                gpu.probe_resident(bytes, 1)?;
            }
            Ok::<(), runner::Error>(())
        })?;
        let each = allocation.median / TRIPS;
        let mark = common::mark(allocation);

        println!(
            "{label:>8} {:>16} {:>18}",
            format!("{}{mark}", micros(each)),
            format!("{}{mark}", micros(each * 3))
        );
    }

    println!(
        "\nfixed cost, from the smallest buffer: {}{}\n{}",
        micros(fixed),
        if fixed_steady { "" } else { "!" },
        common::LEGEND
    );
    println!(
        "That is what a persistent-resource API could remove. It is charged once per `run` call,\n\
         and `run` allocates three buffers and builds a pipeline every time."
    );

    // And what the same dispatch costs when the setup is amortised across many of them, which is
    // what `Gpu::time` with a high iteration count measures.
    let input = vec![1_u32; 65_536];
    gpu.time(&empty, &input, 1, 1)?;
    let batched = gpu.time(&empty, &input, 1, 1_000)? / 1_000;
    println!(
        "\nthe same empty dispatch, amortised over a thousand of them: {}",
        micros(batched)
    );
    println!(
        "The gap between that and the fixed cost above is the whole argument for reusing\n\
         buffers and pipelines rather than rebuilding them."
    );

    Ok(())
}

/// Microseconds, which is the scale everything here lands on.
fn micros(duration: Duration) -> String {
    format!("{:.1} us", duration.as_secs_f64() * 1e6)
}
