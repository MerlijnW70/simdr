//! `simdr` pointed at a real workload: a chess engine's NNUE output layer.
//!
//! The engine is `H:\schaak` — zero dependencies, no `unsafe`, same discipline as this project.
//! Its network is `768 → 256×2 → 1`, quantised, and the whole per-evaluation arithmetic once the
//! accumulator is current is two 256-element clipped-ReLU dot products. That is what runs here,
//! at those dimensions, with that clamp.
//!
//! # What this measures, and what it cannot
//!
//! Two numbers, and the gap between them is the answer:
//!
//! - **Latency**: one evaluation, asked for and waited on. What a game-tree search would pay.
//! - **Throughput**: many evaluations at once. What a trainer or a validation sweep would pay.
//!
//! The engine's own recorded figures (`H:\schaak\NNUE.md`) are the comparison: 5.0 M evals/s on
//! one CPU thread at full refresh, 199 ns each. Those are single-threaded and this is a whole GPU,
//! so a throughput win here is not a like-for-like victory — it is an upper bound on what
//! offloading could ever be worth, measured rather than assumed.
//!
//! Nothing here plugs into the engine. It runs the engine's arithmetic at the engine's size, which
//! is what decides whether plugging in would be worth doing.

use runner::Gpu;
use runner::kernels::network::{Layer, bits, clipped_dot, unclipped_dot};
use std::time::Instant;

/// One layer: 256 elements, the engine's `HIDDEN`.
const WIDTH: usize = 256;

/// Two layers per position — one per perspective — which one workgroup covers.
const PER_POSITION: usize = 2 * WIDTH;

/// The engine's single-threaded full-refresh rate, from `H:\schaak\NNUE.md:154`.
const ENGINE_EVALS_PER_SECOND: f64 = 5.0e6;

/// How many round trips to time.
const ROUND_TRIPS: u32 = 100;

/// How many batched dispatches to time together.
const BATCHED: u32 = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    println!("{} — subgroup {}", limits.name, limits.subgroup_size);
    if limits.subgroup_size != 32 {
        println!("this example is written for a 32-wide subgroup; skipping");
        return Ok(());
    }
    println!("layer: {WIDTH} elements, clamp [0, {}], i32\n", Layer::QA);

    latency(&gpu)?;
    throughput(&gpu)?;
    clamp_cost(&gpu)?;
    verdict();

    Ok(())
}

/// One evaluation, asked for and waited on.
fn latency(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    let spirv = clipped_dot::<256>(32, PER_POSITION as u32, Layer::QA)?;
    let input = payload(1);

    gpu.run_u32(&spirv, &input, 1)?;

    let started = Instant::now();
    for _ in 0..ROUND_TRIPS {
        gpu.run_u32(&spirv, &input, 1)?;
    }
    let each = started.elapsed() / ROUND_TRIPS;
    let rate = 1.0 / each.as_secs_f64();

    println!("one position, host waits for the answer");
    println!("  {:>12.1} us per evaluation", each.as_secs_f64() * 1e6);
    println!("  {:>12.0} evaluations/s", rate);
    println!(
        "  {:>12.0}x slower than the engine's 5.0 M/s on one CPU thread\n",
        ENGINE_EVALS_PER_SECOND / rate
    );

    Ok(())
}

/// As many evaluations at once as the device will take.
///
/// The largest size here holds 268 MB of operands, which is past the point where this project's
/// measurements have ever been trustworthy — the `spread` column is what says so on the spot,
/// rather than the reader finding out by running it twice.
fn throughput(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    println!("many positions at once, dispatch time only");
    println!(
        "{:>10} {:>10} {:>12} {:>15} {:>11} {:>8}",
        "positions", "operands", "per dispatch", "evaluations/s", "vs engine", "spread"
    );

    for positions in [1_u32, 64, 1_024, 8_192, 65_536] {
        let spirv = clipped_dot::<256>(32, positions * PER_POSITION as u32, Layer::QA)?;
        let input = payload(positions as usize);
        let megabytes = (input.len() * size_of::<u32>()) as f64 / 1e6;

        gpu.time(&spirv, &input, positions, 1)?;
        let timing = gpu.time_repeated(&spirv, &input, positions, BATCHED, 5)?;
        let each = timing.best / BATCHED;
        let rate = f64::from(positions) / each.as_secs_f64();

        println!(
            "{positions:>10} {:>10} {:>12} {:>15} {:>10.1}x {:>7.1}x{}",
            format!("{megabytes:.0} MB"),
            format!("{:.1} us", each.as_secs_f64() * 1e6),
            format!("{:.2} M", rate / 1e6),
            rate / ENGINE_EVALS_PER_SECOND,
            timing.spread(),
            if timing.is_steady() { "" } else { "  !" }
        );
    }
    println!("  ! — the repeats disagreed; that row is not a measurement of the kernel\n");

    Ok(())
}

/// What the clamp costs, since there is no elementwise min or max to do it in one.
///
/// Two kernels differing by four instructions per element, timed back to back. The pair is run
/// several times over rather than once each: an A-then-B comparison at one size is exactly the
/// shape that produced two retracted claims in `notes/FINDINGS.md`, and the only defence is to
/// show the spread and refuse the ratio when it is wide.
fn clamp_cost(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    // Small enough that the working set is not the thing being measured: 8 MB of operands.
    let positions = 2_048_u32;
    let offset = positions * PER_POSITION as u32;
    let input = payload(positions as usize);

    let with = clipped_dot::<256>(32, offset, Layer::QA)?;
    let without = unclipped_dot::<256>(32, offset)?;

    println!("the clamp, priced");
    let mut measured = Vec::new();
    for (label, spirv) in [
        ("with the clamp", &with),
        ("without (a linear map, not a network)", &without),
    ] {
        gpu.time(spirv, &input, positions, 1)?;
        let timing = gpu.time_repeated(spirv, &input, positions, BATCHED, 9)?;
        let each = timing.best / BATCHED;

        println!(
            "  {:<38} {:>9} spread {:>5.2}x{}",
            label,
            format!("{:.2} us", each.as_secs_f64() * 1e6),
            timing.spread(),
            if timing.is_steady() { "" } else { "  !" }
        );
        measured.push((each, timing.is_steady(), timing.spread()));
    }

    // A ratio is worth printing only when the gap between the two is larger than the wobble
    // within either. 1% apart with a 3% spread is not a small difference — it is no difference,
    // and reporting it as "1.01x" would be inventing a digit the measurement does not have.
    match (measured.first(), measured.get(1)) {
        (Some(&(clipped, steady_a, spread_a)), Some(&(plain, steady_b, spread_b))) => {
            let ratio = clipped.as_secs_f64() / plain.as_secs_f64();
            let noise = spread_a.max(spread_b);

            if !steady_a || !steady_b {
                println!(
                    "  no ratio: a side wandered between repeats, so what separates them is not \
                     the clamp\n"
                );
            } else if (ratio - 1.0).abs() < noise - 1.0 {
                println!(
                    "  no measurable cost. The two differ by {:.1}% and either alone wobbles by \
                     {:.1}%,\n  so the four extra instructions per element are hidden — this \
                     kernel is waiting on memory,\n  not on arithmetic.\n",
                    (ratio - 1.0).abs() * 100.0,
                    (noise - 1.0) * 100.0
                );
            } else {
                println!("  {ratio:.2}x — larger than either side's spread, so it is real\n");
            }
        }
        _ => println!("  nothing measured\n"),
    }

    Ok(())
}

/// State the conclusion, so a reader of the output does not have to infer it.
fn verdict() {
    println!("what this says");
    println!("  - in search: no. One answer costs more than the engine spends on thousands,");
    println!("    and alpha-beta cannot ask for two at a time.");
    println!("  - in a trainer or a validation sweep: worth measuring against the real thing,");
    println!("    because those do have thousands of independent positions in hand.");
    println!("  - the comparison is one GPU against one CPU thread, and the engine's own");
    println!("    numbers say evaluation is ~20% of search time. Free eval caps the win at ~25%.");
}

/// `positions` positions' worth of activations followed by the same many weights.
///
/// The values are the shape a real accumulator has — some below the floor, some above the
/// ceiling — because a payload that never triggered the clamp would be timing a different kernel.
fn payload(positions: usize) -> Vec<u32> {
    let count = positions * PER_POSITION;
    let mut words = Vec::with_capacity(count * 2);
    words.extend((0..count).map(|index| bits(activation(index))));
    words.extend((0..count).map(|index| bits((index % 255) as i32 - 127)));
    words
}

/// One accumulator value.
fn activation(index: usize) -> i32 {
    match index % 4 {
        0 => -(index as i32 % 500) - 1,
        1 => (index % 200) as i32,
        2 => 250 + (index % 20) as i32,
        _ => 1_000 + (index % 3_000) as i32,
    }
}
