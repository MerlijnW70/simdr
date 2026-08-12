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

use runner::reduction::dispatches_for;
use runner::{Gpu, Timing};
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

    // The largest size's held time, kept so the breakdown below reports shares of a number this
    // run measured rather than one written down from a previous one.
    let mut largest_held = Duration::ZERO;

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
        largest_held = held;

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

    breakdown(&gpu, largest_held)?;
    Ok(())
}

/// Where a held reduction's remaining time actually goes.
///
/// `notes/NEXT.md` proposed shortening the between-pass copies on the grounds that they were
/// probably most of what `Reducer::sum` still costs at 2²⁰, and said to time them first. This is
/// the timing, and it also measures the two things the copies would have to beat.
///
/// Each row is a *difference between two calls that differ in one thing*, rather than a subtraction
/// from an estimate:
///
/// - **between-pass copies** — a chain of empty kernels at [`LONG`] passes against the same chain
///   at one. Both do nothing else; the difference is the copies and nothing but.
/// - **host upload / download** — `Session::write` and `Session::read` on a session that is already
///   built, so no allocation and no pipeline creation is in the number.
///
/// `whole` is what `Reducer::sum` took at the same size in the table above, so the shares are
/// against a number measured in this run rather than one copied from a previous one.
///
/// # Why the long chain
///
/// The copy row is a difference between two ~2 ms numbers. At 15 passes that difference is about
/// a tenth of either, and the repeats are about a tenth apart — so the signal and the noise were
/// the same size, and two runs of this file reported 188 µs and 337 µs for the same thing.
///
/// [`LONG`] makes the difference roughly half of the measurement instead of a tenth, which is the
/// whole fix: the repeats are no steadier, they just no longer swamp what is being measured. Both
/// spreads are printed so a reader can check that rather than take it on trust.
fn breakdown(gpu: &Gpu, whole: Duration) -> Result<(), Box<dyn std::error::Error>> {
    const ELEMENTS: usize = 1 << 20;
    const REPEATS: usize = 20;
    /// Passes in the long chain, and therefore `LONG - 1` copies.
    const LONG: usize = 61;

    let megabytes = (ELEMENTS * size_of::<u32>()) as f64 / 1e6;
    let empty = runner::kernels::empty(gpu.limits().subgroup_size)?;
    let input: Vec<u32> = vec![0; ELEMENTS];

    println!("\nwhere a held reduction over {megabytes:.0} MB spends its time:");

    // Three chains, differing in one thing each:
    //   one pass          — no copies, no barriers
    //   LONG, whole buffer — LONG-1 copies of 4 MB, each between two pipeline barriers
    //   LONG, one word     — the same LONG-1 barrier pairs, carrying almost nothing
    //
    // The last is what separates the copy from the barriers around it, and it is the reason the
    // end-to-end saving from shortening the copies is smaller than the copies measure.
    let mut chained = Vec::with_capacity(3);
    for (passes, outputs) in [(1_usize, None), (LONG, None), (LONG, Some(1_usize))] {
        let chain: Vec<runner::Pass<'_>> = (0..passes)
            .map(|_| match outputs {
                None => runner::Pass::new(&empty, 1),
                Some(words) => runner::Pass::writing(&empty, 1, words),
            })
            .collect();
        gpu.run_chain(&chain, &input)?;

        let mut samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = Instant::now();
            gpu.run_chain(&chain, &input)?;
            samples.push(started.elapsed());
        }
        chained.push(Timing::of(&samples).ok_or("no samples")?);
    }

    let (short, long, thin) = (
        chained.first().ok_or("no short chain")?,
        chained.get(1).ok_or("no long chain")?,
        chained.get(2).ok_or("no thin chain")?,
    );
    let steps = LONG as u32 - 1;
    let each = long.median.saturating_sub(short.median) / steps;
    let barrier_each = thin.median.saturating_sub(short.median) / steps;
    let payload_each = each.saturating_sub(barrier_each);
    // Fourteen full-buffer copies is what a 15-fold reduction over this buffer used to record.
    let copies = each * 14;
    // And what it records now: fourteen barrier pairs that stay whatever the copies carry, plus
    // one buffer's worth of payload between them — the copy before each pass is as long as that
    // pass reads, and those halve to 2^19 + 2^18 + … + 64 words in total.
    let shortened = barrier_each * 14 + payload_each;

    println!(
        "  one whole-buffer step costs about {} — medians of {} at {LONG} passes against {} at\n\
         one, over {REPEATS} repeats each. Of that, about {} is the two pipeline barriers around\n\
         the copy and about {} is the {megabytes:.0} MB itself.",
        micros(each),
        micros(long.median),
        micros(short.median),
        micros(barrier_each),
        micros(payload_each),
    );
    println!(
        "  worst repeat over best: {:.2}×, {:.2}×, {:.2}× — {}",
        long.spread(),
        short.spread(),
        thin.spread(),
        if long.is_steady() && short.is_steady() && thin.is_steady() {
            "steady, so the differences above are worth quoting"
        } else {
            "NOT steady; the differences above are noise as much as measurement"
        }
    );
    println!();

    // The host copies, on a session that is already built.
    let mut session = gpu.session(&empty, &[ELEMENTS, ELEMENTS])?;
    session.write(0, &input)?;
    session.read(1, ELEMENTS)?;

    let started = Instant::now();
    for _ in 0..REPEATS {
        session.write(0, &input)?;
    }
    let upload = started.elapsed() / REPEATS as u32;

    let started = Instant::now();
    for _ in 0..REPEATS {
        session.read(1, ELEMENTS)?;
    }
    let download = started.elapsed() / REPEATS as u32;

    println!(
        "{:>34} {:>12} {:>10}   (of {})",
        "",
        "per call",
        "share",
        micros(whole)
    );
    for (name, taken) in [
        ("fourteen FULL-buffer copies (was)", copies),
        ("the shortened copies (is)", shortened),
        ("host upload of the input", upload),
        ("host download of the output", download),
    ] {
        let share = taken.as_secs_f64() / whole.as_secs_f64() * 100.0;
        println!("{name:>34} {:>12} {share:>9.0}%", micros(taken));
    }

    println!(
        "\n  The first row is what the chain did before `Pass::writing`. The second is what it\n\
         does now, and the difference between them is the *payload* only — the fourteen barrier\n\
         pairs are in both, because a pass still has to wait for the one before it whatever it is\n\
         handed. That is why the end-to-end number moved by far less than the first row suggests,\n\
         and it is what a ping-pong across two descriptor sets would remove instead.\n\n\
       \x20 Shares are against the `Reducer::sum` time in the table above, same run, same device.\n\
         `notes/NEXT.md` expected the copies to dominate. The two host transfers do."
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
