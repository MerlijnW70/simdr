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
    mapped(&gpu)?;
    Ok(())
}

/// What not crossing the bus twice is worth.
///
/// The breakdown above says the host upload and download are most of what a held reduction still
/// costs, and that no device-side change touches them. This is the change that does: Σ x² as one
/// chain instead of three host crossings.
///
/// Both columns compute the same number and `runner/tests/reducer.rs` asserts they agree, so what
/// is being timed is only where the intermediate went.
///
/// # The left column is given every advantage
///
/// It would be easy — and wrong — to write it as `gpu.run(&square, …)`, which allocates three
/// buffers and builds a pipeline on every call. That is ~900 µs of setup this file has already
/// measured, and charging it to the old route would make the new one look two to three times
/// better than it is. So the map runs through a held [`Session`] and the reduction through a held
/// `Reducer`: nothing is allocated or built in either column, and the only difference left is that
/// one of them carries the intermediate across the bus and back.
fn mapped(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    let width = gpu.limits().subgroup_size;
    let square = runner::kernels::square(width)?;

    println!("\nΣ x² — the map on the device against the map through the host:");
    println!(
        "{:>12} {:>16} {:>16} {:>10}",
        "elements", "three crossings", "one crossing", "faster"
    );

    for (elements, repeats) in SIZES {
        // **Values of 0, 1 and 2, and both halves of that matter.** Σ x² has to stay inside the
        // 2²⁴ an `f32` counts exactly or the comparison below is a tolerance wearing an equals
        // sign — 0..7 over 2²⁰ elements sums to 18.3 million and does not, which is how this
        // assertion earned its place by failing. And 2 has to be in there: x² and x agree on 0 and
        // 1, so a map that had stopped squaring would go unnoticed without it.
        let input: Vec<f32> = (0..elements).map(|index| (index % 3) as f32).collect();
        let expected: f32 = input.iter().map(|value| value * value).sum();
        let groups = elements as u32 / runner::kernels::WORKGROUP_SIZE;

        let mut plain = gpu.reducer(elements)?;
        let mut fused = gpu.reducer_of(elements, &square)?;

        // The map, held: no allocation and no pipeline creation inside the timed loop.
        let mut mapping = gpu.session(&square, &[elements, elements])?;
        let words: Vec<u32> = input.iter().map(|value| value.to_bits()).collect();

        // Warm every path, so none pays for the driver's first compile of these modules.
        mapping.write(0, &words)?;
        mapping.dispatch(groups, 1)?;
        mapping.read(1, elements)?;
        plain.sum(&input)?;
        fused.sum(&input)?;

        let started = Instant::now();
        for _ in 0..repeats {
            // The best a caller can do today: send the input, run the map, bring the squares
            // home, send them back to be reduced.
            mapping.write(0, &words)?;
            mapping.dispatch(groups, 1)?;
            let squares: Vec<f32> = mapping
                .read(1, elements)?
                .into_iter()
                .map(f32::from_bits)
                .collect();

            let total = plain.sum(&squares)?.total;
            assert_eq!(total, expected, "the two-step sum of squares is wrong");
        }
        let stepwise = started.elapsed() / repeats;

        let started = Instant::now();
        for _ in 0..repeats {
            let total = fused.sum(&input)?.total;
            assert_eq!(total, expected, "the fused sum of squares is wrong");
        }
        let chained = started.elapsed() / repeats;

        println!(
            "{:>12} {:>16} {:>16} {:>10}",
            thousands(elements),
            micros(stepwise),
            micros(chained),
            format!("{:.1}x", stepwise.as_secs_f64() / chained.as_secs_f64())
        );
    }

    println!(
        "\n  The left column uploads the input, downloads the squares and uploads them again —\n\
         three crossings of which two are the whole buffer. The right one is a single chain: the\n\
         map is its first pass and its output is handed to the first fold on the device."
    );

    Ok(())
}

/// Where a held reduction's remaining time actually goes.
///
/// `notes/NEXT.md` asked twice about this half of the call — first whether the between-pass copies
/// dominated it (they did not; they were a fifth) and then whether the barriers around those copies
/// did (they were two thirds of the fifth). The copies are gone now, replaced by a ping-pong across
/// two descriptor sets, and one barrier per pass is what is left.
///
/// Each row is a *difference between two calls that differ in one thing*, rather than a subtraction
/// from an estimate:
///
/// - **chained step** — a chain of empty kernels at [`LONG`] passes against the same chain at one.
///   Both do nothing else, so the difference is one dispatch and the barrier before it.
/// - **host upload / download** — `Session::write` and `Session::read` on a session that is already
///   built, so no allocation and no pipeline creation is in the number.
/// - **the `f32` copy** and **one submission** — added because the rows above accounted for only
///   about half the call and the rest was going somewhere unnamed. See below.
///
/// # The rows that were missing, and why a breakdown has to add up
///
/// With three rows this table came to roughly *half* of the `Reducer::sum` it was breaking down,
/// and that gap went unremarked through three separate optimisations. A breakdown whose rows do not
/// approach the whole is not a breakdown; it is a list of things that were easy to measure.
///
/// The two that were missing are both things the *measurement* skipped rather than the call:
///
/// - `Reducer::sum` takes `&[f32]` and the buffer holds words, so it builds a `Vec<u32>` of the
///   whole input on every call. The upload row hoisted that conversion out of its timed loop —
///   measuring a cost the real call does not have, and missing one it does.
/// - The step row is a *difference* between two chains, so every fixed cost of submitting at all
///   cancels out of it. One submission and its fence is paid once per call and appeared nowhere.
///
/// `whole` is what `Reducer::sum` took at the same size in the table above, so the shares are
/// against a number measured in this run rather than one copied from a previous one.
///
/// # Why the long chain
///
/// The step row is a difference between two ~2 ms numbers. At 15 passes that difference is about a
/// tenth of either, and the repeats are about a tenth apart — so signal and noise were the same
/// size, and two runs of this file once reported 188 us and 337 us for the same quantity.
///
/// [`LONG`] makes the difference roughly half of the measurement instead of a tenth, which is the
/// whole fix: the repeats are no steadier, they just no longer swamp what is being measured. Both
/// spreads are printed so a reader can check that rather than take it on trust.
fn breakdown(gpu: &Gpu, whole: Duration) -> Result<(), Box<dyn std::error::Error>> {
    const ELEMENTS: usize = 1 << 20;
    const REPEATS: usize = 20;
    /// Passes in the long chain, and therefore `LONG - 1` barriers.
    const LONG: usize = 61;

    let megabytes = (ELEMENTS * size_of::<u32>()) as f64 / 1e6;
    let empty = runner::kernels::empty(gpu.limits().subgroup_size)?;
    let input: Vec<u32> = vec![0; ELEMENTS];

    println!("\nwhere a held reduction over {megabytes:.0} MB spends its time:");

    let mut chained = Vec::with_capacity(2);
    for passes in [1_usize, LONG] {
        let chain: Vec<runner::Pass<'_>> =
            (0..passes).map(|_| runner::Pass::new(&empty, 1)).collect();
        gpu.run_chain(&chain, &input)?;

        let mut samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = Instant::now();
            gpu.run_chain(&chain, &input)?;
            samples.push(started.elapsed());
        }
        chained.push(Timing::of(&samples).ok_or("no samples")?);
    }

    let (short, long) = (
        chained.first().ok_or("no short chain")?,
        chained.get(1).ok_or("no long chain")?,
    );
    let each = long.median.saturating_sub(short.median) / (LONG as u32 - 1);
    let fourteen = each * 14;

    println!(
        "  one chained step — a dispatch and the barrier before it — costs about {}, from\n\
         medians of {} at {LONG} passes against {} at one, over {REPEATS} repeats each.",
        micros(each),
        micros(long.median),
        micros(short.median),
    );
    println!(
        "  worst repeat over best: {:.2}x and {:.2}x — {}",
        long.spread(),
        short.spread(),
        if long.is_steady() && short.is_steady() {
            "steady, so the difference above is worth quoting"
        } else {
            "NOT steady; the difference above is noise as much as measurement"
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
    let whole_download = started.elapsed() / REPEATS as u32;

    // What a reduction actually brings home: one `f32`. It used to be the row above, which is the
    // same buffer for the same one number.
    let started = Instant::now();
    for _ in 0..REPEATS {
        session.read(1, 1)?;
    }
    let download = started.elapsed() / REPEATS as u32;

    // The `Vec<u32>` `Reducer::sum` builds from its `&[f32]` on every call. Pure host work: a
    // four-megabyte allocation and a copy, to reinterpret bits that are already the right bits.
    let floats: Vec<f32> = input.iter().map(|_| 1.0).collect();
    let started = Instant::now();
    for _ in 0..REPEATS {
        let words: Vec<u32> = floats.iter().map(|value| value.to_bits()).collect();
        std::hint::black_box(&words);
    }
    let conversion = started.elapsed() / REPEATS as u32;

    // One submission and the fence it waits on, with nothing in it. Every chain pays this once,
    // and the step row above cannot see it: that row is a *difference* between two chains, so
    // anything paid once per call cancels out of it exactly.
    let started = Instant::now();
    for _ in 0..REPEATS {
        session.dispatch(1, 1)?;
    }
    let submission = started.elapsed() / REPEATS as u32;

    println!(
        "{:>36} {:>12} {:>10}   (of {})",
        "",
        "per call",
        "share",
        micros(whole)
    );
    let accounted = fourteen + upload + download + submission;
    for (name, taken) in [
        ("fourteen chained steps", fourteen),
        ("one submission and its fence", submission),
        ("host upload of the input", upload),
        ("host download of the answer", download),
        ("---- accounted for ----", accounted),
        ("(a whole-buffer download, unpaid)", whole_download),
        ("(the f32 -> u32 copy, unpaid)", conversion),
    ] {
        let share = taken.as_secs_f64() / whole.as_secs_f64() * 100.0;
        println!("{name:>36} {:>12} {share:>9.0}%", micros(taken));
    }

    println!(
        "\n  A chained step used to be a copy between two pipeline barriers — 27.5 us on an RTX\n\
         4080, of which 19.0 us was the barriers. It is one barrier and no copy now, at 16.7 us,\n\
         and removing the second barrier turned out to save about 2 us rather than half of 19.\n\
         Paired against the old build on the same machine: no measurable difference on the 4080\n\
         or on lavapipe, and 5.5% on the integrated Radeon, where bandwidth is scarce enough for\n\
         4 MB of copying to show. notes/FINDINGS.md has the runs.\n\n\
       \x20 Shares are against the `Reducer::sum` time in the table above, same run, same device.\n\
         The two bracketed rows are costs this call used to pay and does not: the whole buffer\n\
         copied home so `.first()` could be called on it, and a `Vec<u32>` built from the\n\
         caller's `&[f32]` to reinterpret bits that were already the right bits.\n\n\
       \x20 `accounted for` comes to a little over the whole because the rows are measured\n\
         separately and overlap — the upload row maps and unmaps the staging buffer, and so does\n\
         the call. Over is the honest direction for that; it was 52% under before the two rows\n\
         below the line were found, and that gap is what a breakdown is for."
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
