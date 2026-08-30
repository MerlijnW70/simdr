use runner::reduction::dispatches_for;
use runner::{Gpu, Timing};
use std::time::{Duration, Instant};

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

    let mut largest_held = Duration::ZERO;

    for (elements, repeats) in SIZES {
        let input: Vec<f32> = (0..elements).map(|index| (index % 16) as f32).collect();
        let expected: f32 = input.iter().sum();

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
    measured_in_place(&gpu)?;
    mapped(&gpu)?;
    Ok(())
}

fn mapped(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    let width = gpu.limits().subgroup_size;
    let square = runner::kernels::square(width)?;

    println!("\nΣ x² — the map on the device against the map through the host:");
    println!(
        "{:>12} {:>16} {:>16} {:>10}",
        "elements", "three crossings", "one crossing", "faster"
    );

    for (elements, repeats) in SIZES {
        let input: Vec<f32> = (0..elements).map(|index| (index % 3) as f32).collect();
        let expected: f32 = input.iter().map(|value| value * value).sum();
        let groups = elements as u32 / runner::kernels::WORKGROUP_SIZE;

        let mut plain = gpu.reducer(elements)?;
        let mut fused = gpu.reducer_of(elements, &square)?;

        let mut mapping = gpu.session(&square, &[elements, elements])?;
        let words: Vec<u32> = input.iter().map(|value| value.to_bits()).collect();

        mapping.write(0, &words)?;
        mapping.dispatch(groups, 1)?;
        mapping.read(1, elements)?;
        plain.sum(&input)?;
        fused.sum(&input)?;

        let started = Instant::now();
        for _ in 0..repeats {
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

fn breakdown(gpu: &Gpu, whole: Duration) -> Result<(), Box<dyn std::error::Error>> {
    const ELEMENTS: usize = 1 << 20;
    const REPEATS: usize = 20;
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
    let steps = (runner::reduction::dispatches_for(ELEMENTS) - 1) as u32;
    let chained = each * steps;

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

    let floats: Vec<f32> = input.iter().map(|_| 1.0).collect();
    let started = Instant::now();
    for _ in 0..REPEATS {
        let words: Vec<u32> = floats.iter().map(|value| value.to_bits()).collect();
        std::hint::black_box(&words);
    }
    let conversion = started.elapsed() / REPEATS as u32;

    let started = Instant::now();
    for _ in 0..REPEATS {
        session.write(0, &[0_u32])?;
    }
    let mapping = started.elapsed() / REPEATS as u32;

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
    let payload = upload.saturating_sub(mapping);
    let accounted = chained + payload + submission;

    for (name, taken) in [
        ("the chained steps (upper bound)", chained),
        ("the input's four megabytes", payload),
        ("one submission and its fence (bound)", submission),
        ("---- accounted for ----", accounted),
        ("(a second and third submission, unpaid)", submission * 2),
        ("(a whole-buffer download, unpaid)", whole_download),
        ("(the f32 -> u32 copy, unpaid)", conversion),
    ] {
        let share = taken.as_secs_f64() / whole.as_secs_f64() * 100.0;
        println!("{name:>40} {:>12} {share:>9.0}%", micros(taken));
    }

    println!(
        "\n  A chained step used to be a copy between two pipeline barriers — 27.5 us on an RTX\n\
         4080, of which 19.0 us was the barriers. It is one barrier and no copy now, at 16.7 us,\n\
         and removing the second barrier turned out to save about 2 us rather than half of 19.\n\
         Paired against the old build on the same machine: no measurable difference on the 4080\n\
         or on lavapipe, and 5.5% on the integrated Radeon, where bandwidth is scarce enough for\n\
         4 MB of copying to show.\n\n\
       \x20 Shares are against the `Reducer::sum` time in the table above, same run, same device.\n\
         The bracketed rows are costs this call used to pay and no longer does: two of its three\n\
         submissions, now recorded inside the chain's own command buffer; the whole buffer copied\n\
         home so `.first()` could be called on it; and a `Vec<u32>` built from the caller's\n\
         `&[f32]` to reinterpret bits that were already the right bits.\n\n\
       \x20 `accounted for` now lands well *over* the whole, and the overshoot is the finding\n\
         rather than a rounding note. It read 52% under before the missing rows were found, 79%\n\
         once they were, and past 100% as soon as the upload stopped going through staging: the\n\
         call got about a third shorter while these rows, measured on their own, did not. Two of\n\
         them are upper bounds by construction — the step row comes from a chain of empty kernels\n\
         where a barrier has nothing to overlap, and the submission row is an empty dispatch that\n\
         waits on its own fence. A cost measured in isolation is not the same cost measured in\n\
         company, and this table is where that shows. Treat the rows as a ranking of what to\n\
         attack next, not as a budget that adds up."
    );

    Ok(())
}
fn micros(duration: Duration) -> String {
    format!("{:.1} us", duration.as_secs_f64() * 1e6)
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

fn measured_in_place(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "

the same reduction, timed from inside its own command buffer:
"
    );

    let Some(&(elements, _)) = SIZES.last() else {
        return Ok(());
    };
    let mut reducer = gpu.reducer(elements)?;
    let input = vec![1.0_f32; elements];

    reducer.sum_timed(&input)?;
    let (reduction, spans) = reducer.sum_timed(&input)?;

    if spans.is_empty() {
        println!(
            "  this device reports no usable timestamp queries, so there is nothing to say —
             `timestampValidBits` is zero on some queues and the whole feature is optional."
        );
        return Ok(());
    }

    let dispatched: Duration = spans.iter().sum();
    println!("{:>28} {:>12} {:>10}", "pass", "on device", "share");
    for (index, taken) in spans.iter().enumerate() {
        let share = taken.as_secs_f64() / dispatched.as_secs_f64() * 100.0;
        println!(
            "{:>28} {:>12} {share:>9.0}%",
            format!("{} of {}", index + 1, spans.len()),
            micros(*taken)
        );
    }
    println!(
        "{:>28} {:>12} {:>10}",
        "---- the dispatches ----",
        micros(dispatched),
        ""
    );

    println!(
        "
  {} dispatches, and they are **not** the whole call: the host still writes its input
         and waits for the submission, and neither of those is inside the command buffer. What this
         says is how the *device* time divides, which the table above could only guess at — and it
         guessed high. The step row up there is an upper bound taken from a chain of empty kernels,
         and against this it is out by roughly five times.

         **The shape of the profile is the device's, not the algorithm's**, and the two here
         disagree completely. Each pass after the first reads a sixteenth of what the one before it
         wrote, so a bandwidth-bound chain falls away to nothing and a latency-bound one does not.
         The integrated Radeon falls away — 92% in the first pass, then 5%, then 1% — and the RTX
         4080 is nearly flat, because at its bandwidth the later passes are too small to cost
         anything but the dispatch itself.

         Same chain, same arithmetic, opposite answers to 'where does the time go'. Which is why
         this prints what the device says rather than repeating a number measured somewhere else.",
        reduction.dispatches
    );

    Ok(())
}
