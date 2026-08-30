use runner::Gpu;
use runner::kernels::network::{Layer, bits, clipped_dot, unclipped_dot};
use std::time::Instant;

const WIDTH: usize = 256;

const PER_POSITION: usize = 2 * WIDTH;

const ENGINE_EVALS_PER_SECOND: f64 = 5.0e6;

const ROUND_TRIPS: u32 = 100;

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

fn clamp_cost(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
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

fn verdict() {
    println!("what this says");
    println!("  - in search: no. One answer costs more than the engine spends on thousands,");
    println!("    and alpha-beta cannot ask for two at a time.");
    println!("  - in a trainer or a validation sweep: worth measuring against the real thing,");
    println!("    because those do have thousands of independent positions in hand.");
    println!("  - the comparison is one GPU against one CPU thread, and the engine's own");
    println!("    numbers say evaluation is ~20% of search time. Free eval caps the win at ~25%.");
}

fn payload(positions: usize) -> Vec<u32> {
    let count = positions * PER_POSITION;
    let mut words = Vec::with_capacity(count * 2);
    words.extend((0..count).map(|index| bits(activation(index))));
    words.extend((0..count).map(|index| bits((index % 255) as i32 - 127)));
    words
}

fn activation(index: usize) -> i32 {
    match index % 4 {
        0 => -(index as i32 % 500) - 1,
        1 => (index % 200) as i32,
        2 => 250 + (index % 20) as i32,
        _ => 1_000 + (index % 3_000) as i32,
    }
}
