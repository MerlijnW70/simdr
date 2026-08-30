use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;
use std::time::{Duration, Instant};

const ROUND_TRIPS: u32 = 200;

const WIDE: usize = 65_536;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    let width = limits.subgroup_size;
    println!("{} — subgroup {width}", limits.name);
    println!("every figure below is host wall clock: submit, wait, and read the answer back.\n");

    if width < 32 {
        println!(
            "SKIPPED latency: the round trip is timed over `Simd<f32, 32>`, one strip at a \
             subgroup of 32 and four at {width} — which needs four times the buffer this fixed \
             input holds. The numbers are a device's anyway, and a software rasteriser has none \
             to give."
        );
        return Ok(());
    }

    let spirv = kernels::lane_sum::<F32, 32>(width)?;
    let one = vec![1.0_f32; WORKGROUP_SIZE as usize];
    let answers_in_one = (WORKGROUP_SIZE / width.max(1)).max(1) as f64;

    heading();

    let each = timed(ROUND_TRIPS, || {
        gpu.run(&spirv, &one, 1)?;
        Ok(())
    })?;
    row(
        &format!("{}, built per call", answers_of(answers_in_one)),
        each,
        answers_in_one,
    );

    let words = to_words(&one);
    let mut session = gpu.session(&spirv, &[words.len(), words.len()])?;
    session.write(0, &words)?;

    let mut device_time = Duration::ZERO;
    let mut calls = 0_u32;
    let each = timed(ROUND_TRIPS, || {
        device_time += session.dispatch(1, 1)?;
        calls += 1;
        session.read(1, words.len())?;
        Ok(())
    })?;
    row(
        &format!("{}, held session", answers_of(answers_in_one)),
        each,
        answers_in_one,
    );
    beside("of which the device itself", device_time / calls, each);

    println!();
    heading();

    let wide = to_words(&vec![1.0_f32; WIDE]);
    let workgroups = WIDE as u32 / WORKGROUP_SIZE;
    let answers = (WIDE / width.max(1) as usize) as f64;

    let mut session = gpu.session(&spirv, &[wide.len(), wide.len()])?;
    session.write(0, &wide)?;

    let mut device_time = Duration::ZERO;
    let mut calls = 0_u32;
    let each = timed(ROUND_TRIPS, || {
        device_time += session.dispatch(workgroups, 1)?;
        calls += 1;
        session.read(1, wide.len())?;
        Ok(())
    })?;
    row(
        &format!("{}, held session", answers_of(answers)),
        each,
        answers,
    );
    beside("of which the device itself", device_time / calls, each);

    crossover(each, answers);
    Ok(())
}

fn timed(
    calls: u32,
    mut work: impl FnMut() -> Result<(), Box<dyn std::error::Error>>,
) -> Result<Duration, Box<dyn std::error::Error>> {
    work()?;

    let mut best = Duration::MAX;
    for _ in 0..3 {
        let started = Instant::now();
        for _ in 0..calls {
            work()?;
        }
        best = best.min(started.elapsed() / calls);
    }
    Ok(best)
}

fn heading() {
    println!(
        "{:<32} {:>11} {:>12} {:>14}",
        "what", "per call", "per answer", "answers/s"
    );
}

fn row(label: &str, each: Duration, answers: f64) {
    let per_answer = each.as_secs_f64() / answers;
    println!(
        "{label:<32} {:>11} {:>12} {:>14}",
        format!("{:.1} us", each.as_secs_f64() * 1e6),
        format!("{:.3} us", per_answer * 1e6),
        format!("{:.0}", 1.0 / per_answer)
    );
}

fn beside(label: &str, device: Duration, wall: Duration) {
    let share = device.as_secs_f64() / wall.as_secs_f64() * 100.0;
    println!(
        "  {label:<30} {:>11}   {share:.1}% of the wall clock",
        format!("{:.1} us", device.as_secs_f64() * 1e6)
    );
}

fn crossover(each: Duration, answers: f64) {
    let round_trip = each.as_secs_f64();
    let device_per_answer = round_trip / answers;

    println!("\nindependent answers needed before this beats a CPU that takes:");
    for cpu_ns in [50.0_f64, 100.0, 1_000.0, 10_000.0] {
        let cpu = cpu_ns * 1e-9;
        let needed = if cpu <= device_per_answer {
            String::from("never — the device is slower per answer too")
        } else {
            format!("{:.0}", round_trip / (cpu - device_per_answer))
        };
        println!("  {cpu_ns:>8.0} ns per answer   {needed}");
    }
}

fn answers_of(count: f64) -> String {
    if count == 1.0 {
        String::from("1 answer")
    } else {
        format!("{count:.0} answers")
    }
}

fn to_words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}
