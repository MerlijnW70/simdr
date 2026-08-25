//! What one answer costs when you need it *now*.
//!
//! Every other benchmark here amortises: hundreds of dispatches in one submission, divided out.
//! That is the right measurement for throughput and the wrong one for a caller that has a single
//! question and cannot continue until it is answered — a game-tree search asking for one
//! evaluation, say.
//!
//! So this measures the whole round trip from the host's point of view: submit, wait, read back.
//! It is deliberately the least flattering number this project can produce, and it is the one that
//! decides whether a latency-bound workload belongs on a GPU at all. `decisions/DR-0008` is what
//! this table was used to settle.
//!
//! # Every row is the same clock, and that had to be fixed
//!
//! The first version put a **host** round trip and a **device** timestamp under one `per answer`
//! heading, and divided only one of them by the answers it produced. The batched row therefore read
//! about 500× better than any caller would ever see: 1.9 µs was what the device spent, not what the
//! host waited, and it was per *dispatch* rather than per answer.
//!
//! Two numbers under one heading, one of them measuring something the caller never experiences, is
//! the same failure this project keeps finding elsewhere — a figure that reads as evidence and was
//! produced by an instrument that cannot see the thing being claimed. So every row here is wall
//! clock around the whole operation, and the device figure is printed *beside* it as the gap rather
//! than in place of it.
//!
//! # And the GPU is given its best case
//!
//! [`Gpu::run`] allocates buffers and builds a pipeline on every call, which is the honest cost of
//! a one-shot caller and an unfair one for a caller in a loop. So the held [`Session`] is measured
//! too — no allocation, no pipeline creation, only submit, wait and read back. Understating the
//! device's case would make the conclusion worthless.
//!
//! Where a figure is still not exact, the direction is stated rather than smoothed. The batched row
//! reads the **whole** output buffer back, because `lane_sum` writes its total to every invocation
//! and this kernel produces 32 copies of each answer. A kernel that compacted them would move a
//! thirty-second of the bytes, so the batched round trip here is *pessimistic* — which is the safe
//! direction for a table whose conclusion is "batch or stay on the CPU".
//!
//! [`Session`]: runner::Session

use runner::Gpu;
mod common;

use runner::Timing;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;
use std::time::Duration;

/// How many round trips to time, after a warm-up.
const ROUND_TRIPS: u32 = 200;

/// Elements in the batched run, and therefore `WIDE / subgroup` independent answers at once.
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

    let spirv = kernels::lane_sum::<F32, 32>(width)?;
    let one = vec![1.0_f32; WORKGROUP_SIZE as usize];
    // One subgroup sum per subgroup in the workgroup, which is how many answers a dispatch of one
    // workgroup actually produces. Dividing by anything else is what the first version did.
    let answers_in_one = (WORKGROUP_SIZE / width.max(1)).max(1) as f64;

    heading();

    // A one-shot caller: `Gpu::run` allocates and builds a pipeline every time.
    let each = timed(ROUND_TRIPS, || {
        gpu.run(&spirv, &one, 1)?;
        Ok(())
    })?;
    row(
        &format!("{}, built per call", answers_of(answers_in_one)),
        each,
        answers_in_one,
    );

    // The same question from a held session: the allocations and the pipeline happen once.
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

    // And the same device asked for two thousand answers at once — the only shape that lets a
    // round trip be divided by anything.
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

/// The best of a few runs of `work`, per call.
///
/// Best-of rather than mean, for the reason `notes/SPEED.md` in the sibling project states: noise
/// only ever adds time, so the smallest reading is the closest to what the machine can do.
fn timed(
    calls: u32,
    mut work: impl FnMut() -> Result<(), Box<dyn std::error::Error>>,
) -> Result<Timing, Box<dyn std::error::Error>> {
    // Warm up: the first call of a module pays for pipeline creation, and the first read faults in
    // the staging mapping.
    work()?;

    // This took the *best* of three, which is a defensible choice and an undeclared one: the best
    // of three hides a disagreement between the three exactly as well as a single sample hides
    // having no second opinion at all. The median of five, with the spread carried alongside, says
    // the same thing about a steady measurement and says something different about a wandering one.
    let batches = common::host(common::SAMPLES, || {
        for _ in 0..calls {
            work()?;
        }
        Ok::<(), Box<dyn std::error::Error>>(())
    })?;

    Ok(Timing {
        best: batches.best / calls,
        median: batches.median / calls,
        worst: batches.worst / calls,
        repeats: batches.repeats,
    })
}

fn heading() {
    println!(
        "{:<32} {:>11} {:>12} {:>14}",
        "what", "per call", "per answer", "answers/s"
    );
}

/// One measured row, with the per-answer figure derived rather than assumed.
fn row(label: &str, each: Timing, answers: f64) {
    let per_answer = each.median.as_secs_f64() / answers;
    let mark = common::mark(each);
    println!(
        "{label:<32} {:>11} {:>12} {:>14}",
        format!("{:.1} us{mark}", each.median.as_secs_f64() * 1e6),
        format!("{:.3} us{mark}", per_answer * 1e6),
        format!("{:.0}{mark}", 1.0 / per_answer)
    );
}

/// The device's own clock beside the wall clock, as a share of it.
///
/// Printed rather than substituted: the difference between these two is the submission, the fence
/// and the copy back, and it is the whole of what a latency-bound caller pays.
fn beside(label: &str, device: Duration, wall: Timing) {
    let share = device.as_secs_f64() / wall.median.as_secs_f64() * 100.0;
    println!(
        "  {label:<30} {:>11}   {share:.1}% of the wall clock",
        format!("{:.1} us", device.as_secs_f64() * 1e6)
    );
}

/// How many answers a caller must have pending at once before this device is worth asking.
///
/// The question the table above exists to answer, made arithmetic. A round trip is a fixed cost, so
/// the device wins only where a CPU producing one answer every `cpu` nanoseconds would take longer
/// than the whole round trip — that is `round_trip / cpu` answers, and they must be **independent**,
/// because a caller that needs answer *n* before it knows whether to ask for *n + 1* has one.
fn crossover(each: Timing, answers: f64) {
    let round_trip = each.median.as_secs_f64();
    let device_per_answer = round_trip / answers;

    println!("\n{}", common::LEGEND);
    println!("\nindependent answers needed before this beats a CPU that takes:");
    for cpu_ns in [50.0_f64, 100.0, 1_000.0, 10_000.0] {
        let cpu = cpu_ns * 1e-9;
        // Below the device's own per-answer cost no batch is ever large enough — the device loses
        // on the arithmetic alone, before the round trip is counted.
        let needed = if cpu <= device_per_answer {
            String::from("never — the device is slower per answer too")
        } else {
            format!("{:.0}", round_trip / (cpu - device_per_answer))
        };
        println!("  {cpu_ns:>8.0} ns per answer   {needed}");
    }
}

/// `n answers`, or `1 answer` — a 64-wide subgroup makes the singular reachable.
fn answers_of(count: f64) -> String {
    if count == 1.0 {
        String::from("1 answer")
    } else {
        format!("{count:.0} answers")
    }
}

/// Floats as the words a buffer holds.
fn to_words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}
