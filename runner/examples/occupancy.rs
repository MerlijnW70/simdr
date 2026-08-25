//! How many subgroups should a workgroup hold?
//!
//! `kernels::WORKGROUP_SIZE` has been 64 since the first kernel in this project and was chosen
//! once, from nothing. On the three devices here that is eight subgroups, two, or one — so it is
//! not even the same quantity twice.
//!
//! This sweeps it. Every row holds the total invocation count fixed and varies only how many of
//! them share a workgroup: same element type, same lane mapping (`WholeSubgroup`, one strip), same
//! buffers, same total work. The dispatch count moves to compensate.
//!
//! # Three kernel shapes, because one would answer for itself
//!
//! - **elementwise** — one multiply per element loaded. Memory-bound.
//! - **repeated** — 512 clamped multiply-adds per element loaded. Arithmetic-bound.
//! - **reduction** — one subgroup instruction and one store per invocation.
//!
//! `notes/FINDINGS.md` already carries a workgroup-size number that came from one elementwise
//! kernel on one device and had to be qualified twice afterwards. Three shapes is the cheapest way
//! not to do that again — and the second of them turned out to say something the first does not.
//!
//! # The arithmetic row was not arithmetic for two attempts
//!
//! It first read *identically* to the elementwise row, to the hundredth of a microsecond, which is
//! not what a kernel doing 512× the work looks like. `kernels::occupancy::sized_repeated_scale`
//! documents both folds and what stopped them. The check that caught it is the cheap one: run the
//! loop at 64 and at 512 and see whether the number moves.
//!
//! # Why every cell is five repeats
//!
//! It was one, and printed bare. On 2026-08-25 a single run of this table was read off as "one
//! subgroup is the worst column on the Radeon, 12-20% on the reduction", which is the opposite of
//! what `notes/FINDINGS.md` had recorded from the same sweep. Five runs settled it: one subgroup
//! is the *best* column there in four of five, and the reduction row wanders by more than any
//! difference between its columns. Nothing had changed but the luck of one sample.
//!
//! Every other measurement here that reports a ratio — `sweep`, `nnue`, `reducer` — already
//! repeated and already marked its unsteady rows. This one asked to be misread and was.
//!
//! # What this cannot say
//!
//! One device per run, and three synthetic kernels. It says which sizes are worth trying and that
//! the answer is not portable; it does not say what any particular kernel should use.

use runner::{Gpu, kernels};
use simdr::lanes::LaneError;
use std::time::Duration;

/// One of the shapes being swept: a name, and how to build it at a given workgroup size.
///
/// A named type rather than the tuple written inline: the closure's signature is four lines on its
/// own and it appears in two places.
type Shape<'a> = (&'a str, &'a dyn Fn(u32) -> Result<Vec<u32>, LaneError>);

/// How many subgroups to put in a workgroup.
const MULTIPLES: [u32; 6] = [1, 2, 4, 8, 16, 32];

/// Roughly how many invocations to run, and how many timed iterations at that size.
///
/// Rounded down to a whole number of workgroups per case, so the count each case actually ran is
/// printed rather than assumed.
const SIZES: [(u32, u32); 2] = [(1 << 14, 200), (1 << 18, 50)];

/// Multiply-adds per element in the arithmetic-bound kernel.
const REPEATS: u32 = 512;

/// How many times each cell is measured before one of them is printed.
///
/// Five, the same as `runner/examples/sweep.rs`, and for the reason that example gives: one timing
/// on a shared device is a sample, and a table of samples reads as a table of results.
const SAMPLES: u32 = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    let width = limits.subgroup_size;
    let ceiling = limits.max_workgroup_invocations;
    println!(
        "{} — subgroup {width}, workgroup ceiling {ceiling} ({} subgroups)\n",
        limits.name,
        ceiling.checked_div(width).unwrap_or(0)
    );

    for (target, iterations) in SIZES {
        println!("about {} invocations:", thousands(target as usize));
        sweep(&gpu, width, ceiling, target, iterations)?;
        println!();
    }

    println!(
        "Every column runs the same total work over the same elements and differs only in how\n\
         many subgroups share a workgroup. A dash is a size past this device's ceiling, or one\n\
         that does not divide the invocation count. Each cell is the median of five repeats, and\n\
         `!` marks one whose repeats disagreed by more than a fifth: not evidence, whichever way\n\
         it leans."
    );

    Ok(())
}

/// One block of the table: three kernel shapes across every workgroup size.
fn sweep(
    gpu: &Gpu,
    width: u32,
    ceiling: u32,
    target: u32,
    iterations: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    print!("{:>12}", "subgroups");
    for multiple in MULTIPLES {
        print!("{multiple:>12}");
    }
    println!();

    let shapes: [Shape<'_>; 3] = [
        ("elementwise", &|workgroup| {
            kernels::flat_scale(width, workgroup, 3)
        }),
        ("repeated", &|workgroup| {
            kernels::sized_repeated_scale(width, workgroup, REPEATS, 3)
        }),
        ("reduction", &|workgroup| {
            kernels::sized_lane_sum(width, workgroup)
        }),
    ];

    for (name, build) in shapes {
        print!("{name:>12}");
        for multiple in MULTIPLES {
            print!(
                "{:>12}",
                one(gpu, width, ceiling, target, iterations, multiple, build)?
            );
        }
        println!();
    }

    Ok(())
}

/// Time one kernel at one workgroup size, or say why it was not run.
fn one(
    gpu: &Gpu,
    width: u32,
    ceiling: u32,
    target: u32,
    iterations: u32,
    multiple: u32,
    build: &dyn Fn(u32) -> Result<Vec<u32>, LaneError>,
) -> Result<String, Box<dyn std::error::Error>> {
    let workgroup = width * multiple;
    if workgroup > ceiling {
        return Ok("-".to_owned());
    }

    // Whole workgroups only, and the same element count for every column of a row: `target` is
    // rounded down to a multiple of the *largest* workgroup this device will run, so every case
    // covers exactly the same elements rather than nearly the same.
    let largest = width
        * MULTIPLES
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .min(ceiling / width);
    let invocations = (target / largest) * largest;
    if invocations == 0 {
        return Ok("-".to_owned());
    }

    let input: Vec<u32> = (0..invocations).collect();
    let spirv = build(workgroup)?;
    let workgroups = invocations / workgroup;

    // One untimed pass, so the driver's lazy pipeline work stays out of the measurement.
    gpu.time(&spirv, &input, workgroups, 1)?;
    let timing = gpu.time_repeated(&spirv, &input, workgroups, iterations, SAMPLES)?;

    // The median rather than the best, and a mark when the repeats did not agree. Printed as one
    // unqualified number, this table was read off once and quoted as a 12% finding that five
    // repeats then refused to reproduce — see `notes/FINDINGS.md`, 2026-08-25.
    Ok(format!(
        "{}{}",
        micros(timing.median / iterations),
        if timing.is_steady() { "" } else { "!" }
    ))
}

/// Microseconds, which is the scale these land on.
fn micros(duration: Duration) -> String {
    format!("{:.2} us", duration.as_secs_f64() * 1e6)
}

/// A count with separators.
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
