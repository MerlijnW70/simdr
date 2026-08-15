//! `cargo run --release` — the quantised layer against its own oracle, at every dot and every
//! mapping.
//!
//! The test beside this asserts; this reports. A tool you can point at a device and read is worth
//! having separately, because the interesting output is the **coverage** — which combinations ran
//! at all, and which were refused, unsupported, invalid or *errored* — and a passing test prints
//! none of that.

use proeftuin::batch::Answer;
use proeftuin::{Dot, check};
use runner::Gpu;
use std::collections::BTreeMap;

/// Seeds per combination. Larger than the test's, because this is not on anybody's critical path.
///
/// **And all of them in one dispatch since 2026-08-16.** They were thirty-two round trips per
/// combination — 384 in all — which `decisions/DR-0008` prices at about 38 ms of waiting for a
/// millisecond of work. Twelve dispatches now, one per combination.
const SEEDS: u64 = 32;

/// How one (dot, mapping) pair fared, in the five ways it can.
#[derive(Default)]
struct Tally {
    agreed: u32,
    disagreed: u32,
    refused: u32,
    unsupported: u32,
    invalid: u32,
    errored: u32,
    /// The first thing worth reading, whichever kind it was.
    note: Option<String>,
}

impl Tally {
    /// Whether anything actually reached the device.
    const fn executed(&self) -> bool {
        self.agreed + self.disagreed > 0
    }

    /// Whether anything went wrong, as opposed to not happening.
    const fn broken(&self) -> u32 {
        self.disagreed + self.invalid + self.errored
    }
}

/// The three lane counts around a width, as the three mappings.
macro_rules! mappings {
    ($gpu:expr, $kind:expr, $seeds:expr, $half:literal, $whole:literal, $double:literal) => {
        vec![
            (
                "clustered",
                check::<$half>($gpu, $kind, "clustered", $seeds),
            ),
            ("whole", check::<$whole>($gpu, $kind, "whole", $seeds)),
            (
                "strip-mined",
                check::<$double>($gpu, $kind, "strip-mined", $seeds),
            ),
        ]
    };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}\n", gpu.limits().name);

    let mut tally: BTreeMap<(&str, &str), Tally> = BTreeMap::new();

    let seeds: Vec<u64> = (0..SEEDS).collect();

    for kind in Dot::ALL {
        let outcomes = match width {
            4 => mappings!(&gpu, kind, &seeds, 2, 4, 8),
            8 => mappings!(&gpu, kind, &seeds, 4, 8, 16),
            16 => mappings!(&gpu, kind, &seeds, 8, 16, 32),
            32 => mappings!(&gpu, kind, &seeds, 16, 32, 64),
            64 => mappings!(&gpu, kind, &seeds, 32, 64, 128),
            other => {
                println!("no lane counts written for a subgroup of {other}");
                return Ok(());
            }
        };

        for (mapping, outcome) in outcomes {
            let entry = tally.entry((kind.name(), mapping)).or_default();

            // **The reason it did not run is counted once for the batch**, where it used to be
            // counted once per seed. A refusal is a property of the module and the seeds share
            // one, so thirty-two identical refusals were thirty-two copies of one fact.
            let Answer::Ran(checked) = outcome else {
                let why = outcome.why().unwrap_or_default();
                match outcome {
                    Answer::Refused(_) => entry.refused += 1,
                    Answer::Unsupported(_) => entry.unsupported += 1,
                    Answer::Invalid(_) => entry.invalid += 1,
                    Answer::Errored(_) => entry.errored += 1,
                    Answer::Ran(_) => {}
                }
                entry.note.get_or_insert(why);
                continue;
            };

            for (seed, one) in checked.iter().enumerate() {
                if one.agreed() {
                    entry.agreed += 1;
                } else {
                    entry.disagreed += 1;
                    entry
                        .note
                        .get_or_insert_with(|| format!("seed {seed}: {one:?}"));
                }
            }
        }
    }

    println!(
        "{:<14} {:<13} {:>7} {:>10} {:>8} {:>12} {:>8} {:>8}",
        "dot", "mapping", "agreed", "disagreed", "refused", "unsupported", "invalid", "errored"
    );
    for ((kind, mapping), t) in &tally {
        println!(
            "{kind:<14} {mapping:<13} {:>7} {:>10} {:>8} {:>12} {:>8} {:>8}",
            t.agreed, t.disagreed, t.refused, t.unsupported, t.invalid, t.errored
        );
    }

    let notes: Vec<String> = tally
        .iter()
        .filter_map(|((kind, mapping), t)| {
            t.note.as_ref().map(|n| format!("{kind} {mapping}: {n}"))
        })
        .collect();
    if !notes.is_empty() {
        println!();
        for note in notes {
            println!("{note}");
        }
    }

    // **The number worth acting on is how many combinations never reached the device**, and a tally
    // that only counts agreements cannot show it — which is the whole reason `Outcome` has five
    // arms instead of a `Result`.
    let executed = tally.values().filter(|t| t.executed()).count();
    println!("\n{executed} of {} combinations executed here", tally.len());

    if tally.values().map(Tally::broken).sum::<u32>() > 0 {
        std::process::exit(1);
    }
    Ok(())
}
