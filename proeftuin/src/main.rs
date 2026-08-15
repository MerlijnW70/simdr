//! `cargo run --release` — the quantised layer against its own oracle, at every mapping.
//!
//! The test beside this asserts; this reports. A tool you can point at a device and read is worth
//! having separately, because the interesting output is the **coverage** — which mappings ran at
//! all, and which were refused, unsupported or *errored* — and a passing test prints none of that.

use proeftuin::{Outcome, check};
use runner::Gpu;
use std::collections::BTreeMap;

/// Seeds per mapping. Larger than the test's, because this is not on anybody's critical path.
const SEEDS: u64 = 64;

/// How each mapping fared, in the four ways it can.
#[derive(Default)]
struct Tally {
    agreed: u32,
    disagreed: u32,
    refused: u32,
    unsupported: u32,
    errored: u32,
    invalid: u32,
    /// The first thing worth reading, whichever kind it was.
    note: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}\n", gpu.limits().name);

    let mut tally: BTreeMap<&str, Tally> = BTreeMap::new();

    for seed in 0..SEEDS {
        let outcomes = match width {
            4 => vec![
                ("clustered", check::<2>(&gpu, "clustered", seed)),
                ("whole", check::<4>(&gpu, "whole", seed)),
                ("strip-mined", check::<8>(&gpu, "strip-mined", seed)),
            ],
            8 => vec![
                ("clustered", check::<4>(&gpu, "clustered", seed)),
                ("whole", check::<8>(&gpu, "whole", seed)),
                ("strip-mined", check::<16>(&gpu, "strip-mined", seed)),
            ],
            16 => vec![
                ("clustered", check::<8>(&gpu, "clustered", seed)),
                ("whole", check::<16>(&gpu, "whole", seed)),
                ("strip-mined", check::<32>(&gpu, "strip-mined", seed)),
            ],
            32 => vec![
                ("clustered", check::<16>(&gpu, "clustered", seed)),
                ("whole", check::<32>(&gpu, "whole", seed)),
                ("strip-mined", check::<64>(&gpu, "strip-mined", seed)),
            ],
            64 => vec![
                ("clustered", check::<32>(&gpu, "clustered", seed)),
                ("whole", check::<64>(&gpu, "whole", seed)),
                ("strip-mined", check::<128>(&gpu, "strip-mined", seed)),
            ],
            other => {
                println!("no lane counts written for a subgroup of {other}");
                return Ok(());
            }
        };

        for (mapping, outcome) in outcomes {
            let entry = tally.entry(mapping).or_default();
            match outcome {
                Outcome::Ran(checked) if checked.agreed() => entry.agreed += 1,
                Outcome::Ran(checked) => {
                    entry.disagreed += 1;
                    entry
                        .note
                        .get_or_insert_with(|| format!("first at seed {seed}: {checked:?}"));
                }
                Outcome::Refused(why) => {
                    entry.refused += 1;
                    entry.note.get_or_insert_with(|| format!("refused: {why}"));
                }
                Outcome::Unsupported(missing) => {
                    entry.unsupported += 1;
                    entry
                        .note
                        .get_or_insert_with(|| format!("device does not offer {missing:?}"));
                }
                Outcome::Invalid(complaint) => {
                    entry.invalid += 1;
                    entry.note.get_or_insert_with(|| format!("spirv-val rejected it: {complaint}"));
                }
                Outcome::Errored(error) => {
                    entry.errored += 1;
                    entry
                        .note
                        .get_or_insert_with(|| format!("the driver took the module and failed: {error}"));
                }
            }
        }
    }

    println!(
        "{:<14} {:>7} {:>10} {:>8} {:>12} {:>8}",
        "mapping", "agreed", "disagreed", "refused", "unsupported", "invalid"
    );
    for (mapping, t) in &tally {
        println!(
            "{mapping:<14} {:>7} {:>10} {:>8} {:>12} {:>8}",
            t.agreed, t.disagreed, t.refused, t.unsupported, t.invalid + t.errored
        );
    }

    let notes: Vec<(&&str, &String)> = tally
        .iter()
        .filter_map(|(m, t)| t.note.as_ref().map(|n| (m, n)))
        .collect();
    if !notes.is_empty() {
        println!();
        for (mapping, note) in notes {
            println!("{mapping}: {note}");
        }
    }

    // A mapping that never *ran* is the number worth acting on, and a tally that only counts
    // agreements cannot show it.
    let ran = tally.values().filter(|t| t.agreed + t.disagreed > 0).count();
    if ran < tally.len() {
        println!("\n{ran} of {} mappings actually executed here", tally.len());
    }

    let broken: u32 = tally.values().map(|t| t.disagreed + t.errored + t.invalid).sum();
    if broken > 0 {
        std::process::exit(1);
    }
    Ok(())
}
