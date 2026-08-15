//! The quantised layer, at every dot product and every mapping the device can hold it in.
//!
//! `runner::kernels::dot` builds every packed dot through `whole_subgroup!`, which fixes `LANES` to
//! the device's width — so `OpSDot`, `OpUDot`, `OpSUDot` and `OpSDotAccSat` have only ever *run* as
//! whole-subgroup vectors. Clustered they fold inside a cluster; strip-mined they fold the strips
//! first, and the saturating one saturates **per strip**. Twelve combinations, of which four had
//! ever executed.

use proeftuin::batch::Answer;
use proeftuin::{Dot, check};
use runner::Gpu;
use std::collections::BTreeSet;

/// The three lane counts around a width, as the three mappings.
macro_rules! mappings {
    ($gpu:expr, $kind:expr, $seeds:expr, $half:literal, $whole:literal, $double:literal) => {
        vec![
            check::<$half>($gpu, $kind, "clustered", $seeds),
            check::<$whole>($gpu, $kind, "whole", $seeds),
            check::<$double>($gpu, $kind, "strip-mined", $seeds),
        ]
    };
}

#[test]
fn the_quantised_layer_agrees_at_every_dot_and_mapping() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        eprintln!("SKIPPED quantised: no Vulkan device");
        return Ok(());
    };

    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}", gpu.limits().name);

    let mut executed: BTreeSet<(&str, &str)> = BTreeSet::new();
    let mut complaints: Vec<String> = Vec::new();

    // **Six seeds in one dispatch, not six dispatches.** They vary only the data, and
    // `decisions/DR-0008` is what makes that worth saying: the device's share of a round trip is
    // 2.9% here, so seventy-two dispatches were seventy-two waits for one dispatch of work.
    let seeds: Vec<u64> = (0..6).collect();

    for kind in Dot::ALL {
        let outcomes = match width {
            4 => mappings!(&gpu, kind, &seeds, 2, 4, 8),
            8 => mappings!(&gpu, kind, &seeds, 4, 8, 16),
            16 => mappings!(&gpu, kind, &seeds, 8, 16, 32),
            32 => mappings!(&gpu, kind, &seeds, 16, 32, 64),
            64 => mappings!(&gpu, kind, &seeds, 32, 64, 128),
            other => {
                eprintln!("SKIPPED quantised: no lane counts written for a subgroup of {other}");
                return Ok(());
            }
        };

        for outcome in outcomes {
            match outcome {
                Answer::Ran(batch) => {
                    for (seed, checked) in batch.iter().enumerate() {
                        executed.insert((kind.name(), checked.mapping));
                        if !checked.agreed() {
                            complaints.push(format!("{} seed {seed}: {checked:?}", kind.name()));
                        }
                    }
                }
                // Lost coverage rather than failure, and printed rather than counted silently:
                // a skipped check that looks green is worse than a red one. One line per batch
                // now rather than one per seed — the reason belongs to the module, and the seeds
                // share it.
                Answer::Refused(why) => {
                    eprintln!("  {} not run: refused — {why}", kind.name());
                }
                Answer::Unsupported(missing) => {
                    eprintln!("  {} not run: device lacks {missing:?}", kind.name());
                }
                // Failures, and which one it is decides who has to fix it: an invalid module is
                // this crate's mistake, a driver erroring on a *validated* one is the device's.
                Answer::Invalid(complaint) => complaints.push(format!(
                    "{}: spirv-val rejected it — {complaint}",
                    kind.name()
                )),
                Answer::Errored(error) => complaints.push(format!(
                    "{}: the driver failed after accepting a valid module — {error}",
                    kind.name()
                )),
            }
        }
    }

    // **Without this the test is vacuous.** Every combination being refused would print tidy lines
    // and assert nothing, which is the failure this whole repository is about. Eight of the twelve
    // is the floor rather than the target: a device without the dot-product extension reaches none,
    // and one with it should reach all twelve.
    assert!(
        executed.len() >= 8,
        "only {} of twelve combinations executed — {executed:?}",
        executed.len()
    );
    assert!(
        complaints.is_empty(),
        "the quantised layer did not come back right:\n{}",
        complaints.join("\n")
    );
    Ok(())
}
