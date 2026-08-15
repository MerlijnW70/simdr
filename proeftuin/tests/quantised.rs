//! The quantised layer, at every mapping the device can hold it in.
//!
//! `runner::kernels::dot` builds every packed dot product through `whole_subgroup!`, which fixes
//! `LANES` to the device's width — so `OpSDot`, `OpUDot` and `OpSUDot` have only ever *run* as a
//! whole-subgroup vector. Clustered they fold inside a cluster; strip-mined they fold the strips
//! first. Same call, three instruction sequences, and two of them had never been executed.

use proeftuin::{Outcome, check};
use runner::Gpu;
use std::collections::BTreeSet;

/// Every mapping this width can express, as the three `LANES` around it.
macro_rules! three_mappings {
    ($gpu:expr, $seed:expr, $half:literal, $whole:literal, $double:literal) => {
        vec![
            check::<$half>($gpu, "clustered", $seed),
            check::<$whole>($gpu, "whole", $seed),
            check::<$double>($gpu, "strip-mined", $seed),
        ]
    };
}

#[test]
fn the_quantised_layer_agrees_at_every_mapping() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        eprintln!("SKIPPED quantised: no Vulkan device");
        return Ok(());
    };

    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}", gpu.limits().name);

    let mut executed: BTreeSet<&str> = BTreeSet::new();
    let mut complaints: Vec<String> = Vec::new();

    for seed in 0..8_u64 {
        let outcomes = match width {
            4 => three_mappings!(&gpu, seed, 2, 4, 8),
            8 => three_mappings!(&gpu, seed, 4, 8, 16),
            16 => three_mappings!(&gpu, seed, 8, 16, 32),
            32 => three_mappings!(&gpu, seed, 16, 32, 64),
            64 => three_mappings!(&gpu, seed, 32, 64, 128),
            other => {
                eprintln!("SKIPPED quantised: no lane counts written for a subgroup of {other}");
                return Ok(());
            }
        };

        for outcome in outcomes {
            match outcome {
                Outcome::Ran(checked) => {
                    executed.insert(checked.mapping);
                    if !checked.agreed() {
                        complaints.push(format!("seed {seed}: {checked:?}"));
                    }
                }
                // A refusal is the mapping working and an unsupported instruction is the device
                // being honest. Both cost coverage, and both are printed rather than counted
                // silently — a skipped check that looks green is worse than a red one.
                Outcome::Refused(why) => eprintln!("  not run at seed {seed}: refused — {why}"),
                Outcome::Unsupported(missing) => {
                    eprintln!("  not run at seed {seed}: device lacks {missing:?}");
                }
                // Both of these are failures rather than lost coverage. An invalid module is this
                // crate's own mistake, and a driver that errors on a *validated* one is the
                // device's — and telling them apart is why the validator runs first.
                Outcome::Invalid(complaint) => {
                    complaints.push(format!("seed {seed}: spirv-val rejected it — {complaint}"));
                }
                Outcome::Errored(error) => complaints.push(format!(
                    "seed {seed}: the driver failed after accepting a valid module — {error}"
                )),
            }
        }
    }

    // **Without this the test is vacuous.** Every mapping being refused would print tidy lines and
    // assert nothing, which is the failure this whole repository is about.
    assert!(
        executed.len() >= 2,
        "only {executed:?} executed, so this proves one instruction sequence at most"
    );
    assert!(
        complaints.is_empty(),
        "the quantised layer did not come back right:\n{}",
        complaints.join("\n")
    );
    Ok(())
}
