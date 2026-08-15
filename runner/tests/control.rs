//! Uniform branches, run on a real device.
//!
//! `decisions/DR-0003` argues that a branch whose condition came from a vote is safe and one that
//! did not is not offered. These check the safe half actually works: the whole subgroup takes the
//! branch together, and the two subgroups of a workgroup can take it differently.

mod common;

use common::{device, ramp, runnable};
use runner::kernels::{self, WORKGROUP_SIZE};

#[test]
fn a_uniform_branch_is_taken_by_a_whole_subgroup_or_by_none_of_it() {
    let Some(gpu) = device("uniform-branch") else {
        return;
    };
    let limits = gpu.limits().clone();

    // **The gate that made this worth converting.** It said `subgroup_arithmetic` while its message
    // said "no subgroup vote support" — and the kernel needs both, because `any_uniform` declares
    // `GroupNonUniformVote`. On a device offering arithmetic and no vote this would have run and
    // failed at pipeline creation instead of skipping. `runnable` asks the module.
    let spirv = kernels::scale_if_any_above(limits.subgroup_size, 40.0).expect("built");
    if !runnable(&gpu, "uniform-branch", &[&spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    // A ramp of 0..64 over two 32-wide subgroups: the first holds 0..31 and the second 32..63, so
    // a threshold of 40 is exceeded by the second and not the first.
    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            let highest = (first + width - 1).min(count - 1);
            if highest as f32 > 40.0 {
                lane as f32 * 10.0
            } else {
                lane as f32
            }
        })
        .collect();

    assert_eq!(output, expected);

    // Discriminator: the two subgroups must have taken different paths, or the branch was not
    // doing anything a per-subgroup condition could not have done unconditionally.
    let low = output.first().copied().unwrap_or_default();
    let high = output.last().copied().unwrap_or_default();
    assert_eq!(low, 0.0, "the first subgroup did not take the branch");
    assert_eq!(high, 63.0 * 10.0, "the second subgroup did take it");
}

#[test]
fn a_threshold_nobody_meets_leaves_every_lane_alone() {
    let Some(gpu) = device("branch-never") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::scale_if_any_above(limits.subgroup_size, 1_000.0).expect("built");
    if !runnable(&gpu, "branch-never", &[&spirv]) {
        return;
    }

    let input = ramp(WORKGROUP_SIZE as usize);
    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    assert_eq!(output, input, "no subgroup qualified, so nothing changed");
}

#[test]
fn a_threshold_everyone_meets_scales_every_lane() {
    let Some(gpu) = device("branch-always") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::scale_if_any_above(limits.subgroup_size, -1.0).expect("built");
    if !runnable(&gpu, "branch-always", &[&spirv]) {
        return;
    }

    let input = ramp(WORKGROUP_SIZE as usize);
    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<f32> = input.iter().map(|value| value * 10.0).collect();
    assert_eq!(output, expected);
}

/// The `if_uniform` helper's own block structure, run rather than only validated.
///
/// Its body writes nothing, so the output is the unconditional store — which is exactly what
/// makes this a test of the *structure*: a malformed selection would not run at all.
#[test]
fn a_helper_built_selection_runs_and_leaves_the_result_alone() {
    let Some(gpu) = device("branch-only") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input = ramp(WORKGROUP_SIZE as usize);

    // Every threshold's module, then one gate: they are the same kernel with a different constant,
    // so they declare the same capabilities — but asking the modules rather than assuming that is
    // the whole point of the change.
    let built: Vec<Vec<u32>> = [-1.0_f32, 40.0, 1_000.0]
        .into_iter()
        .map(|threshold| kernels::branch_only(limits.subgroup_size, threshold).expect("built"))
        .collect();
    let modules: Vec<&[u32]> = built.iter().map(Vec::as_slice).collect();
    if !runnable(&gpu, "branch-only", &modules) {
        return;
    }

    for spirv in &built {
        let output = gpu.run(spirv, &input, 1).expect("dispatched");

        assert_eq!(
            output, input,
            "the body computes and discards, whatever the threshold"
        );
    }
}
