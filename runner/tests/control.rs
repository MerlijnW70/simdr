mod common;

use common::{device, ramp, runnable};
use runner::kernels::{self, WORKGROUP_SIZE};

#[test]
fn a_uniform_branch_is_taken_by_a_whole_subgroup_or_by_none_of_it() {
    let Some(gpu) = device("uniform-branch") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::scale_if_any_above(limits.subgroup_size, 40.0).expect("built");
    if !runnable(&gpu, "uniform-branch", &[&spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

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

#[test]
fn a_helper_built_selection_runs_and_leaves_the_result_alone() {
    let Some(gpu) = device("branch-only") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input = ramp(WORKGROUP_SIZE as usize);

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
