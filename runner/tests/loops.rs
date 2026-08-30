mod common;

use common::{device, ramp, runnable};
use runner::kernels::{self, WORKGROUP_SIZE};

#[test]
fn an_unrolled_loop_reimplements_the_subgroup_sum_exactly() {
    let Some(gpu) = device("butterfly-tree") else {
        return;
    };
    let limits = gpu.limits().clone();

    let tree_spirv = kernels::butterfly_tree_sum(limits.subgroup_size).expect("built");
    let builtin_spirv =
        kernels::lane_sum_whole::<simdr::lanes::F32>(limits.subgroup_size).expect("built");
    if !runnable(&gpu, "butterfly-tree", &[&tree_spirv, &builtin_spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let tree = gpu.run(&tree_spirv, &input, 1).expect("dispatched");

    let expected = common::grouped_sums(count, width);
    assert_eq!(tree, expected, "the tree and the reference disagree");

    let builtin = gpu.run(&builtin_spirv, &input, 1).expect("dispatched");
    assert_eq!(tree, builtin);
}

#[test]
fn a_butterfly_tree_inside_a_cluster_agrees_with_the_clustered_reduce() {
    use simdr::lanes::{F32, LaneError};

    let Some(gpu) = device("butterfly-cluster") else {
        return;
    };
    let limits = gpu.limits().clone();

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    type Builder = fn(u32) -> Result<Vec<u32>, LaneError>;
    let cases: [(u32, Builder); 3] = [
        (2, kernels::lane_sum::<F32, 2>),
        (4, kernels::lane_sum::<F32, 4>),
        (8, kernels::lane_sum::<F32, 8>),
    ];

    for (cluster, builtin) in cases {
        if cluster >= width {
            eprintln!("SKIPPED clusters of {cluster}: not narrower than a {width}-wide subgroup");
            continue;
        }

        let tree_spirv = kernels::butterfly_cluster_sum(width, cluster).expect("built");
        let reduce_spirv = builtin(width).expect("built");
        if !runnable(&gpu, "butterfly-cluster", &[&tree_spirv, &reduce_spirv]) {
            return;
        }

        let tree = gpu.run(&tree_spirv, &input, 1).expect("dispatched");
        let reduced = gpu.run(&reduce_spirv, &input, 1).expect("dispatched");

        assert_eq!(
            tree,
            common::grouped_sums(count, cluster as usize),
            "the clustered tree disagrees with the reference at cluster {cluster}"
        );
        assert_eq!(
            tree, reduced,
            "the clustered tree and the clustered reduce disagree at cluster {cluster}"
        );
    }
}

#[test]
fn a_rolled_loop_carries_its_value_round_the_back_edge() {
    let Some(gpu) = device("rolled-loop") else {
        return;
    };
    let width = gpu.limits().subgroup_size;
    let input = ramp(WORKGROUP_SIZE as usize);

    for times in [0_u32, 1, 3, 5] {
        let output = gpu
            .run(
                &kernels::rolled_doubling(width, times).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

        let factor = 2_f32.powi(times as i32);
        let expected: Vec<f32> = input.iter().map(|value| value * factor).collect();

        assert_eq!(output, expected, "after {times} iterations");
    }
}

#[test]
fn a_value_computed_in_one_arm_arrives_at_the_merge() {
    let Some(gpu) = device("sum-or-max") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::sum_or_max(limits.subgroup_size, 40.0).expect("built");
    if !runnable(&gpu, "sum-or-max", &[&spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let sums = common::grouped_sums(count, width);
    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            let highest = (first + width - 1).min(count - 1);
            if highest as f32 > 40.0 {
                sums.get(lane).copied().unwrap_or_default()
            } else {
                highest as f32
            }
        })
        .collect();

    assert_eq!(output, expected);

    if count > width {
        let low = output.first().copied().unwrap_or_default();
        let high = output.last().copied().unwrap_or_default();
        let first_max = (width - 1) as f32;
        let last_sum: f32 = ((count - width)..count).map(|value| value as f32).sum();

        assert_eq!(low, first_max, "the first subgroup took the max arm");
        assert_eq!(high, last_sum, "the last summed");
    } else {
        eprintln!(
            "sum-or-max: one subgroup of {width} in a workgroup of {count}, so only the arm it \
             took is exercised here"
        );
        assert_eq!(
            output.first().copied(),
            Some((0..count as u32).sum::<u32>() as f32),
            "the only subgroup exceeds the threshold, so it summed"
        );
    }
}

#[test]
fn each_arm_is_the_whole_answer_when_every_subgroup_agrees() {
    let Some(gpu) = device("sum-or-max-uniform") else {
        return;
    };
    let limits = gpu.limits().clone();

    let always = kernels::sum_or_max(limits.subgroup_size, -1.0).expect("built");
    let never = kernels::sum_or_max(limits.subgroup_size, 1_000.0).expect("built");
    if !runnable(&gpu, "sum-or-max-uniform", &[&always, &never]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let summed = gpu.run(&always, &input, 1).expect("dispatched");
    assert_eq!(summed, common::grouped_sums(count, width));

    let maxed = gpu.run(&never, &input, 1).expect("dispatched");
    let expected: Vec<f32> = (0..count)
        .map(|lane| ((lane / width * width) + width - 1).min(count - 1) as f32)
        .collect();
    assert_eq!(maxed, expected);
}

#[test]
fn a_rolled_body_can_see_which_iteration_it_is() {
    let Some(gpu) = device("rolled-counter") else {
        return;
    };
    let width = gpu.limits().subgroup_size;
    let input: Vec<u32> = (0..WORKGROUP_SIZE).collect();

    for times in [0_u32, 1, 4, 9] {
        let output = gpu
            .run_u32(
                &kernels::rolled_counter_sum(width, times).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

        let added = times * times.saturating_sub(1) / 2;
        let expected: Vec<u32> = input.iter().map(|value| value + added).collect();

        assert_eq!(output, expected, "after {times} iterations");
    }
}

#[test]
fn control_flow_nests_both_ways_round() {
    let Some(gpu) = device("nesting") else {
        return;
    };
    let limits = gpu.limits().clone();

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    let times = 3_u32;

    let branch_in_loop = kernels::branch_in_loop(limits.subgroup_size, times, 40.0).expect("built");
    let loop_in_branch = kernels::loop_in_branch(limits.subgroup_size, times, 40.0).expect("built");
    if !runnable(&gpu, "nesting", &[&branch_in_loop, &loop_in_branch]) {
        return;
    }

    let takes_branch = |lane: usize| {
        let highest = (lane / width * width + width - 1).min(count - 1);
        highest as f32 > 40.0
    };

    let output = gpu.run(&branch_in_loop, &input, 1).expect("dispatched");

    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let start = lane as f32;
            if takes_branch(lane) {
                start * 2_f32.powi(times as i32)
            } else {
                start + times as f32
            }
        })
        .collect();
    assert_eq!(output, expected, "a branch inside a loop");

    let output = gpu.run(&loop_in_branch, &input, 1).expect("dispatched");

    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let start = lane as f32;
            if takes_branch(lane) {
                start * 2_f32.powi(times as i32)
            } else {
                start
            }
        })
        .collect();
    assert_eq!(output, expected, "a loop inside a branch");

    assert_ne!(output.first(), output.last());
}
