mod common;

use common::device;
use runner::kernels::{self, occupancy::LIMIT};

fn step(running: u32, factor: u32, salt: u32) -> u32 {
    running.wrapping_mul(factor).wrapping_add(salt).min(LIMIT)
}

fn chain(start: u32, factor: u32, times: u32) -> u32 {
    (0..times).fold(start, |running, salt| step(running, factor, salt))
}

#[test]
fn a_sized_kernel_computes_the_same_answer_at_every_workgroup_size() {
    let Some(gpu) = device("occupancy-scale") else {
        return;
    };
    let width = gpu.limits().subgroup_size;
    let ceiling = gpu.limits().max_workgroup_invocations;

    let input: Vec<u32> = (0..512).collect();

    for multiple in [1, 2, 4, 8] {
        let workgroup = width * multiple;
        if workgroup > ceiling || !512_u32.is_multiple_of(workgroup) {
            continue;
        }

        let output = gpu
            .run_u32(
                &kernels::flat_scale(width, workgroup, 3).expect("built"),
                &input,
                512 / workgroup,
            )
            .expect("dispatched");

        let expected: Vec<u32> = input.iter().map(|value| value.wrapping_mul(3)).collect();
        assert_eq!(output, expected, "{multiple} subgroups per workgroup");
    }
}

#[test]
fn the_arithmetic_kernel_runs_its_loop_rather_than_a_closed_form() {
    let Some(gpu) = device("occupancy-repeated") else {
        return;
    };
    let width = gpu.limits().subgroup_size;
    let input: Vec<u32> = (0..width).collect();

    for times in [1, 8, 64] {
        let output = gpu
            .run_u32(
                &kernels::sized_repeated_scale(width, width, times, 3).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

        let expected: Vec<u32> = input.iter().map(|start| chain(*start, 3, times)).collect();
        assert_eq!(output, expected, "{times} iterations");
    }
}

#[test]
fn the_clamp_actually_clamps_which_is_the_whole_reason_it_is_there() {
    let Some(gpu) = device("occupancy-clamp") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    let input: Vec<u32> = std::iter::repeat_n(1, width as usize).collect();

    let output = gpu
        .run_u32(
            &kernels::sized_repeated_scale(width, width, 64, 3).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    assert!(
        output
            .iter()
            .take(width as usize)
            .all(|value| *value == LIMIT),
        "nothing reached the clamp, so the minimum is the identity again: {:?}",
        output.first()
    );
}

#[test]
fn the_sized_reduction_sums_its_own_subgroup_at_every_workgroup_size() {
    let Some(gpu) = device("occupancy-reduce") else {
        return;
    };
    let width = gpu.limits().subgroup_size;
    let ceiling = gpu.limits().max_workgroup_invocations;

    let count = 512_u32;
    let input: Vec<u32> = (0..count).collect();

    for multiple in [1, 2, 4, 8] {
        let workgroup = width * multiple;
        if workgroup > ceiling || !count.is_multiple_of(workgroup) {
            continue;
        }

        let output = gpu
            .run_u32(
                &kernels::sized_lane_sum(width, workgroup).expect("built"),
                &input,
                count / workgroup,
            )
            .expect("dispatched");

        let expected: Vec<u32> = (0..count)
            .map(|index| {
                let base = (index / width) * width;
                (base..base + width).sum()
            })
            .collect();

        assert_eq!(output, expected, "{multiple} subgroups per workgroup");
    }
}
