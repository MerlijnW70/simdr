//! The kernels whose workgroup size is an argument, on a real device.
//!
//! `runner/examples/occupancy.rs` times these across every workgroup size a device will run. A
//! benchmark that is wrong is worse than no benchmark, and two of the three kernels below have
//! already been wrong in a way that made the numbers meaningless — see
//! `kernels::occupancy::sized_repeated_scale`, whose loop was folded away twice before it stopped
//! being.
//!
//! So these check the answers. **The arithmetic one is checked at two loop lengths**, because the
//! failure that mattered was not a wrong answer — it was a right answer arriving too fast.

mod common;

use common::device;
use runner::kernels::{self, occupancy::LIMIT};

/// The host's version of one iteration of `sized_repeated_scale`.
///
/// `wrapping_*` because `OpIMul` and `OpIAdd` wrap on overflow and nothing about the clamp changes
/// that — the clamp happens after.
fn step(running: u32, factor: u32, salt: u32) -> u32 {
    running.wrapping_mul(factor).wrapping_add(salt).min(LIMIT)
}

/// The whole chain, `times` iterations of it.
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

    // 512 elements covers every size below with whole workgroups.
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
    // Two loop lengths, and the reference is the loop written out on the host. A driver that folds
    // `times` iterations into one expression still gets the answer right — that is what folding
    // means — so this cannot catch the fold directly. What it does catch is the loop having been
    // built wrong, which is the other way the benchmark stops meaning anything.
    //
    // Whether the fold happened is a *timing* question, and `runner/examples/occupancy.rs` asks it
    // the only way it can be asked: run 64 and run 512 and see whether the number moves.
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
    // A `min` that never fires is one a compiler deletes, and deleting it put the affine chain
    // back and made the arithmetic-bound row read as memory-bound. So this asserts the property
    // the benchmark depends on rather than the answer: after enough iterations of multiplying by
    // three, every lane is at the limit.
    let Some(gpu) = device("occupancy-clamp") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    // Start at 1 so nothing is already at the limit, and run long enough that 3^n passes it.
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

        // Every invocation holds its own subgroup's total, whatever the workgroup size — that is
        // the claim, and a workgroup size that leaked into the reduction would break it.
        let expected: Vec<u32> = (0..count)
            .map(|index| {
                let base = (index / width) * width;
                (base..base + width).sum()
            })
            .collect();

        assert_eq!(output, expected, "{multiple} subgroups per workgroup");
    }
}
