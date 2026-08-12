//! Two-dimensional dispatch, on a real device.
//!
//! The emitter's own tests count instructions, which says the address *expression* is the one that
//! was meant. It says nothing about whether that expression agrees with `vkCmdDispatch` — the
//! kernel computes `group.y × rows + local.y` and the dispatch decides what `group.y` ranges over,
//! and those two are written in different crates against different documents.
//!
//! # The three shapes
//!
//! Every test runs at least two of these, because a wrong address arithmetic can agree with a
//! right one on any single shape:
//!
//! - **one workgroup row per group** (`rows = 1`), where the row *is* the workgroup index and the
//!   emitter folds the multiply away entirely;
//! - **several rows per group** (`rows > 1`), which is the only shape that exercises `local.y`;
//! - **several workgroups across** (`pitch > width`), which is the only shape where a row spans
//!   more than one workgroup and the x arithmetic has to survive being added to a row offset.

mod common;

use common::device;
use runner::{Grid, kernels};

/// The matrix these tests use: `height` rows of `pitch` elements, filled so that no two elements
/// are equal and no row is a rotation of another.
///
/// `row × 1000 + column` rather than a running index: a kernel that computed the column right and
/// the row wrong lands on a number whose thousands digit says which row it actually read.
fn matrix(pitch: u32, height: u32) -> Vec<u32> {
    (0..height)
        .flat_map(|row| (0..pitch).map(move |column| row * 1000 + column))
        .collect()
}

#[test]
fn every_cell_of_a_grid_is_visited_exactly_once() {
    let Some(gpu) = device("plane-scale") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    // Three shapes of the same 8-row matrix. `rows` splits the height between the workgroup's y
    // and the dispatch's y; `pitch` splits the width between lanes and the dispatch's x.
    let height = 8;
    for (pitch, rows) in [(width, 1), (width, 2), (width * 3, 1), (width * 2, 4)] {
        let input = matrix(pitch, height);
        let grid = Grid::new(pitch / width, height / rows);

        let output = gpu
            .run_grid(
                &kernels::row_scale(width, pitch, rows, 3).expect("built"),
                &input,
                grid,
            )
            .expect("dispatched");

        let expected: Vec<u32> = input.iter().map(|value| value * 3).collect();
        assert_eq!(
            output, expected,
            "pitch {pitch}, {rows} rows per workgroup, grid {grid:?}"
        );
    }
}

#[test]
fn an_invocation_knows_which_row_it_is_on() {
    // The narrowest claim: the kernel writes its row index and nothing derived from what it read,
    // so an address that is wrong in the same way on the way in and the way out cannot cancel.
    let Some(gpu) = device("plane-index") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    let height = 8;
    for (pitch, rows) in [(width, 1), (width, 2), (width * 2, 4), (width, 8)] {
        let input = matrix(pitch, height);
        let grid = Grid::new(pitch / width, height / rows);

        let output = gpu
            .run_grid(
                &kernels::row_index(width, pitch, rows).expect("built"),
                &input,
                grid,
            )
            .expect("dispatched");

        let expected: Vec<u32> = (0..height)
            .flat_map(|row| std::iter::repeat_n(row, pitch as usize))
            .collect();
        assert_eq!(
            output, expected,
            "pitch {pitch}, {rows} rows per workgroup, grid {grid:?}"
        );
    }
}

#[test]
fn a_row_wise_reduction_reduces_its_own_row_and_no_other() {
    let Some(gpu) = device("plane-sum") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    // The pitch is the subgroup width here and cannot be anything else: one subgroup per row is
    // what makes `reduce_sum` a row total rather than a fragment of one.
    let height = 8;
    for rows in [1, 2, 4, 8] {
        let input = matrix(width, height);
        let grid = Grid::new(1, height / rows);

        let output = gpu
            .run_grid(&kernels::row_sum(width, rows).expect("built"), &input, grid)
            .expect("dispatched");

        // Every column of an output row holds that row's total, because every lane of the
        // reduction does. Checking all of them is what says the reduction was uniform across the
        // row rather than merely non-empty.
        let expected: Vec<u32> = (0..height)
            .flat_map(|row| {
                let total: u32 = (0..width).map(|column| row * 1000 + column).sum();
                std::iter::repeat_n(total, width as usize)
            })
            .collect();

        assert_eq!(output, expected, "{rows} rows per workgroup");
    }
}

#[test]
fn a_row_wise_reduction_would_notice_if_it_summed_the_whole_matrix() {
    // The failure the test above is really guarding against is a kernel that reduces across rows
    // rather than along them — on one row those are the same number, and on eight they are not.
    let Some(gpu) = device("plane-sum-discriminator") else {
        return;
    };
    let width = gpu.limits().subgroup_size;
    let height = 8;

    let output = gpu
        .run_grid(
            &kernels::row_sum(width, 1).expect("built"),
            &matrix(width, height),
            Grid::new(1, height),
        )
        .expect("dispatched");

    let whole: u32 = matrix(width, height).iter().sum();
    let first: u32 = (0..width).sum();

    assert_eq!(output.first().copied(), Some(first), "row zero's own total");
    assert_ne!(
        output.first().copied(),
        Some(whole),
        "that is every row summed together"
    );
}

#[test]
fn a_named_row_is_read_instead_of_this_ones() {
    let Some(gpu) = device("plane-bias") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    let height = 8;
    for (pitch, rows) in [(width, 1), (width * 2, 2), (width, 4)] {
        let input = matrix(pitch, height);
        let grid = Grid::new(pitch / width, height / rows);

        let output = gpu
            .run_grid(
                &kernels::row_bias(width, pitch, rows).expect("built"),
                &input,
                grid,
            )
            .expect("dispatched");

        let expected: Vec<u32> = (0..height)
            .flat_map(|row| (0..pitch).map(move |column| (row * 1000 + column) + column))
            .collect();
        assert_eq!(
            output, expected,
            "pitch {pitch}, {rows} rows per workgroup, grid {grid:?}"
        );

        // And the discriminator: a `load_row_at` that ignored its argument would double every row.
        let doubled: Vec<u32> = input.iter().map(|value| value * 2).collect();
        assert_ne!(
            output, doubled,
            "every row read itself rather than row zero"
        );
    }
}

#[test]
fn a_linear_kernel_still_runs_when_the_dispatch_has_a_second_axis() {
    // `Grid::linear` is what every existing call site became, and it has to be exactly what those
    // call sites did before: y of one, and nothing about the module changed.
    let Some(gpu) = device("plane-linear") else {
        return;
    };
    let width = gpu.limits().subgroup_size;
    let input: Vec<f32> = (0..kernels::WORKGROUP_SIZE)
        .map(|index| index as f32)
        .collect();

    let spirv = kernels::scale(width, 2.0).expect("built");
    let through_grid = gpu
        .run_grid(
            &spirv,
            &input
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            Grid::linear(1),
        )
        .expect("dispatched");
    let through_count = gpu.run(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u32> = input.iter().map(|value| (value * 2.0).to_bits()).collect();
    assert_eq!(through_grid, expected);
    assert_eq!(
        through_count
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
}
