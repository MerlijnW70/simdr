mod common;

use common::{device, runnable};
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;

fn ramp_with(count: usize, at: usize, value: f32) -> Vec<f32> {
    let mut input: Vec<f32> = (0..count).map(|index| index as f32).collect();
    if let Some(slot) = input.get_mut(at) {
        *slot = value;
    }
    input
}

fn inside_first_subgroup(width: usize) -> usize {
    width / 2
}

#[test]
fn a_sum_containing_an_infinity_is_infinite_and_does_not_corrupt_the_other_subgroup() {
    let Some(gpu) = device("infinity-sum") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::lane_sum_whole::<F32>(limits.subgroup_size).expect("built");
    if !runnable(&gpu, "infinity-sum", &[&spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp_with(count, inside_first_subgroup(width), f32::INFINITY);

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    assert_eq!(output.first().copied(), Some(f32::INFINITY));

    if count > width {
        let second: f32 = (width..(width * 2).min(count))
            .map(|value| value as f32)
            .sum();
        assert_eq!(output.get(width).copied(), Some(second));
    }
}

#[test]
fn a_sum_containing_a_nan_is_nan_in_that_subgroup_only() {
    let Some(gpu) = device("nan-sum") else { return };
    let limits = gpu.limits().clone();

    let spirv = kernels::lane_sum_whole::<F32>(limits.subgroup_size).expect("built");
    if !runnable(&gpu, "nan-sum", &[&spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp_with(count, inside_first_subgroup(width), f32::NAN);

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    assert!(
        output.first().copied().is_some_and(f32::is_nan),
        "a NaN in the first subgroup should have reached its total"
    );

    if count > width {
        let second: f32 = (width..(width * 2).min(count))
            .map(|value| value as f32)
            .sum();
        assert_eq!(
            output.get(width).copied(),
            Some(second),
            "and it should not have crossed into the second"
        );
    }
}

#[test]
fn a_maximum_containing_a_nan_behaves_the_same_on_both_reduction_paths() {
    let Some(gpu) = device("nan-max") else { return };
    let limits = gpu.limits().clone();

    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED nan-max: the lane counts here are written for a 32-wide subgroup");
        return;
    }

    let direct_spirv = kernels::lane_max::<F32, 32>(32).expect("built");
    let folded_spirv = kernels::lane_max::<F32, 64>(32).expect("built");
    if !runnable(&gpu, "nan-max", &[&direct_spirv, &folded_spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let direct_input = ramp_with(count, inside_first_subgroup(width), f32::NAN);
    let direct = gpu
        .run(&direct_spirv, &direct_input, 1)
        .expect("dispatched")
        .first()
        .copied()
        .expect("an answer");

    let folded_input = ramp_with(count * 2, inside_first_subgroup(width), f32::NAN);
    let folded = gpu
        .run(&folded_spirv, &folded_input, 1)
        .expect("dispatched")
        .first()
        .copied()
        .expect("an answer");

    eprintln!("nan-max: FMax path gave {direct:?}, compare-and-select path gave {folded:?}");

    assert!(
        direct.is_nan() || direct == 31.0,
        "the first subgroup holds 0..31 with a NaN at {}, so the answer is 31 or NaN, not {direct}",
        inside_first_subgroup(width)
    );
    assert!(
        folded.is_nan() || folded == 95.0,
        "the strip-mined vector spans 0..31 and 64..95, so the answer is 95 or NaN, not {folded}"
    );
}

#[test]
fn a_sum_of_negative_zeros_is_zero_and_its_sign_is_the_implementations_business() {
    let Some(gpu) = device("signed-zero") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::lane_sum_whole::<F32>(limits.subgroup_size).expect("built");
    if !runnable(&gpu, "signed-zero", &[&spirv]) {
        return;
    }

    let count = WORKGROUP_SIZE as usize;
    let input = vec![-0.0_f32; count];

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let total = output.first().copied().expect("an answer");
    eprintln!(
        "signed-zero: the sum of {count} negative zeros has sign bit {}",
        total.to_bits() >> 31
    );

    assert_eq!(total, 0.0, "numerically zero");

    if total.is_sign_negative() {
        eprintln!("signed-zero: this implementation preserves it");
    } else {
        eprintln!(
            "signed-zero: this implementation does not — permitted, and worth knowing about              before trusting a sign bit that came off a GPU"
        );
    }
}

#[test]
fn a_very_large_value_does_not_disturb_the_lanes_around_it() {
    let Some(gpu) = device("large-value") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::lane_sum_whole::<F32>(limits.subgroup_size).expect("built");
    if !runnable(&gpu, "large-value", &[&spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp_with(count, 0, 1.0e30);

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    assert_eq!(
        output.first().copied(),
        Some(1.0e30),
        "the small values are all below the large one's precision, so they leave no trace"
    );

    if count > width {
        let second: f32 = (width..(width * 2).min(count))
            .map(|value| value as f32)
            .sum();
        assert_eq!(output.get(width).copied(), Some(second));
    }
}
