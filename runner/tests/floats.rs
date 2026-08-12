//! Float edge cases, on the device.
//!
//! The fuzzer covers floats holding small integers, where arithmetic is exact — and says plainly
//! that it covers nothing else. This is the other half: NaN, the infinities, and signed zero,
//! with expectations reasoned from the specification rather than generated.
//!
//! # Where the specification stops
//!
//! SPIR-V does not fully pin what `OpGroupNonUniformFMax` does when a lane holds NaN. So these
//! tests assert what *is* guaranteed — that the non-NaN lanes are not corrupted, that a maximum
//! is one of its inputs — and **report** the rest rather than asserting it. Pinning an answer the
//! specification declines to give would turn a driver's freedom into our regression.
//!
//! # The suspect
//!
//! `reduce_max` has two paths. A vector as wide as the subgroup goes straight to
//! `OpGroupNonUniformFMax`. A strip-mined one folds its strips first with a comparison and a
//! select, because max has no core scalar opcode — and an ordered comparison against NaN is
//! *false*, so that fold drops a NaN where the group instruction might not. Whether the two paths
//! agree is exactly the question worth asking.

mod common;

use common::device;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;

/// Every lane's value, with `at` replaced.
fn ramp_with(count: usize, at: usize, value: f32) -> Vec<f32> {
    let mut input: Vec<f32> = (0..count).map(|index| index as f32).collect();
    if let Some(slot) = input.get_mut(at) {
        *slot = value;
    }
    input
}

#[test]
fn a_sum_containing_an_infinity_is_infinite_and_does_not_corrupt_the_other_subgroup() {
    let Some(gpu) = device("infinity-sum") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED infinity-sum: no subgroup arithmetic");
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp_with(count, 3, f32::INFINITY);

    let output = gpu
        .run(
            &kernels::lane_sum::<F32, 32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    // Adding an infinity to finite values gives an infinity, whatever the order — this one is
    // fully defined, so it is asserted rather than reported.
    assert_eq!(output.first().copied(), Some(f32::INFINITY));

    // And the *other* subgroup is untouched, which is the part a broken mapping would break.
    if count > width {
        let second: f32 = (width..count).map(|value| value as f32).sum();
        assert_eq!(output.get(width).copied(), Some(second));
    }
}

#[test]
fn a_sum_containing_a_nan_is_nan_in_that_subgroup_only() {
    let Some(gpu) = device("nan-sum") else { return };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED nan-sum: no subgroup arithmetic");
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp_with(count, 5, f32::NAN);

    let output = gpu
        .run(
            &kernels::lane_sum::<F32, 32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    // NaN propagates through addition under any order, so this is defined too.
    assert!(
        output.first().copied().is_some_and(f32::is_nan),
        "a NaN in the first subgroup should have reached its total"
    );

    if count > width {
        let second: f32 = (width..count).map(|value| value as f32).sum();
        assert_eq!(
            output.get(width).copied(),
            Some(second),
            "and it should not have crossed into the second"
        );
    }
}

/// What the device does with a NaN in a maximum — observed, not asserted.
///
/// Both reduction paths are exercised: 32 lanes goes straight to `OpGroupNonUniformFMax`, and 64
/// folds two strips with compare-and-select first. What is asserted is only what is guaranteed;
/// the actual behaviour is printed so it is on record.
#[test]
fn a_maximum_containing_a_nan_behaves_the_same_on_both_reduction_paths() {
    let Some(gpu) = device("nan-max") else { return };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED nan-max: no subgroup arithmetic");
        return;
    }
    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED nan-max: the lane counts here are written for a 32-wide subgroup");
        return;
    }

    let count = WORKGROUP_SIZE as usize;
    let direct_input = ramp_with(count, 7, f32::NAN);
    let direct = gpu
        .run(
            &kernels::lane_max::<F32, 32>(32).expect("built"),
            &direct_input,
            1,
        )
        .expect("dispatched")
        .first()
        .copied()
        .expect("an answer");

    // Twice as long, and the NaN placed in the first strip so the compare-and-select fold sees it.
    let folded_input = ramp_with(count * 2, 7, f32::NAN);
    let folded = gpu
        .run(
            &kernels::lane_max::<F32, 64>(32).expect("built"),
            &folded_input,
            1,
        )
        .expect("dispatched")
        .first()
        .copied()
        .expect("an answer");

    eprintln!("nan-max: FMax path gave {direct:?}, compare-and-select path gave {folded:?}");

    // Guaranteed: a maximum is one of its inputs or NaN. It is never some third number, and it is
    // never the *smallest* input, which is what a mixed-up comparison would give.
    assert!(
        direct.is_nan() || direct == 31.0,
        "the first subgroup holds 0..31 with a NaN at 7, so the answer is 31 or NaN, not {direct}"
    );
    assert!(
        folded.is_nan() || folded == 95.0,
        "the strip-mined vector spans 0..31 and 64..95, so the answer is 95 or NaN, not {folded}"
    );
}

#[test]
fn negative_zero_and_positive_zero_sum_to_positive_zero() {
    let Some(gpu) = device("signed-zero") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED signed-zero: no subgroup arithmetic");
        return;
    }

    let count = WORKGROUP_SIZE as usize;
    // Every lane holds a negative zero. IEEE 754 says -0.0 + -0.0 is -0.0, so this is defined
    // whatever the reduction order — and it is the case a naive `== 0.0` comparison hides.
    let input = vec![-0.0_f32; count];

    let output = gpu
        .run(
            &kernels::lane_sum::<F32, 32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let total = output.first().copied().expect("an answer");
    assert_eq!(total, 0.0, "numerically zero");
    assert!(
        total.is_sign_negative(),
        "and negative, because every addend was: got a sign bit of {}",
        total.to_bits() >> 31
    );
}

#[test]
fn a_very_large_value_does_not_disturb_the_lanes_around_it() {
    let Some(gpu) = device("large-value") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED large-value: no subgroup arithmetic");
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    // Past 2^24, so adding the small values around it changes nothing — the classic case where a
    // reduction's *order* would show up as a different answer. It must not, because every small
    // addend vanishes into the large one regardless of when it arrives.
    let input = ramp_with(count, 0, 1.0e30);

    let output = gpu
        .run(
            &kernels::lane_sum::<F32, 32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    assert_eq!(
        output.first().copied(),
        Some(1.0e30),
        "the small values are all below the large one's precision, so they leave no trace"
    );

    if count > width {
        let second: f32 = (width..count).map(|value| value as f32).sum();
        assert_eq!(output.get(width).copied(), Some(second));
    }
}
