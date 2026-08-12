//! The GLSL.std.450 instructions, run on a real device.
//!
//! `tests/kernels.rs` in the emitter proves these modules are *valid*; this proves they compute
//! what their names say. Every kernel here is elementwise, so the expected answer is a `map` over
//! the input and no subgroup mapping sits between the instruction and the number.
//!
//! # What the unit tests cannot see
//!
//! `UMax` and `SMax` agree on every value below 2³¹. A transposition between them is invisible to
//! any test whose numbers are small — which is every test that reads back an emitted module — so
//! the discriminating case is here, with a value that has its top bit set.

mod common;

use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{F32, I32, U32};

use common::device;

/// The invocations one workgroup runs, as a `usize`.
fn count() -> usize {
    WORKGROUP_SIZE as usize
}

/// `value` as the bits a `u32` buffer carries it in.
fn signed_bits(value: i32) -> u32 {
    u32::from_ne_bytes(value.to_ne_bytes())
}

/// The `i32` a buffer word holds.
fn as_signed(word: u32) -> i32 {
    i32::from_ne_bytes(word.to_ne_bytes())
}

#[test]
fn a_clamp_holds_every_element_between_its_bounds() {
    let Some(gpu) = device("clamp") else { return };
    let limits = gpu.limits().clone();

    // A ramp that starts below the low bound and ends above the high one, so all three arms of a
    // clamp are exercised in one dispatch — an input entirely inside the bounds would pass against
    // a kernel that did nothing at all.
    let input: Vec<u32> = (0..count())
        .map(|index| signed_bits(index as i32 - 16))
        .collect();

    let output = gpu
        .run_u32(
            &kernels::clamped::<I32, 32>(limits.subgroup_size, signed_bits(0), signed_bits(20))
                .expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<i32> = (0..count())
        .map(|index| (index as i32 - 16).clamp(0, 20))
        .collect();
    let got: Vec<i32> = output.iter().copied().map(as_signed).collect();

    assert_eq!(got, expected);
    assert!(
        expected.contains(&0) && expected.contains(&20) && expected.contains(&5),
        "the input has to reach all three arms for this to mean anything"
    );
}

#[test]
fn a_magnitude_is_the_value_without_its_sign() {
    let Some(gpu) = device("abs") else { return };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..count())
        .map(|index| signed_bits(index as i32 - 32))
        .collect();

    let output = gpu
        .run_u32(
            &kernels::magnitude::<I32, 32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<i32> = (0..count())
        .map(|index| (index as i32 - 32).abs())
        .collect();
    let got: Vec<i32> = output.iter().copied().map(as_signed).collect();

    assert_eq!(got, expected);
}

#[test]
fn an_unsigned_maximum_is_not_a_signed_one() {
    // The discriminator. Every element has its top bit set, so as unsigned it is enormous and as
    // signed it is negative. `UMax(x, 7)` is `x`; `SMax(x, 7)` is 7. Nothing that reads back an
    // emitted module can tell the two instructions apart, and nothing with small numbers in it can
    // either.
    let Some(gpu) = device("unsigned-max") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..count())
        .map(|index| 0x8000_0000_u32 + index as u32)
        .collect();

    let output = gpu
        .run_u32(
            &kernels::larger::<U32, 32>(limits.subgroup_size, 7).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    assert_eq!(output, input, "an unsigned maximum keeps the large values");
    assert!(
        !output.contains(&7),
        "a signed maximum would have replaced every one of them with 7"
    );
}

#[test]
fn an_unsigned_minimum_is_not_a_signed_one() {
    // The other half: `UMin(x, 7)` is 7 for the same inputs, and `SMin` would keep them.
    let Some(gpu) = device("unsigned-min") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..count())
        .map(|index| 0x8000_0000_u32 + index as u32)
        .collect();

    let output = gpu
        .run_u32(
            &kernels::smaller::<U32, 32>(limits.subgroup_size, 7).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    assert_eq!(output, vec![7; count()]);
}

#[test]
fn a_signed_maximum_reads_the_same_bits_as_negative_numbers() {
    // And the same input under `I32`, which must give the opposite answer. Two kernels differing
    // in one type parameter, one dispatch each, disagreeing — that is the whole claim.
    let Some(gpu) = device("signed-max") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..count())
        .map(|index| 0x8000_0000_u32 + index as u32)
        .collect();

    let output = gpu
        .run_u32(
            &kernels::larger::<I32, 32>(limits.subgroup_size, 7).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    assert_eq!(
        output,
        vec![7; count()],
        "read as signed, every element is negative, so 7 is the larger"
    );
}

#[test]
fn a_square_root_is_within_the_precision_vulkan_promises() {
    let Some(gpu) = device("sqrt") else { return };
    let limits = gpu.limits().clone();

    let input: Vec<f32> = (0..count()).map(|index| index as f32).collect();

    let output = gpu
        .run(
            &kernels::root::<32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    // **Not asserted exactly, and the reason is in the specification.** Vulkan pins `Sqrt` at 3
    // ULP rather than correctly rounded, so demanding `f32::sqrt`'s answer to the bit would be
    // asserting something no implementation promises — and it would pass here and fail on the next
    // device, which is the worst way for a test to be wrong.
    for (index, (&got, &value)) in output.iter().zip(&input).enumerate() {
        let want = value.sqrt();
        let tolerance = 3.0 * f32::EPSILON * want.max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "sqrt({value}) gave {got}, wanted {want} within {tolerance} (element {index})"
        );
    }

    assert_eq!(output.first().copied(), Some(0.0), "sqrt(0) is exact");
    assert_eq!(output.get(4).copied(), Some(2.0), "and so is sqrt(4)");
}

#[test]
fn a_fused_multiply_add_rounds_once_and_the_two_instruction_spelling_does_not() {
    let Some(gpu) = device("fma") else { return };
    let limits = gpu.limits().clone();

    // Values chosen so the intermediate product is *not* exact — with an exact product the two
    // spellings agree and the test would prove nothing. Which of them differ is worked out on the
    // CPU below rather than assumed.
    let input: Vec<f32> = (0..count())
        .map(|index| 1.0 + (index as f32) * 0.100_000_1)
        .collect();

    let output = gpu
        .run(
            &kernels::fused_square::<32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let differing = input
        .iter()
        .filter(|&&value| value.mul_add(value, value) != value * value + value)
        .count();
    assert!(
        differing > 0,
        "these inputs never make the two spellings disagree, so nothing here is being tested"
    );

    for (&got, &value) in output.iter().zip(&input) {
        assert_eq!(
            got,
            value.mul_add(value, value),
            "the device rounded differently from the host's own fused multiply-add"
        );
    }
}

/// What the device does with a NaN in an elementwise extreme — observed, not asserted.
///
/// GLSL.std.450 says which operand is returned "is undefined if one of the operands is a NaN",
/// so both orders are run and printed. Pinning an answer the specification declines to give would
/// turn a driver's freedom into our regression, which is the same reasoning `floats.rs` uses for
/// the group instruction.
#[test]
fn an_extreme_containing_a_nan_is_observed_rather_than_asserted() {
    let Some(gpu) = device("nan-extreme") else {
        return;
    };
    let limits = gpu.limits().clone();

    let mut with_nan: Vec<f32> = (0..count()).map(|index| index as f32).collect();
    if let Some(slot) = with_nan.get_mut(3) {
        *slot = f32::NAN;
    }

    // The NaN is in the buffer, so it is the *first* operand of `FMax(in[i], 1.0)`.
    let nan_first = gpu
        .run(
            &kernels::larger::<F32, 32>(limits.subgroup_size, 1.0_f32.to_bits()).expect("built"),
            &with_nan,
            1,
        )
        .expect("dispatched");

    // And here it is the second: every input is finite, the constant is the NaN.
    let finite: Vec<f32> = (0..count()).map(|index| index as f32).collect();
    let nan_second = gpu
        .run(
            &kernels::larger::<F32, 32>(limits.subgroup_size, f32::NAN.to_bits()).expect("built"),
            &finite,
            1,
        )
        .expect("dispatched");

    let first = nan_first.get(3).copied().expect("an answer");
    let second = nan_second.get(3).copied().expect("an answer");
    eprintln!("nan-extreme: FMax(NaN, 1.0) gave {first:?}, FMax(3.0, NaN) gave {second:?}");

    // Guaranteed either way: the lanes that held no NaN are untouched, and the answer is one of
    // its two operands rather than some third number.
    assert_eq!(
        nan_first.get(10).copied(),
        Some(10.0),
        "a NaN in one lane must not disturb the others"
    );
    assert!(
        first.is_nan() || first == 1.0,
        "FMax(NaN, 1.0) gave {first}, which is neither operand"
    );
    assert!(
        second.is_nan() || second == 3.0,
        "FMax(3.0, NaN) gave {second}, which is neither operand"
    );
}

#[test]
fn a_strip_mined_clamp_bounds_every_strip_and_not_just_the_first() {
    let Some(gpu) = device("clamp-strips") else {
        return;
    };
    let limits = gpu.limits().clone();

    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED clamp-strips: the lane count here is written for a 32-wide subgroup");
        return;
    }

    // 128 lanes on a 32-wide subgroup: four strips, four `SClamp` instructions. A loop that
    // emitted one and copied the rest would be right in the first quarter of the buffer.
    let elements = count() * 4;
    let input: Vec<u32> = (0..elements)
        .map(|index| signed_bits(index as i32 - 64))
        .collect();

    let output = gpu
        .run_u32(
            &kernels::clamped::<I32, 128>(32, signed_bits(-8), signed_bits(8)).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<i32> = (0..elements)
        .map(|index| (index as i32 - 64).clamp(-8, 8))
        .collect();
    let got: Vec<i32> = output.iter().copied().map(as_signed).collect();

    assert_eq!(got, expected);
    assert!(
        got.iter().rev().take(16).all(|&value| value == 8),
        "the last strip has to be bounded too"
    );
}
