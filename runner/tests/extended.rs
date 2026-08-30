mod common;

use runner::kernels;
use simdr::lanes::{F32, I32, U32};

use common::{device, elements};

fn signed_bits(value: i32) -> u32 {
    u32::from_ne_bytes(value.to_ne_bytes())
}

fn as_signed(word: u32) -> i32 {
    i32::from_ne_bytes(word.to_ne_bytes())
}

#[test]
fn a_clamp_holds_every_element_between_its_bounds() {
    let Some(gpu) = device("clamp") else { return };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..elements(limits.subgroup_size, 32))
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

    let expected: Vec<i32> = (0..elements(limits.subgroup_size, 32))
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

    let input: Vec<u32> = (0..elements(limits.subgroup_size, 32))
        .map(|index| signed_bits(index as i32 - 32))
        .collect();

    let output = gpu
        .run_u32(
            &kernels::magnitude::<I32, 32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<i32> = (0..elements(limits.subgroup_size, 32))
        .map(|index| (index as i32 - 32).abs())
        .collect();
    let got: Vec<i32> = output.iter().copied().map(as_signed).collect();

    assert_eq!(got, expected);
}

#[test]
fn an_unsigned_maximum_is_not_a_signed_one() {
    let Some(gpu) = device("unsigned-max") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..elements(limits.subgroup_size, 32))
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
    let Some(gpu) = device("unsigned-min") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..elements(limits.subgroup_size, 32))
        .map(|index| 0x8000_0000_u32 + index as u32)
        .collect();

    let output = gpu
        .run_u32(
            &kernels::smaller::<U32, 32>(limits.subgroup_size, 7).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    assert_eq!(output, vec![7; elements(limits.subgroup_size, 32)]);
}

#[test]
fn a_signed_maximum_reads_the_same_bits_as_negative_numbers() {
    let Some(gpu) = device("signed-max") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..elements(limits.subgroup_size, 32))
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
        vec![7; elements(limits.subgroup_size, 32)],
        "read as signed, every element is negative, so 7 is the larger"
    );
}

#[test]
fn a_square_root_is_within_the_precision_vulkan_promises() {
    let Some(gpu) = device("sqrt") else { return };
    let limits = gpu.limits().clone();

    let input: Vec<f32> = (0..elements(limits.subgroup_size, 32))
        .map(|index| index as f32)
        .collect();

    let output = gpu
        .run(
            &kernels::root::<32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

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

    let input: Vec<f32> = (0..elements(limits.subgroup_size, 32))
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

    let fused = input.iter().map(|value| value.mul_add(*value, *value));
    let twice = input.iter().map(|value| value * value + value);

    let matches_fused = output.iter().copied().eq(fused);
    let matches_twice = output.iter().copied().eq(twice);

    eprintln!(
        "fma: the device matches {}",
        if matches_fused {
            "a fused multiply-add"
        } else if matches_twice {
            "a multiply then an add — two roundings"
        } else {
            "neither spelling"
        }
    );
    assert!(
        matches_fused || matches_twice,
        "the device computed something that is neither a * b + c rounded once nor rounded twice"
    );
}

#[test]
fn an_extreme_containing_a_nan_is_observed_rather_than_asserted() {
    let Some(gpu) = device("nan-extreme") else {
        return;
    };
    let limits = gpu.limits().clone();

    let mut with_nan: Vec<f32> = (0..elements(limits.subgroup_size, 32))
        .map(|index| index as f32)
        .collect();
    if let Some(slot) = with_nan.get_mut(3) {
        *slot = f32::NAN;
    }

    let nan_first = gpu
        .run(
            &kernels::larger::<F32, 32>(limits.subgroup_size, 1.0_f32.to_bits()).expect("built"),
            &with_nan,
            1,
        )
        .expect("dispatched");

    let finite: Vec<f32> = (0..elements(limits.subgroup_size, 32))
        .map(|index| index as f32)
        .collect();
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

    let elements = elements(limits.subgroup_size, 32) * 4;
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
