mod common;

use common::{device, elements, runnable};
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::half;
use simdr::lanes::{F16, I8, I16, U8, U16};

#[test]
fn a_byte_kernel_adds_at_eight_bits_and_wraps_there() {
    let Some(gpu) = device("i8-add") else { return };
    let limits = gpu.limits().clone();

    let spirv = kernels::narrow_add::<I8, 32>(limits.subgroup_size, 100).expect("built");
    if !runnable(&gpu, "i8-add", &[&spirv]) {
        return;
    }

    let input: Vec<u8> = (0..elements(limits.subgroup_size, 32))
        .map(|index| index as u8)
        .collect();

    let output = gpu.run_bytes(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u8> = (0..elements(limits.subgroup_size, 32))
        .map(|index| (index as i8).wrapping_add(100) as u8)
        .collect();

    assert_eq!(output, expected);
    assert!(
        expected.iter().any(|&byte| byte >= 0x80),
        "the input must reach past 127 for the wrap to be tested at all"
    );
}

#[test]
fn an_unsigned_byte_kernel_wraps_the_other_way() {
    let Some(gpu) = device("u8-add") else { return };
    let limits = gpu.limits().clone();

    let spirv = kernels::narrow_add::<U8, 32>(limits.subgroup_size, 200).expect("built");
    if !runnable(&gpu, "u8-add", &[&spirv]) {
        return;
    }

    let input: Vec<u8> = (0..elements(limits.subgroup_size, 32))
        .map(|index| (index * 4) as u8)
        .collect();

    let output = gpu.run_bytes(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u8> = input.iter().map(|byte| byte.wrapping_add(200)).collect();
    assert_eq!(output, expected);
}

#[test]
fn every_element_of_a_byte_buffer_is_its_own_byte() {
    let Some(gpu) = device("i8-stride") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::narrow_add::<U8, 32>(limits.subgroup_size, 0).expect("built");
    if !runnable(&gpu, "i8-stride", &[&spirv]) {
        return;
    }

    let input: Vec<u8> = (0..elements(limits.subgroup_size, 32))
        .map(|index| (index % 61) as u8)
        .collect();

    let output = gpu.run_bytes(&spirv, &input, 1).expect("dispatched");

    assert_eq!(
        output, input,
        "adding zero must be the identity element-wise"
    );
}

#[test]
fn a_16_bit_kernel_adds_at_sixteen_bits() {
    let Some(gpu) = device("i16-add") else { return };
    let limits = gpu.limits().clone();

    let spirv = kernels::narrow_add::<I16, 32>(limits.subgroup_size, 30_000).expect("built");
    if !runnable(&gpu, "i16-add", &[&spirv]) {
        return;
    }

    let input: Vec<u16> = (0..elements(limits.subgroup_size, 32))
        .map(|index| (index % 60) as u16 * 1000)
        .collect();

    let output = gpu.run_halves(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u16> = input
        .iter()
        .map(|value| (*value as i16).wrapping_add(30_000) as u16)
        .collect();

    assert_eq!(output, expected);
    assert!(
        input
            .iter()
            .any(|&value| (value as i16).checked_add(30_000).is_none()),
        "the input must overflow an i16 somewhere, or the width is untested"
    );
}

#[test]
fn an_unsigned_16_bit_clamp_holds_its_bounds() {
    let Some(gpu) = device("u16-clamp") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv =
        kernels::narrow_clamp::<U16, 32>(limits.subgroup_size, WORKGROUP_SIZE, 1_000, 20_000)
            .expect("built");
    if !runnable(&gpu, "u16-clamp", &[&spirv]) {
        return;
    }

    let input: Vec<u16> = (0..elements(limits.subgroup_size, 32))
        .map(|index| (index % 90) as u16 * 700)
        .collect();

    let output = gpu.run_halves(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u16> = input
        .iter()
        .map(|value| (*value).clamp(1_000, 20_000))
        .collect();
    assert_eq!(output, expected);
}

#[test]
fn a_half_kernel_computes_in_halves_and_not_in_floats() {
    let Some(gpu) = device("f16-add") else { return };
    let limits = gpu.limits().clone();

    let spirv =
        kernels::narrow_add::<F16, 32>(limits.subgroup_size, u32::from(half::from_f32(1.0)))
            .expect("built");
    if !runnable(&gpu, "f16-add", &[&spirv]) {
        return;
    }

    let input: Vec<u16> = (0..elements(limits.subgroup_size, 32))
        .map(|index| half::from_f32(2048.0 + index as f32))
        .collect();

    let output = gpu.run_halves(&spirv, &input, 1).expect("dispatched");

    let got: Vec<f32> = output.iter().copied().map(half::to_f32).collect();

    assert_eq!(got.first().copied(), Some(2048.0));
    assert_ne!(got.first().copied(), Some(2049.0), "that would be an f32");
    assert_eq!(got.get(2).copied(), Some(2052.0));

    let expected: Vec<f32> = input
        .iter()
        .map(|bits| half::to_f32(half::from_f32(half::to_f32(*bits) + 1.0)))
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn a_narrow_reduction_runs_when_the_device_has_extended_types() {
    let Some(gpu) = device("i8-sum") else { return };
    let limits = gpu.limits().clone();

    if !limits.narrow.subgroup_extended_types {
        eprintln!(
            "SKIPPED i8-sum: narrow subgroup operations need shaderSubgroupExtendedTypes, \
             which no capability in the module can express"
        );
        return;
    }

    let width = limits.subgroup_size as usize;
    let spirv = kernels::narrow_sum_whole::<I8>(limits.subgroup_size).expect("built");
    if !runnable(&gpu, "i8-sum", &[&spirv]) {
        return;
    }

    let input: Vec<u8> = (0..elements(limits.subgroup_size, limits.subgroup_size))
        .map(|index| (index % 4) as u8)
        .collect();

    let output = gpu.run_bytes(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u8> = (0..elements(limits.subgroup_size, limits.subgroup_size))
        .map(|lane| {
            let first = lane / width * width;
            let total: i32 = (first..first + width).map(|index| (index % 4) as i32).sum();
            total as i8 as u8
        })
        .collect();

    assert_eq!(output, expected);
    assert!(
        expected.iter().all(|&byte| byte < 0x80),
        "these totals should not have wrapped; if they did the test is measuring the wrong thing"
    );
}

#[test]
fn a_strip_mined_byte_kernel_reaches_every_strip() {
    let Some(gpu) = device("i8-strips") else {
        return;
    };
    let limits = gpu.limits().clone();

    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED i8-strips: the lane count here is written for a 32-wide subgroup");
        return;
    }

    let spirv = kernels::narrow_add::<U8, 128>(32, 1).expect("built");
    if !runnable(&gpu, "i8-strips", &[&spirv]) {
        return;
    }

    let elements = elements(limits.subgroup_size, 32) * 4;
    let input: Vec<u8> = (0..elements).map(|index| (index % 61) as u8).collect();

    let output = gpu.run_bytes(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u8> = input.iter().map(|byte| byte.wrapping_add(1)).collect();
    assert_eq!(output, expected);
}
