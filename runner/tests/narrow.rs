//! Elements narrower than a lane, on a real device.
//!
//! `tests/kernels.rs` in the emitter proves these modules are valid, and validity is the weaker
//! half here than usual: **the feature that decides whether a narrow reduction runs leaves no
//! trace in the module**. `shaderSubgroupExtendedTypes` is a Vulkan device feature with no SPIR-V
//! capability, so a module reducing over `i8` validates identically whether or not any device in
//! the world would accept it. Only a dispatch can tell.
//!
//! # What is being checked
//!
//! That `decisions/DR-0004` is true: a narrow element is one element per lane, the arithmetic is
//! at the type's own width, and the buffer holds one element per byte. The last of those is the
//! one worth doubting — a stride the device disagreed with would give every fourth element and
//! look like a mapping bug.

mod common;

use common::{device, elements};
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::half;
use simdr::lanes::{F16, I8, I16, U8, U16};

#[test]
fn a_byte_kernel_adds_at_eight_bits_and_wraps_there() {
    let Some(gpu) = device("i8-add") else { return };
    let limits = gpu.limits().clone();

    if !limits.narrow.byte_kernel() {
        eprintln!("SKIPPED i8-add: no shaderInt8 or storageBuffer8BitAccess");
        return;
    }

    // A ramp that crosses 127, so the last elements wrap into the negatives. That wrap is the
    // claim: if the device were computing at 32 bits and truncating on the way out it would give
    // the same answer, but if the *buffer* were being read at the wrong stride it would not.
    let input: Vec<u8> = (0..elements(limits.subgroup_size, 32))
        .map(|index| index as u8)
        .collect();

    let output = gpu
        .run_bytes(
            &kernels::narrow_add::<I8, 32>(limits.subgroup_size, 100).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

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

    if !limits.narrow.byte_kernel() {
        eprintln!("SKIPPED u8-add: no shaderInt8 or storageBuffer8BitAccess");
        return;
    }

    let input: Vec<u8> = (0..elements(limits.subgroup_size, 32))
        .map(|index| (index * 4) as u8)
        .collect();

    let output = gpu
        .run_bytes(
            &kernels::narrow_add::<U8, 32>(limits.subgroup_size, 200).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<u8> = input.iter().map(|byte| byte.wrapping_add(200)).collect();
    assert_eq!(output, expected);
}

#[test]
fn every_element_of_a_byte_buffer_is_its_own_byte() {
    // The stride, stated as an answer rather than as a decoration. Every element is distinct, so a
    // buffer read four bytes apart would return every fourth input and a shape that still looks
    // plausible.
    let Some(gpu) = device("i8-stride") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.narrow.byte_kernel() {
        eprintln!("SKIPPED i8-stride: no shaderInt8 or storageBuffer8BitAccess");
        return;
    }

    let input: Vec<u8> = (0..elements(limits.subgroup_size, 32))
        .map(|index| (index % 61) as u8)
        .collect();

    let output = gpu
        .run_bytes(
            &kernels::narrow_add::<U8, 32>(limits.subgroup_size, 0).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    assert_eq!(
        output, input,
        "adding zero must be the identity element-wise"
    );
}

#[test]
fn a_16_bit_kernel_adds_at_sixteen_bits() {
    let Some(gpu) = device("i16-add") else { return };
    let limits = gpu.limits().clone();

    if !limits.narrow.short_kernel() {
        eprintln!("SKIPPED i16-add: no shaderInt16 or storageBuffer16BitAccess");
        return;
    }

    let input: Vec<u16> = (0..elements(limits.subgroup_size, 32))
        // Bounded, because the buffer is eight times longer on a four-wide device and `index *
        // 1000` leaves a `u16` at 66.
        .map(|index| (index % 60) as u16 * 1000)
        .collect();

    let output = gpu
        .run_halves(
            &kernels::narrow_add::<I16, 32>(limits.subgroup_size, 30_000).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

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

    if !limits.narrow.short_kernel() {
        eprintln!("SKIPPED u16-clamp: no shaderInt16 or storageBuffer16BitAccess");
        return;
    }

    let input: Vec<u16> = (0..elements(limits.subgroup_size, 32))
        // Bounded, as above.
        .map(|index| (index % 90) as u16 * 700)
        .collect();

    let output = gpu
        .run_halves(
            &kernels::narrow_clamp::<U16, 32>(limits.subgroup_size, WORKGROUP_SIZE, 1_000, 20_000)
                .expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

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

    if !limits.narrow.half_kernel() {
        eprintln!("SKIPPED f16-add: no shaderFloat16 or storageBuffer16BitAccess");
        return;
    }

    // 2048 is where a half's precision runs out: it steps by two from there, so 2048 + 1 is 2048
    // and an `f32` computing the same sum would give 2049. That difference is the assertion — it
    // is the only way to tell a real `f16` add from a widened one.
    let input: Vec<u16> = (0..elements(limits.subgroup_size, 32))
        .map(|index| half::from_f32(2048.0 + index as f32))
        .collect();

    let output = gpu
        .run_halves(
            &kernels::narrow_add::<F16, 32>(limits.subgroup_size, u32::from(half::from_f32(1.0)))
                .expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let got: Vec<f32> = output.iter().copied().map(half::to_f32).collect();

    // 2048 + 1 is a tie between 2048 and 2050, and ties go to even — so the increment vanishes.
    // An `f32` add would have given 2049, which is the whole difference being tested.
    assert_eq!(got.first().copied(), Some(2048.0));
    assert_ne!(got.first().copied(), Some(2049.0), "that would be an f32");
    // 2050 + 1 ties between 2050 and 2052, and 2052 is the even one this time.
    assert_eq!(got.get(2).copied(), Some(2052.0));

    // And the whole vector, from the same rule applied on the host. The intermediate is exact in
    // an `f32` at these magnitudes, so rounding once at the end is what the device does too.
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

    if !limits.narrow.byte_kernel() || !limits.narrow.subgroup_extended_types {
        eprintln!(
            "SKIPPED i8-sum: narrow subgroup operations need shaderSubgroupExtendedTypes \
             as well as shaderInt8 and storageBuffer8BitAccess"
        );
        return;
    }
    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED i8-sum: no subgroup arithmetic reported");
        return;
    }

    // The whole subgroup, whatever the subgroup is. A fixed 32 would be one vector on one device,
    // a cluster on another and four strips on a third, and the reduction covers a different number
    // of lanes in each.
    let width = limits.subgroup_size as usize;
    // Small values, so the total of a subgroup stays inside an i8 and the answer is a sum rather
    // than a statement about wrapping.
    // `narrow_sum_whole` is built for the device's own width, so it is one element per invocation
    // whatever that width is — unlike every other kernel in this file, which is built for 32.
    let input: Vec<u8> = (0..elements(limits.subgroup_size, limits.subgroup_size))
        .map(|index| (index % 4) as u8)
        .collect();

    let output = gpu
        .run_bytes(
            &kernels::narrow_sum_whole::<I8>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

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
    // Four elements per lane over a byte buffer: the strip stride is in *elements*, and the byte
    // stride is what the buffer says, so the two multiply. Getting either wrong lands the last
    // strip somewhere else entirely.
    let Some(gpu) = device("i8-strips") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.narrow.byte_kernel() {
        eprintln!("SKIPPED i8-strips: no shaderInt8 or storageBuffer8BitAccess");
        return;
    }
    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED i8-strips: the lane count here is written for a 32-wide subgroup");
        return;
    }

    let elements = elements(limits.subgroup_size, 32) * 4;
    let input: Vec<u8> = (0..elements).map(|index| (index % 61) as u8).collect();

    let output = gpu
        .run_bytes(
            &kernels::narrow_add::<U8, 128>(32, 1).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<u8> = input.iter().map(|byte| byte.wrapping_add(1)).collect();
    assert_eq!(output, expected);
}
