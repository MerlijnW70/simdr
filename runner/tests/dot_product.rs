mod common;

use common::device;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{pack, signed_bytes, unsigned_bytes};

fn count() -> usize {
    WORKGROUP_SIZE as usize
}

fn corpus() -> Vec<u32> {
    (0..count())
        .map(|index| {
            let base = index as i32;
            pack([
                (base % 127) - 63,
                -(base % 100),
                ((base * 7) % 255) - 128,
                (base % 5) * 25,
            ])
        })
        .collect()
}

fn supported(gpu: &runner::Gpu, label: &str) -> bool {
    if gpu.limits().narrow.integer_dot_product {
        return true;
    }
    eprintln!("SKIPPED {label}: no shaderIntegerDotProduct");
    false
}

#[test]
fn one_instruction_agrees_with_the_four_it_replaces() {
    let Some(gpu) = device("dot-packed") else {
        return;
    };
    if !supported(&gpu, "dot-packed") {
        return;
    }
    let width = gpu.limits().subgroup_size;
    let input = corpus();

    let packed = gpu
        .run_u32(&kernels::packed_dot(width).expect("built"), &input, 1)
        .expect("dispatched");
    let unpacked = gpu
        .run_u32(&kernels::unpacked_dot(width).expect("built"), &input, 1)
        .expect("dispatched");

    let expected: Vec<u32> = input
        .iter()
        .map(|word| {
            signed_bytes(*word)
                .iter()
                .map(|byte| byte * byte)
                .sum::<i32>() as u32
        })
        .collect();

    assert_eq!(packed.get(..count()), Some(expected.as_slice()));
    assert_eq!(
        unpacked.get(..count()),
        Some(expected.as_slice()),
        "the four-instruction spelling disagrees with the host"
    );
    assert_eq!(packed, unpacked, "the two spellings disagree");
}

#[test]
fn each_byte_position_is_the_one_it_says_it_is() {
    let Some(gpu) = device("dot-positions") else {
        return;
    };
    if !supported(&gpu, "dot-positions") {
        return;
    }
    let width = gpu.limits().subgroup_size;

    let input: Vec<u32> = (0..count())
        .map(|index| {
            let base = index as i32;
            pack([
                (base % 61) - 30,
                (base % 47) + 40,
                -((base % 53) + 1),
                (base % 29) - 100,
            ])
        })
        .collect();

    for byte in 0..4_u32 {
        let output = gpu
            .run_u32(
                &kernels::byte_component(width, byte).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

        let expected: Vec<u32> = input
            .iter()
            .map(|word| signed_bytes(*word)[byte as usize] as u32)
            .collect();

        assert_eq!(
            output.get(..count()),
            Some(expected.as_slice()),
            "byte {byte} is not the byte the kernel read"
        );
    }
}

#[test]
fn the_negatives_are_where_it_matters() {
    let Some(gpu) = device("dot-signs") else {
        return;
    };
    if !supported(&gpu, "dot-signs") {
        return;
    }
    let width = gpu.limits().subgroup_size;

    let input = vec![pack([-1, 1, 0, 0]); count()];

    let output = gpu
        .run_u32(&kernels::packed_dot(width).expect("built"), &input, 1)
        .expect("dispatched");

    assert_eq!(
        output.first().copied(),
        Some(2),
        "the bytes should be read as -1 and 1, giving 1 + 1"
    );
    assert_ne!(
        output.first().copied(),
        Some(65_026),
        "that is the unsigned reading, which is a different instruction"
    );
}

#[test]
fn the_mixed_form_reads_one_operand_signed_and_the_other_not() {
    let Some(gpu) = device("dot-mixed") else {
        return;
    };
    if !supported(&gpu, "dot-mixed") {
        return;
    }
    let width = gpu.limits().subgroup_size;

    let weights: Vec<u32> = (0..count())
        .map(|index| pack([-(index as i32 % 100), 1, -1, 2]))
        .collect();
    let activations: Vec<u32> = (0..count())
        .map(|index| pack([(index as i32 % 200) - 100, 3, 4, 5]))
        .collect();

    let mut input = weights.clone();
    input.extend(&activations);

    let output = gpu
        .run_u32(
            &kernels::mixed_dot(width, WORKGROUP_SIZE).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<u32> = weights
        .iter()
        .zip(&activations)
        .map(|(weight, activation)| {
            signed_bytes(*weight)
                .iter()
                .zip(unsigned_bytes(*activation).iter())
                .map(|(signed, unsigned)| signed * unsigned)
                .sum::<i32>() as u32
        })
        .collect();

    assert_eq!(output.get(..count()), Some(expected.as_slice()));

    let transposed: Vec<u32> = weights
        .iter()
        .zip(&activations)
        .map(|(weight, activation)| {
            unsigned_bytes(*weight)
                .iter()
                .zip(signed_bytes(*activation).iter())
                .map(|(unsigned, signed)| unsigned * signed)
                .sum::<i32>() as u32
        })
        .collect();
    assert_ne!(
        output.get(..count()),
        Some(transposed.as_slice()),
        "the operands are the wrong way round"
    );
}

#[test]
fn the_repeated_kernels_agree_with_each_other_and_with_the_host() {
    let Some(gpu) = device("dot-repeated") else {
        return;
    };
    if !supported(&gpu, "dot-repeated") {
        return;
    }
    let width = gpu.limits().subgroup_size;
    let times = 8_u32;

    let input: Vec<u32> = (0..count())
        .map(|index| pack([index as i32 % 100 - 50, 2, -3, 4]))
        .collect();

    let packed = gpu
        .run_u32(
            &kernels::repeated_packed_dot(width, times).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");
    let unpacked = gpu
        .run_u32(
            &kernels::repeated_unpacked_dot(width, times).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<u32> = input
        .iter()
        .map(|word| {
            (0..times)
                .map(|step| {
                    let salted = word.wrapping_add(step);
                    signed_bytes(salted)
                        .iter()
                        .map(|byte| byte * byte)
                        .sum::<i32>()
                })
                .sum::<i32>() as u32
        })
        .collect();

    assert_eq!(packed.get(..count()), Some(expected.as_slice()));
    assert_eq!(unpacked.get(..count()), Some(expected.as_slice()));
    assert_eq!(
        packed, unpacked,
        "the saturating accumulator and the wrapping one have parted company, \
         so the sums have grown past what an i32 holds"
    );
}

#[test]
fn a_strip_mined_dot_covers_every_strip() {
    let Some(gpu) = device("dot-strips") else {
        return;
    };
    if !supported(&gpu, "dot-strips") {
        return;
    }
    let width = gpu.limits().subgroup_size;

    let workgroups = 4;
    let elements = count() * workgroups as usize;
    let input: Vec<u32> = (0..elements)
        .map(|index| pack([index as i32 % 100 - 50, 2, -3, 4]))
        .collect();

    let output = gpu
        .run_u32(
            &kernels::packed_dot(width).expect("built"),
            &input,
            workgroups,
        )
        .expect("dispatched");

    let expected: Vec<u32> = input
        .iter()
        .map(|word| {
            signed_bytes(*word)
                .iter()
                .map(|byte| byte * byte)
                .sum::<i32>() as u32
        })
        .collect();

    assert_eq!(output, expected);
}
