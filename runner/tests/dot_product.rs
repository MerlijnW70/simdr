//! The packed integer dot product, on a real device.
//!
//! `OpSDot` replaces four bitcasts, four shifts up, four arithmetic shifts down, four multiplies and
//! three adds — nineteen instructions, counted by differencing the two modules — with one
//! instruction. The strongest thing that can be said about it is that both spellings agree, so
//! every test here runs the pair and compares — the same discipline `loops.rs` uses for the
//! butterfly tree against the built-in reduction.
//!
//! # What the CPU reference is for
//!
//! Agreement between two GPU kernels would still leave both wrong the same way. So the answer is
//! also computed on the host, from `simdr::lanes::signed_bytes`, which is the function the test
//! data was packed with.

mod common;

use common::device;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{pack, signed_bytes, unsigned_bytes};

/// The invocations one workgroup runs, as a `usize`.
fn count() -> usize {
    WORKGROUP_SIZE as usize
}

/// Packed words whose bytes cover the signed range, including the negatives.
///
/// Every fourth element is deliberately at or past 128 unsigned, which is where `OpSDot` and
/// `OpUDot` part company — a corpus of small positive bytes would run both and prove neither.
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

/// Whether this device can run the dot-product kernels at all.
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

    // And the host, so that two wrong kernels cannot agree their way past this.
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
    // Every other test here folds all four positions together with a sum, which does not care
    // which byte is which — so a shift that lands on the wrong byte gives the right total. That is
    // not hypothetical: `24 - byte * 8` mutated to `24 + byte * 8` produces shift counts of 32, 40
    // and 48, this device masks them to 0, 8 and 16, and the result is bytes 0, 3, 2, 1 squared and
    // summed. The same number, from the wrong bytes, and nothing here could see it.
    //
    // So this reads one position at a time and compares against `signed_bytes`, which is the host
    // function the corpus was packed with.
    let Some(gpu) = device("dot-positions") else {
        return;
    };
    if !supported(&gpu, "dot-positions") {
        return;
    }
    let width = gpu.limits().subgroup_size;

    // A corpus whose four bytes are all different in every word, so no two positions can be
    // confused for each other by luck.
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
    // A sum of squares is positive whatever the signs, so the test above would pass against a
    // kernel that read every byte as unsigned. This one puts a negative and a positive byte of the
    // same magnitude in the same word: signed they contribute equally, unsigned they do not.
    let Some(gpu) = device("dot-signs") else {
        return;
    };
    if !supported(&gpu, "dot-signs") {
        return;
    }
    let width = gpu.limits().subgroup_size;

    // -1 and 1 square to the same thing signed; unsigned, -1 is 255 and squares to 65025.
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
    // `OpSUDot` is the shape a quantised layer has: signed weights, unsigned activations. It is
    // not symmetric, and the reference here is the only thing that says which side is which.
    let Some(gpu) = device("dot-mixed") else {
        return;
    };
    if !supported(&gpu, "dot-mixed") {
        return;
    }
    let width = gpu.limits().subgroup_size;

    // Two halves of one buffer: the first is read signed, the second unsigned.
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

    // The discriminator: reading the operands the other way round gives a different answer, so a
    // kernel that swapped them would fail here rather than agreeing by symmetry.
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
    // The pair `runner/examples/dot.rs` times. They differ in one way beyond the instruction: the
    // packed one accumulates with `OpSDotAccSat`, which saturates, and the written-out one adds
    // normally. At these magnitudes neither overflows — and that claim is exactly what this test
    // is for, because if it stopped being true the two would part company silently.
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

    // The host does the same thing: salt, unpack, square, sum, accumulate.
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

    // `packed_dot` is built at the subgroup's width, so this is one element per lane; the strips
    // come from dispatching more workgroups than one.
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
