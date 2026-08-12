//! Plain dot products, run on a real device.
//!
//! The inner loop of every dense layer, without the clipped ReLU that `network.rs` puts on top of
//! it. Integers, so the comparison is exact by construction — a float dot product over 32 terms
//! would need an argument about associativity that says nothing about whether the kernel is right.

mod common;

use common::device;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::I32;

/// A dot product, in the integers a quantised network would use.
///
/// The inner loop of a dense layer: multiply elementwise, sum across the lanes. Integers so the
/// comparison is exact by construction — a float dot product over 32 terms would need an argument
/// about associativity that says nothing about whether the kernel is right.
#[test]
fn a_subgroup_dot_product_matches_the_reference_exactly() {
    let Some(gpu) = device("dot-product") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED dot-product: no subgroup arithmetic reported");
        return;
    }

    // The vector is 32 lanes wide whatever the device is, so on a 64-wide subgroup this is a
    // *cluster* and the reduction covers 32 lanes rather than the subgroup. That is the clustered
    // mapping doing its job, and until there was a 64-wide device to run on, `LANES` and the
    // subgroup width were the same number and nothing distinguished them.
    let width = 32.min(limits.subgroup_size) as usize;
    let count = WORKGROUP_SIZE as usize;

    // Two concatenated vectors in one buffer: weights then activations, which is how a caller
    // with two arrays hands them over.
    let mut input: Vec<u32> = Vec::with_capacity(count * 2);
    input.extend((0..count).map(|index| (index % 7) as u32));
    input.extend((0..count).map(|index| (index % 5 + 1) as u32));

    let output = gpu
        .run_u32(
            &kernels::dot_product::<I32, 32>(limits.subgroup_size, WORKGROUP_SIZE).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    // Every lane of a subgroup holds that subgroup's whole dot product.
    let expected: Vec<u32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            (first..first + width)
                .map(|index| (index % 7) * (index % 5 + 1))
                .sum::<usize>() as u32
        })
        .collect();

    // The buffer is twice as long as the dispatch writes — it holds both operands — so only the
    // written prefix is an answer. The rest is whatever the upload left there.
    assert_eq!(output.get(..count), Some(expected.as_slice()));

    // Discriminator: the two subgroups must disagree, or the reduction could have been over the
    // whole workgroup and nobody would know.
    assert_ne!(
        output.first(),
        output.last(),
        "both subgroups produced the same total, so this proves nothing about the mapping"
    );
}

/// The same, strip-mined: 128 products per subgroup instead of 32.
#[test]
fn a_strip_mined_dot_product_folds_four_products_per_lane() {
    let Some(gpu) = device("dot-product-strips") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED dot-product-strips: no subgroup arithmetic reported");
        return;
    }

    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED dot-product-strips: written for a 32-wide subgroup");
        return;
    }

    // 64 invocations × 4 strips = 256 elements per operand.
    let per_operand = WORKGROUP_SIZE as usize * 4;
    let mut input: Vec<u32> = Vec::with_capacity(per_operand * 2);
    input.extend((0..per_operand).map(|index| (index % 7) as u32));
    input.extend((0..per_operand).map(|index| (index % 5 + 1) as u32));

    let output = gpu
        .run_u32(
            &kernels::dot_product::<I32, 128>(32, per_operand as u32).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    // Subgroup 0 covers invocations 0..32, which read strips at 0..32, 64..96, 128..160, 192..224.
    // Subgroup 1 covers 32..64, reading 32..64, 96..128, 160..192, 224..256.
    let term = |index: usize| (index % 7) * (index % 5 + 1);
    let expected: Vec<u32> = (0..WORKGROUP_SIZE as usize)
        .map(|lane| {
            let first = lane / 32 * 32;
            (first..first + 32)
                .flat_map(|base| (0..4).map(move |strip| base + strip * WORKGROUP_SIZE as usize))
                .map(term)
                .sum::<usize>() as u32
        })
        .collect();

    // As above: only the invocations that ran wrote anything.
    assert_eq!(
        output.get(..WORKGROUP_SIZE as usize),
        Some(expected.as_slice())
    );
}
