mod common;

use common::{device, runnable};
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::I32;

#[test]
fn a_subgroup_dot_product_matches_the_reference_exactly() {
    let Some(gpu) = device("dot-product") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv =
        kernels::dot_product_whole::<I32>(limits.subgroup_size, WORKGROUP_SIZE).expect("built");
    if !runnable(&gpu, "dot-product", &[&spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;

    let mut input: Vec<u32> = Vec::with_capacity(count * 2);
    input.extend((0..count).map(|index| (index % 7) as u32));
    input.extend((0..count).map(|index| (index % 5 + 1) as u32));

    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            (first..first + width)
                .map(|index| (index % 7) * (index % 5 + 1))
                .sum::<usize>() as u32
        })
        .collect();

    assert_eq!(output.get(..count), Some(expected.as_slice()));

    assert_ne!(
        output.first(),
        output.last(),
        "both subgroups produced the same total, so this proves nothing about the mapping"
    );
}

#[test]
fn a_strip_mined_dot_product_folds_four_products_per_lane() {
    let Some(gpu) = device("dot-product-strips") else {
        return;
    };
    let limits = gpu.limits().clone();

    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED dot-product-strips: written for a 32-wide subgroup");
        return;
    }

    let per_operand = WORKGROUP_SIZE as usize * 4;
    let mut input: Vec<u32> = Vec::with_capacity(per_operand * 2);
    input.extend((0..per_operand).map(|index| (index % 7) as u32));
    input.extend((0..per_operand).map(|index| (index % 5 + 1) as u32));

    let spirv = kernels::dot_product::<I32, 128>(32, per_operand as u32).expect("built");
    if !runnable(&gpu, "dot-product-strips", &[&spirv]) {
        return;
    }

    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

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

    assert_eq!(
        output.get(..WORKGROUP_SIZE as usize),
        Some(expected.as_slice())
    );
}
