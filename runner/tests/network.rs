//! Dot products, and the quantised network layer built out of one.
//!
//! The layer is `H:\schaak\src\nnue.rs`'s `clipped_dot` at its own dimensions: 256 elements,
//! activations clamped to `[0, 255]`, `i8`-range weights, `i32` accumulator. Integers throughout,
//! so every comparison here is exact — a disagreement is a bug rather than a rounding question,
//! which is not true of the float reductions in `lanes.rs`.

mod common;

use common::device;
use runner::kernels::WORKGROUP_SIZE;
use runner::kernels::network::{Layer, bits, clipped_dot, clipped_dot_split, reference};

/// How many elements one subgroup folds, and therefore one layer's width.
const WIDTH: usize = 256;

/// Activations spanning the whole interesting range: below the floor, inside, above the ceiling.
///
/// A ramp that stayed inside `[0, qa]` would pass with the clamp deleted entirely, which is the
/// one thing this test exists to catch.
fn activations(count: usize) -> Vec<i32> {
    (0..count)
        .map(|index| match index % 4 {
            0 => -(index as i32) - 1,
            1 => (index % 200) as i32,
            2 => 250 + (index % 20) as i32,
            _ => 1_000 + index as i32,
        })
        .collect()
}

/// Weights in the `i8` range the engine quantises to, both signs.
fn weights(count: usize) -> Vec<i32> {
    (0..count).map(|index| (index % 255) as i32 - 127).collect()
}

/// The two arrays as one buffer, which is how the kernel reads them.
fn packed(activations: &[i32], weights: &[i32]) -> Vec<u32> {
    activations
        .iter()
        .chain(weights)
        .map(|&value| bits(value))
        .collect()
}

#[test]
fn a_clipped_dot_product_matches_the_engines_own_loop() {
    let Some(gpu) = device("clipped-dot") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED clipped-dot: no subgroup arithmetic reported");
        return;
    }
    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED clipped-dot: written for a 32-wide subgroup");
        return;
    }

    // One workgroup covers 64 invocations × 8 strips = 512 elements: two subgroups, one 256-wide
    // layer each. That is exactly the engine's two `clipped_dot` calls per position.
    let per_operand = WORKGROUP_SIZE as usize * 8;
    let activations = activations(per_operand);
    let weights = weights(per_operand);
    let input = packed(&activations, &weights);

    let output = gpu
        .run_u32(
            &clipped_dot::<256>(32, per_operand as u32, Layer::QA).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    // Which elements each subgroup covers: the layout blocks by workgroup and strides within it,
    // so subgroup 0 reads lanes 0..32 of every strip and subgroup 1 reads lanes 32..64.
    let covered = |subgroup: usize| -> Vec<usize> {
        let first = subgroup * 32;
        (first..first + 32)
            .flat_map(|lane| (0..8).map(move |strip| lane + strip * WORKGROUP_SIZE as usize))
            .collect()
    };

    for subgroup in 0..2 {
        let indices = covered(subgroup);
        let mine: Vec<i32> = indices.iter().map(|&i| activations[i]).collect();
        let theirs: Vec<i32> = indices.iter().map(|&i| weights[i]).collect();
        let expected = reference(&mine, &theirs, Layer::QA);

        assert_eq!(indices.len(), WIDTH, "one layer per subgroup");

        // Every lane of the subgroup holds the whole total.
        for lane in 0..32 {
            let slot = subgroup * 32 + lane;
            assert_eq!(
                output[slot] as i32, expected,
                "subgroup {subgroup}, lane {lane}"
            );
        }
    }

    // Discriminator: the two subgroups must disagree, or this would pass for a reduction over the
    // whole workgroup.
    assert_ne!(output.first(), output.last());
}

/// Three buffers, which the runner could not bind until now.
///
/// Same layer, same numbers, operands in their own buffers. It has to agree with the concatenated
/// form exactly — two routes to one answer, and the comparison between them is worth more than
/// either against a reference.
#[test]
fn the_split_form_agrees_with_the_concatenated_one() {
    let Some(gpu) = device("clipped-dot-split") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic || limits.subgroup_size != 32 {
        eprintln!("SKIPPED clipped-dot-split: needs a 32-wide subgroup with arithmetic");
        return;
    }

    let per_operand = WORKGROUP_SIZE as usize * 8;
    let activations = activations(per_operand);
    let weights = weights(per_operand);

    let joined = gpu
        .run_u32(
            &clipped_dot::<256>(32, per_operand as u32, Layer::QA).expect("built"),
            &packed(&activations, &weights),
            1,
        )
        .expect("dispatched");

    let as_words = |values: &[i32]| -> Vec<u32> { values.iter().map(|&v| bits(v)).collect() };
    let split = gpu
        .run_bound(
            &clipped_dot_split::<256>(32, Layer::QA).expect("built"),
            &[&as_words(&activations), &as_words(&weights)],
            per_operand,
            1,
        )
        .expect("dispatched");

    assert_eq!(
        split.get(..WORKGROUP_SIZE as usize),
        joined.get(..WORKGROUP_SIZE as usize),
        "the same layer over three buffers gave a different answer"
    );

    // And it is not passing because both are zero.
    assert!(split.iter().take(WORKGROUP_SIZE as usize).any(|&v| v != 0));
}

#[test]
fn an_empty_binding_list_is_refused() {
    let Some(gpu) = device("bindings-refusal") else {
        return;
    };

    assert!(matches!(
        gpu.run_bound(&[], &[], 4, 1),
        Err(runner::Error::NoPipeline)
    ));
    assert!(matches!(
        gpu.run_bound(&[], &[&[1_u32]], 0, 1),
        Err(runner::Error::NoPipeline)
    ));
}

#[test]
fn the_clamp_is_actually_applied_and_not_merely_present() {
    let Some(gpu) = device("clipped-dot-clamp") else {
        return;
    };
    if gpu.limits().subgroup_size != 32 || !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED clipped-dot-clamp: needs a 32-wide subgroup with arithmetic");
        return;
    }

    let per_operand = WORKGROUP_SIZE as usize * 8;
    let spirv = clipped_dot::<256>(32, per_operand as u32, Layer::QA).expect("built");

    // All activations far above the ceiling, all weights one: the answer must be the *clamped*
    // total, not the raw one. Without the clamp it would be a hundred times larger.
    let high = vec![100_000_i32; per_operand];
    let ones = vec![1_i32; per_operand];
    let output = gpu
        .run_u32(&spirv, &packed(&high, &ones), 1)
        .expect("dispatched");

    let clamped = Layer::QA * WIDTH as i32;
    assert_eq!(output[0] as i32, clamped, "the ceiling did not hold");

    // And all of them below the floor: every product is zero, so the layer contributes nothing.
    let low = vec![-100_000_i32; per_operand];
    let output = gpu
        .run_u32(&spirv, &packed(&low, &ones), 1)
        .expect("dispatched");

    assert_eq!(output[0], 0, "the floor did not hold");
}

#[test]
fn a_batch_of_positions_each_get_their_own_answer() {
    // What the whole exercise is for: many independent layers at once. Each workgroup is one
    // position, and no position may see another's activations.
    let Some(gpu) = device("clipped-dot-batch") else {
        return;
    };
    if gpu.limits().subgroup_size != 32 || !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED clipped-dot-batch: needs a 32-wide subgroup with arithmetic");
        return;
    }

    let positions = 8;
    let per_position = WORKGROUP_SIZE as usize * 8;
    let total = positions * per_position;

    // Position p holds the constant p+1 in every activation, so a subgroup reading another
    // position's data would return a visibly wrong multiple.
    let activations: Vec<i32> = (0..total)
        .map(|index| (index / per_position) as i32 + 1)
        .collect();
    let weights = vec![1_i32; total];

    let output = gpu
        .run_u32(
            &clipped_dot::<256>(32, total as u32, Layer::QA).expect("built"),
            &packed(&activations, &weights),
            positions as u32,
        )
        .expect("dispatched");

    for position in 0..positions {
        let expected = (position as i32 + 1) * WIDTH as i32;
        let slot = position * WORKGROUP_SIZE as usize;
        assert_eq!(
            output[slot] as i32, expected,
            "position {position} read the wrong data"
        );
    }
}
