mod common;

use common::{device, runnable};
use runner::kernels::WORKGROUP_SIZE;
use runner::kernels::network::{Layer, bits, clipped_dot, clipped_dot_split, reference};

const WIDTH: usize = 256;

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

fn weights(count: usize) -> Vec<i32> {
    (0..count).map(|index| (index % 255) as i32 - 127).collect()
}

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

    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED clipped-dot: written for a 32-wide subgroup");
        return;
    }

    let per_operand = WORKGROUP_SIZE as usize * 8;
    let activations = activations(per_operand);
    let weights = weights(per_operand);
    let input = packed(&activations, &weights);

    let spirv = clipped_dot::<256>(32, per_operand as u32, Layer::QA).expect("built");
    if !runnable(&gpu, "clipped-dot", &[&spirv]) {
        return;
    }

    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

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

        for lane in 0..32 {
            let slot = subgroup * 32 + lane;
            assert_eq!(
                output[slot] as i32, expected,
                "subgroup {subgroup}, lane {lane}"
            );
        }
    }

    assert_ne!(output.first(), output.last());
}

#[test]
fn the_split_form_agrees_with_the_concatenated_one() {
    let Some(gpu) = device("clipped-dot-split") else {
        return;
    };
    let limits = gpu.limits().clone();

    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED clipped-dot-split: written for a 32-wide subgroup");
        return;
    }

    let per_operand = WORKGROUP_SIZE as usize * 8;
    let activations = activations(per_operand);
    let weights = weights(per_operand);

    let joined_spirv = clipped_dot::<256>(32, per_operand as u32, Layer::QA).expect("built");
    let split_spirv = clipped_dot_split::<256>(32, Layer::QA).expect("built");
    if !runnable(&gpu, "clipped-dot-split", &[&joined_spirv, &split_spirv]) {
        return;
    }

    let joined = gpu
        .run_u32(&joined_spirv, &packed(&activations, &weights), 1)
        .expect("dispatched");

    let as_words = |values: &[i32]| -> Vec<u32> { values.iter().map(|&v| bits(v)).collect() };
    let split = gpu
        .run_bound(
            &split_spirv,
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
    if gpu.limits().subgroup_size != 32 {
        eprintln!("SKIPPED clipped-dot-clamp: written for a 32-wide subgroup");
        return;
    }

    let per_operand = WORKGROUP_SIZE as usize * 8;
    let spirv = clipped_dot::<256>(32, per_operand as u32, Layer::QA).expect("built");
    if !runnable(&gpu, "clipped-dot-clamp", &[&spirv]) {
        return;
    }

    let high = vec![100_000_i32; per_operand];
    let ones = vec![1_i32; per_operand];
    let output = gpu
        .run_u32(&spirv, &packed(&high, &ones), 1)
        .expect("dispatched");

    let clamped = Layer::QA * WIDTH as i32;
    assert_eq!(output[0] as i32, clamped, "the ceiling did not hold");

    let low = vec![-100_000_i32; per_operand];
    let output = gpu
        .run_u32(&spirv, &packed(&low, &ones), 1)
        .expect("dispatched");

    assert_eq!(output[0], 0, "the floor did not hold");
}

#[test]
fn a_batch_of_positions_each_get_their_own_answer() {
    let Some(gpu) = device("clipped-dot-batch") else {
        return;
    };
    if gpu.limits().subgroup_size != 32 {
        eprintln!("SKIPPED clipped-dot-batch: written for a 32-wide subgroup");
        return;
    }

    let positions = 8;
    let per_position = WORKGROUP_SIZE as usize * 8;
    let total = positions * per_position;

    let activations: Vec<i32> = (0..total)
        .map(|index| (index / per_position) as i32 + 1)
        .collect();
    let weights = vec![1_i32; total];

    let batch_spirv = clipped_dot::<256>(32, total as u32, Layer::QA).expect("built");
    if !runnable(&gpu, "clipped-dot-batch", &[&batch_spirv]) {
        return;
    }

    let output = gpu
        .run_u32(
            &batch_spirv,
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
