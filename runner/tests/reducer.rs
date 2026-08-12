//! The reduction with its pipelines held, on a real device.
//!
//! `reduction.rs` checks that a full-buffer sum is *right*. This checks that keeping the pipelines
//! and the buffers between calls does not change the answer — which is the only thing a cache is
//! allowed to do, and the thing a cache most easily gets wrong.
//!
//! # The failure a cache has
//!
//! State left behind. A reducer whose buffers still hold the last call's data would answer the
//! second question with a mixture of both, and the mixture would be a plausible number. So the
//! tests here run *different* inputs through one reducer and compare each against its own answer,
//! rather than running the same input twice and finding it stable.

mod common;

use common::device;
use runner::kernels::WORKGROUP_SIZE;
use runner::reduction::dispatches_for;
use runner::{BadLength, Error};

/// Values whose every partial sum stays inside the 24 bits an `f32` carries.
fn payload(count: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|index| (index % 16) as f32 * scale)
        .collect()
}

#[test]
fn a_held_reducer_gives_the_same_answer_as_a_fresh_sum() {
    let Some(gpu) = device("reducer-agrees") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED reducer-agrees: no subgroup arithmetic reported");
        return;
    }

    let count = 65_536;
    let input = payload(count, 1.0);
    let expected: f32 = input.iter().sum();

    let once = gpu.sum(&input).expect("reduced");
    let mut reducer = gpu.reducer(count).expect("built");
    let held = reducer.sum(&input).expect("reduced");

    assert_eq!(once.total, expected);
    assert_eq!(
        held.total, expected,
        "the held pipelines gave a different sum"
    );
    assert_eq!(held.dispatches, once.dispatches);
    assert_eq!(held.dispatches, dispatches_for(count));
}

#[test]
fn a_reducer_reused_does_not_return_the_first_answer_again() {
    // The failure a cache has. Three different inputs through one reducer, each answered on its
    // own terms — the same discipline `session.rs` uses, and for the same reason: a stale buffer
    // returns a number rather than an error.
    let Some(gpu) = device("reducer-reuse") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED reducer-reuse: no subgroup arithmetic reported");
        return;
    }

    let count = 8_192;
    let mut reducer = gpu.reducer(count).expect("built");

    let mut seen = Vec::new();
    for scale in [1.0_f32, 2.0, 7.0] {
        let input = payload(count, scale);
        let expected: f32 = input.iter().sum();

        let reduction = reducer.sum(&input).expect("reduced");

        assert_eq!(reduction.total, expected, "at scale {scale}");
        seen.push(reduction.total);
    }

    seen.dedup();
    assert_eq!(seen.len(), 3, "three inputs gave fewer than three answers");
}

#[test]
fn a_reducer_refuses_an_input_that_is_not_the_length_it_was_built_for() {
    // A shorter slice would leave the tail of the buffer holding the last call's data, and the
    // answer would be this call's sum plus part of that one's. Refused rather than truncated,
    // which is what this crate does everywhere a length disagrees.
    let Some(gpu) = device("reducer-length") else {
        return;
    };

    let count = 4_096;
    let mut reducer = gpu.reducer(count).expect("built");

    assert_eq!(reducer.elements(), count);
    assert!(matches!(
        reducer.sum(&payload(count / 2, 1.0)),
        Err(Error::TooLarge { .. })
    ));
    assert!(matches!(
        reducer.sum(&payload(count * 2, 1.0)),
        Err(Error::TooLarge { .. })
    ));

    // And the right length still works afterwards, so a refusal leaves nothing broken behind.
    let input = payload(count, 1.0);
    let expected: f32 = input.iter().sum();
    assert_eq!(reducer.sum(&input).expect("reduced").total, expected);
}

#[test]
fn a_reducer_refuses_the_lengths_a_one_shot_sum_refuses() {
    // The same guards as `Gpu::sum`, checked here because a reducer that accepted a length it
    // could not fold would move the failure from construction to the first call — later, and
    // further from the mistake.
    let Some(gpu) = device("reducer-shape") else {
        return;
    };

    let minimum = 2 * WORKGROUP_SIZE as usize;

    assert!(matches!(
        gpu.reducer(1_000),
        Err(Error::BadLength(BadLength::NotAPowerOfTwo(1_000)))
    ));
    assert!(matches!(
        gpu.reducer(minimum / 2),
        Err(Error::BadLength(BadLength::TooSmall { .. }))
    ));
    assert!(gpu.reducer(minimum).is_ok(), "the smallest shape it takes");
}

#[test]
fn two_reducers_of_different_lengths_do_not_interfere() {
    // Each owns its own buffers and its own pipelines, so holding two at once has to be safe —
    // and the descriptor sets of one must not name the other's memory.
    let Some(gpu) = device("reducer-two") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED reducer-two: no subgroup arithmetic reported");
        return;
    }

    let mut small = gpu.reducer(4_096).expect("built");
    let mut large = gpu.reducer(16_384).expect("built");

    let small_input = payload(4_096, 1.0);
    let large_input = payload(16_384, 1.0);

    // Interleaved, so a shared buffer would show up as one answer polluting the other.
    let first_small = small.sum(&small_input).expect("reduced").total;
    let first_large = large.sum(&large_input).expect("reduced").total;
    let second_small = small.sum(&small_input).expect("reduced").total;

    assert_eq!(first_small, small_input.iter().sum::<f32>());
    assert_eq!(first_large, large_input.iter().sum::<f32>());
    assert_eq!(second_small, first_small, "the large reducer disturbed it");
    assert_ne!(
        first_small, first_large,
        "the two lengths should not give the same total"
    );

    assert_eq!(small.dispatches(), dispatches_for(4_096));
    assert_eq!(large.dispatches(), dispatches_for(16_384));
}
