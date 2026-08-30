mod common;

use common::device;
use runner::kernels::WORKGROUP_SIZE;
use runner::reduction::dispatches_for;
use runner::{BadLength, Error};

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

    let input = payload(count, 1.0);
    let expected: f32 = input.iter().sum();
    assert_eq!(reducer.sum(&input).expect("reduced").total, expected);
}

#[test]
fn a_reducer_refuses_the_lengths_a_one_shot_sum_refuses() {
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

#[test]
fn a_reducer_reads_the_end_of_the_pair_its_last_pass_wrote() {
    let Some(gpu) = device("reducer-parity") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED reducer-parity: no subgroup arithmetic reported");
        return;
    }

    let mut odd = 0;
    let mut even = 0;

    for power in 8..=16 {
        let count = 1_usize << power;
        if count < 2 * WORKGROUP_SIZE as usize {
            continue;
        }
        if dispatches_for(count).is_multiple_of(2) {
            even += 1;
        } else {
            odd += 1;
        }

        let mut reducer = match gpu.reducer(count) {
            Ok(reducer) => reducer,
            Err(error) => panic!("{count} elements: {error}"),
        };

        let heavy = payload(count, 4.0);
        let light = payload(count, 1.0);

        let first = reducer.sum(&heavy).expect("reduced");
        assert_eq!(
            first.total,
            heavy.iter().sum::<f32>(),
            "{count} elements, {} passes, first call",
            dispatches_for(count)
        );

        let second = reducer.sum(&light).expect("reduced");
        assert_eq!(
            second.total,
            light.iter().sum::<f32>(),
            "{count} elements, {} passes, second call",
            dispatches_for(count)
        );
    }

    assert!(
        odd > 0 && even > 0,
        "only one parity was covered: {odd} odd, {even} even"
    );
}

#[test]
fn a_chain_of_either_parity_returns_the_buffer_its_last_pass_wrote() {
    let Some(gpu) = device("chain-parity") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    let doubling = runner::kernels::scale(width, 2.0).expect("built");
    let count = 4 * WORKGROUP_SIZE as usize;
    let input: Vec<u32> = (0..count).map(|index| (index as f32).to_bits()).collect();

    for passes in 1..=4_u32 {
        let chain: Vec<runner::Pass<'_>> = (0..passes)
            .map(|_| runner::Pass::new(&doubling, count as u32 / WORKGROUP_SIZE))
            .collect();

        let output = gpu.run_chain(&chain, &input).expect("dispatched");

        let factor = 2.0_f32.powi(passes as i32);
        let expected: Vec<u32> = (0..count)
            .map(|index| (index as f32 * factor).to_bits())
            .collect();
        assert_eq!(output, expected, "{passes} passes, factor {factor}");
    }
}

#[test]
fn a_head_read_returns_that_many_words_and_the_right_ones() {
    let Some(gpu) = device("chain-head") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    let doubling = runner::kernels::scale(width, 2.0).expect("built");
    let count = 4 * WORKGROUP_SIZE as usize;
    let input: Vec<u32> = (0..count).map(|index| (index as f32).to_bits()).collect();
    let passes = [runner::Pass::new(&doubling, count as u32 / WORKGROUP_SIZE)];

    let whole = gpu.run_chain(&passes, &input).expect("dispatched");
    assert_eq!(whole.len(), count, "the full read is still the full buffer");

    for head in [1_usize, 2, 17, count - 1, count] {
        let prefix = gpu
            .run_chain_head(&passes, &input, head)
            .expect("dispatched");

        assert_eq!(prefix.len(), head, "asked for {head}");
        assert_eq!(
            prefix.as_slice(),
            whole.get(..head).expect("in range"),
            "the first {head} words are not the first {head} words"
        );
    }
}

#[test]
fn a_head_larger_than_the_buffer_returns_the_buffer() {
    let Some(gpu) = device("chain-head-clamp") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    let doubling = runner::kernels::scale(width, 2.0).expect("built");
    let count = 2 * WORKGROUP_SIZE as usize;
    let input: Vec<u32> = (0..count).map(|index| (index as f32).to_bits()).collect();
    let passes = [runner::Pass::new(&doubling, count as u32 / WORKGROUP_SIZE)];

    let output = gpu
        .run_chain_head(&passes, &input, count * 100)
        .expect("dispatched");

    assert_eq!(output.len(), count);
}

#[test]
fn a_mapped_reducer_computes_the_sum_of_squares_without_a_round_trip() {
    let Some(gpu) = device("reducer-mapped") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED reducer-mapped: no subgroup arithmetic reported");
        return;
    }
    let width = gpu.limits().subgroup_size;
    let square = runner::kernels::square(width).expect("built");

    for power in 8..=14 {
        let count = 1_usize << power;
        if count < 2 * WORKGROUP_SIZE as usize {
            continue;
        }

        let input: Vec<f32> = (0..count).map(|index| (index % 8) as f32).collect();
        let expected: f32 = input.iter().map(|value| value * value).sum();

        let mut mapped = gpu.reducer_of(count, &square).expect("built");
        let total = mapped.sum(&input).expect("reduced").total;
        assert_eq!(total, expected, "{count} elements");

        let squares = gpu
            .run(&square, &input, count as u32 / WORKGROUP_SIZE)
            .expect("mapped");
        let stepwise = gpu
            .reducer(count)
            .expect("built")
            .sum(&squares)
            .expect("reduced");
        assert_eq!(
            total, stepwise.total,
            "{count} elements, against the two-step route"
        );
    }
}

#[test]
fn a_mapped_reducer_runs_one_more_dispatch_than_a_plain_one() {
    let Some(gpu) = device("reducer-mapped-count") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED reducer-mapped-count: no subgroup arithmetic reported");
        return;
    }
    let width = gpu.limits().subgroup_size;
    let square = runner::kernels::square(width).expect("built");

    let count = 8_192;
    let plain = gpu.reducer(count).expect("built");
    let mapped = gpu.reducer_of(count, &square).expect("built");

    assert_eq!(mapped.dispatches(), plain.dispatches() + 1);
    assert_eq!(plain.dispatches(), dispatches_for(count));
}
