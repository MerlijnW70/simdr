mod common;

use common::device;
use runner::kernels::WORKGROUP_SIZE;
use runner::reduction::dispatches_for;
use runner::{BadLength, Error};

fn payload(count: usize) -> Vec<f32> {
    (0..count).map(|index| (index % 16) as f32).collect()
}

#[test]
fn a_buffer_larger_than_a_workgroup_reduces_to_one_number() {
    let Some(gpu) = device("full-reduction") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED full-reduction: no subgroup arithmetic reported");
        return;
    }

    let count = 65_536;
    let input = payload(count);
    let expected: f32 = input.iter().sum();

    let reduction = gpu.sum(&input).expect("reduced");

    assert_eq!(reduction.total, expected);
    assert_eq!(reduction.dispatches, dispatches_for(count));
    assert_eq!(
        reduction.host_combined, 1,
        "the host should be reading one number, not assembling it"
    );

    assert!(reduction.total > 0.0);
}

#[test]
fn every_size_from_two_workgroups_upward_agrees_with_the_host() {
    let Some(gpu) = device("reduction-sizes") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED reduction-sizes: no subgroup arithmetic reported");
        return;
    }

    for power in 7..=18 {
        let count = 1_usize << power;
        let input = payload(count);
        let expected: f32 = input.iter().sum();

        let reduction = gpu.sum(&input).expect("reduced");

        assert_eq!(reduction.total, expected, "at 2^{power} elements");
        assert_eq!(
            reduction.dispatches,
            dispatches_for(count),
            "at 2^{power} elements"
        );
    }
}

#[test]
fn a_single_nonzero_element_survives_every_fold() {
    let Some(gpu) = device("reduction-needle") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED reduction-needle: no subgroup arithmetic reported");
        return;
    }

    let count = 8_192;
    for position in [0, 1, 63, 64, 4_095, 4_096, count - 1] {
        let mut input = vec![0.0_f32; count];
        if let Some(slot) = input.get_mut(position) {
            *slot = 1.0;
        }

        let reduction = gpu.sum(&input).expect("reduced");

        assert_eq!(reduction.total, 1.0, "a needle at {position} went missing");
    }
}

#[test]
fn a_length_that_cannot_be_halved_is_refused_by_name() {
    let Some(gpu) = device("reduction-refusal") else {
        return;
    };

    let odd = vec![1.0_f32; 1_000];
    assert!(matches!(
        gpu.sum(&odd),
        Err(Error::BadLength(BadLength::NotAPowerOfTwo(1_000)))
    ));

    let tiny = vec![1.0_f32; 64];
    assert!(matches!(
        gpu.sum(&tiny),
        Err(Error::BadLength(BadLength::TooSmall { length: 64, .. }))
    ));

    let smallest = vec![1.0_f32; 2 * WORKGROUP_SIZE as usize];
    assert_eq!(
        gpu.sum(&smallest).expect("reduced").total,
        2.0 * f64::from(WORKGROUP_SIZE) as f32
    );
}

#[test]
fn an_empty_chain_is_refused_rather_than_returning_the_input() {
    let Some(gpu) = device("empty-chain") else {
        return;
    };

    assert!(matches!(
        gpu.run_chain(&[], &[1, 2, 3]),
        Err(Error::NoPipeline)
    ));
}
