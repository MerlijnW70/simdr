mod common;

use common::device;
use runner::kernels::{self, WORKGROUP_SIZE};

fn count() -> usize {
    WORKGROUP_SIZE as usize
}

const BINS: u32 = 8;

fn counted(gpu: &runner::Gpu, spirv: &[u32], input: &[u32], workgroups: u32) -> Vec<u32> {
    let mut session = gpu
        .session(spirv, &[input.len(), input.len()])
        .expect("opened");
    session.write(0, input).expect("uploaded");
    session
        .write(1, &vec![0_u32; input.len()])
        .expect("zeroed the counters");
    session.dispatch(workgroups, 1).expect("dispatched");
    session.read(1, input.len()).expect("read back")
}

#[test]
fn a_histogram_counts_every_input_into_the_bin_it_belongs_in() {
    let Some(gpu) = device("histogram") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..count() as u32).map(|index| index % 5).collect();

    let output = counted(
        &gpu,
        &kernels::histogram(limits.subgroup_size, WORKGROUP_SIZE, BINS).expect("built"),
        &input,
        1,
    );

    let mut expected = vec![0_u32; BINS as usize];
    for value in &input {
        if let Some(bin) = expected.get_mut(*value as usize) {
            *bin += 1;
        }
    }

    assert_eq!(output.get(..BINS as usize), Some(expected.as_slice()));
    assert_eq!(
        output.iter().take(BINS as usize).sum::<u32>(),
        count() as u32,
        "every input was counted exactly once"
    );
    assert!(
        expected.iter().filter(|&&count| count > 0).count() > 1,
        "the input has to reach more than one bin for the shape to mean anything"
    );
}

#[test]
fn counting_by_increment_agrees_with_counting_by_adding_one() {
    let Some(gpu) = device("histogram-increment") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..count() as u32).map(|index| index % 5).collect();

    let added = counted(
        &gpu,
        &kernels::histogram(limits.subgroup_size, WORKGROUP_SIZE, BINS).expect("built"),
        &input,
        1,
    );
    let incremented = counted(
        &gpu,
        &kernels::histogram_incrementing(limits.subgroup_size, WORKGROUP_SIZE, BINS)
            .expect("built"),
        &input,
        1,
    );

    assert_eq!(added.get(..BINS as usize), incremented.get(..BINS as usize));
}

#[test]
fn an_out_of_range_value_is_clamped_into_the_last_bin_rather_than_out_of_the_buffer() {
    let Some(gpu) = device("histogram-clamp") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..count() as u32).map(|index| index * 1_000).collect();

    let output = counted(
        &gpu,
        &kernels::histogram(limits.subgroup_size, WORKGROUP_SIZE, BINS).expect("built"),
        &input,
        1,
    );

    let expected: Vec<u32> = {
        let mut bins = vec![0_u32; BINS as usize];
        for value in &input {
            let bin = (*value).min(BINS - 1) as usize;
            if let Some(slot) = bins.get_mut(bin) {
                *slot += 1;
            }
        }
        bins
    };

    assert_eq!(output.get(..BINS as usize), Some(expected.as_slice()));
    assert_eq!(
        output.get((BINS - 1) as usize).copied(),
        Some(count() as u32 - 1),
        "everything above the ceiling should be in the last bin"
    );
}

#[test]
fn every_invocation_claims_a_different_slot_and_together_they_claim_all_of_them() {
    let Some(gpu) = device("claim") else { return };
    let limits = gpu.limits().clone();

    let input = vec![0_u32; count() + 1];

    let output = counted(
        &gpu,
        &kernels::claim_slots(limits.subgroup_size).expect("built"),
        &input,
        1,
    );

    assert_eq!(
        output.first().copied(),
        Some(count() as u32),
        "the counter should have been incremented once per invocation"
    );

    let mut claimed: Vec<u32> = output
        .get(1..=count())
        .expect("the buffer holds every claim")
        .to_vec();
    claimed.sort_unstable();

    assert_eq!(
        claimed,
        (0..count() as u32).collect::<Vec<u32>>(),
        "two invocations claimed the same slot, so an atomic was lost"
    );
}

#[test]
fn a_histogram_over_several_workgroups_still_counts_everything() {
    let Some(gpu) = device("histogram-groups") else {
        return;
    };
    let limits = gpu.limits().clone();

    let workgroups = 4;
    let elements = count() * workgroups as usize;
    let input: Vec<u32> = (0..elements as u32).map(|index| index % 5).collect();

    let output = counted(
        &gpu,
        &kernels::histogram(limits.subgroup_size, WORKGROUP_SIZE, BINS).expect("built"),
        &input,
        workgroups,
    );

    let mut expected = vec![0_u32; BINS as usize];
    for value in &input {
        if let Some(bin) = expected.get_mut(*value as usize) {
            *bin += 1;
        }
    }

    assert_eq!(output.get(..BINS as usize), Some(expected.as_slice()));
    assert_eq!(
        output.iter().take(BINS as usize).sum::<u32>(),
        elements as u32
    );
}

#[test]
fn an_exchange_hands_out_every_value_exactly_once() {
    let Some(gpu) = device("exchange-chain") else {
        return;
    };
    let limits = gpu.limits().clone();

    const MARKER: u32 = 9_999;

    let mut session = gpu
        .session(
            &kernels::exchange_chain(limits.subgroup_size).expect("built"),
            &[count() + 1, count() + 1],
        )
        .expect("opened");
    let mut initial = vec![0_u32; count() + 1];
    if let Some(head) = initial.first_mut() {
        *head = MARKER;
    }
    session.write(0, &initial).expect("uploaded");
    session.write(1, &initial).expect("seeded the chain");
    session.dispatch(1, 1).expect("dispatched");
    let output = session.read(1, count() + 1).expect("read back");

    let mut seen: Vec<u32> = output.iter().skip(1).copied().collect();
    seen.push(output.first().copied().expect("the slot"));
    seen.sort_unstable();

    let mut expected: Vec<u32> = (0..count() as u32).collect();
    expected.push(MARKER);
    expected.sort_unstable();

    assert_eq!(
        seen, expected,
        "the chain lost, duplicated or invented a value"
    );
}

#[test]
fn an_atomic_load_gathers_through_an_index_the_data_chose() {
    let Some(gpu) = device("atomic-gather") else {
        return;
    };
    let limits = gpu.limits().clone();

    let input: Vec<u32> = (0..count() as u32)
        .map(|index| (index.wrapping_mul(7).wrapping_add(3)) % count() as u32)
        .collect();
    assert!(
        input.iter().enumerate().all(|(at, to)| at as u32 != *to),
        "a fixed point would make a wrong address look right"
    );

    let output = gpu
        .run_u32(
            &kernels::atomic_gather(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<u32> = input
        .iter()
        .map(|to| input.get(*to as usize).copied().unwrap_or_default())
        .collect();

    assert_eq!(output, expected);
}
