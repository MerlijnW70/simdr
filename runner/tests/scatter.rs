//! Atomics, on a real device.
//!
//! These are the first kernels here whose correctness is not a `map` over the input. Several
//! invocations reach one slot, and what has to hold is that **every** contribution lands — which
//! is precisely what a non-atomic read-modify-write gets wrong, and gets wrong intermittently.
//!
//! # Why a total is a weak assertion and a permutation is a strong one
//!
//! A histogram whose bins sum to the input length could still have put the counts in the wrong
//! bins. So the shape of the histogram is checked as well, and `claim_slots` goes further: it
//! asserts that the values every invocation received are `0..n` **exactly**, with no repeats. A
//! lost atomic shows up there as a duplicate rather than as a total that is one short.

mod common;

use common::device;
use runner::kernels::{self, WORKGROUP_SIZE};

/// The invocations one workgroup runs, as a `usize`.
fn count() -> usize {
    WORKGROUP_SIZE as usize
}

/// How many bins the histogram tests use.
const BINS: u32 = 8;

/// Run a counting kernel with its output buffer **zeroed first**, and read the bins back.
///
/// A histogram *accumulates*: it adds to whatever the output buffer already holds. `Gpu::run`
/// allocates that buffer and does not initialise it, and Vulkan says nothing about what is in a
/// fresh allocation — two drivers here hand back zeros and a third hands back whatever was there.
///
/// So the zeroing is the test's job, and it needs a `Session`, which is the only path that can
/// write a binding the kernel reads *and* writes. Getting this wrong looked like a broken atomic.
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

    // Every value below `BINS`, so the clamp is the identity and the bins are the values. Chosen
    // so the counts differ per bin: a uniform input would agree with a kernel that put everything
    // in one bin and divided.
    let input: Vec<u32> = (0..count() as u32).map(|index| index % 5).collect();

    // Binding 1 starts at zero and is where the counts accumulate. It is the same length as the
    // input because `run_u32` sizes the output to match, and only the first `BINS` are read.
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
    // Two instructions, one answer. `OpAtomicIIncrement` takes no value operand and
    // `OpAtomicIAdd` does, so they are different encodings of the same intent — and if either
    // were wrong about its operand order the two would part company here.
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
    // The other half of letting the data choose an address. Values far above the bin count must
    // land in the last bin, not past the end of anything.
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

    // Only element 0 is below the ceiling; everything else clamps to bin 7.
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
    // The strongest statement an atomic test can make. `OpAtomicIAdd` returns what the slot held
    // *before*, so the values handed out are `0..n` with no repeats — and a lost atomic shows up
    // as two invocations writing to one slot, which leaves a hole somewhere else.
    let Some(gpu) = device("claim") else { return };
    let limits = gpu.limits().clone();

    // Slot 0 is the counter and starts at zero; the claims land from slot 1.
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

    // Every invocation wrote its own local index into the slot it claimed, so the claims are a
    // permutation of `0..count` — in some order the scheduler chose, which is why it is sorted
    // before comparing rather than compared as it stands.
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
    // The reason the atomic's scope is the device rather than the workgroup. Four workgroups all
    // reach the same bins, and a workgroup-scoped atomic would order only the invocations that
    // share one — which is a test that passes at one workgroup and fails at four.
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
