//! The kernels that cross *between* subgroups, over inputs nobody chose.
//!
//! Everything the differential fuzzer reaches stops at the subgroup: a `Program`'s finish is a
//! reduction, a scan or a vote, and every one of those is one subgroup's own instruction. So the
//! kernels that go wider — `workgroup_sum`, `scan_workgroup` and `scan_workgroup_exclusive`, which
//! write a partial to **workgroup shared memory**, take a `barrier`, and fold across it — have only
//! ever been checked by hand.
//!
//! That is the state `reduce_min` was in when the fuzzer found it folding its strips with a
//! *maximum*: right for every mapping but one, so no hand-written test had looked. These have had
//! one input each — `ramp(WORKGROUP_SIZE)` — one dispatch, and one workgroup.
//!
//! # Three axes, and each was fixed at one value
//!
//! * **The input.** A ramp is one shape. These draw from a seed, so a failure names it.
//! * **The dispatch.** One workgroup, so nothing ever asked whether a second one folds *its own*
//!   shared memory rather than the first one's — which is the mistake the shared slot invites.
//! * **The run.** Once. A barrier that is missing or in the wrong place is a **race**, and a race is
//!   the one defect that `spirv-val` cannot see, a CPU reference cannot predict and a single run
//!   usually survives. Repeating a dispatch and comparing is not a proof, but it is the only
//!   instrument here that points at it at all.
//!
//! # Why the answers are exact
//!
//! Wrapping `u32` addition is associative and commutative, so whatever order the hardware folds in,
//! the total and every prefix are the same number. The float kernels are checked over small
//! integers for the reason [`runner::fuzz::Domain::ceiling`] gives: they stay well inside the range
//! an `f32` counts exactly, so comparing exactly is legitimate rather than lucky.

mod common;

use common::{device, runnable};
use runner::fuzz::Rng;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{F32, U32};

/// How many random inputs each kernel is checked over.
const DRAWS: u64 = 16;

/// How many times a dispatch is repeated when looking for a race.
const REPEATS: u32 = 24;

/// `count` words drawn from `seed`, full range.
///
/// Full range on purpose: these fold with wrapping addition, so the top bits are as much under test
/// as the bottom ones, and a ramp never sets them.
fn drawn(seed: u64, count: usize) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..count).map(|_| rng.next() as u32).collect()
}

/// The same, as small integers a float counts exactly.
fn drawn_small(seed: u64, count: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..count).map(|_| rng.below(64) as f32).collect()
}

/// Each workgroup's own total, repeated for every invocation in it.
fn totals_per_group(input: &[u32], workgroup: usize) -> Vec<u32> {
    input
        .chunks(workgroup)
        .flat_map(|group| {
            let total = group.iter().copied().fold(0_u32, u32::wrapping_add);
            std::iter::repeat_n(total, group.len())
        })
        .collect()
}

/// An inclusive prefix within each workgroup.
fn prefix_per_group(input: &[u32], workgroup: usize, exclusive: bool) -> Vec<u32> {
    input
        .chunks(workgroup)
        .flat_map(|group| {
            let mut running = 0_u32;
            group
                .iter()
                .map(|&value| {
                    let before = running;
                    running = running.wrapping_add(value);
                    if exclusive { before } else { running }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn a_workgroup_sum_agrees_over_inputs_nobody_chose() {
    let Some(gpu) = device("crossing-sum") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::workgroup_sum::<U32>(width).expect("built");
    if !runnable(&gpu, "crossing-sum", &[&spirv]) {
        return;
    }

    let workgroup = WORKGROUP_SIZE as usize;
    for seed in 0..DRAWS {
        // **Several workgroups, which these two kernels had never been dispatched over.** Not the
        // shared memory — Vulkan gives every workgroup its own instance of that, so there is no
        // reading of a neighbour's slots to get wrong. It is the *input* addressing: each group has
        // to fold its own run, and `scan_blocks` is checked this way one file over for exactly that
        // reason — "every block writing to slot zero is correct at one workgroup and wrong at two".
        // The reasoning existed; it had not been pointed here.
        for groups in [1_u32, 2, 5] {
            let input = drawn(seed, workgroup * groups as usize);
            let output = gpu.run_u32(&spirv, &input, groups).expect("dispatched");

            assert_eq!(
                output,
                totals_per_group(&input, workgroup),
                "workgroup_sum disagreed at seed {seed} over {groups} workgroups"
            );
        }
    }
}

#[test]
fn a_workgroup_scan_agrees_over_inputs_nobody_chose() {
    let Some(gpu) = device("crossing-scan") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let inclusive = kernels::scan::scan_workgroup::<U32>(width).expect("built");
    let exclusive = kernels::scan::scan_workgroup_exclusive::<U32>(width).expect("built");
    if !runnable(&gpu, "crossing-scan", &[&inclusive, &exclusive]) {
        return;
    }

    let workgroup = WORKGROUP_SIZE as usize;
    for seed in 0..DRAWS {
        for groups in [1_u32, 2, 5] {
            let input = drawn(seed, workgroup * groups as usize);

            // Both directions, because they are two modules and the exclusive one is the form a
            // long scan's block offsets need — `notes/NEXT.md` records that it stopped being a
            // nicety when a float scan turned out not to recover it by subtraction.
            for (spirv, is_exclusive, name) in [
                (&inclusive, false, "inclusive"),
                (&exclusive, true, "exclusive"),
            ] {
                let output = gpu.run_u32(spirv, &input, groups).expect("dispatched");
                assert_eq!(
                    output,
                    prefix_per_group(&input, workgroup, is_exclusive),
                    "the {name} workgroup scan disagreed at seed {seed} over {groups} workgroups"
                );
            }
        }
    }
}

#[test]
fn the_float_forms_agree_too() {
    // The integer forms above prove the *mapping*; these prove the same kernels reach it through
    // `OpGroupNonUniformFAdd` and a float shared slot, which are different instructions and a
    // different type in the shared array.
    let Some(gpu) = device("crossing-float") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let sum = kernels::workgroup_sum::<F32>(width).expect("built");
    let scan = kernels::scan::scan_workgroup::<F32>(width).expect("built");
    if !runnable(&gpu, "crossing-float", &[&sum, &scan]) {
        return;
    }

    let workgroup = WORKGROUP_SIZE as usize;
    for seed in 0..DRAWS {
        let input = drawn_small(seed, workgroup * 2);

        let totals = gpu.run(&sum, &input, 2).expect("dispatched");
        let expected: Vec<f32> = input
            .chunks(workgroup)
            .flat_map(|group| {
                let total: f32 = group.iter().sum();
                std::iter::repeat_n(total, group.len())
            })
            .collect();
        assert_eq!(
            totals, expected,
            "the float workgroup sum disagreed at {seed}"
        );

        let scanned = gpu.run(&scan, &input, 2).expect("dispatched");
        let expected: Vec<f32> = input
            .chunks(workgroup)
            .flat_map(|group| {
                group
                    .iter()
                    .scan(0.0_f32, |running, value| {
                        *running += value;
                        Some(*running)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            scanned, expected,
            "the float workgroup scan disagreed at {seed}"
        );
    }
}

#[test]
fn the_same_dispatch_twice_gives_the_same_answer() {
    // **The one defect no other layer here can see.** These kernels write a partial to shared
    // memory, take a `barrier`, and read their neighbours' slots. A barrier that is missing, or
    // placed where not every invocation reaches it, is a data race — and a race validates cleanly,
    // has no CPU reference to disagree with, and usually survives one run.
    //
    // Repeating a dispatch and comparing does not prove there is no race. It is the only instrument
    // in this suite that points at one at all, and it costs a few milliseconds.
    let Some(gpu) = device("crossing-repeat") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let sum = kernels::workgroup_sum::<U32>(width).expect("built");
    let scan = kernels::scan::scan_workgroup::<U32>(width).expect("built");
    if !runnable(&gpu, "crossing-repeat", &[&sum, &scan]) {
        return;
    }

    let input = drawn(99, WORKGROUP_SIZE as usize * 4);
    for (spirv, name) in [(&sum, "workgroup_sum"), (&scan, "scan_workgroup")] {
        let first = gpu.run_u32(spirv, &input, 4).expect("dispatched");
        for repeat in 1..REPEATS {
            let again = gpu.run_u32(spirv, &input, 4).expect("dispatched");
            assert_eq!(
                again, first,
                "{name} gave a different answer on run {repeat}, which is a race rather than a \
                 wrong number — nothing else in this suite can see one"
            );
        }
    }
}
