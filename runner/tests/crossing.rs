mod common;

use common::{device, runnable};
use runner::fuzz::Rng;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{F32, U32};

const DRAWS: u64 = 16;

const REPEATS: u32 = 24;

fn drawn(seed: u64, count: usize) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..count).map(|_| rng.next() as u32).collect()
}

fn drawn_small(seed: u64, count: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..count).map(|_| rng.below(64) as f32).collect()
}

fn totals_per_group(input: &[u32], workgroup: usize) -> Vec<u32> {
    input
        .chunks(workgroup)
        .flat_map(|group| {
            let total = group.iter().copied().fold(0_u32, u32::wrapping_add);
            std::iter::repeat_n(total, group.len())
        })
        .collect()
}

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
