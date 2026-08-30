mod common;

use common::{device, ramp};
use runner::kernels::{self, WORKGROUP_SIZE};
use runner::{Error, Grid, Timing};
use std::time::Duration;

fn scaling(gpu: &runner::Gpu) -> Vec<u32> {
    kernels::scale(gpu.limits().subgroup_size, 2.0).expect("built")
}

fn words(input: &[f32]) -> Vec<u32> {
    input.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn a_timed_grid_dispatch_runs_the_kernel_and_leaves_the_same_answer_behind() {
    let Some(gpu) = device("time-grid") else {
        return;
    };

    let input = ramp(WORKGROUP_SIZE as usize);
    let spirv = scaling(&gpu);
    let expected: Vec<f32> = input.iter().map(|value| value * 2.0).collect();

    let before = gpu.run(&spirv, &input, 1).expect("dispatched");
    let elapsed = gpu
        .time_grid(&spirv, &words(&input), Grid::linear(1), 4)
        .expect("timed");
    let after = gpu.run(&spirv, &input, 1).expect("dispatched");

    assert_eq!(before, expected, "the plain run did not scale");
    assert_eq!(
        after, expected,
        "the answer changed across a timed dispatch, so the timing path left something behind"
    );
    assert!(
        elapsed > Duration::ZERO,
        "a submit and a fence wait cannot take no time at all"
    );
}

#[test]
fn a_timed_grid_counts_both_axes_when_it_bounds_what_it_dispatches() {
    let Some(gpu) = device("time-grid-axes") else {
        return;
    };

    let spirv = scaling(&gpu);
    let input = ramp(WORKGROUP_SIZE as usize);

    let two_axes = gpu
        .time_grid(&spirv, &words(&input), Grid::new(8, 8), 1)
        .expect_err("64 workgroups over a one-workgroup buffer");
    let one_axis = gpu
        .time_grid(&spirv, &words(&input), Grid::linear(64), 1)
        .expect_err("the same 64 workgroups along one axis");

    let (Error::Overrun { needed: across, .. }, Error::Overrun { needed: along, .. }) =
        (&two_axes, &one_axis)
    else {
        panic!("a dispatch past the end of a binding is an Overrun: {two_axes:?}, {one_axis:?}");
    };
    assert_eq!(
        across, along,
        "8 by 8 asked for a different extent than 64 by 1, so one of the axes did not count"
    );

    let wider = ramp(WORKGROUP_SIZE as usize * 2);
    let elapsed = gpu
        .time_grid(&spirv, &words(&wider), Grid::new(1, 2), 1)
        .expect("two workgroups over a two-workgroup buffer");
    assert!(elapsed > Duration::ZERO);
}

#[test]
fn a_repeated_timing_summarises_exactly_the_repeats_it_took() {
    let Some(gpu) = device("time-repeated") else {
        return;
    };

    let spirv = scaling(&gpu);
    let input = ramp(WORKGROUP_SIZE as usize);
    let repeats = 5;

    let timing: Timing = gpu
        .time_repeated(&spirv, &words(&input), 1, 2, repeats)
        .expect("timed");

    assert_eq!(
        timing.repeats, repeats as usize,
        "the summary counted a different number of samples than were taken"
    );
    assert!(
        timing.best <= timing.median && timing.median <= timing.worst,
        "best {:?}, median {:?}, worst {:?} are not in order",
        timing.best,
        timing.median,
        timing.worst
    );
    assert!(timing.best > Duration::ZERO);
}

#[test]
fn a_repeated_timing_asked_for_no_repeats_still_takes_one() {
    let Some(gpu) = device("time-repeated-zero") else {
        return;
    };

    let spirv = scaling(&gpu);
    let input = ramp(WORKGROUP_SIZE as usize);

    let timing = gpu
        .time_repeated(&spirv, &words(&input), 1, 1, 0)
        .expect("zero repeats is floored to one rather than refused");

    assert_eq!(timing.repeats, 1);
    assert_eq!(
        timing.best, timing.worst,
        "one sample is its own best and worst"
    );
}

#[test]
fn a_timed_sum_gives_the_same_reduction_as_an_untimed_one() {
    let Some(gpu) = device("sum-timed") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED sum-timed: no subgroup arithmetic reported");
        return;
    }

    let count = 65_536;
    let input: Vec<f32> = (0..count).map(|index| (index % 16) as f32).collect();
    let expected: f32 = input.iter().sum();

    let mut reducer = gpu.reducer(count).expect("built");
    let plain = reducer.sum(&input).expect("summed");
    let (timed, spans) = reducer.sum_timed(&input).expect("summed and timed");

    assert_eq!(timed.total, expected);
    assert_eq!(
        timed.total, plain.total,
        "asking for the timing changed the answer"
    );
    assert_eq!(timed.dispatches, plain.dispatches);
    assert_eq!(timed.host_combined, plain.host_combined);

    if spans.is_empty() {
        eprintln!("sum-timed: no timestamp spans on {}", gpu.limits().name);
    } else {
        assert_eq!(
            spans.len(),
            timed.dispatches,
            "a span per pass is what the timestamps are written between"
        );
        assert!(
            spans.iter().all(|span| *span > Duration::ZERO),
            "a pass that ran took some time: {spans:?}"
        );
    }
}

#[test]
fn a_timed_scan_gives_the_same_answer_as_an_untimed_one() {
    let Some(gpu) = device("scan-timed") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED scan-timed: no subgroup arithmetic reported");
        return;
    }

    let count = 4_096;
    let input: Vec<f32> = (0..count).map(|index| (index % 8) as f32).collect();

    let mut scanner = gpu.scanner(count).expect("built");
    let plain = scanner.scan(&input).expect("scanned");
    let (timed, spans) = scanner.scan_timed(&input).expect("scanned and timed");

    assert_eq!(
        timed, plain,
        "asking for the timing changed the scan somewhere"
    );

    let mut running = 0.0_f32;
    let reference: Vec<f32> = input
        .iter()
        .map(|value| {
            running += value;
            running
        })
        .collect();
    assert_eq!(timed, reference, "the timed scan is not a prefix sum");

    if spans.is_empty() {
        eprintln!("scan-timed: no timestamp spans on {}", gpu.limits().name);
    } else {
        assert!(
            spans.iter().all(|span| *span > Duration::ZERO),
            "a pass that ran took some time: {spans:?}"
        );
    }
}

#[test]
fn a_device_offers_memory_types_and_at_least_one_of_them_is_device_local() {
    let Some(gpu) = device("memory-types") else {
        return;
    };

    let types = gpu.memory_types();
    assert!(
        !types.is_empty(),
        "a device that reported no memory type could not have allocated the buffer that opened it"
    );

    for (position, kind) in types.iter().enumerate() {
        assert_eq!(
            kind.index, position as u32,
            "a memory type's index is its position in the list"
        );
    }

    assert!(
        types.iter().any(|kind| kind.device_local),
        "Vulkan requires a device-local type and this device reports none: {types:?}"
    );

    let cached = types
        .iter()
        .filter(|kind| kind.host_visible && kind.host_cached)
        .count();
    eprintln!(
        "memory-types: {} types on {}, {cached} host-visible and cached",
        types.len(),
        gpu.limits().name
    );
}

#[test]
fn a_memory_probe_answers_for_a_size_the_device_can_certainly_hold() {
    let Some(gpu) = device("probe-memory") else {
        return;
    };

    let placement = gpu.probe_memory(1024 * 1024).expect("probed");

    assert!(
        placement.device_local,
        "a megabyte did not land in device-local memory, which every reported heap can hold"
    );
    assert!(
        placement.largest_device_heap > 0,
        "a device with a device-local memory type reports no device-local heap"
    );
    assert_eq!(
        placement.resident, 1,
        "probe_memory asks about one buffer, and says so"
    );
}

#[test]
fn a_resident_probe_holds_the_count_it_was_given_and_agrees_with_the_single_one() {
    let Some(gpu) = device("probe-resident") else {
        return;
    };

    let bytes = 1024 * 1024;

    let alone = gpu.probe_memory(bytes).expect("probed");
    let also_alone = gpu.probe_resident(bytes, 1).expect("probed");
    assert_eq!(alone, also_alone);

    let together = gpu.probe_resident(bytes, 3).expect("probed");
    assert_eq!(together.resident, 3);
    assert_eq!(together.largest_device_heap, alone.largest_device_heap);

    let none = gpu.probe_resident(bytes, 0).expect("probed");
    assert_eq!(none.resident, 1, "zero buffers is one buffer, not an error");
}
