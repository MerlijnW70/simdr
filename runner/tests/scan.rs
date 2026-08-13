//! The prefix sum, run on a device and compared against the CPU element by element.
//!
//! **A scan is a stricter test of the lane mapping than a reduction is.** A reduction sums the same
//! set whatever order the lanes are in, so a mapping that pairs the wrong lanes still returns the
//! right total. A scan does not: element `i` of the answer depends on exactly which elements the
//! hardware considers to come before `i`, so a shuffled mapping produces a wrong number at almost
//! every position while still summing to the same grand total at the end.
//!
//! That last clause is the trap this file is written around. Checking only the final element would
//! pass for a mapping that is wrong everywhere else, so every element is compared.

mod common;

use common::{device, ramp};
use runner::kernels::{self, WORKGROUP_SIZE};

/// The answer, on the host: element `i` is the sum of everything up to and including it.
fn inclusive(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .scan(0.0_f32, |running, value| {
            *running += value;
            Some(*running)
        })
        .collect()
}

#[test]
fn a_workgroup_scan_returns_what_the_cpu_would() {
    let Some(gpu) = device("scan") else { return };
    eprintln!("device: {}", gpu.limits().name);

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scan::scan_workgroup::<simdr::lanes::F32>(width).expect("built");
    let input = ramp(WORKGROUP_SIZE as usize);

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    assert_eq!(output, inclusive(&input));
}

#[test]
fn every_element_is_checked_and_not_just_the_total() {
    // The mapping test. A scan whose lanes are paired wrongly still ends at the same grand total,
    // so the last element agreeing proves nothing on its own — and this asserts the middle first,
    // so a failure reports the position rather than the sum.
    let Some(gpu) = device("scan elements") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scan::scan_workgroup::<simdr::lanes::F32>(width).expect("built");
    let input = ramp(WORKGROUP_SIZE as usize);
    let expected = inclusive(&input);

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
        assert_eq!(got, want, "element {index} of {}", output.len());
    }
    assert_eq!(output.len(), expected.len());
}

#[test]
fn the_subgroup_boundary_is_where_a_workgroup_scan_goes_wrong() {
    // A scan that never combined across subgroups would restart at every boundary: correct within
    // each subgroup and short by the running total at the start of the next. With ones as input
    // the two are trivially told apart — a working scan reads 1, 2, 3, … all the way across, and a
    // restarting one drops back to 1 at the boundary.
    let Some(gpu) = device("scan boundary") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scan::scan_workgroup::<simdr::lanes::F32>(width).expect("built");
    let input = vec![1.0_f32; WORKGROUP_SIZE as usize];

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<f32> = (1..=WORKGROUP_SIZE).map(|index| index as f32).collect();
    assert_eq!(output, expected, "subgroup width {width}");
}

#[test]
fn a_scan_of_zeros_stays_zero_and_a_scan_of_one_element_is_that_element() {
    // The identity cases, which a version that added an uninitialised offset would fail. Shared
    // memory is not zeroed by anyone — `notes/FINDINGS.md` records that two drivers hand back
    // zeros and lavapipe does not — so an offset read before it was written shows up here.
    let Some(gpu) = device("scan zeros") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scan::scan_workgroup::<simdr::lanes::F32>(width).expect("built");

    let zeros = vec![0.0_f32; WORKGROUP_SIZE as usize];
    assert_eq!(
        gpu.run(&spirv, &zeros, 1).expect("dispatched"),
        zeros,
        "a scan of nothing is nothing, however many subgroups it took"
    );

    let mut one = vec![0.0_f32; WORKGROUP_SIZE as usize];
    if let Some(first) = one.first_mut() {
        *first = 7.0;
    }
    let output = gpu.run(&spirv, &one, 1).expect("dispatched");
    assert!(
        output
            .iter()
            .all(|value| (*value - 7.0).abs() < f32::EPSILON),
        "one non-zero at the front should carry all the way across: {output:?}"
    );
}
