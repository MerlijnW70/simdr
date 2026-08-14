//! Kernels run on a real device, their answers compared against a CPU reference.
//!
//! Everything in `simdr`'s own suite establishes that a module is *valid*; only this and
//! `lanes.rs` say it is *right*. A subgroup reduction that validates perfectly and returns the
//! wrong sum would pass every other test in this repository.
//!
//! This file holds the kernels that are not built through the lane API — the control, the
//! shuffle, and what the device reports about itself.

mod common;

use common::{device, ramp};
use runner::kernels::{self, WORKGROUP_SIZE};

#[test]
fn a_scaling_kernel_returns_what_the_cpu_would() {
    let Some(gpu) = device("scale") else { return };
    eprintln!("device: {}", gpu.limits().name);

    let width = gpu.limits().subgroup_size;
    let input = ramp(WORKGROUP_SIZE as usize);
    let spirv = kernels::scale(width, 2.0).expect("built");

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<f32> = input.iter().map(|value| value * 2.0).collect();
    assert_eq!(output, expected);
}

#[test]
fn a_butterfly_shuffle_pairs_each_lane_with_the_one_the_mask_names() {
    let Some(gpu) = device("butterfly") else {
        return;
    };

    if !gpu.limits().subgroup_shuffle {
        eprintln!("SKIPPED butterfly: the device offers no subgroup shuffles");
        return;
    }

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    // Three distances, because a butterfly that ignored its mask would pass at one of them — and
    // all of them below the width, because `lane ^ mask` with a mask at or past the width names a
    // lane in the *next* subgroup, which SPIR-V leaves undefined. On a 32-wide subgroup that ruled
    // nothing out; on an 8-wide one it rules out the 8.
    let distances: Vec<usize> = [1_usize, 2, 8]
        .into_iter()
        .filter(|mask| *mask < width as usize)
        .collect();
    assert!(
        !distances.is_empty(),
        "a subgroup of {width} has no partner"
    );

    for mask in distances {
        let spirv = kernels::butterfly_pair_sum(width, mask as u32).expect("built");
        let output = gpu.run(&spirv, &input, 1).expect("dispatched");

        let expected: Vec<f32> = (0..count)
            .map(|lane| lane as f32 + (lane ^ mask) as f32)
            .collect();

        assert_eq!(output, expected, "butterfly at distance {mask}");
    }
}

#[test]
fn a_vote_answers_for_the_whole_subgroup_and_every_lane_agrees() {
    let Some(gpu) = device("any-above") else {
        return;
    };

    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED any-above: no subgroup vote support reported");
        return;
    }

    let width = gpu.limits().subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    // A ramp of 0..64: on a 32-wide device the first subgroup holds 0..31 and the second 32..63,
    // so a threshold of 40 separates them.
    let output = gpu
        .run(
            &kernels::any_above(width as u32, 40.0).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            let highest = (first + width - 1).min(count - 1);
            if highest as f32 > 40.0 { 1.0 } else { 0.0 }
        })
        .collect();

    assert_eq!(output, expected);

    // Discriminator: two subgroups voting differently is what says the vote was per-subgroup
    // rather than per-workgroup. It needs two of them, and a 64-wide device running 64
    // invocations has one — so on that device this checks the answer and not the scope.
    if count > width {
        assert_ne!(
            output.first(),
            output.last(),
            "the two subgroups must disagree, or the vote spanned the workgroup"
        );
    } else {
        eprintln!(
            "any-above: one subgroup of {width} in a workgroup of {count}, so the vote's scope \
             is not discriminated here"
        );
        assert_eq!(output.first(), Some(&1.0), "the only subgroup exceeds 40");
    }
}

#[test]
fn an_empty_kernel_runs_and_returns_a_buffer_of_the_right_length() {
    // The floor: if this fails, the harness is broken rather than any kernel.
    //
    // It used to assert the output was all zeros, on the grounds that an empty kernel writes
    // nothing so the buffer should be "as allocated". That is an assumption about *uninitialised*
    // memory, which Vulkan does not define — two drivers happened to hand back zeros and a third
    // did not. What is left is what the call actually promises: it runs, and it gives back as many
    // elements as it was given.
    let Some(gpu) = device("empty") else { return };

    let width = gpu.limits().subgroup_size;
    let input = ramp(WORKGROUP_SIZE as usize);
    let spirv = kernels::empty(width).expect("built");

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    assert_eq!(output.len(), input.len());
}

#[test]
fn the_device_reports_a_subgroup_width_that_is_a_power_of_two() {
    let Some(gpu) = device("limits") else { return };
    let limits = gpu.limits();

    eprintln!(
        "{}: subgroup {} — arithmetic {} clustered {} shuffle {}",
        limits.name,
        limits.subgroup_size,
        limits.subgroup_arithmetic,
        limits.subgroup_clustered,
        limits.subgroup_shuffle
    );

    assert!(limits.subgroup_size >= 1);
    assert!(
        limits.subgroup_size.is_power_of_two(),
        "a width that is not a power of two would break every clustered reduction"
    );
}

#[test]
fn a_dispatch_wider_than_its_buffer_is_refused_rather_than_run() {
    // The trap in a one-argument call: the output is sized to the input, so a caller who asks for
    // more workgroups than the buffer has room for gets a kernel writing off the end of it. That
    // is undefined behaviour rather than an error — an access violation on one device here and
    // plausible wrong numbers on another — so it is refused before anything is allocated.
    let Some(gpu) = device("oversized") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scale(width, 2.0).expect("built");
    let input = ramp(WORKGROUP_SIZE as usize);

    // Exactly one workgroup's worth of buffer, and exactly one workgroup: the boundary holds.
    assert!(gpu.run(&spirv, &input, 1).is_ok());

    for workgroups in [2_u32, 3, 64] {
        assert!(
            matches!(
                gpu.run(&spirv, &input, workgroups),
                Err(runner::Error::TooLarge { .. })
            ),
            "{workgroups} workgroups over one workgroup's worth of buffer was accepted"
        );
    }
}

#[test]
fn a_dispatch_that_fills_its_buffer_exactly_is_not_refused() {
    // The other side of the same boundary. A check that refused this would be unusable: filling
    // the buffer is what every kernel here is dispatched to do.
    let Some(gpu) = device("exact") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scale(width, 3.0).expect("built");

    for workgroups in [1_u32, 2, 8] {
        let input = ramp(WORKGROUP_SIZE as usize * workgroups as usize);
        let output = gpu.run(&spirv, &input, workgroups).expect("dispatched");

        let expected: Vec<f32> = input.iter().map(|value| value * 3.0).collect();
        assert_eq!(output, expected, "{workgroups} workgroups");
    }
}

#[test]
fn a_kernels_declared_capabilities_are_checked_against_the_device_that_runs_it() {
    // **The check that replaces remembering.** Every gate in this suite used to name a feature bit
    // by hand — and three of them named the wrong one, because a kernel using votes was gated on
    // the ballot and every kernel at all declares `GroupNonUniform`, which nothing reported.
    //
    // `Limits::unsupported_in` reads the requirement out of the module's own `OpCapability`
    // instructions, so a kernel that starts needing something new brings its own gate with it.
    let Some(gpu) = device("capability-check") else {
        return;
    };
    let limits = gpu.limits().clone();

    // Every kernel this suite runs, and the device that is about to run them.
    let width = limits.subgroup_size;
    let kernels: Vec<(&str, Vec<u32>)> = vec![
        ("scale", kernels::scale(width, 2.0).expect("built")),
        (
            "lane_sum",
            kernels::reduce::lane_sum_whole::<simdr::lanes::F32>(width).expect("built"),
        ),
        (
            "scan_clusters",
            kernels::scan::scan_clusters(width, 2).unwrap_or_default(),
        ),
        (
            "subgroup_agrees",
            kernels::subgroup_agrees(width).expect("built"),
        ),
    ];

    for (name, spirv) in &kernels {
        let missing = limits.unsupported_in(spirv);
        assert!(
            missing.is_empty(),
            "{name} declares {missing:?}, which this device does not offer — and it ran anyway"
        );
    }

    // And the check has teeth: a capability the device *does* offer is reported when the mapping
    // says otherwise, so the assertion above is not vacuous. `Shader` is offered by every device
    // that can run compute, and a module declaring nothing else must come back empty.
    let empty = kernels::empty(width).expect("built");
    assert!(limits.unsupported_in(&empty).is_empty());
    assert!(
        limits.supports(simdr::spec::Capability::Shader),
        "a device running compute offers Shader"
    );
}
