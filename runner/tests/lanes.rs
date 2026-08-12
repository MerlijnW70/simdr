//! The lane API, run on a real device.
//!
//! `kernels::lane_sum` is four lines that name no reduction shape, no cluster size and no opcode:
//! [`simdr::kernel::Kernel`] and [`simdr::lanes::Lanes`] derive all of them from the lane count
//! against the device's width and from the element type. These are the tests where that stops
//! being a design argument and becomes a number.

mod common;

use common::{device, grouped_sums, ramp};
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{F32, U32};

/// A reduction across the *whole workgroup*, through shared memory.
///
/// Every subgroup instruction stops at the subgroup, so combining two of them needs a handover
/// nothing here could express until `Kernel::shared` and `Kernel::barrier` existed. The answer is
/// the sum of all 64 invocations rather than of one 32-lane subgroup, and every invocation holds
/// it — which is what makes a wrong barrier visible as a wrong number rather than as a hang.
#[test]
fn a_workgroup_reduction_crosses_between_subgroups() {
    let Some(gpu) = device("workgroup-sum") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED workgroup-sum: no subgroup arithmetic reported");
        return;
    }

    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    let whole: f32 = input.iter().sum();

    let output = gpu
        .run(
            &kernels::workgroup_sum::<F32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    assert_eq!(
        output,
        vec![whole; count],
        "every invocation should hold the whole workgroup's total"
    );

    // The discriminator that matters: a subgroup reduction over the same input gives a *different*
    // answer, so this is not passing because the two happen to coincide.
    let per_subgroup = gpu
        .run(
            &kernels::lane_sum::<F32, 32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");
    assert_ne!(
        output, per_subgroup,
        "the workgroup sum equals the subgroup sum, so nothing crossed between them"
    );
    assert_eq!(output.first(), output.last(), "and it is uniform");
}

/// The same over integers, where the comparison is exact by construction.
#[test]
fn a_workgroup_reduction_is_exact_over_integers() {
    let Some(gpu) = device("workgroup-sum-u32") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED workgroup-sum-u32: no subgroup arithmetic reported");
        return;
    }

    let count = WORKGROUP_SIZE as usize;
    let input: Vec<u32> = (0..count as u32).map(|index| index * 3 + 1).collect();
    let whole: u32 = input.iter().sum();

    let output = gpu
        .run_u32(
            &kernels::workgroup_sum::<U32>(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    assert_eq!(output, vec![whole; count]);
}

/// One source, three lane counts, three different answers — each what the mapping predicts.
#[test]
fn the_lane_api_reduces_over_exactly_the_lanes_its_width_names() {
    let Some(gpu) = device("lane-sum") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic || !limits.subgroup_clustered {
        eprintln!("SKIPPED lane-sum: the device lacks clustered subgroup arithmetic");
        return;
    }

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let four = gpu
        .run(
            &kernels::lane_sum::<F32, 4>(width).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");
    assert_eq!(four, grouped_sums(count, 4));

    let eight = gpu
        .run(
            &kernels::lane_sum::<F32, 8>(width).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");
    assert_eq!(eight, grouped_sums(count, 8));

    let full = match width {
        32 => gpu
            .run(&kernels::lane_sum::<F32, 32>(32).expect("built"), &input, 1)
            .expect("dispatched"),
        64 => gpu
            .run(&kernels::lane_sum::<F32, 64>(64).expect("built"), &input, 1)
            .expect("dispatched"),
        other => {
            eprintln!("SKIPPED lane-sum: no full-width case written for a subgroup of {other}");
            return;
        }
    };
    assert_eq!(full, grouped_sums(count, width as usize));

    // The three must genuinely differ, or the lane counts were being ignored.
    assert_ne!(four.first(), eight.first());
    assert_ne!(eight.first(), full.first());
}

/// A vector wider than the subgroup: each lane holds several elements, folded before the reduce.
///
/// The layout is the kernel's, not the caller's: workgroup `w` owns a contiguous run of
/// `WORKGROUP_SIZE × strips` elements and strides within it. With one workgroup dispatched, that
/// makes invocation `i` read `i` and `i + WORKGROUP_SIZE`.
#[test]
fn a_strip_mined_vector_reduces_over_more_elements_than_there_are_lanes() {
    let Some(gpu) = device("strip-mined") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED strip-mined: no subgroup arithmetic");
        return;
    }

    let width = limits.subgroup_size as usize;
    let strips = 2_usize;
    let stride = WORKGROUP_SIZE as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count * strips);

    let spirv = match limits.subgroup_size {
        32 => kernels::lane_sum::<F32, 64>(32).expect("built"),
        64 => kernels::lane_sum::<F32, 128>(64).expect("built"),
        other => {
            eprintln!("SKIPPED strip-mined: no case written for a subgroup of {other}");
            return;
        }
    };

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    // Each lane contributes both of its elements, and the subgroup reduces all of them.
    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            (first..first + width)
                .flat_map(|other| (0..strips).map(move |strip| (other + strip * stride) as f32))
                .sum()
        })
        .collect();

    assert_eq!(&output[..count], &expected[..count]);

    // Discriminator: a kernel that ignored the second strip would give the plain subgroup sum.
    let one_strip_only: f32 = (0..width).map(|other| other as f32).sum();
    assert_ne!(
        output.first(),
        Some(&one_strip_only),
        "the second strip was not folded in"
    );
}

/// More than one workgroup, which is what the blocked layout exists for.
///
/// With two workgroups the second owns the run starting at `WORKGROUP_SIZE × strips`, so this is
/// the test that would fail if the base were computed from a dispatch-wide stride instead.
#[test]
fn a_second_workgroup_reads_its_own_run_rather_than_the_first_ones() {
    let Some(gpu) = device("two-workgroups") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED two-workgroups: no subgroup arithmetic");
        return;
    }

    let width = limits.subgroup_size as usize;
    let groups = 2_usize;
    let count = WORKGROUP_SIZE as usize * groups;
    let input = ramp(count);

    let spirv = match limits.subgroup_size {
        32 => kernels::lane_sum::<F32, 32>(32).expect("built"),
        64 => kernels::lane_sum::<F32, 64>(64).expect("built"),
        other => {
            eprintln!("SKIPPED two-workgroups: no case written for a subgroup of {other}");
            return;
        }
    };

    let output = gpu.run(&spirv, &input, groups as u32).expect("dispatched");

    // One strip, so the layout collapses to the plain global index and the reference is the same
    // grouped sum as ever — over twice as many invocations.
    assert_eq!(output, grouped_sums(count, width));

    // Discriminator: the last subgroup must differ from the first, or the second workgroup read
    // the first one's data.
    assert_ne!(
        output.first(),
        output.last(),
        "both workgroups produced the same total"
    );
}

/// The same reduction over unsigned integers, which is a different instruction end to end.
#[test]
fn an_integer_reduction_uses_the_integer_instruction_and_still_adds_up() {
    let Some(gpu) = device("integer-sum") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED integer-sum: no subgroup arithmetic");
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input: Vec<u32> = (0..count as u32).collect();

    let spirv = match limits.subgroup_size {
        32 => kernels::lane_sum::<U32, 32>(32).expect("built"),
        64 => kernels::lane_sum::<U32, 64>(64).expect("built"),
        other => {
            eprintln!("SKIPPED integer-sum: no case written for a subgroup of {other}");
            return;
        }
    };

    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            (first..(first + width).min(count)).sum::<usize>() as u32
        })
        .collect();

    assert_eq!(output, expected);
}

/// `reduce_max`, whose strip fold has no core opcode and goes through compare-and-select.
#[test]
fn a_maximum_reduction_finds_the_largest_element_in_each_group() {
    let Some(gpu) = device("lane-max") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic || !limits.subgroup_clustered {
        eprintln!("SKIPPED lane-max: the device lacks clustered subgroup arithmetic");
        return;
    }

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let output = gpu
        .run(
            &kernels::lane_max::<F32, 8>(width).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    // A ramp's largest element in each group of eight is the last one.
    let expected: Vec<f32> = (0..count).map(|lane| (lane / 8 * 8 + 7) as f32).collect();

    assert_eq!(output, expected);
}

/// Elementwise work only, which should never reach a subgroup instruction.
#[test]
fn an_elementwise_kernel_computes_per_element_and_crosses_no_lane() {
    let Some(gpu) = device("affine") else { return };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let output = gpu
        .run(
            &kernels::lane_affine::<32>(width).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<f32> = input.iter().map(|value| value * 2.0 + 1.0).collect();
    assert_eq!(output, expected);
}

/// What the mapping still refuses, refused when the module is built rather than at run time.
#[test]
fn the_lane_api_refuses_the_lane_counts_that_have_no_mapping() {
    let Some(gpu) = device("no-mapping") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    assert!(
        kernels::lane_sum::<F32, 12>(width).is_err(),
        "12 lanes neither divide a subgroup of {width} nor are a multiple of it"
    );
    // `MAX_STRIPS` is 8, so the count that overruns it depends on the width: 512 lanes is sixteen
    // strips on a 32-wide subgroup and only eight on a 64-wide one, where it is accepted. 1024 is
    // past the limit on both, which is what makes this a statement about the mapping rather than
    // about one device.
    assert!(
        kernels::lane_sum::<F32, 1024>(width).is_err(),
        "1024 lanes need more elements per lane than a vector holds inline, at either width"
    );
    if width == 32 {
        assert!(
            kernels::lane_sum::<F32, 512>(width).is_err(),
            "512 lanes is sixteen strips on a 32-wide subgroup, which is past MAX_STRIPS"
        );
    } else {
        assert!(
            kernels::lane_sum::<F32, 512>(width).is_ok(),
            "512 lanes is eight strips on a {width}-wide subgroup, which is exactly the limit"
        );
    }
}
