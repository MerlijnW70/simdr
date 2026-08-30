mod common;

use common::{device, grouped_sums, ramp, runnable};
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{F32, U32};

#[test]
fn a_workgroup_reduction_crosses_between_subgroups() {
    let Some(gpu) = device("workgroup-sum") else {
        return;
    };
    let limits = gpu.limits().clone();

    let workgroup = kernels::workgroup_sum::<F32>(limits.subgroup_size).expect("built");
    let per_subgroup = kernels::reduce::lane_sum_whole::<F32>(limits.subgroup_size).expect("built");
    if !runnable(&gpu, "workgroup-sum", &[&workgroup, &per_subgroup]) {
        return;
    }

    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    let whole: f32 = input.iter().sum();

    let output = gpu.run(&workgroup, &input, 1).expect("dispatched");

    assert_eq!(
        output,
        vec![whole; count],
        "every invocation should hold the whole workgroup's total"
    );

    let per_subgroup = gpu.run(&per_subgroup, &input, 1).expect("dispatched");
    if limits.subgroup_size < WORKGROUP_SIZE {
        assert_ne!(
            output, per_subgroup,
            "the workgroup sum equals the subgroup sum, so nothing crossed between them"
        );
    } else {
        assert_eq!(
            output, per_subgroup,
            "one subgroup fills this workgroup, so the two reductions cover the same lanes"
        );
    }
    assert_eq!(output.first(), output.last(), "and it is uniform");
}

#[test]
fn a_workgroup_reduction_is_exact_over_integers() {
    let Some(gpu) = device("workgroup-sum-u32") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::workgroup_sum::<U32>(limits.subgroup_size).expect("built");
    if !runnable(&gpu, "workgroup-sum-u32", &[&spirv]) {
        return;
    }

    let count = WORKGROUP_SIZE as usize;
    let input: Vec<u32> = (0..count as u32).map(|index| index * 3 + 1).collect();
    let whole: u32 = input.iter().sum();

    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

    assert_eq!(output, vec![whole; count]);
}

#[test]
fn the_lane_api_reduces_over_exactly_the_lanes_its_width_names() {
    let Some(gpu) = device("lane-sum") else {
        return;
    };
    let limits = gpu.limits().clone();

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let mut built: Vec<(u32, Vec<u32>)> = Vec::new();
    for size in [2_u32, 4, 8, width] {
        if size > width || built.iter().any(|(had, _)| *had == size) {
            continue;
        }

        let spirv = match size {
            2 => kernels::lane_sum::<F32, 2>(width),
            4 => kernels::lane_sum::<F32, 4>(width),
            8 => kernels::lane_sum::<F32, 8>(width),
            _ => kernels::lane_sum_whole::<F32>(width),
        }
        .expect("built");
        built.push((size, spirv));
    }

    let modules: Vec<&[u32]> = built.iter().map(|(_, spirv)| spirv.as_slice()).collect();
    if !runnable(&gpu, "lane-sum", &modules) {
        return;
    }

    let mut runs: Vec<(u32, Vec<f32>)> = Vec::new();
    for (size, spirv) in &built {
        let size = *size;
        let output = gpu.run(spirv, &input, 1).expect("dispatched");
        assert_eq!(
            output,
            grouped_sums(count, size as usize),
            "clusters of {size} on a subgroup of {width}"
        );
        runs.push((size, output));
    }

    assert!(
        runs.len() >= 2,
        "only one cluster size was reachable at width {width}, so nothing below discriminates"
    );

    for (index, (size, output)) in runs.iter().enumerate() {
        for (other, theirs) in runs.iter().skip(index + 1) {
            assert_ne!(
                output.first(),
                theirs.first(),
                "clusters of {size} and of {other} gave the same total"
            );
        }
    }
}

#[test]
fn a_strip_mined_vector_reduces_over_more_elements_than_there_are_lanes() {
    let Some(gpu) = device("strip-mined") else {
        return;
    };
    let limits = gpu.limits().clone();

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
    if !runnable(&gpu, "strip-mined", &[&spirv]) {
        return;
    }

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            (first..first + width)
                .flat_map(|other| (0..strips).map(move |strip| (other + strip * stride) as f32))
                .sum()
        })
        .collect();

    assert_eq!(&output[..count], &expected[..count]);

    let one_strip_only: f32 = (0..width).map(|other| other as f32).sum();
    assert_ne!(
        output.first(),
        Some(&one_strip_only),
        "the second strip was not folded in"
    );
}

#[test]
fn a_second_workgroup_reads_its_own_run_rather_than_the_first_ones() {
    let Some(gpu) = device("two-workgroups") else {
        return;
    };
    let limits = gpu.limits().clone();

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
    if !runnable(&gpu, "two-workgroups", &[&spirv]) {
        return;
    }

    let output = gpu.run(&spirv, &input, groups as u32).expect("dispatched");

    assert_eq!(output, grouped_sums(count, width));

    assert_ne!(
        output.first(),
        output.last(),
        "both workgroups produced the same total"
    );
}

#[test]
fn an_integer_reduction_uses_the_integer_instruction_and_still_adds_up() {
    let Some(gpu) = device("integer-sum") else {
        return;
    };
    let limits = gpu.limits().clone();

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
    if !runnable(&gpu, "integer-sum", &[&spirv]) {
        return;
    }

    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            (first..(first + width).min(count)).sum::<usize>() as u32
        })
        .collect();

    assert_eq!(output, expected);
}

#[test]
fn a_maximum_reduction_finds_the_largest_element_in_each_group() {
    let Some(gpu) = device("lane-max") else {
        return;
    };
    let limits = gpu.limits().clone();

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let clusters_of_four = kernels::lane_max::<F32, 4>(width).expect("built");
    if !runnable(&gpu, "lane-max", &[&clusters_of_four]) {
        return;
    }

    let four = gpu.run(&clusters_of_four, &input, 1).expect("dispatched");
    let expected: Vec<f32> = (0..count).map(|lane| (lane / 4 * 4 + 3) as f32).collect();
    assert_eq!(four, expected, "clusters of four");

    if width >= 8 {
        let eight = gpu
            .run(
                &kernels::lane_max::<F32, 8>(width).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

        let expected: Vec<f32> = (0..count).map(|lane| (lane / 8 * 8 + 7) as f32).collect();
        assert_eq!(eight, expected, "clusters of eight");
    }
}

#[test]
fn an_elementwise_kernel_computes_per_element_and_crosses_no_lane() {
    let Some(gpu) = device("affine") else { return };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let output = gpu
        .run(
            &kernels::lane_affine_whole(width).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<f32> = input.iter().map(|value| value * 2.0 + 1.0).collect();
    assert_eq!(output, expected);
}

#[test]
fn the_lane_api_refuses_the_lane_counts_that_have_no_mapping() {
    let Some(gpu) = device("no-mapping") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    assert!(
        kernels::lane_sum::<F32, 6>(width).is_err(),
        "6 lanes neither divide a subgroup of {width} nor are a multiple of it"
    );
    let strips = 512 / width.max(1);
    assert_eq!(
        kernels::lane_sum::<F32, 512>(width).is_ok(),
        strips as usize <= simdr::lanes::MAX_STRIPS,
        "512 lanes is {strips} strips on a {width}-wide subgroup, and MAX_STRIPS is {}",
        simdr::lanes::MAX_STRIPS
    );

    assert!(
        kernels::lane_sum::<F32, 4096>(width).is_err(),
        "4096 lanes need more elements per lane than a vector holds inline, at any width"
    );
}

#[test]
fn subtraction_division_and_negation_compute_per_element_on_the_device() {
    let Some(gpu) = device("arithmetic") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let output = gpu
        .run(&kernels::lane_arithmetic(width).expect("built"), &input, 1)
        .expect("dispatched");

    let expected: Vec<f32> = input.iter().map(|value| -((value - 1.0) / 2.0)).collect();
    assert_eq!(
        output, expected,
        "the three operations run in the order they were written"
    );
    assert!(
        output.iter().any(|value| *value > 0.0),
        "an input that straddles one leaves both signs behind, so a dropped negation shows"
    );
}

#[test]
fn each_of_the_six_comparisons_holds_exactly_where_it_should() {
    let Some(gpu) = device("ordering") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    assert!(
        input.contains(&kernels::ORDERING_THRESHOLD),
        "the equal-to case has to be reached or two of the six go untested"
    );

    let output = gpu
        .run(&kernels::lane_ordering(width).expect("built"), &input, 1)
        .expect("dispatched");

    let expected: Vec<f32> = input
        .iter()
        .map(|value| {
            let threshold = kernels::ORDERING_THRESHOLD;
            let bits = [
                (*value < threshold, 1.0),
                (*value <= threshold, 2.0),
                (*value > threshold, 4.0),
                (*value >= threshold, 8.0),
                ((*value - threshold).abs() < f32::EPSILON, 16.0),
                ((*value - threshold).abs() >= f32::EPSILON, 32.0),
            ];
            bits.iter()
                .filter(|(held, _)| *held)
                .map(|(_, weight)| weight)
                .sum()
        })
        .collect();

    assert_eq!(
        output, expected,
        "each comparison carries its own bit, so a wrong one names itself"
    );
    assert_eq!(output[0], 35.0, "below: less, less-or-equal, not-equal");
    assert_eq!(
        output[4], 26.0,
        "at: less-or-equal, greater-or-equal, equal"
    );
    assert_eq!(
        output[5], 44.0,
        "above: greater, greater-or-equal, not-equal"
    );
}

#[test]
fn the_two_integer_families_divide_with_their_own_instruction() {
    let Some(gpu) = device("division") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let signed: Vec<i32> = (0..count as i32).map(|index| index - 32).collect();
    let words: Vec<u32> = signed
        .iter()
        .map(|value| u32::from_ne_bytes(value.to_ne_bytes()))
        .collect();

    let signed_out = gpu
        .run_u32(
            &kernels::lane_divide_signed(width).expect("built"),
            &words,
            1,
        )
        .expect("dispatched");
    let signed_out: Vec<i32> = signed_out
        .iter()
        .map(|word| i32::from_ne_bytes(word.to_ne_bytes()))
        .collect();

    let expected: Vec<i32> = signed.iter().map(|value| -(value / 2)).collect();
    assert_eq!(
        signed_out, expected,
        "a signed division truncates toward zero and keeps the sign — OpUDiv would not"
    );
    assert!(
        signed_out.iter().any(|value| *value < 0),
        "the inputs have to reach past zero or OpUDiv would pass this too"
    );

    let unsigned_out = gpu
        .run_u32(
            &kernels::lane_divide_unsigned(width).expect("built"),
            &words,
            1,
        )
        .expect("dispatched");
    let expected: Vec<u32> = words.iter().map(|word| word / 2).collect();
    assert_eq!(
        unsigned_out, expected,
        "the same bits read as unsigned halve as unsigned"
    );
}

#[test]
fn the_four_bitwise_operations_each_produce_the_bits_they_are_named_for() {
    let Some(gpu) = device("bitwise") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input: Vec<u32> = (0..count as u32).collect();

    let expected: Vec<u32> = input
        .iter()
        .map(|value| {
            [
                (value & kernels::BITWISE_AND_MASK, 1_u32),
                (value | kernels::BITWISE_OR_MASK, 3),
                (value ^ kernels::BITWISE_XOR_MASK, 5),
                (!value, 7),
            ]
            .iter()
            .fold(0_u32, |total, (term, weight)| {
                total.wrapping_add(term.wrapping_mul(*weight))
            })
        })
        .collect();

    assert!(
        expected.windows(2).any(|pair| pair[0] != pair[1]),
        "an expectation that does not vary with its input is one the identity          `(x | m) == (x & m) + (x ^ m)` has collapsed, and it would pass with two          of the four traded"
    );

    let output = gpu
        .run_u32(&kernels::lane_bitwise(width).expect("built"), &input, 1)
        .expect("dispatched");

    assert_eq!(output, expected);
}

#[test]
fn a_complement_on_a_signed_lane_is_ones_complement_and_not_a_negation() {
    let Some(gpu) = device("complement") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let signed: Vec<i32> = (0..count as i32).map(|index| index - 32).collect();
    let words: Vec<u32> = signed
        .iter()
        .map(|value| u32::from_ne_bytes(value.to_ne_bytes()))
        .collect();

    let output = gpu
        .run_u32(
            &kernels::lane_complement_signed(width).expect("built"),
            &words,
            1,
        )
        .expect("dispatched");
    let output: Vec<i32> = output
        .iter()
        .map(|word| i32::from_ne_bytes(word.to_ne_bytes()))
        .collect();

    let expected: Vec<i32> = signed.iter().map(|value| !value).collect();
    assert_eq!(output, expected);

    for (complemented, value) in output.iter().zip(&signed) {
        assert_eq!(
            *complemented,
            -value - 1,
            "the complement and the negation differ by one, and this is the complement"
        );
    }
}
