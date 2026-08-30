mod common;

type Ordering = (Comparison, fn(f32, f32) -> bool);
type BitwisePair = (Bitwise, fn(u32, u32) -> u32);

use common::{device, grouped_sums, ramp, ramp_u32, runnable};
use runner::kernels::{self, Bitwise, Comparison, WORKGROUP_SIZE};
use simdr::lanes::{F32, I32, U32};

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

#[test]
fn the_three_bitwise_reductions_fold_the_lanes_they_are_named_over() {
    let Some(gpu) = device("bitwise-reduce") else {
        return;
    };

    let width = gpu.limits().subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;

    // Distinct low bits per lane, so `and` clears where `or` sets and the
    // parity of each column decides `xor` — no two of the three agree.
    let input: Vec<u32> = (0..count as u32).map(|index| (index * 7) | 1).collect();

    let modules = [
        kernels::lane_and_whole::<U32>(gpu.limits().subgroup_size).expect("built"),
        kernels::lane_or_whole::<U32>(gpu.limits().subgroup_size).expect("built"),
        kernels::lane_xor_whole::<U32>(gpu.limits().subgroup_size).expect("built"),
    ];
    let borrowed: Vec<&[u32]> = modules.iter().map(Vec::as_slice).collect();
    if !runnable(&gpu, "bitwise-reduce", &borrowed) {
        return;
    }

    let folded = |combine: fn(u32, u32) -> u32| -> Vec<u32> {
        (0..count)
            .map(|lane| {
                let first = lane / width * width;
                input[first..(first + width).min(count)]
                    .iter()
                    .copied()
                    .reduce(combine)
                    .expect("a subgroup holds at least one lane")
            })
            .collect()
    };

    let outputs: Vec<Vec<u32>> = modules
        .iter()
        .map(|spirv| gpu.run_u32(spirv, &input, 1).expect("dispatched"))
        .collect();

    assert_eq!(outputs[0], folded(|a, b| a & b), "and");
    assert_eq!(outputs[1], folded(|a, b| a | b), "or");
    assert_eq!(outputs[2], folded(|a, b| a ^ b), "xor");

    assert_ne!(outputs[0], outputs[1], "and and or must not agree here");
    assert_ne!(outputs[1], outputs[2], "nor or and xor");
    assert_ne!(outputs[0], outputs[2], "nor and and xor");
}

#[test]
fn a_product_reduction_multiplies_the_lanes_rather_than_adding_them() {
    let Some(gpu) = device("product-reduce") else {
        return;
    };

    let width = gpu.limits().subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;

    // Values a wrapping product stays exact over, and never 1, so a dropped
    // multiply cannot hide in an identity.
    let input: Vec<u32> = (0..count).map(|index| (index % 3 + 2) as u32).collect();

    let spirv = kernels::lane_product_whole::<U32>(gpu.limits().subgroup_size).expect("built");
    if !runnable(&gpu, "product-reduce", &[&spirv]) {
        return;
    }

    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            input[first..(first + width).min(count)]
                .iter()
                .fold(1_u32, |total, value| total.wrapping_mul(*value))
        })
        .collect();

    assert_eq!(output, expected);

    let sums: Vec<u32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            input[first..(first + width).min(count)].iter().sum()
        })
        .collect();
    assert_ne!(output, sums, "a product is not a sum over this input");
}

#[test]
fn the_new_reductions_cluster_and_strip_mine_like_every_other_one() {
    let Some(gpu) = device("reduce-shapes") else {
        return;
    };
    let limits = gpu.limits().clone();

    let width = limits.subgroup_size as usize;
    if width < 4 {
        eprintln!("SKIPPED reduce-shapes: a subgroup of {width} has no cluster of four");
        return;
    }
    let count = WORKGROUP_SIZE as usize;
    let input: Vec<u32> = (0..count as u32)
        .map(|index| index.wrapping_mul(37) | 1)
        .collect();

    let in_clusters = |combine: fn(u32, u32) -> u32, seed: Option<u32>| -> Vec<u32> {
        (0..count)
            .map(|lane| {
                let first = lane / 4 * 4;
                let cluster = input[first..(first + 4).min(count)].iter().copied();
                match seed {
                    Some(start) => cluster.fold(start, combine),
                    None => cluster.reduce(combine).expect("a cluster is not empty"),
                }
            })
            .collect()
    };

    let clustered = [
        (
            "and",
            kernels::lane_and::<U32, 4>(limits.subgroup_size).expect("built"),
            in_clusters(|a, b| a & b, None),
        ),
        (
            "or",
            kernels::lane_or::<U32, 4>(limits.subgroup_size).expect("built"),
            in_clusters(|a, b| a | b, None),
        ),
        (
            "product",
            kernels::lane_product::<U32, 4>(limits.subgroup_size).expect("built"),
            in_clusters(u32::wrapping_mul, Some(1)),
        ),
    ];

    let borrowed: Vec<&[u32]> = clustered
        .iter()
        .map(|(_, spirv, _)| spirv.as_slice())
        .collect();
    if !runnable(&gpu, "reduce-shapes", &borrowed) {
        return;
    }

    for (name, spirv, expected) in &clustered {
        let output = gpu.run_u32(spirv, &input, 1).expect("dispatched");
        assert_eq!(output, *expected, "{name} in clusters of four");
    }

    let strips = 2_usize;
    let stride = count;
    let wide = ramp_u32(count * strips);
    let spirv = match limits.subgroup_size {
        32 => kernels::lane_xor::<U32, 64>(32).expect("built"),
        64 => kernels::lane_xor::<U32, 128>(64).expect("built"),
        other => {
            eprintln!("SKIPPED reduce-shapes: no strip-mined case written for {other}");
            return;
        }
    };

    let output = gpu.run_u32(&spirv, &wide, 1).expect("dispatched");
    let expected: Vec<u32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            (first..first + width)
                .flat_map(|other| (0..strips).map(move |strip| other + strip * stride))
                .fold(0_u32, |total, index| total ^ wide[index])
        })
        .collect();

    assert_eq!(&output[..count], &expected[..count], "xor, strip-mined");

    let one_strip_only = (0..width).fold(0_u32, |total, other| total ^ wide[other]);
    assert_ne!(
        output.first(),
        Some(&one_strip_only),
        "the second strip was not folded in"
    );
}

#[test]
fn every_arm_of_the_tours_comparison_and_bitwise_kernels_is_the_one_it_names() {
    let Some(gpu) = device("tour-arms") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let threshold = 4.0_f32;
    let floats: Vec<f32> = (0..count as u32).map(|index| index as f32).collect();

    let orderings: [Ordering; 6] = [
        (Comparison::Less, |x, t| x < t),
        (Comparison::LessEqual, |x, t| x <= t),
        (Comparison::Greater, |x, t| x > t),
        (Comparison::GreaterEqual, |x, t| x >= t),
        (Comparison::Equal, |x, t| (x - t).abs() < f32::EPSILON),
        (Comparison::NotEqual, |x, t| (x - t).abs() >= f32::EPSILON),
    ];

    let mut seen: Vec<Vec<f32>> = Vec::new();
    for (comparison, holds) in orderings {
        let spirv = kernels::lane_compare(width, threshold, comparison).expect("built");
        let output = gpu.run(&spirv, &floats, 1).expect("dispatched");

        let expected: Vec<f32> = floats
            .iter()
            .map(|value| if holds(*value, threshold) { 1.0 } else { 0.0 })
            .collect();
        assert_eq!(output, expected, "{comparison:?}");
        seen.push(output);
    }

    for (index, answer) in seen.iter().enumerate() {
        for (other, another) in seen.iter().enumerate().skip(index + 1) {
            assert_ne!(
                answer, another,
                "{:?} and {:?} answer alike over this input, so one could stand in for the other",
                orderings[index].0, orderings[other].0
            );
        }
    }

    let mask = 0x5_u32;
    let bits: Vec<u32> = (0..count as u32).collect();
    let bitwise: [BitwisePair; 4] = [
        (Bitwise::And, |x, m| x & m),
        (Bitwise::Or, |x, m| x | m),
        (Bitwise::Xor, |x, m| x ^ m),
        (Bitwise::Not, |x, _| !x),
    ];

    let mut seen: Vec<Vec<u32>> = Vec::new();
    for (operation, apply) in bitwise {
        let spirv = kernels::lane_bitwise_with(width, mask, operation).expect("built");
        let output = gpu.run_u32(&spirv, &bits, 1).expect("dispatched");

        let expected: Vec<u32> = bits.iter().map(|value| apply(*value, mask)).collect();
        assert_eq!(output, expected, "{operation:?}");
        seen.push(output);
    }

    for (index, answer) in seen.iter().enumerate() {
        for another in seen.iter().skip(index + 1) {
            assert_ne!(answer, another, "two bitwise arms answer alike");
        }
    }
}

#[test]
fn the_tours_arithmetic_kernels_each_apply_the_operation_they_name() {
    let Some(gpu) = device("tour-arithmetic") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input: Vec<f32> = (0..count as u32).map(|index| index as f32).collect();

    let difference = gpu
        .run(&kernels::lane_sub(width, 1.0).expect("built"), &input, 1)
        .expect("dispatched");
    let quotient = gpu
        .run(&kernels::lane_div(width, 2.0).expect("built"), &input, 1)
        .expect("dispatched");
    let negated = gpu
        .run(&kernels::lane_neg(width).expect("built"), &input, 1)
        .expect("dispatched");

    assert_eq!(
        difference,
        input.iter().map(|value| value - 1.0).collect::<Vec<f32>>()
    );
    assert_eq!(
        quotient,
        input.iter().map(|value| value / 2.0).collect::<Vec<f32>>()
    );
    assert_eq!(
        negated,
        input.iter().map(|value| -value).collect::<Vec<f32>>()
    );

    assert!(
        negated[1] < 0.0 && difference[0] < 0.0,
        "an input that only ever grew would let a dropped sign through"
    );
}

#[test]
fn saturating_arithmetic_clamps_where_the_wrapping_kind_would_wrap() {
    let Some(gpu) = device("saturating") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;

    let unsigned: Vec<u32> = (0..count as u32).collect();
    let near_the_top = u32::MAX - 10;

    let added = gpu
        .run_u32(
            &kernels::lane_saturating_add_whole::<U32>(width, near_the_top).expect("built"),
            &unsigned,
            1,
        )
        .expect("dispatched");
    assert_eq!(
        added,
        unsigned
            .iter()
            .map(|value| value.saturating_add(near_the_top))
            .collect::<Vec<u32>>()
    );
    assert!(
        added.contains(&u32::MAX) && added.iter().any(|value| *value != u32::MAX),
        "the input has to reach the ceiling and also stop short of it"
    );

    let taken = 20_u32;
    let subtracted = gpu
        .run_u32(
            &kernels::lane_saturating_sub_whole::<U32>(width, taken).expect("built"),
            &unsigned,
            1,
        )
        .expect("dispatched");
    assert_eq!(
        subtracted,
        unsigned
            .iter()
            .map(|value| value.saturating_sub(taken))
            .collect::<Vec<u32>>()
    );
    assert_eq!(subtracted[0], 0, "under the floor is the floor, not a wrap");

    let signed: Vec<i32> = (0..count as i32).map(|index| index - 32).collect();
    let words: Vec<u32> = signed
        .iter()
        .map(|value| u32::from_ne_bytes(value.to_ne_bytes()))
        .collect();

    for rhs in [i32::MAX - 5, i32::MIN + 5] {
        let bits = u32::from_ne_bytes(rhs.to_ne_bytes());

        let added = gpu
            .run_u32(
                &kernels::lane_saturating_add_whole::<I32>(width, bits).expect("built"),
                &words,
                1,
            )
            .expect("dispatched");
        let added: Vec<i32> = added
            .iter()
            .map(|word| i32::from_ne_bytes(word.to_ne_bytes()))
            .collect();
        assert_eq!(
            added,
            signed
                .iter()
                .map(|value| value.saturating_add(rhs))
                .collect::<Vec<i32>>(),
            "signed saturating add against {rhs}"
        );

        let subtracted = gpu
            .run_u32(
                &kernels::lane_saturating_sub_whole::<I32>(width, bits).expect("built"),
                &words,
                1,
            )
            .expect("dispatched");
        let subtracted: Vec<i32> = subtracted
            .iter()
            .map(|word| i32::from_ne_bytes(word.to_ne_bytes()))
            .collect();
        assert_eq!(
            subtracted,
            signed
                .iter()
                .map(|value| value.saturating_sub(rhs))
                .collect::<Vec<i32>>(),
            "signed saturating sub against {rhs}"
        );
    }
}

#[test]
fn a_swizzle_moves_every_lane_to_the_one_its_index_named() {
    let Some(gpu) = device("swizzle") else {
        return;
    };

    let limits = gpu.limits().clone();
    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let reversed = gpu
        .run(
            &kernels::lane_reverse(limits.subgroup_size).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            input[first + (width - 1 - (lane - first))]
        })
        .collect();
    assert_eq!(reversed, expected, "each lane read the one opposite it");
    assert_ne!(
        reversed, input,
        "a reversal that changed nothing is not one"
    );

    // The same permutation two ways: a swizzle carrying an index this test
    // computed, and the fixed `rotate_up` that predates it. They share no code
    // below `Lanes`, so agreeing is evidence rather than a tautology.
    let delta = 3;
    let counted: Vec<u32> = (0..count as u32).collect();
    let by_swizzle = gpu
        .run_u32(
            &kernels::lane_rotate_by_swizzle(limits.subgroup_size, delta).expect("built"),
            &counted,
            1,
        )
        .expect("dispatched");
    let by_rotate = gpu
        .run_u32(
            &kernels::rotate_in_cluster(limits.subgroup_size, limits.subgroup_size, delta)
                .expect("built"),
            &counted,
            1,
        )
        .expect("dispatched");

    assert_eq!(
        by_swizzle, by_rotate,
        "a rotation written as a swizzle is the rotation `rotate_up` already emitted"
    );
    assert_ne!(by_swizzle, counted, "and it moved something");
}

#[test]
fn saturation_is_elementwise_so_it_holds_in_clusters_and_across_strips() {
    let Some(gpu) = device("saturating-shapes") else {
        return;
    };

    let limits = gpu.limits().clone();
    if limits.subgroup_size != 32 {
        eprintln!(
            "SKIPPED saturating-shapes: no case written for a subgroup of {}",
            limits.subgroup_size
        );
        return;
    }

    let count = WORKGROUP_SIZE as usize;
    let rhs = u32::MAX - 10;

    // Nothing here crosses a lane, so the answer is the same elementwise
    // function whatever shape the vector takes -- which is the claim.
    let clustered = kernels::lane_saturating_add::<U32, 4>(32, rhs).expect("built");
    let input: Vec<u32> = (0..count as u32).collect();
    let output = gpu.run_u32(&clustered, &input, 1).expect("dispatched");
    assert_eq!(
        output,
        input
            .iter()
            .map(|value| value.saturating_add(rhs))
            .collect::<Vec<u32>>(),
        "in clusters of four"
    );

    let strip_mined = kernels::lane_saturating_sub::<U32, 64>(32, 20).expect("built");
    let wide: Vec<u32> = (0..count as u32 * 2).collect();
    let output = gpu.run_u32(&strip_mined, &wide, 1).expect("dispatched");
    assert_eq!(
        output,
        wide.iter()
            .map(|value| value.saturating_sub(20))
            .collect::<Vec<u32>>(),
        "across two strips"
    );
    assert!(
        output.contains(&0) && output.iter().any(|value| *value > 0),
        "the input has to reach the floor and also stay above it"
    );
}
