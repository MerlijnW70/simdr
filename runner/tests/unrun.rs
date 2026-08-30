mod common;

use common::{device, runnable};
use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{LaneError, U32};

fn ready(label: &'static str) -> Option<(Gpu, u32)> {
    let gpu = device(label)?;
    let limits = gpu.limits().clone();

    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED {label}: written for a 32-wide subgroup");
        return None;
    }

    Some((gpu, limits.subgroup_size))
}

fn ramp(count: usize) -> Vec<u32> {
    (0..count as u32).map(|index| index + 1).collect()
}

#[test]
fn a_prefix_sum_is_inclusive_and_not_exclusive() {
    let Some((gpu, width)) = ready("prefix-sum") else {
        return;
    };

    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let spirv = kernels::prefix_sum::<U32>(width).expect("built");
    if !runnable(&gpu, "prefix-sum", &[&spirv]) {
        return;
    }

    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

    let lanes = width as usize;
    let expected: Vec<u32> = (0..count)
        .map(|lane| {
            let first = lane / lanes * lanes;
            (first..=lane).map(|other| other as u32 + 1).sum()
        })
        .collect();

    assert_eq!(output, expected);

    assert_eq!(output.first(), Some(&1), "lane 0 excluded itself");
    assert_eq!(output.get(lanes), Some(&(lanes as u32 + 1)));
}

#[test]
fn a_broadcast_hands_every_lane_the_source_lanes_value() {
    let Some((gpu, width)) = ready("broadcast") else {
        return;
    };

    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    let lanes = width as usize;

    for source in [0_u32, 1, 7, 31] {
        let spirv = kernels::broadcast::<U32>(width, source).expect("built");
        if !runnable(&gpu, "broadcast", &[&spirv]) {
            return;
        }

        let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

        let expected: Vec<u32> = (0..count)
            .map(|lane| (lane / lanes * lanes + source as usize) as u32 + 1)
            .collect();

        assert_eq!(output, expected, "broadcasting lane {source}");
    }
}

#[test]
fn a_clustered_broadcast_hands_each_vector_its_own_source_lane() {
    let Some(gpu) = device("broadcast-cluster") else {
        return;
    };
    let limits = gpu.limits().clone();
    let width = limits.subgroup_size;

    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    for (cluster, source) in [(1_u32, 0_u32), (2, 1), (4, 0), (4, 3), (8, 3), (16, 9)] {
        if cluster > width {
            continue;
        }

        let spirv = kernels::broadcast_in_cluster::<U32>(width, cluster, source).expect("built");
        if !runnable(&gpu, "broadcast-cluster", &[&spirv]) {
            return;
        }

        let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

        let size = cluster as usize;
        let expected: Vec<u32> = (0..count)
            .map(|lane| (lane / size * size + source as usize) as u32 + 1)
            .collect();

        assert_eq!(
            output, expected,
            "broadcasting position {source} of every {cluster}-lane vector"
        );
    }
}

#[test]
fn a_shift_moves_values_by_the_delta_where_the_source_lane_exists() {
    let Some((gpu, width)) = ready("shift") else {
        return;
    };

    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    let lanes = width as usize;
    let delta = 4_usize;

    let down_spirv = kernels::shift_down::<U32>(width, delta as u32).expect("built");
    let up_spirv = kernels::shift_up::<U32>(width, delta as u32).expect("built");
    if !runnable(&gpu, "shift", &[&down_spirv, &up_spirv]) {
        return;
    }

    let down = gpu.run_u32(&down_spirv, &input, 1).expect("dispatched");
    let up = gpu.run_u32(&up_spirv, &input, 1).expect("dispatched");

    for lane in 0..count {
        let within = lane % lanes;

        if within + delta < lanes {
            assert_eq!(
                down.get(lane).copied(),
                input.get(lane + delta).copied(),
                "shift_down at lane {lane}"
            );
        }
        if within >= delta {
            assert_eq!(
                up.get(lane).copied(),
                input.get(lane - delta).copied(),
                "shift_up at lane {lane}"
            );
        }
    }

    assert_ne!(down, up);
}

#[test]
fn reduce_min_finds_the_smallest_including_when_strips_are_folded() {
    let Some((gpu, width)) = ready("lane-min") else {
        return;
    };

    let count = WORKGROUP_SIZE as usize;
    let lanes = width as usize;

    let input = ramp(count);
    let spirv = kernels::lane_min::<U32, 32>(width).expect("built");
    if !runnable(&gpu, "lane-min", &[&spirv]) {
        return;
    }

    let plain = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u32> = (0..count)
        .map(|lane| (lane / lanes * lanes) as u32 + 1)
        .collect();
    assert_eq!(plain, expected);

    let wide: Vec<u32> = (0..count * 2).map(|index| index as u32 + 1).collect();
    let strip_mined = gpu
        .run_u32(
            &kernels::lane_min::<U32, 64>(width).expect("built"),
            &wide,
            1,
        )
        .expect("dispatched");

    let expected: Vec<u32> = (0..count)
        .map(|lane| (lane / lanes * lanes) as u32 + 1)
        .collect();
    assert_eq!(
        strip_mined.get(..count),
        Some(expected.as_slice()),
        "the strip fold kept the wrong end"
    );

    let largest = gpu
        .run_u32(
            &kernels::lane_max::<U32, 64>(width).expect("built"),
            &wide,
            1,
        )
        .expect("dispatched");
    assert_ne!(strip_mined, largest);
}

#[test]
fn the_two_votes_answer_differently_on_the_same_input() {
    let Some((gpu, width)) = ready("votes") else {
        return;
    };

    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    let lanes = width as usize;

    let threshold = 16_u32;

    let all_spirv = kernels::all_above(width, threshold).expect("built");
    let any_spirv = kernels::any_above(width, threshold as f32).expect("built");
    if !runnable(&gpu, "votes", &[&all_spirv, &any_spirv]) {
        return;
    }

    let all = gpu.run_u32(&all_spirv, &input, 1).expect("dispatched");
    let any = gpu
        .run(
            &any_spirv,
            &input.iter().map(|&value| value as f32).collect::<Vec<_>>(),
            1,
        )
        .expect("dispatched");

    let expected_all: Vec<u32> = (0..count)
        .map(|lane| u32::from((lane / lanes * lanes) as u32 + 1 > threshold))
        .collect();
    assert_eq!(all, expected_all);

    assert_eq!(
        all.first(),
        Some(&0),
        "not every lane of subgroup 0 is over"
    );
    assert_eq!(any.first(), Some(&1.0), "but some lane is");
    assert_eq!(all.get(lanes), Some(&1), "and every lane of subgroup 1 is");
}

#[test]
fn a_ballot_sets_one_bit_per_qualifying_lane() {
    let Some((gpu, width)) = ready("ballot") else {
        return;
    };

    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    let lanes = width as usize;

    for threshold in [0_u32, 16, 40, 1_000] {
        let spirv = kernels::ballot_above(width, threshold).expect("built");
        if !runnable(&gpu, "ballot", &[&spirv]) {
            return;
        }

        let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

        let expected: Vec<u32> = (0..count)
            .map(|lane| {
                let first = lane / lanes * lanes;
                (0..lanes).fold(0_u32, |mask, within| {
                    if (first + within) as u32 + 1 > threshold {
                        mask | (1 << within)
                    } else {
                        mask
                    }
                })
            })
            .collect();

        assert_eq!(output, expected, "at threshold {threshold}");
    }
}

#[test]
fn a_vote_on_a_value_tells_an_agreeing_subgroup_from_a_divergent_one() {
    let Some(gpu) = device("all-equal") else {
        return;
    };
    let limits = gpu.limits().clone();
    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let spirv = kernels::subgroup_agrees(limits.subgroup_size).expect("built");
    if !runnable(&gpu, "all-equal", &[&spirv]) {
        return;
    }

    let agreeing: Vec<u32> = vec![7; count];
    let agreed = gpu.run_u32(&spirv, &agreeing, 1).expect("dispatched");
    assert_eq!(
        agreed,
        vec![1; count],
        "every subgroup held one value and the vote said otherwise"
    );

    let mut divergent = vec![7_u32; count];
    if let Some(odd) = divergent.get_mut(1) {
        *odd = 8;
    }
    let split = gpu.run_u32(&spirv, &divergent, 1).expect("dispatched");

    let expected: Vec<u32> = (0..count).map(|lane| u32::from(lane >= width)).collect();
    assert_eq!(
        split, expected,
        "the vote answered for the dispatch rather than for each subgroup"
    );
}

#[test]
fn an_elementwise_equality_answers_per_element_and_not_per_lane() {
    let Some(gpu) = device("equals") else {
        return;
    };
    let limits = gpu.limits().clone();

    let count = WORKGROUP_SIZE as usize;
    let input: Vec<u32> = (0..count as u32).map(|index| index % 4).collect();

    for wanted in [0_u32, 3, 9] {
        let output = gpu
            .run_u32(
                &kernels::equals(limits.subgroup_size, wanted).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

        let expected: Vec<u32> = input
            .iter()
            .map(|value| u32::from(*value == wanted))
            .collect();
        assert_eq!(output, expected, "comparing against {wanted}");
    }
}

#[test]
fn a_strip_mined_vote_on_a_value_sees_strips_that_differ_from_each_other() {
    let Some(gpu) = device("all-equal-wide") else {
        return;
    };
    let limits = gpu.limits().clone();
    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let elements = count * 2;

    let spirv = match limits.subgroup_size {
        4 => kernels::subgroup_agrees_wide::<8>(4),
        8 => kernels::subgroup_agrees_wide::<16>(8),
        16 => kernels::subgroup_agrees_wide::<32>(16),
        32 => kernels::subgroup_agrees_wide::<64>(32),
        64 => kernels::subgroup_agrees_wide::<128>(64),
        other => {
            eprintln!("SKIPPED all-equal-wide: no lane count listed for a width of {other}");
            return;
        }
    }
    .expect("built");
    if !runnable(&gpu, "all-equal-wide", &[&spirv]) {
        return;
    }

    let answers = |input: &[u32]| {
        gpu.run_u32(&spirv, input, 1)
            .expect("dispatched")
            .get(..count)
            .map(<[u32]>::to_vec)
            .expect("one answer per invocation")
    };

    let agreeing = vec![7_u32; elements];
    assert_eq!(
        answers(&agreeing),
        vec![1; count],
        "a vector of one value everywhere is all equal"
    );

    let mut split = vec![1_u32; elements];
    for slot in split.iter_mut().skip(count) {
        *slot = 2;
    }
    assert_eq!(
        answers(&split),
        vec![0; count],
        "every lane agrees within each strip and the strips differ — a folded vote says otherwise"
    );

    let mut odd = vec![5_u32; elements];
    if let Some(slot) = odd.get_mut(count + 1) {
        *slot = 6;
    }
    let expected: Vec<u32> = (0..count).map(|lane| u32::from(lane >= width)).collect();
    assert_eq!(
        answers(&odd),
        expected,
        "one element of one lane's second strip differs, in the first subgroup only"
    );
}

#[test]
fn a_rotate_wraps_inside_its_own_vector_where_a_shift_would_not() {
    let Some(gpu) = device("rotate") else {
        return;
    };
    let limits = gpu.limits().clone();

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    for (cluster, delta) in [
        (2_u32, 1_u32),
        (4, 1),
        (4, 3),
        (8, 5),
        (16, 4),
        (32, 1),
        (64, 7),
    ] {
        if cluster > width {
            continue;
        }

        let output = gpu
            .run_u32(
                &kernels::rotate_in_cluster(width, cluster, delta).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

        let size = cluster as usize;
        let expected: Vec<u32> = (0..count)
            .map(|lane| {
                let base = lane / size * size;
                let within = (lane + size - delta as usize % size) % size;
                input[base + within]
            })
            .collect();

        assert_eq!(
            output, expected,
            "a rotate of {delta} inside every {cluster}-lane vector"
        );
    }
}

const BLOCKS: u32 = 4;

fn blocks(count: u32) -> Vec<u32> {
    (0..count)
        .flat_map(|block| (0..WORKGROUP_SIZE).map(move |lane| (block + 1) * 1000 + lane))
        .collect()
}

#[test]
fn the_activation_arithmetic_agrees_with_the_host_bit_for_bit() {
    let Some(gpu) = device("centre-and-scale") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    let centre = 8.0_f32;
    let scale = 4.0_f32;

    let spirv = kernels::centre_and_scale(width, centre, scale).expect("built");
    if !runnable(&gpu, "centre-and-scale", &[&spirv]) {
        return;
    }

    let input: Vec<f32> = (0..WORKGROUP_SIZE).map(|index| index as f32).collect();
    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<f32> = input
        .iter()
        .map(|value| -((value - centre) / scale))
        .collect();

    let bits = |values: &[f32]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<u32>>()
    };
    assert_eq!(
        bits(&output),
        bits(&expected),
        "the three instructions do not compose into the expression they were added for"
    );

    assert_eq!(output.first(), Some(&2.0), "in[0] = 0: -((0 - 8) / 4) = 2");
    assert_eq!(
        output.get(8),
        Some(&-0.0),
        "in[8] is the centre, so the subtraction is zero and the negation is what makes it -0.0"
    );
    assert_eq!(output.get(12), Some(&-1.0), "in[12]: -((12 - 8) / 4) = -1");
}

#[test]
fn a_remainder_written_as_divide_multiply_subtract_is_a_remainder() {
    let Some(gpu) = device("remainder") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    let divisor = 7;

    let spirv = kernels::remainder(width, divisor).expect("built");
    if !runnable(&gpu, "remainder", &[&spirv]) {
        return;
    }

    let input: Vec<u32> = (0..WORKGROUP_SIZE).collect();
    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<u32> = input.iter().map(|value| value % divisor).collect();
    assert_eq!(
        output, expected,
        "x - (x / d) * d is not x % d on this device"
    );

    assert_eq!(output.get(21), Some(&0), "21 is three sevens exactly");
    assert_eq!(output.get(20), Some(&6));
}

#[test]
fn a_rolled_loop_reads_a_different_block_each_trip() {
    let Some(gpu) = device("rolled-block-sum") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    let spirv = kernels::rolled_block_sum(width, BLOCKS).expect("built");
    if !runnable(&gpu, "rolled-block-sum", &[&spirv]) {
        return;
    }

    let input = blocks(BLOCKS);
    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

    let lanes = WORKGROUP_SIZE as usize;
    let expected: Vec<u32> = (0..WORKGROUP_SIZE)
        .map(|lane| {
            (0..BLOCKS)
                .map(|block| input[(block * WORKGROUP_SIZE + lane) as usize])
                .sum()
        })
        .collect();

    assert_eq!(
        output.get(..lanes),
        Some(expected.as_slice()),
        "the loop did not read a different block on each trip"
    );

    let first_block: u32 = input.first().copied().expect("the input is not empty");
    assert_ne!(
        output.first(),
        Some(&(BLOCKS * first_block)),
        "every trip read block zero — the counter phi is not reaching the address"
    );
}

#[test]
fn two_running_totals_come_out_of_one_pass() {
    let Some(gpu) = device("rolled-weighted-totals") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    let spirv = kernels::rolled_weighted_totals(width, BLOCKS).expect("built");
    if !runnable(&gpu, "rolled-weighted-totals", &[&spirv]) {
        return;
    }

    let input = blocks(BLOCKS);
    let output = gpu.run_u32(&spirv, &input, 1).expect("dispatched");

    let lanes = WORKGROUP_SIZE as usize;
    let expected: Vec<u32> = (0..WORKGROUP_SIZE)
        .map(|lane| {
            (0..BLOCKS)
                .map(|block| block * input[(block * WORKGROUP_SIZE + lane) as usize])
                .sum()
        })
        .collect();

    assert_eq!(
        output.get(..lanes),
        Some(expected.as_slice()),
        "the two carried totals did not both survive the loop"
    );

    assert_ne!(output.first(), Some(&0), "the two totals came out equal");
}

#[test]
fn a_broadcast_of_a_lane_outside_the_vector_is_refused_by_the_name_its_doc_gives() {
    let outside = kernels::broadcast_in_cluster::<U32>(32, 8, 9);

    assert!(
        matches!(
            outside,
            Err(LaneError::LaneOutOfRange {
                operand: 9,
                lanes: 8,
                ..
            })
        ),
        "a source outside the vector: {outside:?}"
    );

    let unmappable = kernels::broadcast_in_cluster::<U32>(32, 3, 0);
    assert!(
        matches!(
            unmappable,
            Err(LaneError::NoMapping {
                lanes: 3,
                width: 32
            })
        ),
        "a cluster that is not a power of two: {unmappable:?}"
    );
}
