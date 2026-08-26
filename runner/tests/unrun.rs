//! The lane operations that had never been executed.
//!
//! Six of them, found by a coverage sweep rather than by anything failing: `prefix_sum`, `ballot`,
//! `shift_down`, `broadcast`, `all_uniform` and `reduce_min` all had unit tests and no dispatch.
//!
//! A unit test here decodes the emitted module and agrees that the emitter emitted what the test
//! expected. That is a check on one author's understanding against itself. `reduce_min` passed
//! seven of them while folding its strips with a maximum.
//!
//! **And four more, from a sweep that asked the question of the tree rather than of the lane API.**
//! The five arithmetic instructions added on 2026-08-18 and the two rolled loops added on 2026-08-19
//! reached a device nowhere: their only consumers were tests in the emitter, which build a module
//! and hand it to `spirv-val`. Every one of the four tests below was checked by breaking the kernel
//! it covers — an `f_sub` written as an `f_add`, an `i_sub` as an `i_add`, a loop whose every trip
//! reads block zero, a second phi wired to the first — and each one goes red for its own reason.

mod common;

use common::{device, runnable};
use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{LaneError, U32};

/// A device with the subgroup surface these need, or `None` with a reason printed.
fn ready(label: &'static str) -> Option<(Gpu, u32)> {
    let gpu = device(label)?;
    let limits = gpu.limits().clone();

    // **No capability gate here any more, and that is the point.** This asked
    // `subgroup_surface() && subgroup_ballot` — the union of everything *any* kernel in this file
    // reaches — which was itself a correction, because the list had been three of the five. A union
    // over-gates in the silent direction: a device missing one feature skipped every test in the
    // file, including the ones that never touch it.
    //
    // A shared helper cannot know which module its caller is about to build, so the question moved
    // to where the module is. Each test calls `common::runnable`, which reads the requirement out
    // of that module's own `OpCapability` list. What stays here is the width, which no module
    // declares.
    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED {label}: written for a 32-wide subgroup");
        return None;
    }

    Some((gpu, limits.subgroup_size))
}

/// Distinct values, never equal — the property whose absence hid the `reduce_min` bug.
fn ramp(count: usize) -> Vec<u32> {
    (0..count as u32).map(|index| index + 1).collect()
}

#[test]
fn a_prefix_sum_is_inclusive_and_not_exclusive() {
    // The one that matters. An exclusive scan is the same instruction with a different
    // `GroupOperation`, the two differ by exactly one element, and every opcode-counting test
    // passes for either.
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
            // Inclusive: up to *and including* this lane.
            (first..=lane).map(|other| other as u32 + 1).sum()
        })
        .collect();

    assert_eq!(output, expected);

    // And the discriminator, stated separately so a failure says which way it went wrong: the
    // first lane of a subgroup holds its own value under an inclusive scan and zero under an
    // exclusive one.
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
    // **The lane read differs per invocation**, which is the whole of what this checks. A
    // `Simd<u32, 8>` on a 32-wide subgroup is four vectors, and broadcasting position 3 means
    // subgroup lanes 3, 11, 19 and 27 — each to its own seven neighbours.
    //
    // The wrong implementation is the one that treats `source` as a subgroup lane: it agrees here
    // for the first cluster of every subgroup and is wrong in the other three, which is the same
    // shape of failure a clustered scan would have.
    //
    // **Not through `ready`**, which every other test here uses. That helper refuses a device whose
    // subgroup is not 32 lanes, because their expectations are written for one — and this one's is
    // not: clusters repeat every `cluster` lanes whatever the width, so the answer below has no
    // width in it. A test that skipped 64 and 8 would leave the two devices that found this
    // project's last ten bugs looking at nothing.
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

        // Only the lanes with a source are checked. SPIR-V leaves the others undefined, and
        // pinning an undefined value is how a test starts depending on one driver.
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

    // The two must disagree, or a shift of zero would pass both.
    assert_ne!(down, up);
}

#[test]
fn reduce_min_finds_the_smallest_including_when_strips_are_folded() {
    // The operation that had no kernel at all while its strip fold computed a maximum.
    let Some((gpu, width)) = ready("lane-min") else {
        return;
    };

    let count = WORKGROUP_SIZE as usize;
    let lanes = width as usize;

    // Whole-subgroup first: no strip fold, so this is the group instruction alone.
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

    // Strip-mined: two elements per lane, and the fold between them is the part that was wrong.
    // The second strip holds *larger* values throughout, so a fold that kept the maximum returns
    // the second strip's minimum instead of the first's — visibly different.
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

    // Discriminator: the maximum over the same input differs, so this is not passing because both
    // ends happen to agree.
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

    // A threshold the first subgroup's largest exceeds and its smallest does not: `any` says yes,
    // `all` says no. That is the whole difference between the two instructions.
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
    // **The third vote, run.** `all_equal` asks whether the lanes hold the same value, which no
    // predicate can express, and it is the only way to obtain a `Uniform` from a *value* — so the
    // kernel branches on it rather than selecting, and a subgroup that disagrees never reaches the
    // write.
    //
    // Two inputs, because either alone would pass for a broken vote: one where every subgroup
    // agrees, and one where exactly one lane differs. A vote stuck at `true` fails the second, and
    // one stuck at `false` fails the first.
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

    // One lane of the *first* subgroup differs. Every subgroup after it still agrees, which is
    // what separates a per-subgroup answer from a per-dispatch one.
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
    // The comparison the lane API had no spelling for. Run rather than counted: `OpIEqual` and
    // `OpINotEqual` are adjacent numbers, and an opcode-counting test passes for either.
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
    // **The input the refusal existed for.** Two strips, each internally uniform, holding
    // different values: every `AllEqual` says true, and the vector is not all equal. A folded
    // vote answers 1 here; the built one asks the second question — does every strip equal strip 0
    // in my lane — and answers 0.
    //
    // The kernel loads `Simd<u32, 2 × width>`, so lane `l` holds elements `l` and `l + width`.
    let Some(gpu) = device("all-equal-wide") else {
        return;
    };
    let limits = gpu.limits().clone();
    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    // Two strips per lane over one workgroup: the buffer is twice the invocation count, laid out
    // strip after strip, which is the order `Kernel::load` reads.
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

    // Everything the same: both halves of the question say yes.
    // The buffer is twice the invocation count and the kernel writes one slot each, so only the
    // first `count` slots are answers — the rest is the input buffer's own length showing through.
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

    // Each strip uniform, the strips different. Strip 0 is the first `WORKGROUP_SIZE` elements.
    let mut split = vec![1_u32; elements];
    for slot in split.iter_mut().skip(count) {
        *slot = 2;
    }
    assert_eq!(
        answers(&split),
        vec![0; count],
        "every lane agrees within each strip and the strips differ — a folded vote says otherwise"
    );

    // And one lane of the first subgroup differing in strip 1 only, which the first question
    // cannot see either.
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
    // **The operation a cluster's edge was waiting for.** `shift_up` leaves the bottom `delta`
    // lanes undefined and refuses a clustered vector outright, because the lanes it would read
    // belong to the vector next door and the hardware hands them over without a word. A rotate
    // reads only inside its own vector, so it is defined for every lane and allowed for both
    // mappings — and this checks the wrap, which is the half a shift does not have.
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

        // Vector `v` occupies lanes `v * cluster .. (v + 1) * cluster`, and element `i` of it comes
        // from element `i - delta` of the same vector, wrapping. Written as the source index rather
        // than as a rotation of a slice, so the expectation is the addressing rather than a library
        // function that might rotate the other way.
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

/// How many blocks the two rolled-loop kernels below read.
///
/// Four rather than two: a loop that runs once too few or once too many is off by a whole block,
/// and at two trips that is half the answer and hard to tell from a wrong initial value.
const BLOCKS: u32 = 4;

/// `blocks * 64` values where a block's contribution and a lane's are separable.
///
/// Element `block * 64 + lane` is `(block + 1) * 1000 + lane`, so a sum that dropped a block is a
/// round number away from the right one and a sum that read the wrong lane is not. A ramp would
/// have neither property: every block would look like every other, shifted.
fn blocks(count: u32) -> Vec<u32> {
    (0..count)
        .flat_map(|block| (0..WORKGROUP_SIZE).map(move |lane| (block + 1) * 1000 + lane))
        .collect()
}

#[test]
fn the_activation_arithmetic_agrees_with_the_host_bit_for_bit() {
    // `f_sub`, `f_div` and `f_negate` had one consumer between them for a week — a test in the
    // emitter that hands one module to `spirv-val`. That says the words are legal. It cannot say
    // that `OpFNegate` negates, because a wrong opcode number is a *different well-formed
    // instruction* and the validator accepts it.
    let Some(gpu) = device("centre-and-scale") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    // An integer centre and a power-of-two scale, so every step is exact and the comparison can be
    // on bits rather than within a tolerance.
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

    // And the discriminators, so a failure says which of the three went wrong rather than that the
    // vector differs. Each names a lane whose answer only one instruction can produce.
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
    // This crate emits no `OpUMod`. `u_div` and `i_sub` were added for the arithmetic that says
    // which of a batch a lane is working on, and this is that arithmetic run.
    let Some(gpu) = device("remainder") else {
        return;
    };
    let width = gpu.limits().subgroup_size;

    // Seven: not a power of two, so the division cannot be folded into a shift and what runs is
    // `OpUDiv` itself.
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

    // The identity is exact for every input, so the interesting lanes are the ones where the
    // division is not the whole story: a multiple of seven must come back zero, and the value
    // before it must come back six.
    assert_eq!(output.get(21), Some(&0), "21 is three sevens exactly");
    assert_eq!(output.get(20), Some(&6));
}

#[test]
fn a_rolled_loop_reads_a_different_block_each_trip() {
    // `decisions/DR-0010`'s kernel, and the thing that record says is not verified: *"That the
    // emitted loop is valid SPIR-V, on this machine... The validator and a real dispatch are what
    // would settle it."* This is the dispatch.
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

    // One value per invocation, so only the first workgroup's worth of the output is this kernel's
    // — the rest of the buffer is whatever the device's memory held. `Gpu::run` says so.
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

    // The two ways this fails while validating, named apart. A body that re-read block zero every
    // trip returns four times the first block; a counter stepped in the header rather than the
    // continue block skips block zero and reads one past the end.
    let first_block: u32 = input.first().copied().expect("the input is not empty");
    assert_ne!(
        output.first(),
        Some(&(BLOCKS * first_block)),
        "every trip read block zero — the counter phi is not reaching the address"
    );
}

#[test]
fn two_running_totals_come_out_of_one_pass() {
    // `Kernel::repeat_rolled_many`, whose only consumers were its own unit tests. Those assert the
    // decoded module holds two `OpPhi` instructions, which is true of a loop that carries the wrong
    // value on either back edge.
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

    // The kernel keeps a plain total and one weighted by `block + 1` and stores the difference,
    // which is the plain total weighted by `block`.
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

    // A second phi wired to the first would make the difference zero, which is the failure the
    // subtraction exists to expose and the one a single-total kernel could not have.
    assert_ne!(output.first(), Some(&0), "the two totals came out equal");
}

#[test]
fn a_broadcast_of_a_lane_outside_the_vector_is_refused_by_the_name_its_doc_gives() {
    // **The claim that had no producer.** `broadcast_in_cluster` documented `NoSuchForm` for this
    // input and cannot emit one: `Lanes::broadcast` refuses through `Lanes::within_group`, which
    // names the operand and the width it passed. Needs no device — the refusal happens while the
    // module is being built, which is why this sits beside the dispatches rather than among them.
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

    // The other half of the same sentence, which was right and stays checked beside it.
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
