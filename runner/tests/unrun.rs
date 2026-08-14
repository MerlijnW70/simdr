//! The lane operations that had never been executed.
//!
//! Six of them, found by a coverage sweep rather than by anything failing: `prefix_sum`, `ballot`,
//! `shift_down`, `broadcast`, `all_uniform` and `reduce_min` all had unit tests and no dispatch.
//!
//! A unit test here decodes the emitted module and agrees that the emitter emitted what the test
//! expected. That is a check on one author's understanding against itself. `reduce_min` passed
//! seven of them while folding its strips with a maximum.

mod common;

use common::device;
use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::U32;

/// A device with the subgroup surface these need, or `None` with a reason printed.
fn ready(label: &'static str) -> Option<(Gpu, u32)> {
    let gpu = device(label)?;
    let limits = gpu.limits().clone();

    // Every feature the kernels in this file reach for, and it used to be three of the five: the
    // shifts declare `GroupNonUniformShuffleRelative` and the votes declare `GroupNonUniformVote`,
    // and neither was named here. Both are offered by every device that offers the other three, so
    // the gate was right by luck — `Limits::subgroup_surface` is the same list written once.
    if !limits.subgroup_surface() || !limits.subgroup_ballot {
        eprintln!("SKIPPED {label}: the device lacks part of the subgroup surface");
        return None;
    }
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

    let output = gpu
        .run_u32(
            &kernels::prefix_sum::<U32>(width).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

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
        let output = gpu
            .run_u32(
                &kernels::broadcast::<U32>(width, source).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

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
    if !limits.subgroup_shuffle {
        eprintln!("SKIPPED broadcast-cluster: no subgroup shuffle");
        return;
    }
    let width = limits.subgroup_size;

    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    for (cluster, source) in [(1_u32, 0_u32), (2, 1), (4, 0), (4, 3), (8, 3), (16, 9)] {
        if cluster > width {
            continue;
        }

        let output = gpu
            .run_u32(
                &kernels::broadcast_in_cluster::<U32>(width, cluster, source).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

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

    let down = gpu
        .run_u32(
            &kernels::shift_down::<U32>(width, delta as u32).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");
    let up = gpu
        .run_u32(
            &kernels::shift_up::<U32>(width, delta as u32).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

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
    let plain = gpu
        .run_u32(
            &kernels::lane_min::<U32, 32>(width).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");

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

    let all = gpu
        .run_u32(
            &kernels::all_above(width, threshold).expect("built"),
            &input,
            1,
        )
        .expect("dispatched");
    let any = gpu
        .run(
            &kernels::any_above(width, threshold as f32).expect("built"),
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
        let output = gpu
            .run_u32(
                &kernels::ballot_above(width, threshold).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

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
    if !limits.subgroup_vote {
        eprintln!("SKIPPED all-equal: no subgroup vote");
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let spirv = kernels::subgroup_agrees(limits.subgroup_size).expect("built");

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
    if !limits.subgroup_vote {
        eprintln!("SKIPPED all-equal-wide: no subgroup vote");
        return;
    }

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
