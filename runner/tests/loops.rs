//! Loops, values that survive a merge, and control flow that nests — on a real device.
//!
//! Split from `control.rs`, which keeps the plain uniform branches. What is here is everything
//! with a phi in it: the back edge of a rolled loop, a value carried out of a selection, and the
//! two ways those can be nested inside each other.
//!
//! Every one of these is a shape where a wrong answer *validates*. An `OpPhi` naming the wrong
//! predecessor satisfies `spirv-val` and then reads a value from an edge that never carried it,
//! so running them is the only check there is.

mod common;

use common::{device, ramp, runnable};
use runner::kernels::{self, WORKGROUP_SIZE};

/// An unrolled loop, checked against the built-in reduction it reimplements.
///
/// A butterfly tree of `log2(width)` steps gives every lane the subgroup's total — the same
/// answer `reduce_sum` produces, by a different route. If [`simdr::lanes::Lanes::repeat`] threads
/// its carried value wrongly, the two disagree.
#[test]
fn an_unrolled_loop_reimplements_the_subgroup_sum_exactly() {
    let Some(gpu) = device("butterfly-tree") else {
        return;
    };
    let limits = gpu.limits().clone();

    // "Part of the subgroup surface" is what a hand-picked gate has to say when it is guessing.
    // The modules know exactly which part.
    let tree_spirv = kernels::butterfly_tree_sum(limits.subgroup_size).expect("built");
    let builtin_spirv =
        kernels::lane_sum_whole::<simdr::lanes::F32>(limits.subgroup_size).expect("built");
    if !runnable(&gpu, "butterfly-tree", &[&tree_spirv, &builtin_spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let tree = gpu.run(&tree_spirv, &input, 1).expect("dispatched");

    let expected = common::grouped_sums(count, width);
    assert_eq!(tree, expected, "the tree and the reference disagree");

    // And against the built-in, which is the stronger comparison: two implementations of the same
    // operation, neither of them the reference.
    // The *whole* subgroup, not a 32-lane vector: the tree above folds `log2(width)` times, so on a
    // 64-wide device it reduces 64 lanes and a `lane_sum::<_, 32>` would reduce a cluster of 32.
    // The two agreed on every device until there was a second one.
    let builtin = gpu.run(&builtin_spirv, &input, 1).expect("dispatched");
    assert_eq!(tree, builtin);
}

/// The same tree inside a cluster, against the clustered reduce it reimplements.
///
/// **The kernel that could not be written until this week.** `Lanes::butterfly` refused every
/// clustered vector, so the mapping that exists to run four small vectors at once could be reduced
/// by the hardware and not swizzled at all. A mask below the cluster's width cannot leave it — the
/// clusters are aligned runs of a power-of-two size — and this is what says so on a device rather
/// than in a comment.
///
/// Two implementations, neither of them the reference: one `ClusteredReduce` against `log2(cluster)`
/// shuffles and adds. The failure it guards against is a tree that folds `log2(width)` times
/// instead, which returns the *subgroup's* total in every lane and agrees with the reference in the
/// one case where a cluster is the subgroup.
#[test]
fn a_butterfly_tree_inside_a_cluster_agrees_with_the_clustered_reduce() {
    use simdr::lanes::{F32, LaneError};

    let Some(gpu) = device("butterfly-cluster") else {
        return;
    };
    let limits = gpu.limits().clone();

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    // The built-in's lane count is a const generic and the cluster width is a number, so the two
    // are paired here rather than matched on — a `_ =>` arm would be a case this test silently did
    // not run.
    type Builder = fn(u32) -> Result<Vec<u32>, LaneError>;
    let cases: [(u32, Builder); 3] = [
        (2, kernels::lane_sum::<F32, 2>),
        (4, kernels::lane_sum::<F32, 4>),
        (8, kernels::lane_sum::<F32, 8>),
    ];

    for (cluster, builtin) in cases {
        if cluster >= width {
            eprintln!("SKIPPED clusters of {cluster}: not narrower than a {width}-wide subgroup");
            continue;
        }

        // Per case rather than once for the test: a clustered tree and a clustered reduce declare
        // different capabilities — shuffles against clustered arithmetic — and both are dispatched
        // here. The gate this replaced named three bits by hand for exactly that reason, which is
        // three chances to name the wrong one.
        let tree_spirv = kernels::butterfly_cluster_sum(width, cluster).expect("built");
        let reduce_spirv = builtin(width).expect("built");
        if !runnable(&gpu, "butterfly-cluster", &[&tree_spirv, &reduce_spirv]) {
            return;
        }

        let tree = gpu.run(&tree_spirv, &input, 1).expect("dispatched");
        let reduced = gpu.run(&reduce_spirv, &input, 1).expect("dispatched");

        assert_eq!(
            tree,
            common::grouped_sums(count, cluster as usize),
            "the clustered tree disagrees with the reference at cluster {cluster}"
        );
        assert_eq!(
            tree, reduced,
            "the clustered tree and the clustered reduce disagree at cluster {cluster}"
        );
    }
}

/// A rolled loop, which is where the phis and the back edge live.
#[test]
fn a_rolled_loop_carries_its_value_round_the_back_edge() {
    let Some(gpu) = device("rolled-loop") else {
        return;
    };
    let width = gpu.limits().subgroup_size;
    let input = ramp(WORKGROUP_SIZE as usize);

    // Doubling five times is multiplying by 32. A loop that ran the wrong number of times, or
    // dropped the carried value, would give a different power of two — or the input back.
    for times in [0_u32, 1, 3, 5] {
        let output = gpu
            .run(
                &kernels::rolled_doubling(width, times).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

        let factor = 2_f32.powi(times as i32);
        let expected: Vec<f32> = input.iter().map(|value| value * factor).collect();

        assert_eq!(output, expected, "after {times} iterations");
    }
}

/// A value that survives a merge, which is the thing an `OpPhi` is for.
///
/// Both arms end in a subgroup reduction and exactly one of them runs, so the answer says which
/// edge the phi actually read from. A phi naming the wrong predecessor validates cleanly and then
/// reads whatever that other edge left behind — only running it says.
#[test]
fn a_value_computed_in_one_arm_arrives_at_the_merge() {
    let Some(gpu) = device("sum-or-max") else {
        return;
    };
    let limits = gpu.limits().clone();

    // `sum_or_max` votes, so it declares `GroupNonUniformVote` on top of the arithmetic this gate
    // used to name alone — the under-specified shape that made this conversion worth doing.
    let spirv = kernels::sum_or_max(limits.subgroup_size, 40.0).expect("built");
    if !runnable(&gpu, "sum-or-max", &[&spirv]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    // A ramp of 0..64 over two 32-wide subgroups. At 40 the first takes the max arm and the second
    // the sum arm, so one dispatch exercises both edges — and the two answers are far apart.
    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let sums = common::grouped_sums(count, width);
    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let first = lane / width * width;
            let highest = (first + width - 1).min(count - 1);
            if highest as f32 > 40.0 {
                sums.get(lane).copied().unwrap_or_default()
            } else {
                highest as f32
            }
        })
        .collect();

    assert_eq!(output, expected);

    // Discriminator: the arms must have produced different things, or the phi could have named
    // either edge and still looked right.
    //
    // It needs *two* subgroups in the workgroup to say that, and a 64-wide device running 64
    // invocations has one — so the threshold is met by the only subgroup there is and both arms
    // cannot be exercised in a single dispatch. Reported rather than asserted away: the case is
    // covered on a 32-wide device and is genuinely not covered here.
    if count > width {
        // Computed from the width rather than written out: the first subgroup is `0..width` and
        // the last is the `width` elements before `count`, and on an 8-wide device there are eight
        // subgroups between them rather than one.
        let low = output.first().copied().unwrap_or_default();
        let high = output.last().copied().unwrap_or_default();
        let first_max = (width - 1) as f32;
        let last_sum: f32 = ((count - width)..count).map(|value| value as f32).sum();

        assert_eq!(low, first_max, "the first subgroup took the max arm");
        assert_eq!(high, last_sum, "the last summed");
    } else {
        eprintln!(
            "sum-or-max: one subgroup of {width} in a workgroup of {count}, so only the arm it \
             took is exercised here"
        );
        assert_eq!(
            output.first().copied(),
            Some((0..count as u32).sum::<u32>() as f32),
            "the only subgroup exceeds the threshold, so it summed"
        );
    }
}

/// Both thresholds that make every subgroup agree, so each arm is checked alone.
#[test]
fn each_arm_is_the_whole_answer_when_every_subgroup_agrees() {
    let Some(gpu) = device("sum-or-max-uniform") else {
        return;
    };
    let limits = gpu.limits().clone();

    let always = kernels::sum_or_max(limits.subgroup_size, -1.0).expect("built");
    let never = kernels::sum_or_max(limits.subgroup_size, 1_000.0).expect("built");
    if !runnable(&gpu, "sum-or-max-uniform", &[&always, &never]) {
        return;
    }

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);

    let summed = gpu.run(&always, &input, 1).expect("dispatched");
    assert_eq!(summed, common::grouped_sums(count, width));

    let maxed = gpu.run(&never, &input, 1).expect("dispatched");
    let expected: Vec<f32> = (0..count)
        .map(|lane| ((lane / width * width) + width - 1).min(count - 1) as f32)
        .collect();
    assert_eq!(maxed, expected);
}

/// The loop counter, read by a body that is built once.
#[test]
fn a_rolled_body_can_see_which_iteration_it_is() {
    let Some(gpu) = device("rolled-counter") else {
        return;
    };
    let width = gpu.limits().subgroup_size;
    let input: Vec<u32> = (0..WORKGROUP_SIZE).collect();

    for times in [0_u32, 1, 4, 9] {
        let output = gpu
            .run_u32(
                &kernels::rolled_counter_sum(width, times).expect("built"),
                &input,
                1,
            )
            .expect("dispatched");

        // 0 + 1 + … + times-1. A body handed anything other than the live counter phi would add a
        // constant, and every one of these but `times = 1` would disagree.
        let added = times * times.saturating_sub(1) / 2;
        let expected: Vec<u32> = input.iter().map(|value| value + added).collect();

        assert_eq!(output, expected, "after {times} iterations");
    }
}

/// A branch inside a loop, and a loop inside a branch.
///
/// Both nestings at once, because they fail differently. A branch inside a loop makes the loop's
/// own bookkeeping land in the selection's merge block rather than in the body block it opened; a
/// loop inside a branch makes the selection's `OpPhi` name the loop's merge block rather than the
/// arm's. Neither had ever been built, let alone run.
#[test]
fn control_flow_nests_both_ways_round() {
    let Some(gpu) = device("nesting") else {
        return;
    };
    let limits = gpu.limits().clone();

    let width = limits.subgroup_size as usize;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    let times = 3_u32;

    // Both nestings, gated together: they are different modules and this test dispatches each.
    let branch_in_loop = kernels::branch_in_loop(limits.subgroup_size, times, 40.0).expect("built");
    let loop_in_branch = kernels::loop_in_branch(limits.subgroup_size, times, 40.0).expect("built");
    if !runnable(&gpu, "nesting", &[&branch_in_loop, &loop_in_branch]) {
        return;
    }

    // A ramp of 0..64 over two 32-wide subgroups: at 40 the first subgroup fails the vote and the
    // second passes it, so one dispatch exercises both arms.
    let takes_branch = |lane: usize| {
        let highest = (lane / width * width + width - 1).min(count - 1);
        highest as f32 > 40.0
    };

    // Inside a loop: the taken arm doubles each trip, the other adds one each trip.
    let output = gpu.run(&branch_in_loop, &input, 1).expect("dispatched");

    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let start = lane as f32;
            if takes_branch(lane) {
                start * 2_f32.powi(times as i32)
            } else {
                start + times as f32
            }
        })
        .collect();
    assert_eq!(output, expected, "a branch inside a loop");

    // Inside a branch: the taken arm runs the whole loop, the other returns the input.
    let output = gpu.run(&loop_in_branch, &input, 1).expect("dispatched");

    let expected: Vec<f32> = (0..count)
        .map(|lane| {
            let start = lane as f32;
            if takes_branch(lane) {
                start * 2_f32.powi(times as i32)
            } else {
                start
            }
        })
        .collect();
    assert_eq!(output, expected, "a loop inside a branch");

    // Discriminator: the two subgroups took different paths in both kernels, so a phi that always
    // read the same edge would have shown up.
    assert_ne!(output.first(), output.last());
}
