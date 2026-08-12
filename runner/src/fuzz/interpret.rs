//! The CPU reference: what a generated program must produce.
//!
//! This models the mapping rather than the instructions — where each element lives, which lanes
//! share a subgroup, and which of them a reduction covers. If it and the emitter ever disagree
//! about that, the disagreement is the finding, and either could be at fault.
//!
//! Written to be obviously right rather than fast: a reference with a clever bug in it turns
//! every fuzzing round into a false alarm. Arithmetic goes through [`Domain`], so the same
//! structure serves both element types and neither gets its own chance to be subtly wrong.

mod steps;

use self::steps::apply;
use super::program::{Finish, Program};

/// What the device should return for `program` over `input`.
///
/// One value per invocation, matching what the kernel writes with `store_scalar`.
#[must_use]
pub fn reference(program: &Program, input: &[u32]) -> Vec<u32> {
    let invocations = (program.groups * program.workgroup) as usize;
    let strips = strips_of(program);
    let domain = program.domain;

    // Step 1: gather each invocation's elements, using the same address arithmetic the kernel
    // does — workgroup-blocked, strided within the run.
    let mut held: Vec<Vec<u32>> = Vec::with_capacity(invocations);
    for invocation in 0..invocations {
        let group = invocation / program.workgroup as usize;
        let local = invocation % program.workgroup as usize;
        let base = group * program.workgroup as usize * strips;

        held.push(
            (0..strips)
                .map(|strip| {
                    let at = base + local + strip * program.workgroup as usize;
                    input.get(at).copied().unwrap_or_else(|| domain.zero())
                })
                .collect(),
        );
    }

    // Step 2: run the program. Every step is elementwise except the shuffles and the vote, which
    // read across invocations — so the whole state advances together.
    for step in &program.steps {
        held = apply(program, &held, *step);
    }

    // Step 3: reduce. A vector at least as wide as the subgroup reduces over the subgroup after
    // its strips are folded; a narrower one reduces within its cluster. `min` rather than a
    // comparison, for the same reason as `strips_of` below: at equal widths both arms of the
    // branch give the same number, so the branch was unfalsifiable.
    let group_size = program.lanes.min(program.subgroup) as usize;

    // The vote and the reduction do not cover the same lanes. `any_uniform` is subgroup-scoped
    // whatever the mapping is, while a clustered reduction covers only `lanes` of them — so a
    // narrow vector votes across four clusters and then reduces within one. Working the vote out
    // per subgroup, separately, is the only way to model that; reusing the reduction's group would
    // agree with the kernel for every full-width vector and disagree for every clustered one.
    // One entry per *invocation* rather than per subgroup, so reading it needs no index
    // arithmetic and no default for an index that cannot occur. The per-subgroup version needed
    // `voted.get(invocation / width).unwrap_or(false)`, and that default was unreachable — a
    // branch no test could take, which a mutation run duly flipped without anything noticing.
    let width = program.subgroup as usize;
    let voted: Vec<bool> = match program.finish {
        Finish::SumOrMax { when_any_above } => {
            let threshold = domain.encode(when_any_above);
            held.chunks(width)
                .flat_map(|subgroup| {
                    let passed = subgroup
                        .iter()
                        .flatten()
                        .any(|value| domain.greater(*value, threshold));
                    std::iter::repeat_n(passed, subgroup.len())
                })
                .collect()
        }
        // Never read: the only arm that consults it is `Finish::SumOrMax` above. `false` over
        // `true` because it is the answer that would be least surprising if it somehow were.
        //
        // A mutation run flips this to `true` and nothing notices, correctly — it is an
        // equivalent mutant, and the alternatives are worse. Making it unrepresentable means
        // either an index-and-default (the shape this replaced, whose default was *also*
        // unreachable) or nesting the width-chunks and the reduction-chunks inside each other in
        // the reference every other layer is checked against. Neither is worth trading a comment
        // for.
        _ => vec![false; invocations],
    };

    voted
        .iter()
        .enumerate()
        .map(|(invocation, &takes_the_sum)| {
            let first = invocation / group_size * group_size;
            let members = first..(first + group_size).min(invocations);
            let values: Vec<u32> = members
                .flat_map(|other| {
                    held.get(other)
                        .map(|elements| elements.iter().copied())
                        .into_iter()
                        .flatten()
                })
                .collect();

            let sum = || {
                values
                    .iter()
                    .fold(domain.zero(), |total, &value| domain.add(total, value))
            };
            // `reduce`, not `fold` from an identity. A maximum folded from zero is right only
            // while every value is non-negative, which stopped being true when the signed domain
            // arrived — and it would have gone on looking right for most inputs.
            let max = || {
                values
                    .iter()
                    .copied()
                    .reduce(|best, value| domain.max(best, value))
                    .unwrap_or_else(|| domain.smallest())
            };

            match program.finish {
                Finish::Sum => sum(),
                Finish::Max => max(),
                Finish::Min => values
                    .iter()
                    .copied()
                    .reduce(|best, value| domain.min(best, value))
                    .unwrap_or_else(|| domain.largest()),
                Finish::SumOrMax { .. } => {
                    if takes_the_sum {
                        sum()
                    } else {
                        max()
                    }
                }
            }
        })
        .collect()
}

/// How many elements each invocation holds.
///
/// Written without a comparison on purpose. The obvious spelling —
/// `if lanes > subgroup { lanes / subgroup } else { 1 }` — has a branch that cannot be got wrong:
/// at `lanes == subgroup` both arms give one, so flipping `>` to `>=` changes nothing and no test
/// can ever kill that mutant. A survivor that no test could kill is a survivor that teaches
/// nothing, and it was pointing at a branch that did not need to exist.
fn strips_of(program: &Program) -> usize {
    (program.lanes / program.subgroup.max(1)).max(1) as usize
}

#[cfg(test)]
mod tests {
    #![allow(clippy::indexing_slicing)]

    use super::*;
    use crate::fuzz::{Domain, Op};

    fn program(domain: Domain, lanes: u32, steps: Vec<Op>, finish: Finish) -> Program {
        Program {
            domain,
            subgroup: 32,
            workgroup: 64,
            groups: 1,
            lanes,
            steps,
            finish,
        }
    }

    /// A ramp of small integers, encoded for `domain`.
    fn ramp(domain: Domain, count: u32) -> Vec<u32> {
        (0..count).map(|value| domain.encode(value)).collect()
    }

    #[test]
    fn a_plain_subgroup_sum_matches_the_arithmetic_series() {
        let input = ramp(Domain::Unsigned, 64);
        let out = reference(
            &program(Domain::Unsigned, 32, Vec::new(), Finish::Sum),
            &input,
        );

        assert_eq!(out[0], (0..32).sum::<u32>());
        assert_eq!(out[32], (32..64).sum::<u32>());
    }

    #[test]
    fn the_same_program_over_floats_gives_the_same_numbers() {
        // The whole float argument in one assertion: at these magnitudes the two domains agree
        // exactly, so a disagreement later is about instructions rather than about rounding.
        let out = reference(
            &program(Domain::Float, 32, Vec::new(), Finish::Sum),
            &ramp(Domain::Float, 64),
        );

        assert_eq!(f32::from_bits(out[0]), (0..32).sum::<u32>() as f32);
        assert_eq!(f32::from_bits(out[32]), (32..64).sum::<u32>() as f32);
    }

    #[test]
    fn a_clustered_sum_covers_only_its_own_cluster() {
        let input = ramp(Domain::Unsigned, 64);
        let out = reference(
            &program(Domain::Unsigned, 8, Vec::new(), Finish::Sum),
            &input,
        );

        assert_eq!(out[0], (0..8).sum::<u32>());
        assert_eq!(out[8], (8..16).sum::<u32>());
    }

    #[test]
    fn a_strip_mined_sum_covers_both_of_each_lanes_elements() {
        let input = ramp(Domain::Unsigned, 128);
        let out = reference(
            &program(Domain::Unsigned, 64, Vec::new(), Finish::Sum),
            &input,
        );

        // Lanes 0..32 hold {0..32} and {64..96}; the subgroup sums all of them.
        assert_eq!(out[0], (0..32).chain(64..96).sum::<u32>());
    }

    #[test]
    fn elementwise_steps_run_before_the_reduction() {
        let input = vec![Domain::Unsigned.encode(1); 64];
        let out = reference(
            &program(
                Domain::Unsigned,
                32,
                vec![Op::AddConstant(1), Op::MulConstant(2)],
                Finish::Sum,
            ),
            &input,
        );

        // (1 + 1) * 2 = 4, thirty-two times.
        assert_eq!(out[0], 4 * 32);
    }

    #[test]
    fn a_butterfly_pairs_within_the_subgroup_and_not_across_it() {
        let input = ramp(Domain::Unsigned, 64);
        let out = reference(
            &program(Domain::Unsigned, 32, vec![Op::ButterflyAdd(1)], Finish::Max),
            &input,
        );

        // Lanes 30 and 31 pair to 61, the largest pair in the first subgroup.
        assert_eq!(out[0], 61);
        assert_eq!(out[32], 125);
    }

    #[test]
    fn a_uniform_branch_applies_per_subgroup_and_not_per_lane() {
        let input = ramp(Domain::Unsigned, 64);
        let step = Op::AddIfAnyAbove {
            when_any_above: 40,
            add: 100,
        };
        let out = reference(
            &program(Domain::Unsigned, 32, vec![step], Finish::Max),
            &input,
        );

        // The first subgroup's largest element is 31, so it does not qualify and nothing in it
        // moves. The second's is 63, so every lane of it gains 100.
        assert_eq!(out[0], 31);
        assert_eq!(out[32], 163);
    }

    #[test]
    fn a_clamp_raises_the_floor_and_leaves_the_rest() {
        let input = ramp(Domain::Unsigned, 64);
        let out = reference(
            &program(Domain::Unsigned, 32, vec![Op::ClampBelow(10)], Finish::Max),
            &input,
        );
        assert_eq!(out[0], 31, "the maximum is untouched by a floor below it");

        let out = reference(
            &program(Domain::Unsigned, 32, vec![Op::ClampBelow(100)], Finish::Max),
            &input,
        );
        assert_eq!(out[0], 100, "and everything is the floor when it is above");
    }

    #[test]
    fn a_shift_of_zero_changes_nothing() {
        let input = ramp(Domain::Unsigned, 64);
        let plain = reference(
            &program(Domain::Unsigned, 32, Vec::new(), Finish::Sum),
            &input,
        );
        let shifted = reference(
            &program(Domain::Unsigned, 32, vec![Op::ShiftUp(0)], Finish::Sum),
            &input,
        );

        assert_eq!(plain, shifted);
    }
}
