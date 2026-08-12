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

/// What the device should return, and whether it can be checked exactly.
#[derive(Debug, Clone)]
pub struct Reference {
    /// One value per invocation, matching what the kernel writes with `store_scalar`.
    pub values: Vec<u32>,
    /// Whether every value stayed inside the range its domain counts exactly.
    ///
    /// The integer domains are always exact: wrapping is defined and this reference wraps the same
    /// way. The float ones are not — a single counts integers to 2²⁴ and a **half only to 2¹¹** —
    /// and past that a sum is rounded. Comparing a rounded device answer against a rounded host
    /// answer is a comparison of two roundings, which says nothing about which lanes were combined.
    ///
    /// So a round that leaves the range is *refused* rather than loosened. That is what lets
    /// [`super::Domain::Half`] be fuzzed at all, and it is why `Domain::Float` — which had only
    /// ever *assumed* it stayed under 2²⁴ — is now checked too.
    pub exact: bool,
}

/// What the device should return for `program` over `input`.
#[must_use]
pub fn reference(program: &Program, input: &[u32]) -> Reference {
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
    // Checked after the corpus is laid out and again after every step, not only at the end: a
    // value that left the range and was then clamped back would otherwise be compared as though
    // nothing had happened.
    let limit = domain.exact_limit();
    let mut exact = within(domain, limit, &held);

    for step in &program.steps {
        held = apply(program, &held, *step);
        exact = exact && within(domain, limit, &held);
    }

    // Step 3: reduce. A vector at least as wide as the subgroup reduces over the subgroup after
    // its strips are folded; a narrower one reduces within its cluster. `min` rather than a
    // comparison, for the same reason as `strips_of` below: at equal widths both arms of the
    // branch give the same number, so the branch was unfalsifiable.
    let group_size = program.lanes.min(program.subgroup) as usize;

    // The vote and the reduction do not cover the same lanes. `any_uniform` is subgroup-scoped
    // whatever the mapping is, while a clustered reduction covers only `lanes` of them — so a
    // narrow vector votes across four clusters and then reduces within one. Reusing the
    // reduction's group would agree with the kernel for every full-width vector and disagree for
    // every clustered one.
    //
    // **It is worked out inside the one arm that reads it.** This used to be a `voted` vector
    // built up front, with a `_ => vec![false; invocations]` arm for the three finishes that never
    // look at it — a value no test could observe, which a mutation run flipped to `true` without
    // anything noticing, and which two earlier attempts had only moved rather than removed. There
    // is no default here to be wrong about, because there is no value to default.
    //
    // The cost is that the vote is recomputed per invocation rather than once per subgroup. This
    // reference is written to be obviously right rather than fast, and computing something where
    // it is used is the more obvious of the two.
    let width = program.subgroup as usize;

    let values: Vec<u32> = (0..invocations)
        .map(|invocation| {
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
                Finish::SumOrMax { when_any_above } => {
                    // The vote covers this invocation's whole *subgroup*, which is a different set
                    // from the lanes the reduction above combines whenever the vector is narrower
                    // than the subgroup.
                    let threshold = domain.encode(when_any_above);
                    let first = invocation / width * width;
                    let takes_the_sum = held
                        .get(first..(first + width).min(invocations))
                        .into_iter()
                        .flatten()
                        .flatten()
                        .any(|value| domain.greater(*value, threshold));

                    if takes_the_sum { sum() } else { max() }
                }
            }
        })
        .collect();

    // The reduction itself can leave the range even when every element was inside it — that is
    // exactly what a sum over a few hundred halves does — so the answers are checked as well.
    let exact = exact && within(domain, limit, std::slice::from_ref(&values));

    Reference { values, exact }
}

/// Whether every value in `held` is one its domain still counts exactly.
///
/// `None` means the domain has no such limit — the integers, where wrapping is defined and this
/// reference wraps the same way — and everything is exact by construction.
fn within(domain: super::Domain, limit: Option<f32>, held: &[Vec<u32>]) -> bool {
    let Some(limit) = limit else {
        return true;
    };

    held.iter().flatten().all(|&bits| {
        let value = domain.as_f32(bits);
        value.is_finite() && value.abs() <= limit
    })
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

    /// The reference's answers, for a test that is about the arithmetic rather than the range.
    ///
    /// Every case below stays well inside its domain's exact range — they are hand-written with
    /// small integers — so `exact` is asserted once here rather than in each of them, and a test
    /// that started leaving the range would say so rather than quietly comparing rounded values.
    fn values_of(program: &Program, input: &[u32]) -> Vec<u32> {
        let answer = reference(program, input);
        assert!(
            answer.exact,
            "this case left the range {:?} counts exactly",
            program.domain
        );
        answer.values
    }

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
        let out = values_of(
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
        let out = values_of(
            &program(Domain::Float, 32, Vec::new(), Finish::Sum),
            &ramp(Domain::Float, 64),
        );

        assert_eq!(f32::from_bits(out[0]), (0..32).sum::<u32>() as f32);
        assert_eq!(f32::from_bits(out[32]), (32..64).sum::<u32>() as f32);
    }

    #[test]
    fn a_clustered_sum_covers_only_its_own_cluster() {
        let input = ramp(Domain::Unsigned, 64);
        let out = values_of(
            &program(Domain::Unsigned, 8, Vec::new(), Finish::Sum),
            &input,
        );

        assert_eq!(out[0], (0..8).sum::<u32>());
        assert_eq!(out[8], (8..16).sum::<u32>());
    }

    #[test]
    fn a_strip_mined_sum_covers_both_of_each_lanes_elements() {
        let input = ramp(Domain::Unsigned, 128);
        let out = values_of(
            &program(Domain::Unsigned, 64, Vec::new(), Finish::Sum),
            &input,
        );

        // Lanes 0..32 hold {0..32} and {64..96}; the subgroup sums all of them.
        assert_eq!(out[0], (0..32).chain(64..96).sum::<u32>());
    }

    #[test]
    fn elementwise_steps_run_before_the_reduction() {
        let input = vec![Domain::Unsigned.encode(1); 64];
        let out = values_of(
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
        let out = values_of(
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
        let out = values_of(
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
        let out = values_of(
            &program(Domain::Unsigned, 32, vec![Op::ClampBelow(10)], Finish::Max),
            &input,
        );
        assert_eq!(out[0], 31, "the maximum is untouched by a floor below it");

        let out = values_of(
            &program(Domain::Unsigned, 32, vec![Op::ClampBelow(100)], Finish::Max),
            &input,
        );
        assert_eq!(out[0], 100, "and everything is the floor when it is above");
    }

    #[test]
    fn a_shift_of_zero_changes_nothing() {
        let input = ramp(Domain::Unsigned, 64);
        let plain = values_of(
            &program(Domain::Unsigned, 32, Vec::new(), Finish::Sum),
            &input,
        );
        let shifted = values_of(
            &program(Domain::Unsigned, 32, vec![Op::ShiftUp(0)], Finish::Sum),
            &input,
        );

        assert_eq!(plain, shifted);
    }

    #[test]
    fn a_value_exactly_at_the_limit_still_counts_as_exact() {
        // The boundary `within` is written on. 2048 *is* representable in a half — it is the first
        // integer whose successor is not — so a round holding it is comparable, and a `<` there
        // would throw away the one value the limit is named after.
        let input = vec![Domain::Half.encode(2_048); 64];
        let answer = reference(&program(Domain::Half, 32, Vec::new(), Finish::Max), &input);

        assert!(answer.exact, "2048 is exactly representable in a half");
        assert_eq!(answer.values[0], Domain::Half.encode(2_048));
    }

    #[test]
    fn a_value_past_the_limit_is_not_exact() {
        // The other side of the same boundary, so the pair pins it rather than one of them.
        let input = vec![Domain::Half.encode(4_000); 64];
        let answer = reference(&program(Domain::Half, 32, Vec::new(), Finish::Max), &input);

        assert!(!answer.exact, "4000 is not an integer a half can hold");
    }

    #[test]
    fn a_step_that_leaves_the_range_is_noticed_even_though_the_input_was_inside_it() {
        // Exactness accumulates across the steps: it is `and`, not `or`. With `or` a program whose
        // input was in range would be reported exact however far its arithmetic wandered
        // afterwards — which is the whole failure this check exists to prevent, since the values
        // compared are the ones *after* the steps.
        let input = vec![Domain::Half.encode(1_024); 64];

        let untouched = reference(&program(Domain::Half, 32, Vec::new(), Finish::Max), &input);
        assert!(untouched.exact, "1024 alone is inside the range");

        let raised = reference(
            &program(Domain::Half, 32, vec![Op::AddConstant(1_500)], Finish::Max),
            &input,
        );
        assert!(
            !raised.exact,
            "1024 + 1500 leaves what a half counts exactly, and the step is where it happened"
        );

        // **And back again.** The case above is *also* caught by the check on the final answers,
        // so it does not actually prove the per-step accumulation is an `and` — flipping it to
        // `or` left that assertion passing, which is how the mutation gate found this gap.
        //
        // Here the value leaves the range at the first step and is clamped back inside it at the
        // second, so everything that survives to be compared is in range and only the *middle* was
        // not. That is the case the comment beside the loop describes, and it had no test.
        let there_and_back = reference(
            &program(
                Domain::Half,
                32,
                vec![Op::AddConstant(1_500), Op::ClampBoth { low: 0, high: 100 }],
                Finish::Max,
            ),
            &input,
        );
        assert!(
            !there_and_back.exact,
            "the intermediate left the range, and clamping it back does not un-round it"
        );
    }

    #[test]
    fn an_integer_domain_is_exact_whatever_it_holds() {
        // Wrapping is defined and this reference wraps the same way, so there is no range to
        // leave — and a limit accidentally given to an integer domain would start refusing rounds
        // that were perfectly checkable.
        let input = vec![Domain::Unsigned.encode(4_000_000_000); 64];
        let answer = reference(
            &program(
                Domain::Unsigned,
                32,
                vec![Op::AddConstant(9_999)],
                Finish::Sum,
            ),
            &input,
        );

        assert!(answer.exact);
    }
}
