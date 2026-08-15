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

    // Step 3a: a scan, if that is the finish. It keeps every element rather than combining them,
    // so it answers in the buffer's own shape and returns before the reduction below.
    //
    // The `else` is what makes the match below exhaustive without an unreachable arm. An arm that
    // cannot be reached is a lie waiting to become true, and returning the identity from one would
    // turn a future mistake into zeros rather than a compile error.
    let Some(combine) = Combine::of(program.finish) else {
        let exclusive = matches!(program.finish, Finish::ScanExclusive);
        let values = scanned(program, &held, exclusive);
        let exact = exact && within(domain, limit, std::slice::from_ref(&values));
        return Reference { values, exact };
    };

    // Step 3b: reduce. A vector at least as wide as the subgroup reduces over the subgroup after
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

            match combine {
                Combine::Sum => sum(),
                Combine::Max => max(),
                Combine::Min => values
                    .iter()
                    .copied()
                    .reduce(|best, value| domain.min(best, value))
                    .unwrap_or_else(|| domain.largest()),
                Combine::SumOrMax { when_any_above } => {
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

/// The finishes that **combine** the lanes rather than keeping every element.
///
/// A separate type rather than a subset of [`Finish`] matched loosely, so the reduction below is
/// exhaustive over exactly the cases it answers for. Adding a third kind of scan will not compile
/// until somebody decides what it means here.
#[derive(Clone, Copy)]
enum Combine {
    Sum,
    Max,
    Min,
    SumOrMax { when_any_above: u32 },
}

impl Combine {
    /// `None` for the finishes that keep every element, which are answered elsewhere.
    const fn of(finish: Finish) -> Option<Self> {
        match finish {
            Finish::Sum => Some(Self::Sum),
            Finish::Max => Some(Self::Max),
            Finish::Min => Some(Self::Min),
            Finish::SumOrMax { when_any_above } => Some(Self::SumOrMax { when_any_above }),
            Finish::Scan | Finish::ScanExclusive => None,
        }
    }
}

/// What a scan should write, in the buffer's own order.
///
/// **This models the lane order rather than the arithmetic, and that is the whole point.** A
/// reduction combines the same set whichever lane holds what, so its reference never had to know
/// where an element sits. A prefix does: element `j` of the answer is the sum of vector positions
/// `0..=j`, and which element is at position `j` is exactly what a mapping decides.
///
/// `crate::lanes::vector` documents the order this reproduces: lane `l` holds the elements at `l`,
/// `l + width`, `l + 2·width`, so vector position `j` is strip `j / width` of lane `j % width` —
/// strips are consecutive runs of the vector and every element of one comes before every element
/// of the next.
///
/// The vector belongs to a **subgroup**, not a workgroup: nothing in `Lanes` crosses between them,
/// so each run of `width` invocations scans on its own and the next starts from zero again.
///
/// **And a vector narrower than the subgroup is narrower than that.** A `Simd<f32, 8>` on a 32-wide
/// device is four independent vectors sharing one subgroup, and the clustered ladder scans each of
/// them on its own — so the run that scans together is `min(lanes, width)` invocations rather than
/// the width. Same expression as the reduction's `group_size` above, and for the same reason: a
/// reference that used the width here would agree with the kernel in the first cluster of every
/// subgroup and disagree in all the others.
fn scanned(program: &Program, held: &[Vec<u32>], exclusive: bool) -> Vec<u32> {
    let domain = program.domain;
    let group = (program.lanes.min(program.subgroup) as usize).max(1);
    let workgroup = program.workgroup as usize;
    let strips = strips_of(program);
    let invocations = held.len();

    let mut values = vec![domain.zero(); invocations * strips];

    for first in (0..invocations).step_by(group) {
        let members = group.min(invocations.saturating_sub(first));
        let mut running = domain.zero();

        for position in 0..members * strips {
            let lane = position % group;
            let strip = position / group;
            let invocation = first + lane;

            let Some(element) = held.get(invocation).and_then(|held| held.get(strip)) else {
                continue;
            };

            // The inclusive total is what carries forward either way; which of the two is
            // *written* is the only difference between the finishes.
            let inclusive = domain.add(running, *element);
            let answer = if exclusive { running } else { inclusive };
            running = inclusive;

            // Back to where the kernel's own addressing puts it: workgroup-blocked, strided
            // within the run — the same arithmetic step 1 read the corpus with.
            let group = invocation / workgroup.max(1);
            let local = invocation % workgroup.max(1);
            let at = group * workgroup * strips + local + strip * workgroup;
            if let Some(slot) = values.get_mut(at) {
                *slot = answer;
            }
        }
    }

    values
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
    fn the_vote_on_a_value_adds_where_the_subgroup_agrees_and_nowhere_else() {
        // **Both branches, because a generated round only ever reaches one.** The corpus is
        // distinct values by design — that is what makes a wrong answer obvious — so the vote it
        // feeds never passes, and the mutation gate duly found that flipping this condition to
        // `false` changed nothing any sweep could see.
        //
        // A uniform subgroup is not a shape the generator produces, and it is the only shape the
        // operation is *for*.
        let step = Op::AddIfAllEqual { add: 5 };
        let agreeing: Vec<u32> = vec![Domain::Unsigned.encode(7); 64];
        let out = values_of(
            &program(Domain::Unsigned, 32, vec![step], Finish::Sum),
            &agreeing,
        );
        assert_eq!(
            out[0],
            (7 + 5) * 32,
            "every lane agreed, so every lane added"
        );

        // One lane of the first subgroup differs: that subgroup adds nothing, and the second still
        // does. A reference that voted per *lane* would add in thirty-one of the first thirty-two.
        let mut split = agreeing.clone();
        split[1] = Domain::Unsigned.encode(8);
        let out = values_of(
            &program(Domain::Unsigned, 32, vec![step], Finish::Sum),
            &split,
        );
        assert_eq!(out[0], 7 * 31 + 8, "the divergent subgroup adds nothing");
        assert_eq!(out[32], (7 + 5) * 32, "and the one beside it is unaffected");
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
            &program(Domain::Unsigned, 32, vec![Op::ShiftUp], Finish::Sum),
            &input,
        );

        assert_eq!(plain, shifted);
    }

    #[test]
    fn a_scan_that_leaves_the_range_is_not_exact_even_though_every_element_was_inside_it() {
        // The same `and`, on the scan's own path. A prefix is the one finish whose *output* can
        // leave a domain's exact range while every input and every step stayed inside it: 300
        // halves are each representable and their running total passes 2048 part way along.
        //
        // With `or` the round would be reported comparable and both sides would be rounded, which
        // says nothing about which lanes were combined — the failure `exact` exists to prevent.
        let input = vec![Domain::Half.encode(300); 64];
        let answer = reference(&program(Domain::Half, 32, Vec::new(), Finish::Scan), &input);

        assert!(
            !answer.exact,
            "a running total of 300s passes 2048 before the end of a 32-lane scan"
        );

        // And the inputs really were inside it, so this is the accumulation being caught rather
        // than the corpus.
        let each = reference(&program(Domain::Half, 32, Vec::new(), Finish::Max), &input);
        assert!(each.exact, "300 on its own is exactly representable");
    }

    #[test]
    fn a_scan_answers_for_every_element_and_not_every_invocation() {
        // **The shape, which is what tells a scan from a reduction here.** A reduction writes one
        // value per invocation; a scan writes one per element, so its answer is as long as the
        // input and lands in the same places.
        //
        // A length computed with a division rather than a multiplication — which is what the
        // mutation gate tried — leaves the vector short, the extra writes dropped on the floor,
        // and the comparison in `verdict` zipping over a prefix of what it should check.
        for lanes in [32_u32, 64, 128] {
            let scanning = program(Domain::Unsigned, lanes, Vec::new(), Finish::Scan);
            let reducing = program(Domain::Unsigned, lanes, Vec::new(), Finish::Sum);
            let input = ramp(Domain::Unsigned, scanning.input_len() as u32);

            assert_eq!(
                reference(&scanning, &input).values.len(),
                scanning.input_len(),
                "a scan of {lanes} lanes answers for every element"
            );
            assert_eq!(
                reference(&reducing, &input).values.len(),
                (scanning.groups * scanning.workgroup) as usize,
                "a reduction of {lanes} lanes answers once per invocation"
            );
        }
    }

    #[test]
    fn a_scan_is_the_prefix_of_its_subgroup_and_starts_again_at_the_next() {
        // Nothing in `Lanes` crosses between subgroups, so a workgroup of 64 over a 32-wide
        // subgroup holds two scans and not one. The 33rd element starts from its own element
        // again, which is the boundary a reference that scanned the whole workgroup would miss.
        let scanning = program(Domain::Unsigned, 32, Vec::new(), Finish::Scan);
        let input = vec![Domain::Unsigned.encode(1); scanning.input_len()];
        let answer = reference(&scanning, &input);

        assert_eq!(answer.values[0], Domain::Unsigned.encode(1));
        assert_eq!(answer.values[31], Domain::Unsigned.encode(32));
        assert_eq!(
            answer.values[32],
            Domain::Unsigned.encode(1),
            "the second subgroup starts again"
        );
        assert_eq!(answer.values[63], Domain::Unsigned.encode(32));
    }

    #[test]
    fn an_exclusive_scan_leaves_each_element_out_and_starts_at_zero() {
        let scanning = program(Domain::Unsigned, 32, Vec::new(), Finish::ScanExclusive);
        let input = vec![Domain::Unsigned.encode(1); scanning.input_len()];
        let answer = reference(&scanning, &input);

        assert_eq!(answer.values[0], Domain::Unsigned.encode(0));
        assert_eq!(answer.values[31], Domain::Unsigned.encode(31));
        assert_eq!(answer.values[32], Domain::Unsigned.encode(0));
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
