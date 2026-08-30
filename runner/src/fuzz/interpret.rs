mod steps;

use self::steps::apply;
use super::program::{Finish, Fold, Op, Program};

#[derive(Debug, Clone)]
pub struct Reference {
    pub values: Vec<u32>,
    pub exact: bool,
}

#[must_use]
pub fn reference(program: &Program, input: &[u32]) -> Reference {
    let invocations = (program.groups * program.workgroup) as usize;
    let strips = strips_of(program);
    let domain = program.domain;

    assert!(
        input.len() >= program.input_len(),
        "this program reads {} elements and was given {}",
        program.input_len(),
        input.len()
    );

    let mut held: Vec<Vec<u32>> = Vec::with_capacity(invocations);
    for invocation in 0..invocations {
        let group = invocation / program.workgroup as usize;
        let local = invocation % program.workgroup as usize;
        let base = group * program.workgroup as usize * strips;

        held.push(
            (0..strips)
                .map(|strip| {
                    let at = base + local + strip * program.workgroup as usize;
                    input[at]
                })
                .collect(),
        );
    }

    let limit = domain.exact_limit();
    let mut exact = within(domain, limit, &held);

    for step in &program.steps {
        exact = exact && predictable(domain, *step, &held);
        held = apply(program, &held, *step);
        exact = exact && within(domain, limit, &held);
    }

    let Some(combine) = Combine::of(program.finish) else {
        let (fold, exclusive) = match program.finish {
            Finish::ScanBy { fold, exclusive } => (Some(fold), exclusive),
            other => (None, matches!(other, Finish::ScanExclusive)),
        };
        let values = scanned(program, &held, fold, exclusive);
        let exact = exact && within(domain, limit, std::slice::from_ref(&values));
        return Reference { values, exact };
    };

    let group_size = program.lanes.min(program.subgroup) as usize;

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
                Combine::By(fold) => values.iter().fold(domain.identity(fold), |total, &value| {
                    domain.fold(fold, total, value)
                }),
            }
        })
        .collect();

    let exact = exact && within(domain, limit, std::slice::from_ref(&values));

    Reference { values, exact }
}

#[derive(Clone, Copy)]
enum Combine {
    Sum,
    Max,
    Min,
    SumOrMax { when_any_above: u32 },
    By(Fold),
}

impl Combine {
    const fn of(finish: Finish) -> Option<Self> {
        match finish {
            Finish::Sum => Some(Self::Sum),
            Finish::Max => Some(Self::Max),
            Finish::Min => Some(Self::Min),
            Finish::SumOrMax { when_any_above } => Some(Self::SumOrMax { when_any_above }),
            Finish::Scan | Finish::ScanExclusive | Finish::ScanBy { .. } => None,
            Finish::ReduceBy(fold) => Some(Self::By(fold)),
        }
    }
}

fn scanned(program: &Program, held: &[Vec<u32>], fold: Option<Fold>, exclusive: bool) -> Vec<u32> {
    let domain = program.domain;
    let group = (program.lanes.min(program.subgroup) as usize).max(1);
    let workgroup = program.workgroup as usize;
    let strips = strips_of(program);
    let invocations = held.len();

    let seed = fold.map_or_else(|| domain.zero(), |fold| domain.identity(fold));
    let mut values = vec![seed; invocations * strips];

    for first in (0..invocations).step_by(group) {
        let members = group.min(invocations.saturating_sub(first));
        let mut running = seed;

        for position in 0..members * strips {
            let lane = position % group;
            let strip = position / group;
            let invocation = first + lane;

            let Some(element) = held.get(invocation).and_then(|held| held.get(strip)) else {
                continue;
            };

            let inclusive = fold.map_or_else(
                || domain.add(running, *element),
                |fold| domain.fold(fold, running, *element),
            );
            let answer = if exclusive { running } else { inclusive };
            running = inclusive;

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

fn predictable(domain: super::Domain, step: Op, held: &[Vec<u32>]) -> bool {
    if !matches!(step, Op::Absolute) || domain.is_float() {
        return true;
    }

    let smallest = domain.smallest();
    held.iter().flatten().all(|&bits| bits != smallest)
}

fn within(domain: super::Domain, limit: Option<f32>, held: &[Vec<u32>]) -> bool {
    let Some(limit) = limit else {
        return true;
    };

    held.iter().flatten().all(|&bits| {
        let value = domain.as_f32(bits);
        value.is_finite() && value.abs() <= limit
    })
}

fn strips_of(program: &Program) -> usize {
    (program.lanes / program.subgroup.max(1)).max(1) as usize
}

#[cfg(test)]
mod tests {
    #![allow(clippy::indexing_slicing)]

    use super::*;
    use crate::fuzz::{Domain, Op};

    use crate::fuzz::BitShift;

    fn one_lane(domain: Domain, steps: Vec<Op>) -> Program {
        Program {
            domain,
            subgroup: 1,
            workgroup: 1,
            groups: 1,
            lanes: 1,
            steps,
            finish: Finish::Sum,
        }
    }

    #[test]
    fn a_magnitude_with_no_answer_is_refused_and_an_ordinary_one_is_not() {
        let shifted_to_the_top = one_lane(
            Domain::Byte,
            vec![
                Op::BitShift {
                    kind: BitShift::Left,
                    by: 7,
                },
                Op::Absolute,
            ],
        );
        assert!(
            !reference(&shifted_to_the_top, &[1]).exact,
            "a magnitude of the smallest `i8` was compared, and no specification says what it is"
        );

        assert!(
            reference(&shifted_to_the_top, &[2]).exact,
            "a round that never met the minimum was refused, so the domain has no coverage left"
        );

        let no_magnitude = one_lane(
            Domain::Byte,
            vec![Op::BitShift {
                kind: BitShift::Left,
                by: 7,
            }],
        );
        assert!(
            reference(&no_magnitude, &[1]).exact,
            "the smallest `i8` is an ordinary value until something asks for its magnitude"
        );

        let unsigned = one_lane(
            Domain::UnsignedByte,
            vec![Op::BitShift {
                kind: BitShift::Left,
                by: 7,
            }],
        );
        assert!(reference(&unsigned, &[1]).exact);
    }

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

    fn ramp(domain: Domain, count: u32) -> Vec<u32> {
        (0..count).map(|value| domain.encode(value)).collect()
    }

    #[test]
    #[should_panic(expected = "reads 64 elements and was given 63")]
    fn a_short_input_is_refused_rather_than_read_as_zeros() {
        let program = program(Domain::Unsigned, 32, vec![Op::AddConstant(1)], Finish::Sum);
        let _ = reference(&program, &ramp(Domain::Unsigned, 63));
    }

    #[test]
    fn exactly_what_the_program_reads_is_enough() {
        let program = program(Domain::Unsigned, 32, vec![Op::AddConstant(1)], Finish::Sum);
        assert_eq!(program.input_len(), 64);
        assert!(reference(&program, &ramp(Domain::Unsigned, 64)).exact);
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
        let out = values_of(
            &program(Domain::Float, 32, Vec::new(), Finish::Sum),
            &ramp(Domain::Float, 64),
        );

        assert_eq!(f32::from_bits(out[0]), (0..32).sum::<u32>() as f32);
        assert_eq!(f32::from_bits(out[32]), (32..64).sum::<u32>() as f32);
    }

    #[test]
    fn the_vote_on_a_value_adds_where_the_subgroup_agrees_and_nowhere_else() {
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

        assert_eq!(out[0], 4 * 32);
    }

    #[test]
    fn a_butterfly_pairs_within_the_subgroup_and_not_across_it() {
        let input = ramp(Domain::Unsigned, 64);
        let out = values_of(
            &program(Domain::Unsigned, 32, vec![Op::ButterflyAdd(1)], Finish::Max),
            &input,
        );

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
        let input = vec![Domain::Half.encode(300); 64];
        let answer = reference(&program(Domain::Half, 32, Vec::new(), Finish::Scan), &input);

        assert!(
            !answer.exact,
            "a running total of 300s passes 2048 before the end of a 32-lane scan"
        );

        let each = reference(&program(Domain::Half, 32, Vec::new(), Finish::Max), &input);
        assert!(each.exact, "300 on its own is exactly representable");
    }

    #[test]
    fn a_scan_answers_for_every_element_and_not_every_invocation() {
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
        let input = vec![Domain::Half.encode(2_048); 64];
        let answer = reference(&program(Domain::Half, 32, Vec::new(), Finish::Max), &input);

        assert!(answer.exact, "2048 is exactly representable in a half");
        assert_eq!(answer.values[0], Domain::Half.encode(2_048));
    }

    #[test]
    fn a_value_past_the_limit_is_not_exact() {
        let input = vec![Domain::Half.encode(4_000); 64];
        let answer = reference(&program(Domain::Half, 32, Vec::new(), Finish::Max), &input);

        assert!(!answer.exact, "4000 is not an integer a half can hold");
    }

    #[test]
    fn a_step_that_leaves_the_range_is_noticed_even_though_the_input_was_inside_it() {
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
