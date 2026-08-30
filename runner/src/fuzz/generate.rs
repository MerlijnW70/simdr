mod vocabulary;

#[cfg(test)]
mod coverage;

use self::vocabulary::{CLUSTERED, Kind, STRIPPED, WHOLE, by_element, fill};
use super::domain::Domain;
use super::program::{Finish, Program};
use simdr::lanes::{LaneError, Mapping};

#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub const fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub const fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

pub fn generate(rng: &mut Rng, domain: Domain, subgroup: u32, workgroup: u32) -> Program {
    const WIDTHS: [u32; 6] = [2, 4, 8, 16, 32, 64];

    let lanes = WIDTHS[rng.below(WIDTHS.len() as u64) as usize];
    let steps_wanted = 1 + rng.below(4) as usize;

    let mapping = Mapping::of(lanes, subgroup);
    let pool = match mapping {
        Ok(Mapping::Clusters { .. }) => CLUSTERED,
        Ok(Mapping::WholeSubgroup) => WHOLE,
        Ok(Mapping::Strips { .. }) | Err(_) => STRIPPED,
    };
    let by_type = by_element(domain);
    let mut steps = Vec::with_capacity(steps_wanted);

    for _ in 0..steps_wanted {
        let choices = pool.len() + by_type.len();
        let kind = pool
            .iter()
            .chain(by_type)
            .nth(rng.below(choices as u64) as usize)
            .copied()
            .unwrap_or(Kind::AddConstant);

        steps.push(fill(rng, domain, subgroup, lanes, kind));
    }

    Program {
        domain,
        subgroup,
        workgroup,
        groups: 1 + rng.below(2) as u32,
        lanes,
        steps,
        finish: finish(rng, domain, mapping),
    }
}

fn finish(rng: &mut Rng, domain: Domain, mapping: Result<Mapping, LaneError>) -> Finish {
    match rng.below(if matches!(mapping, Ok(Mapping::Clusters { .. })) {
        5
    } else {
        6
    }) {
        0 => Finish::Sum,
        1 => Finish::Max,
        2 => Finish::Min,
        3 => Finish::Scan,
        4 => Finish::ScanExclusive,
        _ => Finish::SumOrMax {
            when_any_above: rng.below(u64::from(domain.ceiling())) as u32,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzz::Op;

    #[test]
    fn the_generator_takes_its_mapping_from_the_crate_under_test() {
        assert_eq!(Mapping::of(4, 8), Ok(Mapping::Clusters { size: 4 }));
        assert_eq!(
            Mapping::of(8, 8),
            Ok(Mapping::WholeSubgroup),
            "equal widths are a whole subgroup, not a cluster of the same size"
        );
        assert_eq!(Mapping::of(16, 8), Ok(Mapping::Strips { count: 2 }));
        assert!(
            Mapping::of(7, 8).is_err(),
            "a width that neither divides nor multiplies is refused, where a comparison called it \
             clustered"
        );
    }

    #[test]
    fn every_width_the_generator_draws_reaches_a_pool_the_emitter_agrees_with() {
        for subgroup in [4_u32, 8, 16, 32, 64] {
            for lanes in [2_u32, 4, 8, 16, 32, 64] {
                match Mapping::of(lanes, subgroup) {
                    Ok(_) => {}
                    Err(LaneError::TooManyStrips { .. }) => {
                        assert!(
                            lanes > subgroup,
                            "only a wide vector can have too many strips"
                        );
                    }
                    Err(other) => assert!(
                        matches!(other, LaneError::TooManyStrips { .. }),
                        "{lanes} lanes on {subgroup} is refused as {other}, which the pools have \
                         no arm for"
                    ),
                }
            }
        }
    }

    #[test]
    fn the_generator_is_deterministic_in_its_seed() {
        let first = generate(&mut Rng::new(42), Domain::Unsigned, 32, 64);
        let second = generate(&mut Rng::new(42), Domain::Unsigned, 32, 64);

        assert_eq!(first, second, "a finding has to be reproducible");
    }

    #[test]
    fn the_domain_reaches_the_program() {
        let floats = generate(&mut Rng::new(1), Domain::Float, 32, 64);
        let integers = generate(&mut Rng::new(1), Domain::Unsigned, 32, 64);

        assert_eq!(floats.domain, Domain::Float);
        assert_eq!(integers.domain, Domain::Unsigned);
    }

    #[test]
    fn different_seeds_give_different_programs() {
        let mut seen = Vec::new();
        for seed in 0..16 {
            seen.push(generate(&mut Rng::new(seed), Domain::Unsigned, 32, 64));
        }
        seen.dedup();

        assert!(seen.len() > 8, "the generator is not stuck");
    }

    #[test]
    fn a_clustered_program_never_asks_for_a_shuffle_or_a_vote() {
        for seed in 0..64 {
            let program = generate(&mut Rng::new(seed), Domain::Float, 32, 64);
            if program.lanes >= 32 {
                continue;
            }
            assert!(
                !program.steps.iter().any(|step| matches!(
                    step,
                    Op::ButterflyAdd(_) | Op::ShiftUp | Op::AddIfAnyAbove { .. }
                )),
                "seed {seed} put a subgroup-wide operation in a clustered program"
            );
        }
    }

    #[test]
    fn every_generated_program_builds_in_both_domains() {
        for domain in [Domain::Unsigned, Domain::Float] {
            for seed in 0..128 {
                let program = generate(&mut Rng::new(seed), domain, 32, 64);
                assert!(
                    program.build().is_ok(),
                    "seed {seed} in {domain:?} produced a program the emitter refused: {program:?}"
                );
            }
        }
    }
}
