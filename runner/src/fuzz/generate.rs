//! Making programs out of a seed.
//!
//! Deterministic and dependency-free: the seed is the caller's, so a disagreement is reproducible
//! by re-running with the seed that found it.

mod vocabulary;

#[cfg(test)]
mod coverage;

use self::vocabulary::{CLUSTERED, INTEGER_ONLY, Kind, STRIPPED, WHOLE, fill};
use super::domain::Domain;
use super::program::{Finish, Program};
use simdr::lanes::{LaneError, Mapping};

/// A small deterministic generator.
///
/// `SplitMix64`, which is four lines and good enough to pick between a handful of choices.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Start from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next value.
    pub const fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value below `bound`, which must not be zero.
    pub const fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// Generate a program in `domain`, for a device of `subgroup` lanes.
///
/// Constants stay small so that a sum over a few hundred elements stays inside the domain's
/// exact range — see [`Domain::ceiling`], which is the whole reason a float comparison can be
/// exact rather than approximate.
pub fn generate(rng: &mut Rng, domain: Domain, subgroup: u32, workgroup: u32) -> Program {
    const WIDTHS: [u32; 6] = [2, 4, 8, 16, 32, 64];

    let lanes = WIDTHS[rng.below(WIDTHS.len() as u64) as usize];
    let steps_wanted = 1 + rng.below(4) as usize;

    // Which operations are legal depends on the mapping, and the generator respects that rather
    // than leaning on `build` to refuse: a run made mostly of refusals tests very little. Votes and
    // shuffles need a vector at least as wide as the subgroup; the rotate needs one that is exactly
    // as wide or narrower, because over strips it would move elements between them.
    //
    // **Three pools rather than two**, since the rotate arrived: the mapping is a three-way choice
    // and it used to be asked as a yes-or-no. `lanes == subgroup` was the case that had no name.
    //
    // And it is asked **once, of the emitter**. This decided the same relationship twice — the pool
    // three ways and the finish as a yes-or-no beside it — and the mutation gate found the second
    // copy. Merging them left one comparison here, which was still a *third* spelling of a rule
    // `simdr::lanes::Mapping` already owned, and not the same rule: `lanes < subgroup` calls a
    // three-lane vector on a 32-wide subgroup clustered, where divisibility refuses it.
    //
    // So the generator asks the crate under test what the mapping is. A refusal means the draw has
    // no mapping at all, and the pool for that is empty — `build` would refuse it too, so the
    // generator would only be making a round that proves nothing.
    let mapping = Mapping::of(lanes, subgroup);
    let pool = match mapping {
        Ok(Mapping::Clusters { .. }) => CLUSTERED,
        Ok(Mapping::WholeSubgroup) => WHOLE,
        // A vector too wide to hold inline is a strip-mined one that `build` then refuses **by
        // name**, which the sweeps count and print rather than hide — 32 of 256 rounds at a
        // four-wide subgroup, all of them `TooManyStrips`. Both widths here are powers of two, so
        // that is the only refusal `Mapping::of` can return, and dropping the round instead would
        // quietly narrow the sweep at exactly the width that needs it most.
        Ok(Mapping::Strips { .. }) | Err(_) => STRIPPED,
    };
    // **A second axis, and it is not the mapping's.** The pools above say which lanes a vector may
    // read. This says which instructions its *element type* has: `Lanes`' three bit shifts take
    // `T: Integer`, and a float domain has no such instruction at all — not a rounding question but
    // a module `spirv-val` rejects. A shift crosses no lane, so it is legal under every mapping,
    // which is why it is one list beside the pool rather than three entries inside them.
    let integer_only: &[Kind] = if domain.is_float() { &[] } else { INTEGER_ONLY };
    let mut steps = Vec::with_capacity(steps_wanted);

    // Loop trip counts stay small. A rolled loop of four is the same shape as one of four hundred
    // — four blocks and two phis — and the short one leaves the sums well inside the float
    // domain's exact range, which is what lets the comparison be exact at all.
    for _ in 0..steps_wanted {
        // Drawn from the two lists as one, so a shift is as likely as any other step rather than
        // being a coin flip on top of the draw — which would have made it a quarter of every
        // integer program and a fifth of the vocabulary's exposure for everything else.
        let choices = pool.len() + integer_only.len();
        let kind = pool
            .iter()
            .chain(integer_only)
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

/// How the program ends.
///
/// `SumOrMax` needs a vote, which has no clustered form and answers for every vector sharing the
/// subgroup — so it is offered only where the vector is at least as wide as the subgroup, the same
/// rule the shuffles follow.
///
/// **The scans are offered at every mapping, and used not to be.** A clustered scan was
/// `Outcome::Refused` by name, so the rounds that would have exercised the ladder ended in a
/// reduction instead — and the ladder is the most intricate thing in the tree with the least
/// differential coverage. It has all three mappings now: an instruction at the width, a carry
/// between strips above it, and the ladder below.
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
            // Straddling the input's range again, so both arms are reached across a sweep.
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
        // **The two rules were not the same rule.** This file decided the mapping with
        // `lanes < subgroup`, and `Mapping::of` decides it by divisibility — so a seven-lane vector
        // on an eight-wide subgroup was "clustered" here and is *refused* there, and only the
        // generator drawing powers of two kept them agreeing.
        //
        // Asserted rather than assumed, because a fuzzer whose idea of the mapping differs from the
        // emitter's is a fuzzer generating rounds the emitter will refuse, and counting them.
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
        // The pools are chosen from the mapping, so a draw whose mapping this crate refuses would
        // be generated from the strip pool and then refused by `build`. That is deliberate — the
        // sweeps count and print those — but it must be the *only* refusal reachable, and it is
        // reachable only above `MAX_STRIPS`.
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
        // Both read or answer for the whole subgroup, and the lane API refuses them on a vector
        // that shares its lanes. The generator should not be producing refusals on purpose.
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
