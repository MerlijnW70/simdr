//! Random lane programs, run on the device and checked against a CPU reference.
//!
//! Every other test in this repository was written by someone who already had an idea of what
//! could go wrong. This one has no idea: it generates a program, works out what the answer must
//! be by interpreting the same program on the CPU, and compares. What it finds is what nobody
//! thought to look for.
//!
//! # Why this can be exact
//!
//! Floating-point addition is not associative, and a subgroup reduction combines lanes in an
//! order the specification does not fix — so comparing arbitrary `f32` sums exactly would be
//! comparing against one arbitrary order.
//!
//! The integer domains have no such problem: addition and multiplication are associative and
//! commutative modulo their width, and wrapping is defined. The float domain earns the same
//! property by generating only small integers, which are exact in an `f32` and stay exact through
//! sums that remain below 2²⁴ — see [`Domain::ceiling`].
//!
//! **`f16` is fuzzed too**, which this paragraph denied for longer than it was true. A half is
//! exact only to 2048 and a sum over sixty-four lanes leaves that range at once — so a round that
//! leaves it is *refused* rather than compared, which is what [`Reference::exact`] is for. Two
//! rounds in 256 are typically refused that way; the rest are compared exactly like any other
//! domain.
//!
//! # What is generated
//!
//! Straight-line programs over one loaded vector and a handful of constants: elementwise
//! arithmetic, comparisons and selects, subgroup shuffles, and one reduction or scan at the end.
//!
//! **And control flow, since 2026-08-11.** `Op::RepeatAdd` and `Op::RolledAdd` do the same
//! arithmetic through an unrolled loop and a real four-block one, so the pair must agree while
//! only one of them has a back edge. `Op::RolledCounterAdd` reads the loop's counter phi.
//! `Finish::SumOrMax` carries a value out of a branch through an `OpPhi` — the failure mode no
//! other layer here catches, because a phi naming the wrong predecessor validates cleanly and then
//! computes the wrong thing.
//!
//! Until that day this module's vocabulary predated three passes of emitter work, and "30 000
//! programs, zero disagreements" was a true statement about the wrong surface.
//!
//! **And the narrow integers, since 2026-08-12.** `i8`, `u8`, `i16` and `u16` reach different
//! conversion and extreme instructions from the same source, and had direct device tests and no
//! fuzzing. The buffer is where they differ from everything else here — four 8-bit elements share
//! a word — and [`check`] is the one place that packs and unpacks.
//!
//! **And the scans, since 2026-08-14.** [`Finish::Scan`] and [`Finish::ScanExclusive`] are the
//! first finishes that keep *every* element rather than combining them, and they are here for a
//! reason the reductions illustrate: a reduction combines the same set whatever order the lanes
//! are in, so a mapping that pairs the wrong lanes still returns the right total. That is how
//! `reduce_min` came to fold its strips with a maximum and agree with every hand-written test but
//! the strip-mined one.
//!
//! A scan cannot hide that. Its answer at position `j` depends on exactly which elements the
//! hardware considers to come before `j`, so the reference has to model the **lane order** and not
//! only the arithmetic — see [`interpret`]. Until this, every test of the scan was hand-written,
//! which is the state the reduction was in when the fuzzer found that bug.

mod domain;
mod generate;
mod interpret;
mod program;

pub use domain::{ALL_DOMAINS, Domain};
pub use generate::{Rng, generate};
pub use interpret::reference;
pub use program::{Finish, Op, Program};

use crate::{Error, Gpu};
use simdr::lanes::LaneError;

/// What a single fuzzing round concluded.
#[derive(Debug)]
pub enum Outcome {
    /// The device and the reference agreed.
    Agreed,
    /// They did not, and here is the program that separated them.
    Disagreed {
        /// The program, so it can be minimised and re-run.
        program: Program,
        /// What the reference said, for the first few lanes.
        expected: Vec<u32>,
        /// What the device said.
        actual: Vec<u32>,
        /// The first index where they differ.
        at: usize,
    },
    /// The program could not be built for this device — a lane count with no mapping, say.
    ///
    /// Not a failure: the generator is allowed to ask for things the mapping refuses, and being
    /// refused *by name* is the correct answer.
    Refused(LaneError),
    /// The reference left the range its domain counts exactly, so the round was not compared.
    ///
    /// Also not a failure, and for a sharper reason than [`Outcome::Refused`]: the *device* would
    /// have answered perfectly well. It is the comparison that cannot be trusted, because both
    /// sides would be rounded and two roundings agreeing says nothing about the mapping.
    ///
    /// This is what lets [`Domain::Half`] be fuzzed at all — a half counts integers only to 2048,
    /// which a sum over a few hundred lanes leaves at once. A caller sweeping should **count
    /// these**: a domain that is refused every round is a domain with no coverage, and it would
    /// otherwise look exactly like a domain that always agreed.
    Unrepresentable,
}

/// Something that stopped a round before it could conclude.
#[derive(Debug)]
#[non_exhaustive]
pub enum FuzzError {
    /// The dispatch failed.
    Run(Error),
}

impl From<Error> for FuzzError {
    fn from(error: Error) -> Self {
        Self::Run(error)
    }
}

impl core::fmt::Display for FuzzError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Run(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for FuzzError {}

/// Build `program`, run it, and compare against the reference.
///
/// # Errors
///
/// [`FuzzError::Run`] if the dispatch itself fails, which is a broken environment rather than a
/// disagreement.
pub fn check(gpu: &Gpu, program: &Program, input: &[u32]) -> Result<Outcome, FuzzError> {
    let spirv = match program.build() {
        Ok(spirv) => spirv,
        Err(refused) => return Ok(Outcome::Refused(refused)),
    };

    // The reference first, and the dispatch only if it can be believed. A round whose arithmetic
    // left the range its domain counts exactly cannot be compared — both sides would be rounded
    // and agreeing or disagreeing would say nothing about which lanes were combined — so it is
    // refused here rather than dispatched and then loosened.
    let expected = reference(program, input);
    if !expected.exact {
        return Ok(Outcome::Unrepresentable);
    }

    let actual = dispatch(gpu, program, &spirv, input)?;
    Ok(verdict(program, expected.values, actual))
}

/// Run `spirv` over `input`, packing the buffer the way this domain's stride requires.
///
/// Everything above this line works in **element values held one per `u32`**, which is the shape
/// the interpreter and the comparison want. The buffer does not: a domain of 8-bit elements has a
/// stride of one byte, so four elements share a word. This is the only place the two meet, and it
/// is a boundary rather than a special case scattered through the harness.
fn dispatch(
    gpu: &Gpu,
    program: &Program,
    spirv: &[u32],
    input: &[u32],
) -> Result<Vec<u32>, FuzzError> {
    let workgroups = program.workgroups();

    match program.domain.bits() {
        8 => {
            let bytes: Vec<u8> = input.iter().map(|value| *value as u8).collect();
            let out = gpu.run_bytes(spirv, &bytes, workgroups)?;
            Ok(out.into_iter().map(u32::from).collect())
        }
        16 => {
            let halves: Vec<u16> = input.iter().map(|value| *value as u16).collect();
            let out = gpu.run_halves(spirv, &halves, workgroups)?;
            Ok(out.into_iter().map(u32::from).collect())
        }
        _ => Ok(gpu.run_u32(spirv, input, workgroups)?),
    }
}

/// Whether two answers agree, and where they first stop agreeing.
///
/// Split out of [`check`] because a device is not needed to decide it and one *was* needed to
/// reach it. A mutation run found that out: flipping the comparison here left the whole suite
/// green, because `Outcome::Disagreed` is never constructed while everything agrees — so the index
/// it reports had nothing checking it, and a fuzzing failure would have pointed at the wrong
/// element.
fn verdict(program: &Program, expected: Vec<u32>, actual: Vec<u32>) -> Outcome {
    match expected
        .iter()
        .zip(&actual)
        .position(|(left, right)| left != right)
    {
        None => Outcome::Agreed,
        Some(at) => Outcome::Disagreed {
            program: program.clone(),
            expected,
            actual,
            at,
        },
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;

    fn program() -> Program {
        Program {
            domain: Domain::Unsigned,
            subgroup: 32,
            workgroup: 64,
            groups: 1,
            lanes: 32,
            steps: Vec::new(),
            finish: Finish::Sum,
        }
    }

    #[test]
    fn identical_answers_agree() {
        let outcome = verdict(&program(), vec![1, 2, 3], vec![1, 2, 3]);
        assert!(matches!(outcome, Outcome::Agreed));
    }

    #[test]
    fn a_disagreement_reports_the_first_index_that_differs() {
        // The index a failure message points at. Nothing checked it until a mutant flipped the
        // comparison and the suite stayed green.
        let outcome = verdict(&program(), vec![1, 2, 3, 4], vec![1, 2, 9, 4]);

        match outcome {
            Outcome::Disagreed { at, .. } => assert_eq!(at, 2),
            other => panic!("expected a disagreement, got {other:?}"),
        }
    }

    #[test]
    fn the_first_difference_is_reported_and_not_the_last() {
        let outcome = verdict(&program(), vec![0, 0, 0], vec![7, 0, 7]);

        match outcome {
            Outcome::Disagreed { at, .. } => assert_eq!(at, 0),
            other => panic!("expected a disagreement, got {other:?}"),
        }
    }

    #[test]
    fn a_shorter_answer_agrees_over_the_part_that_exists() {
        // `zip` stops at the shorter side, which is deliberate: the output buffer is often longer
        // than the dispatch wrote, and the tail is whatever the upload left there.
        let outcome = verdict(&program(), vec![1, 2], vec![1, 2, 999]);
        assert!(matches!(outcome, Outcome::Agreed));
    }
}
