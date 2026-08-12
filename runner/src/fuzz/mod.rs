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
//! order the specification does not fix — so comparing `f32` sums exactly would be comparing
//! against one arbitrary order. Every program here is therefore over **`u32`**, where addition
//! and multiplication are associative and commutative modulo 2³² and the answer does not depend
//! on the order at all. That is what makes an exact comparison legitimate rather than lucky, and
//! it is why a float mode would need a tolerance and a much more careful argument.
//!
//! # What is generated
//!
//! Straight-line programs over one loaded vector and a handful of constants: elementwise
//! arithmetic, comparisons and selects, subgroup shuffles, and one reduction at the end.
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

    let actual = gpu.run_u32(&spirv, input, program.workgroups())?;
    let expected = reference(program, input);

    Ok(verdict(program, expected, actual))
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
