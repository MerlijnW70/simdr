mod domain;
mod generate;
mod interpret;
mod program;

pub use domain::{ALL_DOMAINS, BitShift, Domain};
pub use generate::{Rng, generate};
pub use interpret::{Reference, reference};
pub use program::{Emit, Finish, Op, Program, ProgramError};

use crate::{Error, Gpu};

#[derive(Debug)]
pub enum Outcome {
    Agreed,
    Disagreed {
        program: Program,
        expected: Vec<u32>,
        actual: Vec<u32>,
        at: usize,
    },
    Refused(ProgramError),
    Unrepresentable,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum FuzzError {
    Run(Error),
    ShortInput { needed: usize, given: usize },
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
            Self::ShortInput { needed, given } => {
                write!(
                    f,
                    "this program reads {needed} elements and was given {given}"
                )
            }
        }
    }
}

impl std::error::Error for FuzzError {}

pub fn check(gpu: &Gpu, program: &Program, input: &[u32]) -> Result<Outcome, FuzzError> {
    let needed = program.input_len();
    if input.len() < needed {
        return Err(FuzzError::ShortInput {
            needed,
            given: input.len(),
        });
    }

    let spirv = match program.build() {
        Ok(spirv) => spirv,
        Err(refused) => return Ok(Outcome::Refused(refused)),
    };

    let expected = reference(program, input);
    if !expected.exact {
        return Ok(Outcome::Unrepresentable);
    }

    let actual = dispatch(gpu, program, &spirv, input)?;
    Ok(verdict(program, expected.values, actual))
}

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
        let outcome = verdict(&program(), vec![1, 2], vec![1, 2, 999]);
        assert!(matches!(outcome, Outcome::Agreed));
    }
}
