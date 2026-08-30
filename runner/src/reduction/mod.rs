//! ```text
//!   65536 elements   fold  →  32768  →  16384  →  …  →  64      10 dispatches
//!   64 elements      workgroup_sum   →  one number             1 dispatch
//! ```

mod held;
mod plan;

pub use held::Reducer;

use crate::kernels::{self, WORKGROUP_SIZE};
use crate::{Error, Gpu, Pass};
use simdr::lanes::F32;

#[derive(Debug, Clone, PartialEq)]
pub struct Reduction {
    pub total: f32,
    pub dispatches: usize,
    pub host_combined: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadLength {
    NotAPowerOfTwo(usize),
    TooSmall { length: usize, minimum: usize },
}

impl Gpu {
    pub fn sum(&self, input: &[f32]) -> Result<Reduction, Error> {
        let width = self.limits().subgroup_size;
        let minimum = 2 * WORKGROUP_SIZE as usize;

        if !input.len().is_power_of_two() {
            return Err(Error::BadLength(BadLength::NotAPowerOfTwo(input.len())));
        }
        if input.len() < minimum {
            return Err(Error::BadLength(BadLength::TooSmall {
                length: input.len(),
                minimum,
            }));
        }

        let mut modules = Vec::new();
        for step in folds(input.len()) {
            let words = kernels::fold_by(width, step.factor, step.stride).map_err(Error::Emit)?;
            modules.push((words, step.workgroups));
        }
        let finisher = kernels::workgroup_sum::<F32>(width).map_err(Error::Emit)?;
        modules.push((finisher, 1));

        let passes: Vec<Pass<'_>> = modules
            .iter()
            .map(|(words, workgroups)| Pass::new(words, *workgroups))
            .collect();

        let words: Vec<u32> = input.iter().map(|value| value.to_bits()).collect();
        let output = self.run_chain_head(&passes, &words, 1)?;

        let total = output
            .first()
            .copied()
            .map(f32::from_bits)
            .ok_or(Error::NoPipeline)?;

        Ok(Reduction {
            total,
            dispatches: passes.len(),
            host_combined: 1,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fold {
    pub factor: u32,
    pub stride: u32,
    pub workgroups: u32,
}

pub const MAX_FOLD: u32 = 16;

#[must_use]
pub fn folds(length: usize) -> Vec<Fold> {
    let mut steps = Vec::new();
    let mut remaining = length;

    while remaining > WORKGROUP_SIZE as usize {
        let factor = fold_factor(remaining);
        let stride = (remaining / factor as usize) as u32;

        steps.push(Fold {
            factor,
            stride,
            workgroups: stride / WORKGROUP_SIZE,
        });
        remaining = stride as usize;
    }

    steps
}

fn fold_factor(remaining: usize) -> u32 {
    let groups = (remaining / WORKGROUP_SIZE as usize) as u32;
    groups.min(MAX_FOLD)
}

#[must_use]
pub fn dispatches_for(length: usize) -> usize {
    folds(length).len() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lengths() -> impl Iterator<Item = usize> {
        (7..=24).map(|power| 1_usize << power)
    }

    #[test]
    fn a_buffer_of_two_workgroups_folds_once_and_finishes() {
        assert_eq!(dispatches_for(128), 2);
        assert_eq!(
            folds(128),
            vec![Fold {
                factor: 2,
                stride: 64,
                workgroups: 1
            }]
        );
    }

    #[test]
    fn each_fold_dispatches_exactly_the_invocations_it_needs() {
        for length in lengths() {
            for fold in folds(length) {
                assert_eq!(
                    fold.workgroups * WORKGROUP_SIZE,
                    fold.stride,
                    "at {length}: a fold leaving {} dispatched {} workgroups",
                    fold.stride,
                    fold.workgroups
                );
            }
        }
    }

    #[test]
    fn the_chain_consumes_the_whole_buffer_and_leaves_exactly_one_workgroup() {
        for length in lengths() {
            let steps = folds(length);
            let mut remaining = length;

            for fold in &steps {
                assert_eq!(
                    fold.factor as usize * fold.stride as usize,
                    remaining,
                    "at {length}: a fold read {} of {remaining} elements",
                    fold.factor as usize * fold.stride as usize
                );
                remaining = fold.stride as usize;
            }

            assert_eq!(
                remaining, WORKGROUP_SIZE as usize,
                "at {length}: the finisher was handed {remaining} elements"
            );
        }
    }

    #[test]
    fn every_fold_is_the_widest_that_still_leaves_a_whole_workgroup() {
        for length in lengths() {
            for fold in folds(length) {
                assert!(fold.factor <= MAX_FOLD, "at {length}: {fold:?}");
                assert!(fold.factor >= 2, "at {length}: {fold:?}");

                let wider = fold.factor * 2;
                let leaves = fold.factor as usize * fold.stride as usize / wider as usize;
                assert!(
                    fold.factor == MAX_FOLD || leaves < WORKGROUP_SIZE as usize,
                    "at {length}: {fold:?} could have folded by {wider} and left {leaves}"
                );
            }
        }
    }

    #[test]
    fn a_wider_fold_reads_the_buffer_about_once_where_halving_read_it_twice() {
        let length = 1_usize << 20;
        let read: usize = folds(length)
            .iter()
            .map(|fold| fold.factor as usize * fold.stride as usize)
            .sum();

        assert!(
            read < length * 6 / 5,
            "the chain reads {read} for {length} elements, which is not much better than halving"
        );
        assert!(read >= length, "it has to read every element at least once");
    }

    #[test]
    fn a_buffer_that_is_already_one_workgroup_needs_no_folds() {
        assert!(folds(64).is_empty());
        assert_eq!(dispatches_for(64), 1);
    }

    #[test]
    fn the_chain_grows_with_the_logarithm_of_the_input_and_not_with_its_length() {
        assert_eq!(
            dispatches_for(1 << 20),
            5,
            "2^20: four folds and a finisher"
        );
        assert_eq!(dispatches_for(1 << 24), 6, "sixteen times the elements");

        let mut previous = 0;
        for length in lengths() {
            let next = dispatches_for(length);
            assert!(
                next >= previous,
                "at {length}: a larger buffer took fewer dispatches"
            );
            previous = next;
        }
    }

    #[test]
    fn a_million_elements_take_a_handful_of_dispatches() {
        assert_eq!(dispatches_for(1 << 20), 5);
        assert!(dispatches_for(1 << 30) < 12, "still one command buffer");
    }
}
