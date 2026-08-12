//! A full-buffer sum: the first thing here that is an *algorithm* rather than an operation.
//!
//! Everything else in this crate runs one dispatch and checks one answer. A reduction over more
//! elements than a workgroup holds cannot work that way — Vulkan has no barrier across a dispatch,
//! so the only way one workgroup reads another's output is a second dispatch. That makes this the
//! test of whether the pieces compose: an emitter that builds a different module per step, a
//! [`crate::Pass`] chain that keeps the words on the device between them, and arithmetic that has
//! to agree with a CPU sum to the last bit.
//!
//! # The shape
//!
//! ```text
//!   65536 elements   fold  →  32768  →  16384  →  …  →  64      10 dispatches
//!   64 elements      workgroup_sum   →  one number             1 dispatch
//! ```
//!
//! The folding half is `out[i] = in[i] + in[i + half]`, dispatched at exactly `half` invocations
//! so nothing needs a bounds test — see [`crate::kernels::fold_halves`]. It stops at one workgroup
//! because that is the smallest dispatch there is.
//!
//! # It used to end on the host, and no longer does
//!
//! Until 2026-08-12 the last pass was a *subgroup* reduction, so the final workgroup produced one
//! total per subgroup — two on a 32-wide device — and the host added them. That was recorded as a
//! real boundary rather than hidden, because combining two subgroups needs shared memory and a
//! barrier and the emitter had neither.
//!
//! It has them now. [`crate::kernels::workgroup_sum`] hands every invocation of the final
//! workgroup the whole total, and the host reads one number it did not compute any part of.
//!
//! # Once, or repeatedly
//!
//! [`Gpu::sum`] builds every pipeline it needs and destroys them again, which is right for a
//! caller that asks once and wrong for one that asks in a loop. [`Reducer`] is the same reduction
//! with the pipelines and the buffers held between calls — 5.0× over 8 192 elements, measured in
//! `runner/examples/reducer.rs`.
//!
//! # Why the answer is exact
//!
//! Floating-point addition is not associative, so a GPU reduction and a CPU one agree only if
//! every partial sum is representable. The tests feed values small enough that every intermediate
//! stays inside the 24 bits an `f32` carries, which makes an exact comparison legitimate rather
//! than lucky. Feed it larger values and the right comparison is a tolerance; that is the caller's
//! judgement and this module does not make it for them.

mod held;

pub use held::Reducer;

use crate::kernels::{self, WORKGROUP_SIZE};
use crate::{Error, Gpu, Pass};
use simdr::lanes::F32;

/// A finished reduction, and what it cost.
#[derive(Debug, Clone, PartialEq)]
pub struct Reduction {
    /// The sum.
    pub total: f32,
    /// How many dispatches ran, all in one submission.
    pub dispatches: usize,
    /// How many values the host had to combine to get [`Reduction::total`].
    ///
    /// **One**, meaning none: the device produced the answer and the host read it. This is
    /// reported rather than assumed because it was two until shared memory arrived, and a
    /// reduction that quietly finishes on the CPU is a different claim from one that does not.
    pub host_combined: usize,
}

/// Why a buffer cannot be reduced by this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadLength {
    /// Not a power of two.
    ///
    /// Every fold halves, so anything else would leave an odd element with no partner — and
    /// padding it here would hide the fact from a caller who may prefer to pad differently.
    NotAPowerOfTwo(usize),
    /// Smaller than the one workgroup the final pass needs.
    TooSmall {
        /// What was passed.
        length: usize,
        /// The smallest this accepts.
        minimum: usize,
    },
}

impl Gpu {
    /// Sum every element of `input`, in one submission of however many dispatches it takes.
    ///
    /// `input.len()` must be a power of two and at least `2 × WORKGROUP_SIZE`. Both are checked,
    /// because a reduction that silently dropped a tail would return a plausible wrong number.
    ///
    /// # Errors
    ///
    /// [`Error::BadLength`] if the buffer is not a shape this can fold, [`Error::Emit`] if a pass
    /// cannot be built, otherwise as [`Gpu::run`].
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

        // One module per fold, because the offset is a build-time constant. Built up front so a
        // failure happens before anything is submitted.
        // The third element is how many words each pass writes, which is what the pass after it
        // has to be handed. A fold of `2h` into `h` writes `h`; without it the chain copies the
        // whole buffer between every pass, which `notes/FINDINGS.md` measured at a fifth of a
        // large reduction.
        let mut modules = Vec::new();
        for step in folds(input.len()) {
            let words = kernels::fold_halves(width, step.half).map_err(Error::Emit)?;
            modules.push((words, step.workgroups, step.half as usize));
        }
        // The last pass crosses between the subgroups of the final workgroup, through shared
        // memory and a barrier, so every one of its invocations holds the whole answer. Nothing
        // follows it, so what it writes is never copied anywhere — the count is the workgroup it
        // filled, stated rather than left to a reader to wonder about.
        let finisher = kernels::workgroup_sum::<F32>(width).map_err(Error::Emit)?;
        modules.push((finisher, 1, WORKGROUP_SIZE as usize));

        let passes: Vec<Pass<'_>> = modules
            .iter()
            .map(|(words, workgroups, outputs)| Pass::writing(words, *workgroups, *outputs))
            .collect();

        let words: Vec<u32> = input.iter().map(|value| value.to_bits()).collect();
        let output = self.run_chain(&passes, &words)?;

        // One number, read rather than assembled. Every invocation of the final workgroup wrote
        // the same total, so any of them would do; slot zero is the obvious one.
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

/// One halving pass: how far apart the two operands are, and how many workgroups run it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fold {
    /// The offset between the two elements each invocation adds — and how many invocations there
    /// are, since the dispatch is sized to exactly that.
    pub half: u32,
    /// Workgroups to dispatch, which is `half / WORKGROUP_SIZE` and nothing else.
    pub workgroups: u32,
}

/// The folding passes for a buffer of `length`, largest first.
///
/// **Separated from `Gpu::sum` so the arithmetic is testable without a device**, and it needed to
/// be: a mutation run changed `half / WORKGROUP_SIZE` to `half *` and the whole suite stayed
/// green. Over-dispatching by four thousand times still produces the right answer — the extra
/// invocations write past the buffer and are discarded — so nothing but the wall clock noticed,
/// and the reduction tests do not look at that.
///
/// A wrong dispatch size is invisible in the answer and enormous in the cost. That is exactly the
/// shape of thing that needs pinning rather than inferring.
#[must_use]
pub fn folds(length: usize) -> Vec<Fold> {
    let mut steps = Vec::new();
    let mut remaining = length / 2;

    while remaining >= WORKGROUP_SIZE as usize {
        let half = remaining as u32;
        steps.push(Fold {
            half,
            workgroups: half / WORKGROUP_SIZE,
        });
        remaining /= 2;
    }

    steps
}

/// How many dispatches [`Gpu::sum`] will run for a buffer of `length`.
///
/// The folds, plus the one workgroup reduction that finishes.
#[must_use]
pub fn dispatches_for(length: usize) -> usize {
    folds(length).len() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_of_two_workgroups_folds_once_and_finishes() {
        // 128 → fold at 64 → 64 elements → the workgroup pass.
        assert_eq!(dispatches_for(128), 2);
        assert_eq!(
            folds(128),
            vec![Fold {
                half: 64,
                workgroups: 1
            }]
        );
    }

    #[test]
    fn each_fold_dispatches_exactly_the_invocations_it_needs() {
        // The mutation that survived: `half / WORKGROUP_SIZE` changed to `half *` over-dispatches
        // by four thousand times and still gives the right answer, because the extra invocations
        // write past the buffer and are discarded. Only the clock notices, and no test watches it.
        for fold in folds(65_536) {
            assert_eq!(
                fold.workgroups * WORKGROUP_SIZE,
                fold.half,
                "a fold of {} dispatched {} workgroups",
                fold.half,
                fold.workgroups
            );
        }
    }

    #[test]
    fn the_folds_halve_from_half_the_buffer_down_to_one_workgroup() {
        let steps = folds(1_024);
        let halves: Vec<u32> = steps.iter().map(|fold| fold.half).collect();

        assert_eq!(halves, vec![512, 256, 128, 64]);
        assert_eq!(
            steps.last().map(|fold| fold.workgroups),
            Some(1),
            "the last fold is one workgroup, which is the smallest dispatch there is"
        );
    }

    #[test]
    fn a_buffer_that_is_already_one_workgroup_needs_no_folds() {
        assert!(folds(64).is_empty());
        assert_eq!(dispatches_for(64), 1);
    }

    #[test]
    fn each_doubling_of_the_input_costs_exactly_one_more_dispatch() {
        let mut previous = dispatches_for(128);
        for power in 8..=20 {
            let next = dispatches_for(1 << power);
            assert_eq!(next, previous + 1, "at 2^{power}");
            previous = next;
        }
    }

    #[test]
    fn a_million_elements_take_fewer_than_twenty_dispatches() {
        // The point of halving: the pass count is logarithmic, so the chain stays short enough to
        // fit one command buffer however large the buffer gets.
        assert_eq!(dispatches_for(1 << 20), 15);
    }
}
