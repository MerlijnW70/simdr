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
mod plan;

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
        let mut modules = Vec::new();
        for step in folds(input.len()) {
            let words = kernels::fold_by(width, step.factor, step.stride).map_err(Error::Emit)?;
            modules.push((words, step.workgroups));
        }
        // The last pass crosses between the subgroups of the final workgroup, through shared
        // memory and a barrier, so every one of its invocations holds the whole answer.
        let finisher = kernels::workgroup_sum::<F32>(width).map_err(Error::Emit)?;
        modules.push((finisher, 1));

        let passes: Vec<Pass<'_>> = modules
            .iter()
            .map(|(words, workgroups)| Pass::new(words, *workgroups))
            .collect();

        let words: Vec<u32> = input.iter().map(|value| value.to_bits()).collect();
        // One word home, not the buffer: the answer is a single number and the rest is the last
        // fold's leftovers. Reading it all was 37% of the call — see `notes/FINDINGS.md`.
        let output = self.run_chain_head(&passes, &words, 1)?;

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

/// One folding pass: how many elements each invocation adds, how far apart they are, and how many
/// workgroups run it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fold {
    /// How many elements each invocation adds together.
    ///
    /// Two, until 2026-08-13. See [`MAX_FOLD`] for why it is sixteen where it can be.
    pub factor: u32,
    /// The offset between those elements — and how many invocations there are, and how many
    /// elements the pass leaves behind. All three are the same number.
    pub stride: u32,
    /// Workgroups to dispatch, which is `stride / WORKGROUP_SIZE` and nothing else.
    pub workgroups: u32,
}

/// The most elements one invocation will add in a single pass.
///
/// Halving takes `log₂(N/64)` passes to reduce N elements; folding by sixteen takes a quarter as
/// many. Over 2²⁰ elements that is **five dispatches instead of fifteen**.
///
/// # What it is worth, which is less than it looks
///
/// Measured, paired against the halving build: **~8%** off `Reducer::sum` at 2²⁰ (~442 → ~407 µs)
/// and nothing at all at 8 192, where the chain was short already. `Gpu::sum` gains more — ~6% —
/// and for a different reason: it builds a pipeline per pass, so ten fewer passes is ten fewer
/// pipelines.
///
/// Two arguments for this change were *wrong*, and both were wrong in the optimistic direction:
///
/// - **"It halves the memory traffic."** True as a ratio and irrelevant as a duration. Halving
///   reads ~2N and this reads ~1.07N, but the difference is one buffer's worth — 4 MB at 2²⁰,
///   which is about **6 µs** of bandwidth. The first pass reads N either way and dominates both.
/// - **"Ten fewer dispatches at ~15 µs each is ~150 µs."** That per-step figure comes from a chain
///   of *empty* kernels with nothing to overlap, and it overestimates a real chained step by
///   about four times. The passes removed are the tail, where the dispatches are tiny.
///
/// Sixteen rather than more because the passes have to keep landing on whole workgroups: the last
/// fold has to leave exactly [`WORKGROUP_SIZE`] elements for the finisher, and a larger factor
/// overshoots that sooner and spends the difference on a narrower final fold anyway.
pub const MAX_FOLD: u32 = 16;

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
    let mut remaining = length;

    // Down to exactly one workgroup, which is what the finisher takes. `remaining` starts a power
    // of two of at least two workgroups and every factor is a power of two that divides it, so it
    // stays one and the loop ends on `WORKGROUP_SIZE` rather than stepping past it.
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

/// The widest fold that still leaves a whole workgroup behind.
///
/// A fold by `f` leaves `remaining / f`, and that has to be at least one workgroup — so `f` can be
/// at most **how many workgroups there are**, and at most [`MAX_FOLD`].
///
/// `remaining` is a power of two of more than one workgroup whenever [`folds`] asks, so `groups` is
/// a power of two of at least two: there is no rounding to do and no floor to apply.
///
/// This was a loop that started at `MAX_FOLD` and halved `while factor > 2 && …`. The guard could
/// never fire — at a factor of two the condition beside it is already false, because `remaining` is
/// never less than two workgroups — so the mutation gate reported it as an equivalent mutant, and
/// it was right. Saying the bound directly removes the loop as well as the unfalsifiable branch.
fn fold_factor(remaining: usize) -> u32 {
    let groups = (remaining / WORKGROUP_SIZE as usize) as u32;
    groups.min(MAX_FOLD)
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

    /// Every length the reduction accepts, up to 2²⁴.
    fn lengths() -> impl Iterator<Item = usize> {
        (7..=24).map(|power| 1_usize << power)
    }

    #[test]
    fn a_buffer_of_two_workgroups_folds_once_and_finishes() {
        // 128 → one fold of two → 64 elements → the workgroup pass. Two is the only factor that
        // fits here: anything wider would leave less than a workgroup.
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
        // The mutation that survived once: `stride / WORKGROUP_SIZE` changed to `stride *`
        // over-dispatches by four thousand times and still gives the right answer, because the
        // extra invocations write past the buffer and are discarded. Only the clock notices, and
        // no test watches it.
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
        // The property that makes the answer right rather than merely plausible: the first fold
        // reads every element, each fold after it reads exactly what the one before left, and the
        // last hands the finisher the one workgroup it expects. A chain that lost a level would
        // return a partial sum, which looks like an answer.
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
        // Narrower than it needs to be is a longer chain; wider is a fold that leaves less than a
        // workgroup for the next dispatch, which cannot be dispatched at all.
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
        // The reason for [`MAX_FOLD`], stated as arithmetic rather than as a timing. Halving reads
        // `2h` at every level for h = N/2, N/4, …, which comes to very nearly 2N; folding by
        // sixteen reads N + N/16 + N/256 + …, which is a little over N.
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
        // It used to be one more dispatch per doubling. Four doublings now buy one, because each
        // fold takes sixteen elements to one — so a buffer sixteen times larger costs one pass
        // more rather than four.
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
        // The pass count is logarithmic, so the chain stays short however large the buffer gets —
        // and the base of that logarithm is `MAX_FOLD` rather than two, which is the difference
        // between five dispatches and fifteen. This asserted 15 while the folds halved.
        assert_eq!(dispatches_for(1 << 20), 5);
        assert!(dispatches_for(1 << 30) < 12, "still one command buffer");
    }
}
