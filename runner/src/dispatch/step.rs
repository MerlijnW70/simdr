//! What a chain runs, and how much each pass is handed from the one before it.
//!
//! Split from [`super::chain`], which records and submits. This file decides *what* to record and
//! contains no `unsafe`, which is the difference that matters: `tests/integrity.rs` excuses the
//! FFI half of `runner` from mutation because a mutant there kills the process rather than failing
//! a test, and a wrong copy length is not that — it is a wrong number, and it belongs inside the
//! gate.
//!
//! The same seam was found once before, in `dispatch.rs`, and 200 lines of pure conversion turned
//! out to have been sitting behind a blanket FFI exemption. This is the second time that split has
//! been worth making, which is why it was made before waiting to be told.

/// One dispatch of a chain.
#[derive(Debug, Clone, Copy)]
pub struct Pass<'words> {
    /// The module to run.
    pub spirv: &'words [u32],
    /// How many workgroups of it.
    pub workgroups: u32,
    /// How many words this pass writes, or `None` for "assume the whole buffer".
    ///
    /// Only the copy that feeds the *next* pass reads this, so the last pass's value is never
    /// used. `None` is the safe answer and the default: copying more than the next pass reads is
    /// wasted work, and copying less is a wrong answer.
    pub outputs: Option<usize>,
}

impl<'words> Pass<'words> {
    /// A pass running `workgroups` groups of `spirv`, writing an unknown amount.
    ///
    /// The whole buffer is copied to the next pass. Correct for any kernel, and wasteful for one
    /// that writes a prefix — see [`Pass::writing`].
    #[must_use]
    pub const fn new(spirv: &'words [u32], workgroups: u32) -> Self {
        Self {
            spirv,
            workgroups,
            outputs: None,
        }
    }

    /// The same, saying how many words the pass writes.
    ///
    /// Only that many are handed to the pass after it. **A count smaller than what the next pass
    /// reads is a wrong answer, not a slower one** — the tail it reads would hold whatever the
    /// source buffer held before, which for a repeated call is the previous call's data.
    ///
    /// For a halving fold the count is the fold's own half: a pass folding `2h` elements into `h`
    /// writes `h`, and the pass after it reads `in[j]` and `in[j + h/2]` for `j < h/2` — every one
    /// of them inside the first `h`.
    #[must_use]
    pub const fn writing(spirv: &'words [u32], workgroups: u32, outputs: usize) -> Self {
        Self {
            spirv,
            workgroups,
            outputs: Some(outputs),
        }
    }
}

/// One recorded step: a dispatch, and what to hand it from the pass before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Step {
    /// Workgroups to dispatch.
    pub(crate) workgroups: u32,
    /// Bytes to copy from destination back into source *before* this dispatch.
    ///
    /// Zero on the first step, which has no predecessor — and zero is also how "copy nothing" is
    /// spelled, so [`crate::Gpu::replay`] does not have to special-case the index.
    pub(crate) copy_bytes: u64,
}

impl Step {
    /// The steps for `passes` over a buffer of `bytes`.
    ///
    /// A pass that did not say how much it writes gets the whole buffer copied, which is what every
    /// pass got before [`Pass::writing`] existed.
    pub(crate) fn plan(passes: &[Pass<'_>], bytes: u64) -> Vec<Self> {
        let mut steps = Vec::with_capacity(passes.len());
        let mut previous: Option<&Pass<'_>> = None;

        for pass in passes {
            steps.push(Self {
                workgroups: pass.workgroups,
                copy_bytes: previous.map_or(0, |before| copy_for(before.outputs, bytes)),
            });
            previous = Some(pass);
        }
        steps
    }
}

/// How many bytes to copy on behalf of a pass that wrote `outputs` words.
///
/// Floored at one word and capped at `bytes`: a zero-length `vkCmdCopyBuffer` is not allowed, and
/// a length past the end of the allocation is undefined rather than merely wrong.
fn copy_for(outputs: Option<usize>, bytes: u64) -> u64 {
    match outputs {
        None => bytes,
        Some(words) => ((words.max(1) * size_of::<u32>()) as u64).min(bytes),
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;

    /// Words enough for the tests below to have somewhere to be capped against.
    const BYTES: u64 = 4096;

    /// A pass over an empty module, which none of these dispatch.
    fn pass(outputs: Option<usize>) -> Pass<'static> {
        match outputs {
            None => Pass::new(&[], 1),
            Some(words) => Pass::writing(&[], 1, words),
        }
    }

    #[test]
    fn the_first_step_copies_nothing_because_nothing_ran_before_it() {
        let steps = Step::plan(&[pass(Some(16)), pass(Some(8))], BYTES);

        assert_eq!(steps[0].copy_bytes, 0);
        assert_eq!(
            steps[1].copy_bytes, 64,
            "sixteen words, from the pass before"
        );
    }

    #[test]
    fn each_step_is_handed_what_the_pass_before_it_wrote_and_not_its_own_count() {
        // The mistake that would still run: reading this pass's `outputs` instead of the previous
        // one's. Every count below is different, so the two readings cannot coincide.
        let steps = Step::plan(&[pass(Some(64)), pass(Some(32)), pass(Some(16))], BYTES);

        assert_eq!(steps[1].copy_bytes, 256, "the first pass wrote 64 words");
        assert_eq!(steps[2].copy_bytes, 128, "the second wrote 32");
    }

    #[test]
    fn a_pass_that_says_nothing_hands_on_the_whole_buffer() {
        let steps = Step::plan(&[pass(None), pass(Some(8))], BYTES);

        assert_eq!(steps[1].copy_bytes, BYTES);
    }

    #[test]
    fn a_count_larger_than_the_buffer_is_capped_rather_than_read_past_the_end() {
        let steps = Step::plan(&[pass(Some(1_000_000)), pass(None)], BYTES);

        assert_eq!(steps[1].copy_bytes, BYTES);
    }

    #[test]
    fn a_count_of_zero_becomes_one_word_because_an_empty_copy_is_not_allowed() {
        // Vulkan refuses a zero-size `vkCmdCopyBuffer`, and `copy_bytes == 0` already means "no
        // copy at all" here — so a pass claiming to write nothing must not collide with that.
        let steps = Step::plan(&[pass(Some(0)), pass(None)], BYTES);

        assert_eq!(steps[1].copy_bytes, 4);
        assert_ne!(steps[1].copy_bytes, 0, "that would skip the copy entirely");
    }

    #[test]
    fn every_step_keeps_its_own_pass_workgroup_count() {
        let passes = [
            Pass::writing(&[], 7, 16),
            Pass::writing(&[], 3, 8),
            Pass::new(&[], 1),
        ];
        let steps = Step::plan(&passes, BYTES);

        assert_eq!(
            steps.iter().map(|step| step.workgroups).collect::<Vec<_>>(),
            vec![7, 3, 1]
        );
    }

    #[test]
    fn a_chain_of_one_is_one_step_that_copies_nothing() {
        let steps = Step::plan(&[pass(Some(16))], BYTES);

        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].copy_bytes, 0);
    }

    #[test]
    fn no_passes_is_no_steps() {
        assert!(Step::plan(&[], BYTES).is_empty());
    }
}
