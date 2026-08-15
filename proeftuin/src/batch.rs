//! One module, N independent problems, one dispatch — and the four gates in front of it.
//!
//! Every tool here had the same twenty lines: build, ask the device whether it supports what the
//! module declares, validate, dispatch, and answer four different ways when any of those says no.
//! Three copies, three near-identical outcome enums, and three spellings of the one `match` that
//! picks between `run_u32`, `run_bytes` and `run_halves` by element width.
//!
//! # Why the batching half is here rather than in the engine
//!
//! `decisions/DR-0008` measured what a caller waits for: a round trip is ~100 µs on the discrete
//! device and the device's own share of it is **2.9%**. Raising the answers per round trip from 2
//! to 2 048 is 800×; no kernel change recorded anywhere in `notes/NEXT.md` has been worth 3×.
//!
//! `notes/NEXT.md` has also refused three times to *invent* a batching API, each time for the same
//! reason: it needs a caller. This directory turned out to be one, and not in the flattering way.
//! Of its three tools:
//!
//! * the half sweep already dispatches all 65 536 patterns at once, and had that shape by accident
//!   rather than by design;
//! * the quantised layer ran **seventy-two** dispatches where twelve would do, because the seeds
//!   vary the *data* and the module is the same;
//! * the conversions ran **seventy-two** where six would do, and cannot be fixed the same way —
//!   see below, because the exception is as informative as the rule.
//!
//! So the pressure was real and it was a mistake, twice, in the directory built to find mistakes.
//!
//! # What a batch is, exactly
//!
//! **N problems laid out so that the invocation's own index selects the problem.** That is the
//! whole of it, and it is a constraint on the *kernel* rather than on the buffer: a module that
//! reaches its second operand at a constant offset sized for one workgroup cannot be dispatched
//! over two, because workgroup 1 would read workgroup 0's activations. The offset has to be the
//! batch's, which is why [`Batch::second_operand`] exists and why the layer takes it as an argument
//! rather than computing it.
//!
//! The conversions are the case this does not fit. Their probe value is a *constant in the module*
//! — that is what makes the twelve boundaries twelve modules — so batching them would mean loading
//! the probe from a buffer, which is a different test: a driver may fold a constant conversion and
//! cannot fold a loaded one. Twelve round trips buys a stronger question, and this file is the
//! wrong place to trade it away. They use the gates below and none of the layout.

use crate::spirv_val;
use runner::{Error, Gpu};
use simdr::lanes::LaneError;
use simdr::spec::Capability;

/// A word a device buffer can hold.
///
/// The `match` on element width that each tool wrote out, as one trait. `runner::Gpu` names the
/// three by hand — `run_u32`, `run_bytes`, `run_halves` — because it is a Vulkan wrapper and a
/// generic there would have to be `unsafe` in three shapes; here it is safe to close over them.
pub trait Word: Copy + 'static {
    /// Send `input` through `spirv` over `workgroups` workgroups.
    ///
    /// # Errors
    ///
    /// [`Error`] if the dispatch fails, which is the device's answer rather than this crate's.
    fn dispatch(
        gpu: &Gpu,
        spirv: &[u32],
        input: &[Self],
        workgroups: u32,
    ) -> Result<Vec<Self>, Error>;
}

impl Word for u32 {
    fn dispatch(
        gpu: &Gpu,
        spirv: &[u32],
        input: &[Self],
        workgroups: u32,
    ) -> Result<Vec<Self>, Error> {
        gpu.run_u32(spirv, input, workgroups)
    }
}

impl Word for u8 {
    fn dispatch(
        gpu: &Gpu,
        spirv: &[u32],
        input: &[Self],
        workgroups: u32,
    ) -> Result<Vec<Self>, Error> {
        gpu.run_bytes(spirv, input, workgroups)
    }
}

impl Word for u16 {
    fn dispatch(
        gpu: &Gpu,
        spirv: &[u32],
        input: &[Self],
        workgroups: u32,
    ) -> Result<Vec<Self>, Error> {
        gpu.run_halves(spirv, input, workgroups)
    }
}

/// What a run concluded: four ways of not running, and one of having run.
///
/// `decisions/DR-0009` in one type, and it was three types before this. `Outcome`,
/// `ConversionsFailed` and `Roundtrip` each carried the same four arms with different names, which
/// is the duplication that goes stale quietly: a fifth reason would have had to be added three
/// times, and adding it to two of the three reads exactly like adding it to all.
#[derive(Debug)]
pub enum Answer<T> {
    /// The lane API said no. The mapping working.
    Refused(LaneError),
    /// The device does not offer what the module declares. The device being honest.
    Unsupported(Vec<Capability>),
    /// `spirv-val` rejected it. **This crate's mistake**, and nothing was dispatched.
    Invalid(String),
    /// The driver took a *validated* module and failed. The device's mistake, and the only one of
    /// the four worth reporting upstream.
    Errored(Error),
    /// It ran, and here is what came back.
    Ran(T),
}

impl<T> Answer<T> {
    /// What ran, if anything did.
    pub const fn ran(&self) -> Option<&T> {
        match self {
            Self::Ran(value) => Some(value),
            _ => None,
        }
    }

    /// Turn the answer into another shape, leaving the four refusals alone.
    ///
    /// What every caller does with this: run the comparison over what came back and keep the reason
    /// it did not run otherwise. Written once so that a tool cannot quietly drop an arm — which is
    /// the failure `Outcome` was invented to prevent and then had to be written three times to keep.
    pub fn map<U>(self, then: impl FnOnce(T) -> U) -> Answer<U> {
        match self {
            Self::Ran(value) => Answer::Ran(then(value)),
            Self::Refused(why) => Answer::Refused(why),
            Self::Unsupported(missing) => Answer::Unsupported(missing),
            Self::Invalid(complaint) => Answer::Invalid(complaint),
            Self::Errored(error) => Answer::Errored(error),
        }
    }

    /// Why it did not run, phrased for a report.
    ///
    /// `None` when it ran. A caller printing these is printing **lost coverage**, which is the
    /// distinction this whole type exists for: a check that was skipped and looks green is worse
    /// than one that failed.
    pub fn why(&self) -> Option<String> {
        match self {
            Self::Ran(_) => None,
            Self::Refused(why) => Some(format!("refused: {why}")),
            Self::Unsupported(missing) => Some(format!("unsupported: {missing:?}")),
            Self::Invalid(complaint) => Some(format!("invalid: {complaint}")),
            Self::Errored(error) => Some(format!("errored: {error}")),
        }
    }
}

/// N independent problems of the same size, in one dispatch.
///
/// A description of a *layout* rather than a buffer: the words belong to the caller, because each
/// tool fills them differently and a `Vec` in here would only be a place to copy them to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Batch {
    problems: usize,
    per_problem: usize,
}

impl Batch {
    /// `problems` problems of `per_problem` words each.
    #[must_use]
    pub const fn of(problems: usize, per_problem: usize) -> Self {
        Self {
            problems,
            per_problem,
        }
    }

    /// How many problems.
    #[must_use]
    pub const fn problems(&self) -> usize {
        self.problems
    }

    /// How many words one problem holds.
    #[must_use]
    pub const fn per_problem(&self) -> usize {
        self.per_problem
    }

    /// Words in the whole batch, for one operand.
    #[must_use]
    pub const fn words(&self) -> usize {
        self.problems * self.per_problem
    }

    /// Where a second operand starts, for a binding holding two arrays end to end.
    ///
    /// **The number that made the layer un-batchable**, and the reason this type exists rather than
    /// a loop over `run`. `Kernel::load_offset` takes a constant element offset, and the layer was
    /// passing the size of *one* workgroup's operand — correct for a single dispatch and wrong for
    /// every workgroup after the first, which would have read its neighbour's activations. The
    /// offset belongs to the batch, so it is computed here and passed in.
    #[must_use]
    pub const fn second_operand(&self) -> u32 {
        self.words() as u32
    }

    /// One workgroup per problem, which is what makes the invocation's index select it.
    #[must_use]
    pub const fn workgroups(&self) -> u32 {
        self.problems as u32
    }

    /// Split what came back into one slice per problem.
    ///
    /// `each` is the *answers* per problem, which is not `per_problem`: a reduction answers once
    /// per invocation and a scan answers once per element. Passing it explicitly is the honest
    /// spelling — deriving it from the input would be guessing at the kernel's shape.
    pub fn answers<'a, W>(&self, returned: &'a [W], each: usize) -> impl Iterator<Item = &'a [W]> {
        returned.chunks(each.max(1)).take(self.problems)
    }
}

/// Build, check the capabilities, validate, dispatch.
///
/// The four gates, in the order that makes each one's failure mean something. **The validator runs
/// before the device, every time**: a driver is lenient about things `spirv-val` is not, and an
/// invalid module in this directory once came back as 192 correct-looking answers on one device and
/// an opaque `ERROR_UNKNOWN` on another. That is `runner/tests/validated.rs`'s opening paragraph
/// happening inside the sandbox built to test the thing it is about.
pub fn run<W: Word>(
    gpu: &Gpu,
    label: &str,
    built: Result<Vec<u32>, LaneError>,
    input: &[W],
    workgroups: u32,
) -> Answer<Vec<W>> {
    let spirv = match built {
        Ok(spirv) => spirv,
        Err(refused) => return Answer::Refused(refused),
    };

    let missing = gpu.limits().unsupported_in(&spirv);
    if !missing.is_empty() {
        return Answer::Unsupported(missing);
    }

    if let Err(complaint) = spirv_val::validate(&spirv, label, spirv_val::VULKAN_1_1) {
        return Answer::Invalid(complaint);
    }

    match W::dispatch(gpu, &spirv, input, workgroups) {
        Ok(returned) => Answer::Ran(returned),
        Err(error) => Answer::Errored(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batchs_second_operand_is_the_whole_batch_and_not_one_problem() {
        // The bug this type was extracted to fix, as an assertion. Six problems of 128 words put
        // the activations at 768, not at 128 — and 128 is what the layer passed while dispatching
        // one workgroup, which is why it was right and unbatchable at the same time.
        let batch = Batch::of(6, 128);

        assert_eq!(batch.second_operand(), 768);
        assert_eq!(batch.workgroups(), 6);
        assert_eq!(batch.words(), 768);
        assert_ne!(
            batch.second_operand(),
            batch.per_problem() as u32,
            "a batch of one problem is the only size where the two agree, which is exactly why \
             the mistake survived"
        );
    }

    #[test]
    fn one_problem_is_the_size_the_old_code_was_right_for() {
        let single = Batch::of(1, 128);

        assert_eq!(single.second_operand(), 128);
        assert_eq!(single.workgroups(), 1);
    }

    #[test]
    fn the_answers_are_split_by_what_a_problem_answers_rather_than_by_what_it_reads() {
        let batch = Batch::of(3, 128);
        let returned: Vec<u32> = (0..12).collect();

        // Four answers a problem, not 128: a reduction answers once per invocation.
        let split: Vec<&[u32]> = batch.answers(&returned, 4).collect();

        assert_eq!(split.len(), 3);
        assert_eq!(split[0], &[0, 1, 2, 3]);
        assert_eq!(split[2], &[8, 9, 10, 11]);
    }

    #[test]
    fn a_short_return_gives_back_what_there_is_rather_than_inventing_the_rest() {
        // The direction that matters: a device returning less than asked is a finding, and a split
        // that padded it to the expected shape would compare invented zeros against a reference.
        let batch = Batch::of(4, 8);
        let returned: Vec<u32> = (0..6).collect();

        let split: Vec<&[u32]> = batch.answers(&returned, 4).collect();

        assert_eq!(
            split.len(),
            2,
            "two problems' worth came back, and two did not"
        );
        assert_eq!(split[1], &[4, 5]);
    }

    #[test]
    fn an_answer_keeps_its_refusal_through_a_map() {
        let refused: Answer<u32> = Answer::Refused(LaneError::NoMapping { lanes: 7, width: 8 });
        let mapped = refused.map(|value| value + 1);

        assert!(mapped.ran().is_none());
        assert!(
            mapped.why().is_some_and(|why| why.starts_with("refused")),
            "a mapped refusal has to still say why, or a caller counting lost coverage counts zero"
        );
        assert!(Answer::Ran(1_u32).why().is_none());
    }
}
