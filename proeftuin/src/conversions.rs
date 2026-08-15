//! The conversions, which are one method and five instructions.
//!
//! `Lanes::convert_u32::<T>` emits whatever `T::FROM_U32` names, and that is not one opcode:
//!
//! | target | opcode | what it does |
//! | --- | --- | --- |
//! | `U32` | `OpCopyObject` | nothing |
//! | `I32` | `OpBitcast` | the **bits** |
//! | `I8`, `I16` | `OpSConvert` | truncate, signed |
//! | `U8`, `U16` | `OpUConvert` | truncate, unsigned |
//! | `F32`, `F16` | `OpConvertUToF` | the number, as a float |
//!
//! `src/lanes/narrow.rs` names the reason the middle two are worth separating: *"`OpUConvert`
//! requires a result type whose signedness is 0 and `OpSConvert` does not, so narrowing a `u32`
//! reaches a different opcode depending on whether the target is signed — even though both are the
//! same truncation. That is the kind of asymmetry that assembles cleanly when it is wrong."*
//!
//! None of the three conversions is in the fuzzer's vocabulary.
//!
//! # The probes are boundaries, not samples
//!
//! Every distinction here lives at a boundary and nowhere else. `OpSConvert` and `OpUConvert` agree
//! on every value below 128; a bitcast and a numeric conversion agree on every value below
//! `i32::MAX`; sign extension only shows where the truncated top bit is set. So the probe list is
//! the boundaries themselves — a sweep of random `u32`s would hit them by accident and prove it by
//! accident too.

use crate::batch::{self, Answer, Word};
use runner::Gpu;
use runner::kernels::WORKGROUP_SIZE;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, F32, I8, I16, I32, LaneError, U8, U16, U32};

/// A target this tool has a reference for, and how its answer comes back off the device.
///
/// **Two stringly-typed decisions, replaced by one trait.** `sweep` used to `match T::STRIDE` to
/// pick between `run_bytes`, `run_halves` and `run_u32`, and then compare `target == "i8"` to decide
/// whether to sign-extend what came back. Both are properties of the target type, and both were
/// written as tests on values a caller passes in — so a new target could be added with a wrong
/// string and the sign would silently be the other one.
pub trait Probed: Element {
    /// The buffer word this target's elements are read back as.
    type Word: Word;

    /// One returned word, widened to the `u32` the reference speaks in.
    ///
    /// Where the signedness lives. `OpSConvert`'s answer comes back in the target's own width and
    /// has to be read as *signed* to be compared, and `OpUConvert`'s does not — which is the same
    /// asymmetry the two opcodes have and the reason this is a method rather than a cast.
    fn widened(word: Self::Word) -> u32;
}

impl Probed for U32 {
    type Word = u32;
    fn widened(word: u32) -> u32 {
        word
    }
}

impl Probed for I32 {
    type Word = u32;
    fn widened(word: u32) -> u32 {
        word
    }
}

impl Probed for U8 {
    type Word = u8;
    fn widened(word: u8) -> u32 {
        u32::from(word)
    }
}

impl Probed for I8 {
    type Word = u8;
    fn widened(word: u8) -> u32 {
        i32::from(word as i8) as u32
    }
}

impl Probed for U16 {
    type Word = u16;
    fn widened(word: u16) -> u32 {
        u32::from(word)
    }
}

impl Probed for I16 {
    type Word = u16;
    fn widened(word: u16) -> u32 {
        i32::from(word as i16) as u32
    }
}

/// The values where the five instructions part company.
///
/// Zero and one for the trivial agreement; 127/128/255 for an 8-bit sign boundary and its wrap;
/// 32767/32768/65535 for the 16-bit one; and the three at the top for the difference between a
/// bitcast and a number.
pub const PROBES: [u32; 12] = [
    0,
    1,
    127,
    128,
    255,
    256,
    32_767,
    32_768,
    65_535,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
];

/// What the target type does to a `u32`, written from the **opcode** rather than from the method.
///
/// This is the reference, and it is deliberately a table of instruction behaviours rather than a
/// restatement of what `convert_u32` is documented to do. The doc says "a `u32` value's *number*, as
/// a value of `T`"; `I32` reaches `OpBitcast`, which is its bits. Modelling the doc would agree with
/// the implementation about every value under `i32::MAX` and disagree about the answer.
#[must_use]
pub fn converted(name: &str, value: u32) -> u32 {
    match name {
        // `OpCopyObject`.
        "u32" => value,
        // `OpBitcast` — the same bits, read as signed. Stored back as bits, so identity here, and
        // the distinction is the one the doc makes and the opcode does not.
        "i32" => value,
        // `OpUConvert`: truncate, zero-extended on the way back out.
        "u8" => value & 0xff,
        "u16" => value & 0xffff,
        // `OpSConvert`: truncate, and the top bit of what is left is a sign.
        "i8" => i32::from(((value & 0xff) as u8) as i8) as u32,
        "i16" => i32::from(((value & 0xffff) as u16) as i16) as u32,
        other => unreachable!("no reference written for {other}"),
    }
}

/// A kernel that converts one `u32` constant to `T` and stores it.
///
/// Whole-subgroup only, and that is a scoping decision rather than an omission: a conversion is
/// elementwise, so the mapping cannot reach it. `tests/instructions.rs` sweeps the mappings for the
/// operations where they *are* three instruction sequences.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
pub fn probe<T: Element, const LANES: u32>(
    subgroup: u32,
    value: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(Shape::new(subgroup, WORKGROUP_SIZE, 2))?;

    let converted = {
        let mut lanes = kernel.lanes()?;
        let source = lanes.splat_bits::<U32, LANES>(value)?;
        let narrowed = lanes.convert_u32::<T>(source.id())?;
        lanes.from_lane_value::<T, LANES>(narrowed)?
    };

    kernel.store(1, converted)?;
    kernel.finish()
}

/// One conversion, run and compared.
#[derive(Debug)]
pub struct Conversion {
    /// The target type's name.
    pub target: &'static str,
    /// The `u32` that went in.
    pub probe: u32,
    /// What the device stored, in the target's own width.
    pub actual: u32,
    /// What the opcode says it should be.
    pub expected: u32,
}

impl Conversion {
    /// Whether the device and the reference agree.
    #[must_use]
    pub const fn agreed(&self) -> bool {
        self.actual == self.expected
    }
}

/// Run every probe into `T` and compare.
///
/// Returns [`crate::Outcome::Ran`]-shaped information per probe, or the one reason the whole set did
/// not run — the same four-way split `decisions/DR-0009` argues for, kept here because a conversion
/// that is refused, unsupported or invalid is three different pieces of news.
///
/// # Errors
///
/// [`Error`] if a dispatch fails after the module validated, which is the device's problem rather
/// than this crate's.
pub fn sweep<T: Probed, const LANES: u32>(
    gpu: &Gpu,
    target: &'static str,
) -> Answer<Vec<Conversion>>
where
    Vec<T::Word>: FromIterator<T::Word>,
    T::Word: Default,
{
    let width = gpu.limits().subgroup_size;
    let mut found = Vec::with_capacity(PROBES.len());

    for probe_value in PROBES {
        // **One dispatch per probe, and this is the tool the batch layout does not fit.** The probe
        // is a *constant in the module* — that is what makes twelve boundaries twelve modules — so
        // batching them would mean loading it from a buffer, and a driver may fold a constant
        // conversion where it cannot fold a loaded one. Twelve round trips buys a stronger
        // question, and `batch::run` is the half of that API this uses.
        let input = vec![T::Word::default(); WORKGROUP_SIZE as usize];
        let answer = batch::run(
            gpu,
            &format!("proeftuin-convert-{target}-{probe_value}"),
            probe::<T, LANES>(width, probe_value),
            &input,
            1,
        );

        let Answer::Ran(returned) = answer else {
            // A refusal is a property of the module, and every probe builds the same shape — so
            // the first one to fail is the answer for the sweep rather than the first of twelve
            // identical lines.
            return answer.map(|_| Vec::new());
        };

        found.push(Conversion {
            target,
            probe: probe_value,
            actual: returned.first().copied().map_or(0, T::widened),
            expected: converted(target, probe_value),
        });
    }

    Answer::Ran(found)
}

/// Every target this tool has a reference for.
///
/// `F32` and `F16` are absent on purpose: `OpConvertUToF` answers with a float, and a float's bits
/// are not a number this table can compare with an integer one. They need their own reference and
/// are the obvious next thing rather than a silent omission.
pub fn every_target<const LANES: u32>(gpu: &Gpu) -> Vec<(&'static str, Answer<Vec<Conversion>>)> {
    vec![
        ("u32", sweep::<U32, LANES>(gpu, "u32")),
        ("i32", sweep::<I32, LANES>(gpu, "i32")),
        ("u8", sweep::<U8, LANES>(gpu, "u8")),
        ("i8", sweep::<I8, LANES>(gpu, "i8")),
        ("u16", sweep::<U16, LANES>(gpu, "u16")),
        ("i16", sweep::<I16, LANES>(gpu, "i16")),
    ]
}

/// Named so the unused import above is not a lie: `F32` belongs to the sweep this does not do yet.
const _: Option<F32> = None;
