//! The programs the fuzzer generates, and how they become SPIR-V.
//!
//! A program is a straight line of operations over one accumulator, ending in a reduction. That
//! is narrow, and deliberately so: every operation in it has a CPU meaning that can be stated in
//! one line, which is what makes `interpret` short enough to trust as a reference. A reference
//! with bugs of its own proves nothing.

mod emit;

pub use self::emit::Emit;

use self::emit::apply;
use super::domain::{BitShift, Domain};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{F16, F32, I8, I16, I32, LaneError, U8, U16, U32};
use std::fmt;

/// One step of a generated program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Add a constant to every element.
    AddConstant(u32),
    /// Multiply every element by a constant.
    MulConstant(u32),
    /// Add the value held `mask` lanes away, exclusive-or.
    ButterflyAdd(u32),
    /// Replace each element by the one **zero** lanes below — the identity, and the instruction.
    ///
    /// **It carried a distance and the distance was always zero.** SPIR-V leaves a shift's
    /// out-of-range lanes undefined, so a non-zero one has no reference to compare against: the
    /// generator drew none, the interpreter ignored the operand and returned the values unchanged,
    /// and anything that *did* build one — this type is public — would have been compared against
    /// an answer for a different program.
    ///
    /// So the invariant is in the type rather than in a comment beside it. What is left is worth
    /// keeping: the instruction is emitted, declares its capability, and is proved harmless.
    /// [`Op::RotateUp`] is the shuffle that moves elements and can be checked, because it wraps.
    ShiftUp,
    /// The same, downwards — `Lanes::shift_down`, which no generated program could reach.
    ///
    /// **An asymmetry rather than a decision.** The shift up was here from the beginning and its
    /// twin was not, so of the two directions the fuzzer proved one instruction emitted, declared
    /// and harmless and said nothing at all about the other. That is the shape this project keeps
    /// finding: `reduce_min` folded its strips with a *maximum* and agreed with every hand-written
    /// test but the strip-mined one.
    ///
    /// Zero for the same reason as [`Op::ShiftUp`], and it buys the same thing — no more, and no
    /// less than the direction beside it had.
    ShiftDown,
    /// Replace every element of a vector by the one at position `source` of that vector.
    ///
    /// The cross-lane operation nothing generated could reach, and the one whose answer is *fully
    /// determined*: unlike a shift, every lane reads a lane that exists, so the reference predicts
    /// it exactly rather than steering around an undefined edge.
    ///
    /// It is also the operation whose mapping does the most work. A whole-subgroup vector reads
    /// lane `source` of the subgroup; a **clustered** one reads position `source` of its own
    /// cluster, which is a different lane for every cluster and needs the invocation's own index —
    /// `decisions/DR-0007`; and a strip-mined one does it per strip. A reference that read
    /// `source` as a subgroup lane would agree with all three for the first cluster of every
    /// subgroup, which is exactly how a wrong mapping survives hand-written tests.
    BroadcastLane(u32),
    /// Keep the element where it exceeds a constant, otherwise substitute that constant.
    ///
    /// The *core* spelling of a lower bound: a comparison and a select. [`Op::MaxConstant`] is the
    /// same arithmetic through one extended instruction, and the pair is worth having for the same
    /// reason [`Op::RepeatAdd`] and [`Op::RolledAdd`] are — one answer, two instruction streams,
    /// and they must agree.
    ClampBelow(u32),
    /// The smaller of each element and a constant, through `OpExtInst` `*Min`.
    ///
    /// Which of `FMin`, `UMin` and `SMin` is reached depends on the domain, and the three agree on
    /// every value below 2³¹ — so what this catches is not the choice between them but whether the
    /// instruction computes a minimum at all. The signed-versus-unsigned confusion needs a value
    /// with its top bit set, which is `runner/tests/extended.rs`.
    MinConstant(u32),
    /// The larger of each element and a constant, through `OpExtInst` `*Max`.
    MaxConstant(u32),
    /// Each element held between two constants, through `OpExtInst` `*Clamp`.
    ///
    /// The bounds are generated in order, because a clamp whose low bound is above its high one is
    /// undefined and a reference cannot predict undefined.
    ClampBoth {
        /// The lower bound.
        low: u32,
        /// The upper bound, never below `low`.
        high: u32,
    },
    /// Replace each element by the one `delta` lanes below, **wrapping inside the vector**.
    ///
    /// The one shuffle here that can be generated with a real distance. [`Op::ShiftUp`] is drawn
    /// with a delta of zero and nothing else, because SPIR-V leaves its bottom lanes undefined and
    /// a reference cannot predict undefined — a rotate has no such lane, so every delta is fair
    /// game and the reference knows exactly where each element came from.
    RotateUp(u32),
    /// Replace every element equal to `to` with `then`, and leave the rest alone.
    ///
    /// The elementwise **equality**, which the lane API had no spelling for until the audit above
    /// asked for it. A comparison and a select, the same shape as [`Op::ClampBelow`] — and the one
    /// operation here whose two integer domains reach the *same* instruction, `OpIEqual`, where
    /// every other comparison splits into a signed and an unsigned form.
    ///
    /// `to` is drawn from inside the corpus's range, because an equality nothing ever satisfies is
    /// an identity, and an identity agrees with every reference including a wrong one.
    SelectEqual {
        /// The value an element must equal to be replaced.
        to: u32,
        /// What replaces it.
        then: u32,
    },
    /// Add a constant, but only where **every lane of the subgroup holds the same value**.
    ///
    /// The vote about a value rather than about a predicate, and the second uniform branch here.
    /// [`Op::AddIfAnyAbove`] asks whether a comparison held somewhere; this asks whether the lanes
    /// agree, which no comparison can express — the value a lane would compare against is the one
    /// it is trying to learn.
    ///
    /// Reachable only where the vector is at least as wide as the subgroup, like the other vote:
    /// `all_equal` refuses a clustered vector, where the answer would cover four vectors at once.
    AddIfAllEqual {
        /// What to add where the subgroup agrees.
        add: u32,
    },
    /// Add a constant, but only where some element of the subgroup exceeds `when_any_above`.
    ///
    /// A uniform branch: the condition is a vote, so the whole subgroup takes it or none of it
    /// does. The reference can predict that exactly, which is the only reason this is fuzzable —
    /// a per-lane branch would leave a subgroup operation inside it answering for whichever lanes
    /// happened to be running. See `decisions/DR-0003`.
    AddIfAnyAbove {
        /// The threshold every element is compared against.
        when_any_above: u32,
        /// What to add where the vote passes.
        add: u32,
    },
    /// Add a constant `times` times over, as an *unrolled* loop.
    ///
    /// Arithmetically it is one multiplication, and that is the point: the answer is trivial and
    /// the emitted shape is not. `Lanes::repeat` threads a value through a Rust loop, and a
    /// threading bug would show up here as an off-by-one nobody could mistake for rounding.
    RepeatAdd {
        /// How many iterations.
        times: u32,
        /// What each one adds.
        add: u32,
    },
    /// Add a constant on each of `times` trips of a *rolled* loop.
    ///
    /// The four-block shape with its two `OpPhi`s, in every domain. Same arithmetic as
    /// [`Op::RepeatAdd`] and a completely different instruction stream, which is what makes the
    /// pair worth having: they must agree, and only one of them has a back edge.
    RolledAdd {
        /// How many trips.
        times: u32,
        /// What each one adds.
        add: u32,
    },
    /// Add the iteration number on each of `times` trips of a *rolled* loop.
    ///
    /// The counter is a `u32` whatever the domain is, so this also exercises
    /// `Lanes::convert_u32` — and in the float domain a reinterpretation instead of a conversion
    /// would turn iteration 3 into a denormal, which the reference would catch at once.
    ///
    /// The one op here that cannot be written any other way. `Lanes::repeat_rolled` emits the
    /// four-block shape with two `OpPhi`s, and the body — built once — reads the counter phi. The
    /// total added is `times × (times − 1) / 2`, which is wrong for every mistake worth making: a
    /// body handed a copy of the counter adds zero, an off-by-one trip count lands one triangular
    /// number away.
    RolledCounterAdd {
        /// How many trips.
        times: u32,
    },
    /// Move every element's **bits**, which only an integer domain can do.
    ///
    /// The first step here that is not available in every domain, and the reason the emission below
    /// goes through [`Emit`] rather than through one function generic over `Element`. `Lanes`'
    /// three shifts take `T: Integer`, so a float domain cannot even be asked — which is a bound
    /// that arrived *because* preparing this pass found `shift_left` taking `T: Element` and
    /// building a module `spirv-val` rejects.
    ///
    /// **`by` reaches the top of the element**, deliberately. Everything the generator draws is a
    /// small positive number, and `OpShiftRightLogical` and `OpShiftRightArithmetic` agree on every
    /// one of them — the difference lives entirely in the top bit, so a corpus alone cannot
    /// separate the two instructions. A left shift of 31 puts a bit there, and the next right shift
    /// is then a question with two different answers.
    BitShift {
        /// Which of the three.
        kind: BitShift,
        /// How far, always below the element's own width — see [`ProgramError::ShiftTooFar`].
        by: u32,
    },
    /// Each element's **magnitude**, through GLSL.std.450's `SAbs` or `FAbs`.
    ///
    /// The second gate on [`Emit`], and a different one: `Lanes::abs` takes `T: Signed`, which is
    /// five of the eight domains — neither the six an `Integer` bound reaches nor the two a
    /// float-only operation does. That is what makes the trait a mechanism rather than an
    /// arrangement for one operation.
    ///
    /// **The value it has no answer for is now reachable.** A two's-complement minimum negates to
    /// itself, and before [`Op::BitShift`] existed every signed value here was too small to be one.
    /// A left shift of 31 lands on it exactly, so the reference refuses a round whose magnitude
    /// meets that value instead of asserting an answer GLSL.std.450 does not promise — the same
    /// answer `Domain::exact_limit` gives for a half that leaves its range.
    Absolute,
    /// Multiply by one constant and add another, **fused** — GLSL.std.450's `Fma`.
    ///
    /// The third gate, and the narrowest: `Lanes::fma` takes `Vector<F32, _>` concretely, so of the
    /// eight domains exactly **one** can be asked. Integer-only, five-of-eight, and one-of-eight in
    /// the same trait is the whole argument that the gate belongs on the element type.
    ///
    /// **Its own doc comment argues against fuzzing it**, and is right in general: a fused multiply
    /// and add rounds once where a separate pair rounds twice, so *"a kernel that must agree with a
    /// CPU reference exactly has to make the same choice on both sides"*. This corpus is the case
    /// that sentence does not cover. Every float here is a small integer, both roundings are
    /// exact, and the two spellings give the same bits — so the pair can be held to agreeing, which
    /// is the same trade [`Op::RepeatAdd`] and [`Op::RolledAdd`] are here for.
    FusedMulAdd {
        /// What every element is multiplied by.
        by: u32,
        /// What is added to the product.
        plus: u32,
    },
    /// Add a constant, but only where **every** element of the subgroup exceeds a threshold.
    ///
    /// The third vote, and two of the three were generated. [`Op::AddIfAnyAbove`] asks whether a
    /// comparison held somewhere and [`Op::AddIfAllEqual`] asks whether the lanes agree; this asks
    /// whether it held *everywhere*, which is `Lanes::all_uniform` — a different opcode from
    /// `any_uniform`, with unit tests and, until now, no generated program.
    ///
    /// **The threshold is drawn low on purpose.** `AddIfAnyAbove` straddles the corpus so both arms
    /// are reached; an `all` vote over a few hundred distinct elements almost never passes at the
    /// same threshold, and a step that never fires is an identity — which agrees with every
    /// reference including a wrong one.
    AddIfAllAbove {
        /// The threshold every element must exceed.
        when_all_above: u32,
        /// What to add where the vote passes.
        add: u32,
    },
}

/// Which operation an element type turned out not to have.
///
/// Three gates, three memberships: six domains, five, and one. Naming them apart rather than
/// reporting "unsupported" keeps the refusal as informative as the lane API's own — the difference
/// between *this width has no mapping* and *this type has no such instruction* was worth a type
/// when there was one of these, and it is worth more now there are three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Missing {
    /// One of the three bit shifts. `Lanes` gates these on `Integer`.
    BitShift(BitShift),
    /// `Lanes::abs`, gated on `Signed`.
    Absolute,
    /// `Lanes::fma`, which takes `F32` and nothing else.
    FusedMulAdd,
}

impl fmt::Display for Missing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitShift(kind) => write!(formatter, "a {kind:?} bit shift"),
            Self::Absolute => formatter.write_str("an absolute value"),
            Self::FusedMulAdd => formatter.write_str("a fused multiply-add"),
        }
    }
}

/// How a program's final reduction combines the lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// Sum across the group.
    Sum,
    /// Largest across the group.
    Max,
    /// Smallest across the group.
    Min,
    /// Sum when any element exceeds the threshold, otherwise the largest.
    ///
    /// The only finish that carries a value out of a branch. Both arms end in a subgroup
    /// reduction and exactly one runs, so the answer says which edge the `OpPhi` read from — and a
    /// phi naming the wrong predecessor validates cleanly and computes the wrong thing, which is
    /// the failure no other layer here catches.
    SumOrMax {
        /// The threshold the vote compares against.
        when_any_above: u32,
    },
    /// Running totals: every element keeps the sum of everything up to and including itself.
    ///
    /// **The only finish that keeps every element**, and that is why it is worth generating. A
    /// reduction combines the same set whatever order the lanes are in, so a mapping that pairs
    /// the wrong lanes still returns the right total — which is how `reduce_min` folded its strips
    /// with a *maximum* and agreed with every hand-written test but the strip-mined one. A scan
    /// gets a different number at almost every position and the same grand total at the end.
    Scan,
    /// The same, with each element's own contribution left out.
    ///
    /// Not a shift of [`Finish::Scan`]: SPIR-V has a separate group operation for it, because
    /// deriving one from the other over floats subtracts a large running total back off itself and
    /// loses the low bits the scan just accumulated.
    ScanExclusive,
}

/// Why a generated program could not be emitted.
///
/// **Two kinds, and telling them apart is the point** — `decisions/DR-0009` in one sentence. Until
/// this existed a refusal was always a [`LaneError`], because every operation was available in
/// every domain and the only thing that could go wrong was the *width*. A step that some domains
/// have and others do not is a second kind of no, and folding it into the first would have made a
/// float program holding a bit shift look exactly like a vector too wide to map.
#[derive(Debug)]
#[non_exhaustive]
pub enum ProgramError {
    /// The lane API refused: a width with no mapping, a lane outside the vector.
    ///
    /// Not a failure. The generator is allowed to ask for things the mapping refuses, and being
    /// refused by name is the correct answer.
    Lanes(LaneError),
    /// The element type has no such instruction — a bit shift in a float domain.
    ///
    /// Not reachable from [`super::generate`], which offers the shifts only where they exist. It is
    /// reachable from a [`Program`] somebody wrote by hand, and this crate's answer to a
    /// representable-but-wrong state is to name it rather than to compute something.
    NotInThisDomain {
        /// What was being emitted.
        missing: Missing,
        /// The element type that has no instruction for it.
        element: &'static str,
    },
    /// A shift by at least the element's own width, which SPIR-V leaves **undefined**.
    ///
    /// The reference cannot predict undefined, so a round like this would compare a device's
    /// arbitrary answer against a host's arbitrary answer and report whichever way they fell. That
    /// is the `ButterflyAdd` mistake — a mask that took a lane past the subgroup, right on the two
    /// devices here and wrong on an eight-wide one — and it is refused here rather than drawn
    /// around, because `Op` is public and a caller can write one.
    ShiftTooFar {
        /// The distance asked for.
        by: u32,
        /// The element's width, which it must stay below.
        bits: u32,
    },
}

impl From<LaneError> for ProgramError {
    fn from(error: LaneError) -> Self {
        Self::Lanes(error)
    }
}

impl fmt::Display for ProgramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lanes(error) => write!(formatter, "{error}"),
            Self::NotInThisDomain { missing, element } => {
                write!(formatter, "{element} has no {missing}")
            }
            Self::ShiftTooFar { by, bits } => {
                write!(
                    formatter,
                    "a shift of {by} in a {bits}-bit element is undefined"
                )
            }
        }
    }
}

impl std::error::Error for ProgramError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Lanes(error) => Some(error),
            _ => None,
        }
    }
}

/// A generated program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// Which element type it computes in.
    pub domain: Domain,
    /// The device's subgroup width, which decides the mapping.
    pub subgroup: u32,
    /// Invocations per workgroup.
    pub workgroup: u32,
    /// How many workgroups to dispatch.
    pub groups: u32,
    /// The vector width, as a `LANES` the builder will instantiate.
    pub lanes: u32,
    /// The steps, in order.
    pub steps: Vec<Op>,
    /// How it ends.
    pub finish: Finish,
}

impl Program {
    /// How many workgroups this dispatches.
    #[must_use]
    pub const fn workgroups(&self) -> u32 {
        self.groups
    }

    /// How many elements the input buffer must hold.
    #[must_use]
    pub fn input_len(&self) -> usize {
        // Division and a floor, not a comparison. The branch this replaced asked whether the
        // vector was wider than the subgroup and returned 1 when it was not — and at *equal*
        // widths both arms give 1, so no input could tell them apart. `interpret::strips_of` had
        // the same shape and was fixed the same way; the mutation gate found the second copy.
        let strips = (self.lanes / self.subgroup.max(1)).max(1);
        (self.groups * self.workgroup * strips) as usize
    }

    /// Emit this program as SPIR-V.
    ///
    /// # Errors
    ///
    /// [`ProgramError`] when the lane count has no mapping onto this subgroup, or when a step asks
    /// for an instruction this domain does not have. Both are legitimate answers rather than
    /// failures, and they are different answers — see the type.
    pub fn build(&self) -> Result<Vec<u32>, ProgramError> {
        match self.domain {
            Domain::Unsigned => self.build_in::<U32>(),
            Domain::Signed => self.build_in::<I32>(),
            Domain::Float => self.build_in::<F32>(),
            Domain::UnsignedByte => self.build_in::<U8>(),
            Domain::Byte => self.build_in::<I8>(),
            Domain::UnsignedShort => self.build_in::<U16>(),
            Domain::Short => self.build_in::<I16>(),
            Domain::Half => self.build_in::<F16>(),
        }
    }

    /// The same, with the element type chosen.
    ///
    /// **[`Emit`] rather than `Element`, and that is the whole shape of this pass.** The three bit
    /// shifts take `T: Integer`, so one function generic over `Element` cannot emit them — which is
    /// what kept them out of the generated vocabulary. The alternative was a second copy of the
    /// width ladder below, one for the integer domains and one for the floats, and a relationship
    /// decided twice is what `notes/FINDINGS.md` catalogues most often. So the ladder stays single
    /// and the *element type* carries what it can do.
    fn build_in<T: Emit>(&self) -> Result<Vec<u32>, ProgramError> {
        // `LANES` is a const generic and the generator picks it at runtime, so the widths have to
        // be listed. These are every power of two the mapping can express on a 32- or 64-lane
        // device with `MAX_STRIPS` of headroom.
        match self.lanes {
            1 => self.build_at::<T, 1>(),
            2 => self.build_at::<T, 2>(),
            4 => self.build_at::<T, 4>(),
            8 => self.build_at::<T, 8>(),
            16 => self.build_at::<T, 16>(),
            32 => self.build_at::<T, 32>(),
            64 => self.build_at::<T, 64>(),
            128 => self.build_at::<T, 128>(),
            256 => self.build_at::<T, 256>(),
            other => Err(ProgramError::Lanes(LaneError::NoMapping {
                lanes: other,
                width: self.subgroup,
            })),
        }
    }

    fn build_at<T: Emit, const LANES: u32>(&self) -> Result<Vec<u32>, ProgramError> {
        let mut kernel = Kernel::<T>::new(Shape::new(self.subgroup, self.workgroup, 2))?;
        let mut value = kernel.load::<LANES>(0)?;

        for step in &self.steps {
            value = apply::<T, LANES>(&mut kernel.lanes()?, self.domain, value, *step)?;
        }

        let element = kernel.element();

        // **Each arm stores what it produces**, because the two kinds of finish do not produce the
        // same shape. A reduction is one value per invocation and goes out through `store_scalar`;
        // a scan keeps every element and goes out through `store`, filling exactly the addresses
        // the input came from.
        match self.finish {
            Finish::Sum => {
                let total = kernel.lanes()?.reduce_sum(value)?;
                kernel.store_scalar(1, total)?;
            }
            Finish::Max => {
                let total = kernel.lanes()?.reduce_max(value)?;
                kernel.store_scalar(1, total)?;
            }
            Finish::Min => {
                let total = kernel.lanes()?.reduce_min(value)?;
                kernel.store_scalar(1, total)?;
            }
            Finish::SumOrMax { when_any_above } => {
                let total = {
                    let mut lanes = kernel.lanes()?;
                    let limit = lanes.splat_bits::<T, LANES>(self.domain.encode(when_any_above))?;
                    let above = lanes.greater_than(value, limit)?;
                    let vote = lanes.any_uniform(above)?;

                    lanes.choose_uniform(
                        vote,
                        element,
                        |lanes| lanes.reduce_sum(value),
                        |lanes| lanes.reduce_max(value),
                    )?
                };
                kernel.store_scalar(1, total)?;
            }
            // All three mappings scan now: an instruction at the subgroup's width, a carry between
            // strips above it, and a ladder below. The clustered one used to arrive here as
            // `Outcome::Refused`, which meant every narrow vector the generator produced skipped
            // its scan — and the ladder was checked by hand-written tests only.
            Finish::Scan => {
                let scanned = kernel.lanes()?.prefix_sum(value)?;
                kernel.store(1, scanned)?;
            }
            Finish::ScanExclusive => {
                let scanned = kernel.lanes()?.prefix_sum_exclusive(value)?;
                kernel.store(1, scanned)?;
            }
        }
        Ok(kernel.finish()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_input_length_covers_every_strip() {
        let program = Program {
            domain: Domain::Unsigned,
            subgroup: 32,
            workgroup: 64,
            groups: 2,
            lanes: 128,
            steps: Vec::new(),
            finish: Finish::Sum,
        };

        // Four strips per invocation, 64 invocations per workgroup, two workgroups.
        assert_eq!(program.input_len(), 4 * 64 * 2);
    }

    #[test]
    fn a_float_domain_refuses_a_bit_shift_rather_than_emitting_one() {
        // Not reachable from `generate`, which offers the shifts only where they exist. It is
        // reachable from a `Program` somebody wrote by hand — the type is public and every field
        // with it — and this crate's answer to a representable-but-wrong state is to name it.
        //
        // What the alternative looks like is on the record: `Lanes::shift_left` took `T: Element`
        // for a month, and a shift of a vector of floats built a module `spirv-val` rejects. The
        // bound stops the *call* from compiling; this stops the *program* from being emitted, and
        // the two are different layers of the same no.
        for domain in [Domain::Float, Domain::Half] {
            let program = Program {
                domain,
                subgroup: 32,
                workgroup: 64,
                groups: 1,
                lanes: 32,
                steps: vec![Op::BitShift {
                    kind: BitShift::Left,
                    by: 3,
                }],
                finish: Finish::Sum,
            };

            let refused = program.build().expect_err("a float has no bit shift");
            assert!(
                matches!(refused, ProgramError::NotInThisDomain { .. }),
                "{domain:?} refused a bit shift as {refused}, which reads as a width problem"
            );
        }
    }

    #[test]
    fn a_shift_past_the_element_is_refused_at_every_width() {
        // SPIR-V leaves a shift by at least the operand's width undefined, and a reference cannot
        // predict undefined. The `ButterflyAdd` mask was the same shape: every distance it drew was
        // inside a 32- or 64-wide subgroup, so it was right on both devices here and wrong on an
        // eight-wide one, found by lavapipe on seed 3.
        //
        // The generator cannot draw one of these. `Op` is public, so a caller can write one.
        for domain in [
            Domain::Unsigned,
            Domain::Signed,
            Domain::UnsignedByte,
            Domain::Byte,
            Domain::UnsignedShort,
            Domain::Short,
        ] {
            let program = Program {
                domain,
                subgroup: 32,
                workgroup: 64,
                groups: 1,
                lanes: 32,
                steps: vec![Op::BitShift {
                    kind: BitShift::RightLogical,
                    by: domain.bits(),
                }],
                finish: Finish::Sum,
            };

            let refused = program.build().expect_err("an undefined shift is refused");
            assert!(
                matches!(refused, ProgramError::ShiftTooFar { .. }),
                "{domain:?} took a shift of {} bits and answered {refused}",
                domain.bits()
            );
        }

        // And the width below it builds, or the check above would pass by refusing everything.
        let program = Program {
            domain: Domain::Unsigned,
            subgroup: 32,
            workgroup: 64,
            groups: 1,
            lanes: 32,
            steps: vec![Op::BitShift {
                kind: BitShift::RightLogical,
                by: 31,
            }],
            finish: Finish::Sum,
        };
        assert!(
            program.build().is_ok(),
            "a shift of 31 into a u32 is defined"
        );
    }

    #[test]
    fn the_three_shifts_are_three_different_modules() {
        // One `Op` reaching one instruction, which is what `Emit` exists to arrange. If the match
        // in the macro folded two arms together the programs would still build, still run, and
        // still agree with a reference that made the same mistake — so the claim is made about the
        // words rather than about the answer.
        let base = Program {
            domain: Domain::Signed,
            subgroup: 32,
            workgroup: 64,
            groups: 1,
            lanes: 32,
            steps: Vec::new(),
            finish: Finish::Sum,
        };

        let built = |kind| {
            Program {
                steps: vec![Op::BitShift { kind, by: 3 }],
                ..base.clone()
            }
            .build()
            .expect("built")
        };

        let left = built(BitShift::Left);
        let logical = built(BitShift::RightLogical);
        let arithmetic = built(BitShift::RightArithmetic);

        assert_ne!(left, logical);
        assert_ne!(
            logical, arithmetic,
            "the two right shifts are one opcode apart"
        );
        assert_ne!(left, arithmetic);
    }

    #[test]
    fn the_two_domains_emit_different_instructions_from_one_program() {
        let base = Program {
            domain: Domain::Unsigned,
            subgroup: 32,
            workgroup: 64,
            groups: 1,
            lanes: 32,
            steps: vec![Op::AddConstant(1)],
            finish: Finish::Sum,
        };
        let floats = Program {
            domain: Domain::Float,
            ..base.clone()
        };

        let integer_words = base.build().expect("built");
        let float_words = floats.build().expect("built");

        assert_ne!(
            integer_words, float_words,
            "the same program in two domains must not produce the same module"
        );
    }
}
