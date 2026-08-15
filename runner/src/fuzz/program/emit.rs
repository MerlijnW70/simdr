//! Turning one generated step into instructions.
//!
//! Split from the vocabulary in [`super`] so the enums stay readable: what an `Op` *is* fits on a
//! screen, and what it *emits* does not. Every arm here is one step, and its CPU meaning lives in
//! `interpret/steps.rs` — the two are meant to be read side by side, because a disagreement
//! between them is what the fuzzer reports.
//!
//! # One bound differs, and that is why there is a trait
//!
//! Every step here was emitted by one function generic over [`Element`], and could be, because
//! every instruction it reached exists for all eight domains. The three bit shifts do not: they
//! take `T: Integer`, so `F32` cannot be asked for one.
//!
//! That bound is younger than this file. `Lanes::shift_left` took `T: Element` until preparing this
//! pass, and a shift of a vector of floats compiled, built, and produced a module `spirv-val`
//! rejects — `OpUDot`'s shape exactly. Making it `Integer` fixed the emitter and moved the problem
//! here: the generator could not offer a shift, because the one place that emits a step could not
//! express the difference between a domain that has one and a domain that does not.
//!
//! [`Emit`] is that difference, carried by the element type. The width ladder in [`super`] stays
//! single — a second copy of it, one for integers and one for floats, is the relationship-decided-
//! twice shape `notes/FINDINGS.md` catalogues more often than any other.

use super::{BitShift, Domain, Op, ProgramError};
use simdr::lanes::{Element, F16, F32, I8, I16, I32, LaneError, Lanes, U8, U16, U32, Vector};

/// What an element type can be asked to emit.
///
/// One method, because one bound differs. A blanket `impl<T: Integer> Emit for T` beside an
/// `impl Emit for F32` is what this wants to be, and Rust will not take it: the compiler cannot
/// know `F32` is not an `Integer`, so the two conflict. Hence the two macros below.
pub trait Emit: Element {
    /// Move each element's bits, or refuse because this element has no such instruction.
    ///
    /// # Errors
    ///
    /// [`ProgramError::NotInThisDomain`] for a float element, and whatever the lane API refuses.
    fn bit_shift<const LANES: u32>(
        lanes: &mut Lanes<'_>,
        kind: BitShift,
        value: Vector<Self, LANES>,
        by: Vector<U32, LANES>,
    ) -> Result<Vector<Self, LANES>, ProgramError>;
}

/// The six integer elements, each emitting the shift the lane API names.
///
/// The match is exhaustive over [`BitShift`], so a fourth shift is a compile error here rather than
/// a silent fall-through — which is what a `_` arm would have bought instead.
macro_rules! shifts_for {
    ($($element:ty),+ $(,)?) => {
        $(impl Emit for $element {
            fn bit_shift<const LANES: u32>(
                lanes: &mut Lanes<'_>,
                kind: BitShift,
                value: Vector<Self, LANES>,
                by: Vector<U32, LANES>,
            ) -> Result<Vector<Self, LANES>, ProgramError> {
                Ok(match kind {
                    BitShift::Left => lanes.shift_left(value, by)?,
                    BitShift::RightLogical => lanes.shift_right_logical(value, by)?,
                    BitShift::RightArithmetic => lanes.shift_right_arithmetic(value, by)?,
                })
            }
        })+
    };
}

shifts_for!(U32, I32, U8, I8, U16, I16);

/// The two float elements, which have no bit shift at all.
///
/// SPIR-V's shifts take integer operands and give an integer result. That is not a leniency a
/// driver might wave through and not a rounding question — it is a module the validator rejects.
/// So the answer is a refusal by name rather than a bitcast, which would have worked and meant
/// something else.
macro_rules! no_shifts_for {
    ($($element:ty => $name:literal),+ $(,)?) => {
        $(impl Emit for $element {
            fn bit_shift<const LANES: u32>(
                _lanes: &mut Lanes<'_>,
                kind: BitShift,
                _value: Vector<Self, LANES>,
                _by: Vector<U32, LANES>,
            ) -> Result<Vector<Self, LANES>, ProgramError> {
                Err(ProgramError::NotInThisDomain { kind, element: $name })
            }
        })+
    };
}

no_shifts_for!(F32 => "f32", F16 => "f16");

/// Emit one step.
pub(super) fn apply<T: Emit, const LANES: u32>(
    lanes: &mut Lanes<'_>,
    domain: Domain,
    value: Vector<T, LANES>,
    step: Op,
) -> Result<Vector<T, LANES>, ProgramError> {
    Ok(match step {
        Op::AddConstant(operand) => {
            let constant = lanes.splat_bits::<T, LANES>(domain.encode(operand))?;
            lanes.add(value, constant)?
        }
        Op::MulConstant(operand) => {
            let constant = lanes.splat_bits::<T, LANES>(domain.encode(operand))?;
            lanes.mul(value, constant)?
        }
        Op::ButterflyAdd(mask) => {
            let neighbour = lanes.butterfly(value, mask)?;
            lanes.add(value, neighbour)?
        }
        Op::ShiftUp => lanes.shift_up(value, 0)?,
        Op::ShiftDown => lanes.shift_down(value, 0)?,
        Op::RotateUp(delta) => lanes.rotate_up(value, delta)?,
        Op::BroadcastLane(source) => lanes.broadcast(value, source)?,
        // The one step whose instruction does not exist in every domain, and the only arm here that
        // asks the element type rather than the `Lanes` in front of it.
        Op::BitShift { kind, by } => {
            // **The width check is here rather than in the six impls**, because it is a property of
            // the domain and not of the element type — and because six copies of one comparison is
            // the shape this project spends most of its time deleting. SPIR-V leaves a shift by at
            // least the operand's width undefined, and a reference cannot predict undefined.
            if by >= domain.bits() {
                return Err(ProgramError::ShiftTooFar {
                    by,
                    bits: domain.bits(),
                });
            }
            // The amount is a vector of `u32` whatever the value's element type is — SPIR-V takes
            // it as an operand rather than as a literal, so a constant shift is a splat and a
            // per-lane one would be expressible without a second entry point.
            let amount = lanes.splat_bits::<U32, LANES>(by)?;
            T::bit_shift(lanes, kind, value, amount)?
        }
        Op::ClampBelow(floor) => {
            let limit = lanes.splat_bits::<T, LANES>(domain.encode(floor))?;
            let above = lanes.greater_than(value, limit)?;
            lanes.select(above, value, limit)?
        }
        Op::MinConstant(operand) => {
            let limit = lanes.splat_bits::<T, LANES>(domain.encode(operand))?;
            lanes.min(value, limit)?
        }
        Op::MaxConstant(operand) => {
            let limit = lanes.splat_bits::<T, LANES>(domain.encode(operand))?;
            lanes.max(value, limit)?
        }
        Op::ClampBoth { low, high } => {
            let low = lanes.splat_bits::<T, LANES>(domain.encode(low))?;
            let high = lanes.splat_bits::<T, LANES>(domain.encode(high))?;
            lanes.clamp(value, low, high)?
        }
        Op::SelectEqual { to, then } => {
            let target = lanes.splat_bits::<T, LANES>(domain.encode(to))?;
            let replacement = lanes.splat_bits::<T, LANES>(domain.encode(then))?;
            let same = lanes.equal(value, target)?;
            lanes.select(same, replacement, value)?
        }
        Op::AddIfAllEqual { add } => {
            // The other uniform branch, and the vote that asks about a value. Written as a select
            // on the vote for the same reason the one below is: a branch cannot hand a value out
            // across its merge without an `OpPhi`, and the vote is uniform so both readings agree.
            let vote = lanes.all_equal_uniform(value)?;
            let increment = lanes.splat_bits::<T, LANES>(domain.encode(add))?;
            let raised = lanes.add(value, increment)?;

            let element = lanes.type_of::<T>()?;
            let mut chosen = Vec::with_capacity(value.strip_count());
            for (&taken, &left) in raised.strips().iter().zip(value.strips()) {
                chosen.push(
                    lanes
                        .module()
                        .select(element, vote.id(), taken, left)
                        .map_err(LaneError::Build)?,
                );
            }
            lanes.from_strips(&chosen)?
        }
        Op::AddIfAnyAbove {
            when_any_above,
            add,
        } => {
            let limit = lanes.splat_bits::<T, LANES>(domain.encode(when_any_above))?;
            let above = lanes.greater_than(value, limit)?;
            let vote = lanes.any_uniform(above)?;

            // A branch cannot hand a value out across its merge without an `OpPhi`, so the
            // conditional part is done as a select on the vote instead: the vote is uniform, so
            // both readings agree, and this one keeps the value in a register.
            let increment = lanes.splat_bits::<T, LANES>(domain.encode(add))?;
            let raised = lanes.add(value, increment)?;

            let element = lanes.type_of::<T>()?;
            let mut chosen = Vec::with_capacity(value.strip_count());
            for (&taken, &left) in raised.strips().iter().zip(value.strips()) {
                chosen.push(
                    lanes
                        .module()
                        .select(element, vote.id(), taken, left)
                        .map_err(LaneError::Build)?,
                );
            }
            lanes.from_strips(&chosen)?
        }
        Op::RepeatAdd { times, add } => {
            let increment = lanes.splat_bits::<T, LANES>(domain.encode(add))?;

            // One strip at a time: `repeat` threads a single id, and a vector is one id per strip.
            let mut carried = Vec::with_capacity(value.strip_count());
            for &strip in value.strips() {
                carried.push(lanes.repeat(times, strip, |lanes, held, _| {
                    let one = lanes.from_lane_value::<T, 1>(held)?;
                    let step = lanes.from_lane_value::<T, 1>(increment.id())?;
                    Ok(lanes.add(one, step)?.id())
                })?);
            }
            lanes.from_strips(&carried)?
        }
        Op::RolledAdd { times, add } => {
            let element = lanes.type_of::<T>()?;
            let increment = lanes.splat_bits::<T, LANES>(domain.encode(add))?;

            let mut carried = Vec::with_capacity(value.strip_count());
            for &strip in value.strips() {
                carried.push(
                    lanes.repeat_rolled(times, element, strip, |lanes, held, _| {
                        let one = lanes.from_lane_value::<T, 1>(held)?;
                        let step = lanes.from_lane_value::<T, 1>(increment.id())?;
                        Ok(lanes.add(one, step)?.id())
                    })?,
                );
            }
            lanes.from_strips(&carried)?
        }
        Op::RolledCounterAdd { times } => {
            let element = lanes.type_of::<T>()?;

            let mut carried = Vec::with_capacity(value.strip_count());
            for &strip in value.strips() {
                carried.push(lanes.repeat_rolled(
                    times,
                    element,
                    strip,
                    |lanes, held, iteration| {
                        // The counter is a `u32` whatever `T` is, so it is *converted* rather than
                        // reinterpreted: `OpConvertUToF` for a float. Reading its bits instead
                        // would make iteration 3 a denormal near zero, which is a wrong answer
                        // that reads like a numerical problem. This op was unsigned-only until
                        // the emitter could do the conversion.
                        let converted = lanes.convert_u32::<T>(iteration)?;
                        let one = lanes.from_lane_value::<T, 1>(held)?;
                        let step = lanes.from_lane_value::<T, 1>(converted)?;
                        Ok(lanes.add(one, step)?.id())
                    },
                )?);
            }
            lanes.from_strips(&carried)?
        }
    })
}
