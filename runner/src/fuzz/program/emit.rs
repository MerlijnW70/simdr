//! Turning one generated step into instructions.
//!
//! Split from the vocabulary in [`super`] so the enums stay readable: what an `Op` *is* fits on a
//! screen, and what it *emits* does not. Every arm here is one step, and its CPU meaning lives in
//! `interpret/steps.rs` — the two are meant to be read side by side, because a disagreement
//! between them is what the fuzzer reports.

use super::{Domain, Op};
use simdr::lanes::{Element, LaneError, Lanes, Vector};

/// Emit one step.
pub(super) fn apply<T: Element, const LANES: u32>(
    lanes: &mut Lanes<'_>,
    domain: Domain,
    value: Vector<T, LANES>,
    step: Op,
) -> Result<Vector<T, LANES>, LaneError> {
    match step {
        Op::AddConstant(operand) => {
            let constant = lanes.splat_bits::<T, LANES>(domain.encode(operand))?;
            lanes.add(value, constant)
        }
        Op::MulConstant(operand) => {
            let constant = lanes.splat_bits::<T, LANES>(domain.encode(operand))?;
            lanes.mul(value, constant)
        }
        Op::ButterflyAdd(mask) => {
            let neighbour = lanes.butterfly(value, mask)?;
            lanes.add(value, neighbour)
        }
        Op::ShiftUp => lanes.shift_up(value, 0),
        Op::ShiftDown => lanes.shift_down(value, 0),
        Op::RotateUp(delta) => lanes.rotate_up(value, delta),
        Op::BroadcastLane(source) => lanes.broadcast(value, source),
        Op::ClampBelow(floor) => {
            let limit = lanes.splat_bits::<T, LANES>(domain.encode(floor))?;
            let above = lanes.greater_than(value, limit)?;
            lanes.select(above, value, limit)
        }
        Op::MinConstant(operand) => {
            let limit = lanes.splat_bits::<T, LANES>(domain.encode(operand))?;
            lanes.min(value, limit)
        }
        Op::MaxConstant(operand) => {
            let limit = lanes.splat_bits::<T, LANES>(domain.encode(operand))?;
            lanes.max(value, limit)
        }
        Op::ClampBoth { low, high } => {
            let low = lanes.splat_bits::<T, LANES>(domain.encode(low))?;
            let high = lanes.splat_bits::<T, LANES>(domain.encode(high))?;
            lanes.clamp(value, low, high)
        }
        Op::SelectEqual { to, then } => {
            let target = lanes.splat_bits::<T, LANES>(domain.encode(to))?;
            let replacement = lanes.splat_bits::<T, LANES>(domain.encode(then))?;
            let same = lanes.equal(value, target)?;
            lanes.select(same, replacement, value)
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
                chosen.push(lanes.module().select(element, vote.id(), taken, left)?);
            }
            lanes.from_strips(&chosen)
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
                chosen.push(lanes.module().select(element, vote.id(), taken, left)?);
            }
            lanes.from_strips(&chosen)
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
            lanes.from_strips(&carried)
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
            lanes.from_strips(&carried)
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
            lanes.from_strips(&carried)
        }
    }
}
