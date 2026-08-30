use super::{BitShift, Domain, Missing, Op, ProgramError};
use simdr::lanes::{
    Element, F16, F32, I8, I16, I32, Integer, LaneError, Lanes, Signed, U8, U16, U32, Vector,
};

pub trait Emit: Element {
    fn bit_shift<const LANES: u32>(
        lanes: &mut Lanes<'_>,
        kind: BitShift,
        value: Vector<Self, LANES>,
        by: Vector<U32, LANES>,
    ) -> Result<Vector<Self, LANES>, ProgramError> {
        let _ = (lanes, value, by);
        Err(ProgramError::NotInThisDomain {
            missing: Missing::BitShift(kind),
            element: <Self as Element>::NAME,
        })
    }

    fn absolute<const LANES: u32>(
        lanes: &mut Lanes<'_>,
        value: Vector<Self, LANES>,
    ) -> Result<Vector<Self, LANES>, ProgramError> {
        let _ = (lanes, value);
        Err(ProgramError::NotInThisDomain {
            missing: Missing::Absolute,
            element: <Self as Element>::NAME,
        })
    }

    fn fused_mul_add<const LANES: u32>(
        lanes: &mut Lanes<'_>,
        value: Vector<Self, LANES>,
        by: Vector<Self, LANES>,
        plus: Vector<Self, LANES>,
    ) -> Result<Vector<Self, LANES>, ProgramError> {
        let _ = (lanes, value, by, plus);
        Err(ProgramError::NotInThisDomain {
            missing: Missing::FusedMulAdd,
            element: <Self as Element>::NAME,
        })
    }
}

fn shifted<T: Integer, const LANES: u32>(
    lanes: &mut Lanes<'_>,
    kind: BitShift,
    value: Vector<T, LANES>,
    by: Vector<U32, LANES>,
) -> Result<Vector<T, LANES>, ProgramError> {
    Ok(match kind {
        BitShift::Left => lanes.shift_left(value, by)?,
        BitShift::RightLogical => lanes.shift_right_logical(value, by)?,
        BitShift::RightArithmetic => lanes.shift_right_arithmetic(value, by)?,
    })
}

fn magnitude<T: Signed, const LANES: u32>(
    lanes: &mut Lanes<'_>,
    value: Vector<T, LANES>,
) -> Result<Vector<T, LANES>, ProgramError> {
    Ok(lanes.abs(value)?)
}

macro_rules! emit_for {
    ($element:ty $(, $capability:ident)* $(,)?) => {
        impl Emit for $element {
            $(emit_for!(@can $capability);)*
        }
    };
    (@can shifts) => {
        fn bit_shift<const LANES: u32>(
            lanes: &mut Lanes<'_>,
            kind: BitShift,
            value: Vector<Self, LANES>,
            by: Vector<U32, LANES>,
        ) -> Result<Vector<Self, LANES>, ProgramError> {
            shifted(lanes, kind, value, by)
        }
    };
    (@can magnitude) => {
        fn absolute<const LANES: u32>(
            lanes: &mut Lanes<'_>,
            value: Vector<Self, LANES>,
        ) -> Result<Vector<Self, LANES>, ProgramError> {
            magnitude(lanes, value)
        }
    };
    (@can fused) => {
        fn fused_mul_add<const LANES: u32>(
            lanes: &mut Lanes<'_>,
            value: Vector<Self, LANES>,
            by: Vector<Self, LANES>,
            plus: Vector<Self, LANES>,
        ) -> Result<Vector<Self, LANES>, ProgramError> {
            Ok(lanes.fma(value, by, plus)?)
        }
    };
}

emit_for!(U32, shifts);
emit_for!(I32, shifts, magnitude);
emit_for!(U8, shifts);
emit_for!(I8, shifts, magnitude);
emit_for!(U16, shifts);
emit_for!(I16, shifts, magnitude);
emit_for!(F32, magnitude, fused);
emit_for!(F16, magnitude);

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
        Op::BitShift { kind, by } => {
            if by >= domain.bits() {
                return Err(ProgramError::ShiftTooFar {
                    by,
                    bits: domain.bits(),
                });
            }
            let amount = lanes.splat_bits::<U32, LANES>(by)?;
            T::bit_shift(lanes, kind, value, amount)?
        }
        Op::Absolute => T::absolute(lanes, value)?,
        Op::FusedMulAdd { by, plus } => {
            let factor = lanes.splat_bits::<T, LANES>(domain.encode(by))?;
            let addend = lanes.splat_bits::<T, LANES>(domain.encode(plus))?;
            T::fused_mul_add(lanes, value, factor, addend)?
        }
        Op::AddIfAllAbove {
            when_all_above,
            add,
        } => {
            let limit = lanes.splat_bits::<T, LANES>(domain.encode(when_all_above))?;
            let above = lanes.greater_than(value, limit)?;
            let vote = lanes.all_uniform(above)?;

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
