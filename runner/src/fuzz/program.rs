mod emit;

pub use self::emit::Emit;

use self::emit::apply;
use super::domain::{BitShift, Domain};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{F16, F32, I8, I16, I32, LaneError, U8, U16, U32};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    AddConstant(u32),
    MulConstant(u32),
    ButterflyAdd(u32),
    ShiftUp,
    ShiftDown,
    BroadcastLane(u32),
    ClampBelow(u32),
    MinConstant(u32),
    MaxConstant(u32),
    ClampBoth { low: u32, high: u32 },
    RotateUp(u32),
    SelectEqual { to: u32, then: u32 },
    AddIfAllEqual { add: u32 },
    AddIfAnyAbove { when_any_above: u32, add: u32 },
    RepeatAdd { times: u32, add: u32 },
    RolledAdd { times: u32, add: u32 },
    RolledCounterAdd { times: u32 },
    BitShift { kind: BitShift, by: u32 },
    Absolute,
    FusedMulAdd { by: u32, plus: u32 },
    AddIfAllAbove { when_all_above: u32, add: u32 },
    SubConstant(u32),
    SaturatingAddConstant(u32),
    SaturatingSubConstant(u32),
    AndConstant(u32),
    OrConstant(u32),
    XorConstant(u32),
    NotValue,
    Floor,
    Ceil,
    Trunc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Missing {
    BitShift(BitShift),
    Absolute,
    FusedMulAdd,
    Saturating,
    Bitwise,
    Rounding,
}

impl fmt::Display for Missing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BitShift(kind) => write!(formatter, "a {kind:?} bit shift"),
            Self::Absolute => formatter.write_str("an absolute value"),
            Self::FusedMulAdd => formatter.write_str("a fused multiply-add"),
            Self::Saturating => formatter.write_str("saturating arithmetic"),
            Self::Bitwise => formatter.write_str("a bitwise operation"),
            Self::Rounding => formatter.write_str("a rounding"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Sum,
    Max,
    Min,
    SumOrMax {
        when_any_above: u32,
    },
    Scan,
    ScanExclusive,
    /// A reduction by a fold the flat variants above do not name.
    ReduceBy(Fold),
    /// A running fold, in either form. The sum is `Scan` and `ScanExclusive`
    /// above; every other fold arrives here.
    ScanBy {
        fold: Fold,
        exclusive: bool,
    },
}

/// What a reduction or a running fold combines with.
///
/// The bitwise three are integer-only, and the emitter refuses them on a float
/// the way it refuses a bit shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Fold {
    Product,
    Min,
    Max,
    And,
    Or,
    Xor,
}

impl Fold {
    pub const EVERY: [Self; 6] = [
        Self::Product,
        Self::Min,
        Self::Max,
        Self::And,
        Self::Or,
        Self::Xor,
    ];

    /// Whether this fold can only be compared over an integer.
    ///
    /// The bitwise three because SPIR-V has no float form of them. The product
    /// for a different reason, found by running it: a float product is not
    /// associative, so the answer depends on the order the device folds in,
    /// which this reference cannot know. Over `f16` a handful of lanes reaches
    /// infinity, and one more multiplication by zero makes a NaN out of what
    /// the reference had as nought.
    #[must_use]
    pub const fn needs_an_integer(self) -> bool {
        matches!(self, Self::And | Self::Or | Self::Xor | Self::Product)
    }

    /// Whether [`Finish`] already reduces this way under a flat name.
    ///
    /// `Min` and `Max` are here because a running minimum needs the fold and a
    /// reduction to a minimum predates it, so the generator draws the fold for
    /// scans and the flat variant for reductions, and neither operation is ever
    /// reached under two names.
    #[must_use]
    pub const fn reduces_under_another_name(self) -> bool {
        matches!(self, Self::Min | Self::Max)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Product => "product",
            Self::Min => "fold-min",
            Self::Max => "fold-max",
            Self::And => "fold-and",
            Self::Or => "fold-or",
            Self::Xor => "fold-xor",
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum ProgramError {
    Lanes(LaneError),
    NotInThisDomain {
        missing: Missing,
        element: &'static str,
    },
    ShiftTooFar {
        by: u32,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub domain: Domain,
    pub subgroup: u32,
    pub workgroup: u32,
    pub groups: u32,
    pub lanes: u32,
    pub steps: Vec<Op>,
    pub finish: Finish,
}

impl Program {
    #[must_use]
    pub const fn workgroups(&self) -> u32 {
        self.groups
    }

    #[must_use]
    pub fn input_len(&self) -> usize {
        let strips = (self.lanes / self.subgroup.max(1)).max(1);
        (self.groups * self.workgroup * strips) as usize
    }

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

    fn build_in<T: Emit>(&self) -> Result<Vec<u32>, ProgramError> {
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
            Finish::Scan => {
                let scanned = kernel.lanes()?.prefix_sum(value)?;
                kernel.store(1, scanned)?;
            }
            Finish::ScanExclusive => {
                let scanned = kernel.lanes()?.prefix_sum_exclusive(value)?;
                kernel.store(1, scanned)?;
            }
            Finish::ReduceBy(fold) => {
                let total = T::reduce_by(&mut kernel.lanes()?, fold, value)?;
                kernel.store_scalar(1, total)?;
            }
            Finish::ScanBy { fold, exclusive } => {
                let scanned = T::scan_by(&mut kernel.lanes()?, fold, exclusive, value)?;
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

        assert_eq!(program.input_len(), 4 * 64 * 2);
    }

    #[test]
    fn a_float_domain_refuses_a_bit_shift_rather_than_emitting_one() {
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
