//! What a lane operation refuses, and why.
//!
//! Every variant here is a case the mapping genuinely cannot express, named rather than
//! approximated. That is the whole discipline: a reduction over fewer lanes than were asked for
//! is a wrong answer, not a smaller one.

use super::MAX_STRIPS;
use crate::module::BuildError;
use core::fmt;

/// Something a lane operation would not do.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LaneError {
    /// The module could not be built.
    Build(BuildError),
    /// The subgroup width was reported as zero, or as something that is not a power of two.
    BadWidth {
        /// What was passed to [`super::Lanes::new`].
        width: u32,
    },
    /// `LANES` is neither a divisor nor a multiple of the subgroup width.
    ///
    /// `Simd<f32, 12>` on a 32-lane machine has no mapping: it cannot cluster, because 12 does
    /// not divide 32, and it cannot strip, because 32 does not divide 12.
    NoMapping {
        /// The lane count that was asked for.
        lanes: u32,
        /// What the hardware offers.
        width: u32,
    },
    /// The vector needs more elements per lane than fit inline.
    ///
    /// See [`MAX_STRIPS`]. Refused rather than truncated.
    TooManyStrips {
        /// How many were needed.
        strips: usize,
        /// How many fit.
        limit: usize,
    },
    /// A kernel was asked for a buffer it never bound.
    NoSuchBuffer {
        /// The index that was asked for.
        index: u32,
        /// How many were bound.
        bound: u32,
    },
    /// A kernel shape that cannot describe anything: no buffers, or no invocations.
    BadShape {
        /// Invocations per workgroup.
        workgroup: u32,
        /// How many buffers were asked for.
        buffers: u32,
    },
    /// A grid kernel whose second axis holds no invocations.
    ///
    /// Apart from [`LaneError::BadShape`] because a workgroup of `columns × 0` is not a workgroup
    /// with no invocations in it — it is a caller who wrote the wrong one of two numbers, and
    /// saying which one is the whole value of the message.
    BadRows {
        /// What was passed to [`crate::kernel::Shape::grid`].
        rows: u32,
    },
    /// A two-dimensional access on a kernel that has only one axis.
    ///
    /// [`crate::kernel::Shape::new`] builds a kernel whose address is a single index, and there is
    /// no row to compute one from. [`crate::kernel::Shape::grid`] builds one that has.
    NotAGrid,
    /// A row pitch of zero, which would put every row at the same address.
    ///
    /// Refused rather than treated as one row: a kernel that stacks every row on top of the first
    /// validates, runs, and returns whichever row happened to be written last.
    BadPitch,
    /// The operation has no form for how this vector sits on the subgroup.
    ///
    /// A clustered *scan*, for instance: SPIR-V's clustered form is a reduce, so scanning a
    /// vector narrower than the subgroup would run across lanes belonging to a different vector.
    NoSuchForm {
        /// What was asked for.
        operation: &'static str,
        /// Why it does not exist here.
        because: &'static str,
    },
    /// An arm of a selection produced a value but left no block for the merge to arrive from.
    ///
    /// An `OpPhi` names the block each value came through. An arm that ends in its own branch or
    /// return has no such block, so the value it computed cannot reach the merge — and emitting a
    /// phi that named the wrong predecessor would be a module that validates and computes the
    /// wrong thing.
    NoOpenBlock {
        /// Which arm.
        arm: &'static str,
    },
}

impl LaneError {
    /// The error for a vector that produced no strips at all.
    ///
    /// Unreachable — [`super::Vector`] refuses to exist empty — and spelled once here rather than
    /// four times at the call sites that have to name *something* for the empty case.
    pub(super) const fn no_strips() -> Self {
        Self::TooManyStrips {
            strips: 0,
            limit: MAX_STRIPS,
        }
    }
}

impl From<BuildError> for LaneError {
    fn from(error: BuildError) -> Self {
        Self::Build(error)
    }
}

impl fmt::Display for LaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Build(error) => write!(f, "{error}"),
            Self::BadWidth { width } => {
                write!(f, "a subgroup width of {width} is not a power of two")
            }
            Self::NoMapping { lanes, width } => write!(
                f,
                "{lanes} lanes neither divide nor are a multiple of a subgroup of {width}"
            ),
            Self::TooManyStrips { strips, limit } => write!(
                f,
                "that needs {strips} elements per lane and only {limit} fit inline"
            ),
            Self::NoSuchBuffer { index, bound } => {
                write!(f, "buffer {index} was asked for and only {bound} are bound")
            }
            Self::BadShape { workgroup, buffers } => write!(
                f,
                "a kernel of {workgroup} invocations over {buffers} buffers describes nothing"
            ),
            Self::BadRows { rows } => write!(
                f,
                "a grid {rows} rows deep has no invocations on its second axis"
            ),
            Self::NotAGrid => write!(
                f,
                "this kernel has one axis and no rows; Shape::grid builds one that has"
            ),
            Self::BadPitch => write!(
                f,
                "a row pitch of 0 would stack every row on the address of the first"
            ),
            Self::NoSuchForm { operation, because } => {
                write!(f, "{operation} has no form here: {because}")
            }
            Self::NoOpenBlock { arm } => write!(
                f,
                "the {arm} arm ended its own block, so its value has no edge into the merge"
            ),
        }
    }
}

impl std::error::Error for LaneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Build(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_says_something_a_reader_can_act_on() {
        // Each case pairs an error with the one detail a reader needs out of it. The pairing is
        // the test: this used to assert only that the message held *a* digit somewhere, which any
        // wrong number satisfies as well as the right one.
        let cases = [
            (LaneError::BadWidth { width: 24 }, "24"),
            (
                LaneError::NoMapping {
                    lanes: 12,
                    width: 32,
                },
                "12",
            ),
            (
                LaneError::TooManyStrips {
                    strips: 16,
                    limit: 8,
                },
                "16",
            ),
            (LaneError::NoSuchBuffer { index: 2, bound: 2 }, "buffer 2"),
            (
                LaneError::BadShape {
                    workgroup: 0,
                    buffers: 2,
                },
                "0 invocations",
            ),
            (LaneError::BadRows { rows: 0 }, "0 rows"),
            (LaneError::NotAGrid, "Shape::grid"),
            (LaneError::BadPitch, "pitch of 0"),
            (
                LaneError::NoSuchForm {
                    operation: "prefix_sum",
                    because: "there is no clustered scan",
                },
                "clustered scan",
            ),
            (LaneError::NoOpenBlock { arm: "then" }, "then"),
        ];

        for (case, expected) in cases {
            let message = case.to_string();
            assert!(
                message.contains(expected),
                "{case:?} printed {message:?}, which never says {expected:?}"
            );
        }
    }

    #[test]
    fn the_empty_strip_case_names_the_inline_limit() {
        assert_eq!(
            LaneError::no_strips(),
            LaneError::TooManyStrips {
                strips: 0,
                limit: MAX_STRIPS
            }
        );
    }
}
