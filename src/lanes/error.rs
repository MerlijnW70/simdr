use super::MAX_STRIPS;
use crate::module::BuildError;
use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LaneError {
    Build(BuildError),
    BadWidth {
        width: u32,
    },
    NoMapping {
        lanes: u32,
        width: u32,
    },
    TooManyStrips {
        strips: usize,
        limit: usize,
    },
    NoSuchBuffer {
        index: u32,
        bound: u32,
    },
    BadShape {
        workgroup: u32,
        buffers: u32,
    },
    BadRows {
        rows: u32,
    },
    NotAGrid,
    BadPitch,
    BadCarry {
        given: usize,
        wanted: usize,
    },
    EmptyShared,
    NoSuchForm {
        operation: &'static str,
        because: &'static str,
    },
    LaneOutOfRange {
        operation: &'static str,
        operand: u32,
        lanes: u32,
    },
    AddressOverflow {
        term: &'static str,
        needed: u64,
    },
    NoOpenBlock {
        arm: &'static str,
    },
}

impl LaneError {
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
            Self::BadCarry { given, wanted } => write!(
                f,
                "a rolled body carried {given} value(s) out where {wanted} were promised"
            ),
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
            Self::EmptyShared => write!(
                f,
                "a shared array of 0 elements has no slot that is not past its end"
            ),
            Self::NoSuchForm { operation, because } => {
                write!(f, "{operation} has no form here: {because}")
            }
            Self::LaneOutOfRange {
                operation,
                operand,
                lanes,
            } => write!(
                f,
                "{operation} was given {operand}, which is outside a group of {lanes} lanes"
            ),
            Self::AddressOverflow { term, needed } => write!(
                f,
                "the address term {term} reaches element {needed}, past the {} a 32-bit index holds",
                u32::MAX
            ),
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

    fn wanted(error: &LaneError) -> &'static str {
        match error {
            LaneError::Build(_) => "id",
            LaneError::BadCarry { .. } => "2 value(s) out where 3",
            LaneError::BadWidth { .. } => "24",
            LaneError::NoMapping { .. } => "12",
            LaneError::TooManyStrips { .. } => "16",
            LaneError::NoSuchBuffer { .. } => "buffer 2",
            LaneError::BadShape { .. } => "0 invocations",
            LaneError::BadRows { .. } => "0 rows",
            LaneError::NotAGrid => "Shape::grid",
            LaneError::BadPitch => "pitch of 0",
            LaneError::EmptyShared => "shared array of 0",
            LaneError::NoSuchForm { .. } => "clustered scan",
            LaneError::LaneOutOfRange { .. } => "outside a group of 8 lanes",
            LaneError::AddressOverflow { .. } => "4294967296",
            LaneError::NoOpenBlock { .. } => "then",
        }
    }

    #[test]
    fn every_variant_says_something_a_reader_can_act_on() {
        let cases = [
            LaneError::Build(BuildError::IdSpaceExhausted),
            LaneError::BadCarry {
                given: 2,
                wanted: 3,
            },
            LaneError::BadWidth { width: 24 },
            LaneError::NoMapping {
                lanes: 12,
                width: 32,
            },
            LaneError::TooManyStrips {
                strips: 16,
                limit: 8,
            },
            LaneError::NoSuchBuffer { index: 2, bound: 2 },
            LaneError::BadShape {
                workgroup: 0,
                buffers: 2,
            },
            LaneError::BadRows { rows: 0 },
            LaneError::NotAGrid,
            LaneError::BadPitch,
            LaneError::EmptyShared,
            LaneError::NoSuchForm {
                operation: "prefix_sum",
                because: "there is no clustered scan",
            },
            LaneError::LaneOutOfRange {
                operation: "butterfly",
                operand: 8,
                lanes: 8,
            },
            LaneError::AddressOverflow {
                term: "workgroup × strips",
                needed: 1 << 32,
            },
            LaneError::NoOpenBlock { arm: "then" },
        ];

        for case in &cases {
            let expected = wanted(case);
            let message = case.to_string();
            assert!(
                message.contains(expected),
                "{case:?} printed {message:?}, which never says {expected:?}"
            );
        }

        let details: std::collections::BTreeSet<&str> = cases.iter().map(wanted).collect();
        assert_eq!(
            details.len(),
            cases.len(),
            "two samples share a detail, so one arm of `wanted` is unreached"
        );
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
