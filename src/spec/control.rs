//! Hints on the structured control-flow instructions.

use crate::encode::Word;

/// Hints attached to a selection (`OpSelectionMerge`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectionControl {
    /// No hint. A bitmask in the specification, and this is its empty value.
    None,
    /// Prefer to compute both arms and select, rather than branch.
    Flatten,
    /// Prefer to branch.
    DontFlatten,
}

impl SelectionControl {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::None => 0,
            Self::Flatten => 1,
            Self::DontFlatten => 2,
        }
    }
}

/// Hints attached to a loop (`OpLoopMerge`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoopControl {
    /// No hint.
    None,
    /// Prefer to unroll.
    Unroll,
    /// Prefer not to.
    DontUnroll,
}

impl LoopControl {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::None => 0,
            Self::Unroll => 1,
            Self::DontUnroll => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_control_hint_matches_the_khronos_grammar() {
        assert_eq!(SelectionControl::None.word(), 0);
        assert_eq!(SelectionControl::Flatten.word(), 1);
        assert_eq!(SelectionControl::DontFlatten.word(), 2);

        assert_eq!(LoopControl::None.word(), 0);
        assert_eq!(LoopControl::Unroll.word(), 1);
        assert_eq!(LoopControl::DontUnroll.word(), 2);
    }
}
