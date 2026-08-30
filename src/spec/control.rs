use crate::encode::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectionControl {
    None,
    Flatten,
    DontFlatten,
}

impl SelectionControl {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::None => 0,
            Self::Flatten => 1,
            Self::DontFlatten => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoopControl {
    None,
    Unroll,
    DontUnroll,
}

impl LoopControl {
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
