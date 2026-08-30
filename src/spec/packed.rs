use crate::encode::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedVectorFormat {
    FourEightBit,
}

impl PackedVectorFormat {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::FourEightBit => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_format_matches_the_khronos_grammar() {
        assert_eq!(PackedVectorFormat::FourEightBit.word(), 0);
    }

    #[test]
    fn zero_is_a_value_and_not_an_absence() {
        assert_eq!(PackedVectorFormat::FourEightBit.word(), 0);
        assert_eq!(
            size_of_val(&PackedVectorFormat::FourEightBit.word()),
            size_of::<Word>()
        );
    }
}
