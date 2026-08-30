use crate::encode::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Glsl {
    FAbs,
    SAbs,
    FMin,
    UMin,
    SMin,
    FMax,
    UMax,
    SMax,
    FClamp,
    UClamp,
    SClamp,
    Sqrt,
    InverseSqrt,
    Exp,
    Log,
    Fma,
}

impl Glsl {
    pub const SET_NAME: &'static str = "GLSL.std.450";

    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::FAbs => 4,
            Self::SAbs => 5,
            Self::Exp => 27,
            Self::Log => 28,
            Self::Sqrt => 31,
            Self::InverseSqrt => 32,
            Self::FMin => 37,
            Self::UMin => 38,
            Self::SMin => 39,
            Self::FMax => 40,
            Self::UMax => 41,
            Self::SMax => 42,
            Self::FClamp => 43,
            Self::UClamp => 44,
            Self::SClamp => 45,
            Self::Fma => 50,
        }
    }

    #[must_use]
    pub const fn operands(self) -> usize {
        match self {
            Self::FAbs | Self::SAbs | Self::Sqrt | Self::InverseSqrt | Self::Exp | Self::Log => 1,
            Self::FMin | Self::UMin | Self::SMin | Self::FMax | Self::UMax | Self::SMax => 2,
            Self::FClamp | Self::UClamp | Self::SClamp | Self::Fma => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_instruction_matches_the_khronos_grammar() {
        assert_eq!(Glsl::FAbs.word(), 4);
        assert_eq!(Glsl::SAbs.word(), 5);
        assert_eq!(Glsl::Exp.word(), 27);
        assert_eq!(Glsl::Log.word(), 28);
        assert_eq!(Glsl::Sqrt.word(), 31);
        assert_eq!(Glsl::InverseSqrt.word(), 32);
        assert_eq!(Glsl::FMin.word(), 37);
        assert_eq!(Glsl::UMin.word(), 38);
        assert_eq!(Glsl::SMin.word(), 39);
        assert_eq!(Glsl::FMax.word(), 40);
        assert_eq!(Glsl::UMax.word(), 41);
        assert_eq!(Glsl::SMax.word(), 42);
        assert_eq!(Glsl::FClamp.word(), 43);
        assert_eq!(Glsl::UClamp.word(), 44);
        assert_eq!(Glsl::SClamp.word(), 45);
        assert_eq!(Glsl::Fma.word(), 50);
    }

    #[test]
    fn the_set_name_is_the_string_the_implementation_matches() {
        assert_eq!(Glsl::SET_NAME, "GLSL.std.450");
    }

    #[test]
    fn no_two_instructions_share_a_number() {
        let every = [
            Glsl::FAbs,
            Glsl::SAbs,
            Glsl::Exp,
            Glsl::Log,
            Glsl::Sqrt,
            Glsl::InverseSqrt,
            Glsl::FMin,
            Glsl::UMin,
            Glsl::SMin,
            Glsl::FMax,
            Glsl::UMax,
            Glsl::SMax,
            Glsl::FClamp,
            Glsl::UClamp,
            Glsl::SClamp,
            Glsl::Fma,
        ];
        let mut numbers: Vec<Word> = every.iter().map(|instruction| instruction.word()).collect();
        numbers.sort_unstable();
        let count = numbers.len();
        numbers.dedup();

        assert_eq!(numbers.len(), count);
    }

    #[test]
    fn the_families_agree_on_how_many_operands_they_take() {
        for one in [Glsl::FAbs, Glsl::SAbs, Glsl::Sqrt, Glsl::Exp] {
            assert_eq!(one.operands(), 1);
        }
        for two in [Glsl::FMin, Glsl::UMin, Glsl::SMin, Glsl::FMax] {
            assert_eq!(two.operands(), 2);
        }
        for three in [Glsl::FClamp, Glsl::UClamp, Glsl::SClamp, Glsl::Fma] {
            assert_eq!(three.operands(), 3);
        }
    }
}
