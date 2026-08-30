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
    Pow,
    Fma,
    Round,
    Trunc,
    Floor,
    Ceil,
    Sin,
    Cos,
}

impl Glsl {
    pub const SET_NAME: &'static str = "GLSL.std.450";

    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::Round => 1,
            Self::Trunc => 3,
            Self::FAbs => 4,
            Self::Floor => 8,
            Self::Ceil => 9,
            Self::Sin => 13,
            Self::Cos => 14,
            Self::Pow => 26,
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
            Self::FAbs
            | Self::SAbs
            | Self::Sqrt
            | Self::InverseSqrt
            | Self::Exp
            | Self::Log
            | Self::Round
            | Self::Trunc
            | Self::Floor
            | Self::Ceil
            | Self::Sin
            | Self::Cos => 1,
            Self::FMin
            | Self::UMin
            | Self::SMin
            | Self::FMax
            | Self::UMax
            | Self::SMax
            | Self::Pow => 2,
            Self::FClamp | Self::UClamp | Self::SClamp | Self::Fma => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_numbers_added_later_match_the_khronos_grammar_too() {
        assert_eq!(Glsl::Round.word(), 1);
        assert_eq!(Glsl::Trunc.word(), 3);
        assert_eq!(Glsl::Floor.word(), 8);
        assert_eq!(Glsl::Ceil.word(), 9);
        assert_eq!(Glsl::Sin.word(), 13);
        assert_eq!(Glsl::Cos.word(), 14);
        assert_eq!(Glsl::Pow.word(), 26);
    }

    #[test]
    fn no_two_instructions_share_a_number_and_each_says_how_many_it_takes() {
        let every = [
            Glsl::FAbs,
            Glsl::SAbs,
            Glsl::FMin,
            Glsl::UMin,
            Glsl::SMin,
            Glsl::FMax,
            Glsl::UMax,
            Glsl::SMax,
            Glsl::FClamp,
            Glsl::UClamp,
            Glsl::SClamp,
            Glsl::Sqrt,
            Glsl::InverseSqrt,
            Glsl::Exp,
            Glsl::Log,
            Glsl::Pow,
            Glsl::Fma,
            Glsl::Round,
            Glsl::Trunc,
            Glsl::Floor,
            Glsl::Ceil,
            Glsl::Sin,
            Glsl::Cos,
        ];

        let mut seen = std::collections::BTreeMap::new();
        for instruction in every {
            assert!(
                (1..=3).contains(&instruction.operands()),
                "{instruction:?} takes an operand count the grammar has no form for"
            );
            let clash = seen.insert(instruction.word(), instruction);
            assert!(
                clash.is_none(),
                "{instruction:?} and {clash:?} share the number {}",
                instruction.word()
            );
        }
        assert_eq!(seen.len(), every.len());
    }

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
