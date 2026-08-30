use crate::encode::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    Shader,
    GroupNonUniform,
    GroupNonUniformVote,
    GroupNonUniformArithmetic,
    GroupNonUniformBallot,
    GroupNonUniformShuffle,
    GroupNonUniformShuffleRelative,
    GroupNonUniformClustered,
    Int8,
    Int16,
    Float16,
    StorageBuffer8BitAccess,
    StorageBuffer16BitAccess,
    DotProduct,
    DotProductInput4x8BitPacked,
}

impl Capability {
    pub const ALL: [Self; 15] = [
        Self::Shader,
        Self::GroupNonUniform,
        Self::GroupNonUniformVote,
        Self::GroupNonUniformArithmetic,
        Self::GroupNonUniformBallot,
        Self::GroupNonUniformShuffle,
        Self::GroupNonUniformShuffleRelative,
        Self::GroupNonUniformClustered,
        Self::Int8,
        Self::Int16,
        Self::Float16,
        Self::StorageBuffer8BitAccess,
        Self::StorageBuffer16BitAccess,
        Self::DotProduct,
        Self::DotProductInput4x8BitPacked,
    ];

    #[must_use]
    pub fn from_word(word: Word) -> Option<Self> {
        Self::ALL.into_iter().find(|known| known.word() == word)
    }

    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::Shader => 1,
            Self::GroupNonUniform => 61,
            Self::GroupNonUniformVote => 62,
            Self::GroupNonUniformArithmetic => 63,
            Self::GroupNonUniformBallot => 64,
            Self::GroupNonUniformShuffle => 65,
            Self::GroupNonUniformShuffleRelative => 66,
            Self::GroupNonUniformClustered => 67,
            Self::Int16 => 22,
            Self::Int8 => 39,
            Self::Float16 => 9,
            Self::StorageBuffer16BitAccess => 4433,
            Self::StorageBuffer8BitAccess => 4448,
            Self::DotProductInput4x8BitPacked => 6018,
            Self::DotProduct => 6019,
        }
    }

    #[must_use]
    pub const fn extension(self) -> Option<&'static str> {
        match self {
            Self::StorageBuffer8BitAccess => Some("SPV_KHR_8bit_storage"),
            Self::DotProduct | Self::DotProductInput4x8BitPacked => {
                Some("SPV_KHR_integer_dot_product")
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Capability; 15] = Capability::ALL;

    #[allow(
        dead_code,
        reason = "kept as the list the round-trip below is written against"
    )]
    const SPELLED_OUT: [Capability; 15] = [
        Capability::DotProduct,
        Capability::DotProductInput4x8BitPacked,
        Capability::Shader,
        Capability::GroupNonUniform,
        Capability::GroupNonUniformVote,
        Capability::GroupNonUniformArithmetic,
        Capability::GroupNonUniformBallot,
        Capability::GroupNonUniformShuffle,
        Capability::GroupNonUniformShuffleRelative,
        Capability::GroupNonUniformClustered,
        Capability::Int8,
        Capability::Int16,
        Capability::Float16,
        Capability::StorageBuffer8BitAccess,
        Capability::StorageBuffer16BitAccess,
    ];

    #[test]
    fn every_capability_is_named_by_its_own_word() {
        for capability in Capability::ALL {
            assert_eq!(
                Capability::from_word(capability.word()),
                Some(capability),
                "{capability:?} does not decode back to itself"
            );
        }
        assert_eq!(
            Capability::ALL.len(),
            SPELLED_OUT.len(),
            "a capability was added to the enum and not to `ALL`"
        );
        assert_eq!(
            Capability::from_word(0xFFFF),
            None,
            "and a word nobody claims is nobody's"
        );
    }

    #[test]
    fn every_capability_matches_the_khronos_grammar() {
        assert_eq!(Capability::Shader.word(), 1);
        assert_eq!(Capability::GroupNonUniform.word(), 61);
        assert_eq!(Capability::GroupNonUniformVote.word(), 62);
        assert_eq!(Capability::GroupNonUniformArithmetic.word(), 63);
        assert_eq!(Capability::GroupNonUniformBallot.word(), 64);
        assert_eq!(Capability::GroupNonUniformShuffle.word(), 65);
        assert_eq!(Capability::GroupNonUniformShuffleRelative.word(), 66);
        assert_eq!(Capability::GroupNonUniformClustered.word(), 67);
        assert_eq!(Capability::Float16.word(), 9);
        assert_eq!(Capability::Int16.word(), 22);
        assert_eq!(Capability::Int8.word(), 39);
        assert_eq!(Capability::StorageBuffer16BitAccess.word(), 4433);
        assert_eq!(Capability::StorageBuffer8BitAccess.word(), 4448);
        assert_eq!(Capability::DotProductInput4x8BitPacked.word(), 6018);
        assert_eq!(Capability::DotProduct.word(), 6019);
    }

    #[test]
    fn a_capability_names_the_extension_it_needs_at_this_version() {
        assert_eq!(
            Capability::StorageBuffer8BitAccess.extension(),
            Some("SPV_KHR_8bit_storage")
        );
        assert_eq!(
            Capability::DotProduct.extension(),
            Some("SPV_KHR_integer_dot_product")
        );
        assert_eq!(
            Capability::DotProductInput4x8BitPacked.extension(),
            Some("SPV_KHR_integer_dot_product")
        );
        assert_eq!(Capability::StorageBuffer16BitAccess.extension(), None);

        let needs_one = [
            Capability::StorageBuffer8BitAccess,
            Capability::DotProduct,
            Capability::DotProductInput4x8BitPacked,
        ];
        for capability in ALL {
            if !needs_one.contains(&capability) {
                assert_eq!(capability.extension(), None, "{capability:?}");
            }
        }
    }

    #[test]
    fn the_type_capabilities_are_separate_from_the_storage_ones() {
        assert_ne!(
            Capability::Int8.word(),
            Capability::StorageBuffer8BitAccess.word()
        );
        assert_ne!(
            Capability::Int16.word(),
            Capability::StorageBuffer16BitAccess.word()
        );
    }

    #[test]
    fn no_two_capabilities_share_a_word() {
        let mut words: Vec<Word> = ALL.iter().map(|capability| capability.word()).collect();
        words.sort_unstable();
        let count = words.len();
        words.dedup();

        assert_eq!(words.len(), count, "a copy-paste slip would show up here");
    }
}
