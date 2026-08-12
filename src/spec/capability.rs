//! What a module declares it needs.

use crate::encode::Word;

/// A capability a module declares it needs (`OpCapability`).
///
/// A consumer refuses a module that uses a feature without declaring the capability for it, so
/// these are not documentation — leaving one out is a validation failure. Declaring a *surplus*
/// one is worse: a module that names `GroupNonUniformClustered` will not run on a device that
/// does not offer it, even if nothing in the module would have used it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Shaders: the baseline for anything running under Vulkan.
    Shader,
    /// The subgroup instructions exist at all.
    GroupNonUniform,
    /// Subgroup vote: `OpGroupNonUniformAll`, `Any`, `AllEqual`.
    GroupNonUniformVote,
    /// Subgroup reductions and scans — what `Simd::reduce_sum` lowers to.
    GroupNonUniformArithmetic,
    /// Subgroup ballot and broadcast — what a `Mask` lowers to.
    GroupNonUniformBallot,
    /// Arbitrary subgroup shuffles — what `simd_swizzle!` lowers to.
    GroupNonUniformShuffle,
    /// Relative shuffles: up and down by a delta, so a rotate costs one instruction.
    GroupNonUniformShuffleRelative,
    /// Clustered reductions: several independent vectors packed into one subgroup, which is how
    /// a lane count below the subgroup width avoids idling the rest of the hardware.
    GroupNonUniformClustered,
    /// 8-bit integers exist as a type.
    ///
    /// Core SPIR-V since 1.0 and needs no extension — but Vulkan gates it behind the `shaderInt8`
    /// feature, so a device may validate a module it will not run.
    Int8,
    /// 16-bit integers exist as a type. Vulkan's `shaderInt16`.
    Int16,
    /// 16-bit floats exist as a type. Vulkan's `shaderFloat16`.
    Float16,
    /// A storage buffer may hold 8-bit types.
    ///
    /// Separate from [`Capability::Int8`], and the separation is the whole point of the narrow
    /// types: 8-bit *arithmetic* is one capability and 8-bit *memory* is another, and it is the
    /// second one that makes a buffer a quarter of the size. Needs `SPV_KHR_8bit_storage` below
    /// SPIR-V 1.5, which is every module this crate emits.
    StorageBuffer8BitAccess,
    /// A storage buffer may hold 16-bit types.
    ///
    /// Core in SPIR-V 1.3 — unlike its 8-bit counterpart, which arrived two versions later — so a
    /// module emitting this declares no extension for it.
    StorageBuffer16BitAccess,
}

impl Capability {
    /// The word this encodes to.
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
        }
    }

    /// The SPIR-V extension this capability needs, in the versions this crate emits.
    ///
    /// **Version-dependent, and stated for SPIR-V 1.3.** The 16-bit storage capability was
    /// promoted to core in 1.3 and the 8-bit one only in 1.5, so at the version this crate emits
    /// exactly one of the two still needs an `OpExtension`. Declaring it at 1.5 would be harmless;
    /// leaving it out at 1.3 is a rejected module.
    #[must_use]
    pub const fn extension(self) -> Option<&'static str> {
        match self {
            Self::StorageBuffer8BitAccess => Some("SPV_KHR_8bit_storage"),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Capability; 13] = [
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
    }

    #[test]
    fn only_the_8_bit_storage_capability_needs_an_extension_at_this_version() {
        // The asymmetry that would be easy to get wrong in either direction: 16-bit storage is
        // core in SPIR-V 1.3, 8-bit only in 1.5, and this crate emits 1.3.
        assert_eq!(
            Capability::StorageBuffer8BitAccess.extension(),
            Some("SPV_KHR_8bit_storage")
        );
        assert_eq!(Capability::StorageBuffer16BitAccess.extension(), None);

        for capability in ALL {
            if capability != Capability::StorageBuffer8BitAccess {
                assert_eq!(capability.extension(), None, "{capability:?}");
            }
        }
    }

    #[test]
    fn the_type_capabilities_are_separate_from_the_storage_ones() {
        // Declaring `Int8` says the module computes in 8-bit integers; declaring
        // `StorageBuffer8BitAccess` says a buffer holds them. A device can offer the first and
        // not the second, so a module that conflated them would refuse to run on it.
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
