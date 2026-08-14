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
    /// The integer dot-product instructions exist.
    ///
    /// `OpSDot` and its relatives: several narrow products summed into one wider accumulator, in
    /// a single instruction. This says the *instructions* exist; a second capability says which
    /// input formats they accept.
    DotProduct,
    /// A dot product may take its four 8-bit inputs **packed into a 32-bit integer**.
    ///
    /// The form both devices here report as accelerated. Note what it is not: this does not make
    /// `Simd<i8, N>` four elements per lane — `decisions/DR-0004` is unchanged. The packing is in
    /// the *instruction's operands*, which happen to be 32-bit integers whose bytes it reads.
    DotProductInput4x8BitPacked,
}

impl Capability {
    /// Every capability this crate can declare.
    ///
    /// Here rather than only in the tests because a consumer needs it: `runner` reads the
    /// `OpCapability` instructions back out of a finished module and asks the device whether it
    /// offers each one, which is a loop over this list — and a capability added to the enum and
    /// not to it would be a module requirement nothing checks.
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

    /// The capability a word names, or `None` for one this crate does not know.
    ///
    /// The inverse of [`Capability::word`], and the reason a module can be read back: a device
    /// refuses a pipeline whose module declares something it does not offer, with a message that
    /// names neither. Decoding the declaration is how a caller can be told which one.
    #[must_use]
    pub fn from_word(word: Word) -> Option<Self> {
        Self::ALL.into_iter().find(|known| known.word() == word)
    }

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
            Self::DotProductInput4x8BitPacked => 6018,
            Self::DotProduct => 6019,
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
            // Core only in SPIR-V 1.6, which is three versions above what this crate emits.
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
        // The inverse a consumer reads a module back with. A capability missing from `ALL` would
        // decode to `None` and be reported as a module requirement nobody can name — which is the
        // failure this exists to prevent, so the check is that the round trip is total.
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
        // The asymmetry that would be easy to get wrong in either direction: 16-bit storage is
        // core in SPIR-V 1.3, 8-bit only in 1.5, and the dot product only in 1.6. This crate emits
        // 1.3, so two of the three need an `OpExtension` beside them and one does not.
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
