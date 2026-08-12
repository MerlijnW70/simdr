//! The shapes a module has before any instruction is in it: ids, versions, sections, failures.

use crate::encode::{EncodeError, Word};
use core::fmt;

/// A result id.
///
/// Ids are allocated by [`super::Module::alloc_id`] and start at one — zero is not a valid id,
/// which is why this wraps the number rather than letting a caller pick it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id(u32);

impl Id {
    /// Wrap a raw number.
    ///
    /// Crate-private on purpose: an id a caller invented is an id nothing declared, and the
    /// difference does not show up until a validator reads the module.
    pub(super) const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// The id as the word it encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        self.0
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Which SPIR-V version a module declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    major: u8,
    minor: u8,
}

impl Version {
    /// SPIR-V 1.0 — Vulkan 1.0.
    pub const V1_0: Self = Self { major: 1, minor: 0 };
    /// SPIR-V 1.3 — Vulkan 1.1, and the first version with the subgroup operations this crate
    /// exists to emit. Anything targeting `GroupNonUniform*` needs at least this.
    pub const V1_3: Self = Self { major: 1, minor: 3 };

    /// The version word as it appears in the header (§2.3).
    #[must_use]
    pub const fn word(self) -> Word {
        (self.major as Word) << 16 | (self.minor as Word) << 8
    }
}

/// Where in the logical layout (§2.4) an instruction belongs.
///
/// The order of these variants *is* the required order, and each section is buffered separately
/// so that emitting out of order is impossible rather than merely discouraged. A validator
/// rejects a module whose sections are shuffled, and that is a tedious failure to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Section {
    /// `OpCapability`.
    Capability,
    /// `OpExtension`.
    Extension,
    /// `OpExtInstImport`.
    ExtInstImport,
    /// `OpMemoryModel`.
    MemoryModel,
    /// `OpEntryPoint`.
    EntryPoint,
    /// `OpExecutionMode` and `OpExecutionModeId`.
    ExecutionMode,
    /// Debug instructions: `OpName`, `OpMemberName`, `OpString`, and the source ones.
    Debug,
    /// `OpDecorate` and the rest of the annotations.
    Annotation,
    /// Types, constants, and global variables — one section, because they interleave.
    TypeConstantVariable,
    /// Function declarations and definitions.
    Function,
}

/// Something that stopped a module being built.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuildError {
    /// An instruction could not be encoded.
    Encode(EncodeError),
    /// Every one of the 2³²−1 result ids had been handed out.
    ///
    /// Unreachable in any module a GPU would accept, and present because the alternative is an
    /// overflow that aborts the process.
    IdSpaceExhausted,
}

impl From<EncodeError> for BuildError {
    fn from(error: EncodeError) -> Self {
        Self::Encode(error)
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(error) => write!(f, "{error}"),
            Self::IdSpaceExhausted => f.write_str("the module has used every available result id"),
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encode(error) => Some(error),
            Self::IdSpaceExhausted => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_word_places_major_and_minor_in_their_own_bytes() {
        assert_eq!(Version::V1_0.word(), 0x0001_0000);
        assert_eq!(Version::V1_3.word(), 0x0001_0300);
    }

    #[test]
    fn an_id_displays_the_way_a_disassembly_writes_it() {
        assert_eq!(Id::new(1).to_string(), "%1");
        assert_eq!(Id::new(37).to_string(), "%37");
    }

    #[test]
    fn the_sections_are_ordered_the_way_the_layout_requires() {
        // Their order *is* the specification's, so a reordering of the enum is a silent
        // miscompilation of every module. This is the test that would notice.
        assert!(Section::Capability < Section::MemoryModel);
        assert!(Section::MemoryModel < Section::EntryPoint);
        assert!(Section::EntryPoint < Section::ExecutionMode);
        assert!(Section::ExecutionMode < Section::Debug);
        assert!(Section::Debug < Section::Annotation);
        assert!(Section::Annotation < Section::TypeConstantVariable);
        assert!(Section::TypeConstantVariable < Section::Function);
    }

    #[test]
    fn a_build_error_carries_its_encoding_cause() {
        let error = BuildError::from(EncodeError::InstructionTooLong {
            opcode: 5,
            words: 70_000,
        });

        assert!(std::error::Error::source(&error).is_some());
        assert!(error.to_string().contains("70000"));
    }

    #[test]
    fn running_out_of_ids_has_no_cause_beyond_itself() {
        let error = BuildError::IdSpaceExhausted;

        assert!(std::error::Error::source(&error).is_none());
        assert!(error.to_string().contains("result id"));
    }
}
