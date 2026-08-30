use crate::encode::{EncodeError, Word};
use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id(u32);

impl Id {
    pub(super) const fn new(raw: u32) -> Self {
        Self(raw)
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    major: u8,
    minor: u8,
}

impl Version {
    pub const V1_0: Self = Self { major: 1, minor: 0 };
    pub const V1_3: Self = Self { major: 1, minor: 3 };

    #[must_use]
    pub const fn word(self) -> Word {
        (self.major as Word) << 16 | (self.minor as Word) << 8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Section {
    Capability,
    Extension,
    ExtInstImport,
    MemoryModel,
    EntryPoint,
    ExecutionMode,
    Debug,
    Annotation,
    TypeConstantVariable,
    Function,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuildError {
    Encode(EncodeError),
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
