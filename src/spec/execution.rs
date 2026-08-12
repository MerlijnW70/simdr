//! How a module is executed: its memory model, its entry point, its workgroup.

use crate::encode::Word;

/// How pointers behave (`OpMemoryModel`, first operand).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressingModel {
    /// No numeric addresses: a pointer is an opaque id. The only model Vulkan accepts without an
    /// extension, and the only one this crate emits.
    Logical,
}

impl AddressingModel {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::Logical => 0,
        }
    }
}

/// Which memory model's rules apply (`OpMemoryModel`, second operand).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryModel {
    /// The GLSL 450 model — what nearly every Vulkan shader in the wild declares.
    Glsl450,
    /// The Vulkan memory model: stronger guarantees, and required before the availability and
    /// visibility operands mean anything. Needs `VulkanMemoryModel`, which is not declared here.
    Vulkan,
}

impl MemoryModel {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::Glsl450 => 1,
            Self::Vulkan => 3,
        }
    }
}

/// What kind of pipeline stage an entry point is (`OpEntryPoint`, first operand).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionModel {
    /// A compute shader. The only stage this crate targets: SIMD work has no use for a
    /// rasteriser.
    GlCompute,
}

impl ExecutionModel {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::GlCompute => 5,
        }
    }
}

/// A mode an entry point runs under (`OpExecutionMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    /// The workgroup dimensions, as three literal operands. A compute entry point is invalid
    /// without one.
    LocalSize,
}

impl ExecutionMode {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::LocalSize => 17,
        }
    }
}

/// Hints attached to a function definition (`OpFunction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionControl {
    /// No hint. A bitmask in the specification, and this is its empty value.
    None,
}

impl FunctionControl {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_execution_value_matches_the_khronos_grammar() {
        assert_eq!(AddressingModel::Logical.word(), 0);
        assert_eq!(MemoryModel::Glsl450.word(), 1);
        assert_eq!(MemoryModel::Vulkan.word(), 3);
        assert_eq!(ExecutionModel::GlCompute.word(), 5);
        assert_eq!(ExecutionMode::LocalSize.word(), 17);
        assert_eq!(FunctionControl::None.word(), 0);
    }
}
