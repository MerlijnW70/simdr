use crate::encode::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressingModel {
    Logical,
}

impl AddressingModel {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::Logical => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryModel {
    Glsl450,
    Vulkan,
}

impl MemoryModel {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::Glsl450 => 1,
            Self::Vulkan => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionModel {
    GlCompute,
}

impl ExecutionModel {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::GlCompute => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    LocalSize,
}

impl ExecutionMode {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::LocalSize => 17,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionControl {
    None,
}

impl FunctionControl {
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
