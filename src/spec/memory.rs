use crate::encode::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageClass {
    Input,
    Workgroup,
    Private,
    Function,
    StorageBuffer,
}

impl StorageClass {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::Input => 1,
            Self::Workgroup => 4,
            Self::Private => 6,
            Self::Function => 7,
            Self::StorageBuffer => 12,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemorySemantics {
    None,
    AcquireReleaseWorkgroup,
    AcquireReleaseBuffer,
}

impl MemorySemantics {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::None => 0,
            Self::AcquireReleaseWorkgroup => 0x108,
            Self::AcquireReleaseBuffer => 0x48,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Decoration {
    Block,
    ArrayStride,
    SpecId,
    Offset,
    Binding,
    DescriptorSet,
    BuiltIn,
}

impl Decoration {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::SpecId => 1,
            Self::Block => 2,
            Self::ArrayStride => 6,
            Self::BuiltIn => 11,
            Self::Binding => 33,
            Self::DescriptorSet => 34,
            Self::Offset => 35,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltIn {
    WorkgroupId,
    LocalInvocationId,
    GlobalInvocationId,
    SubgroupSize,
    NumSubgroups,
    SubgroupLocalInvocationId,
}

impl BuiltIn {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::WorkgroupId => 26,
            Self::LocalInvocationId => 27,
            Self::GlobalInvocationId => 28,
            Self::SubgroupSize => 36,
            Self::NumSubgroups => 38,
            Self::SubgroupLocalInvocationId => 41,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_storage_class_matches_the_khronos_grammar() {
        assert_eq!(StorageClass::Input.word(), 1);
        assert_eq!(StorageClass::Workgroup.word(), 4);
        assert_eq!(MemorySemantics::None.word(), 0);
        assert_eq!(MemorySemantics::AcquireReleaseWorkgroup.word(), 264);
        assert_eq!(MemorySemantics::AcquireReleaseBuffer.word(), 72);
        assert_eq!(StorageClass::Private.word(), 6);
        assert_eq!(StorageClass::Function.word(), 7);
        assert_eq!(StorageClass::StorageBuffer.word(), 12);
    }

    #[test]
    fn every_decoration_matches_the_khronos_grammar() {
        assert_eq!(Decoration::SpecId.word(), 1);
        assert_eq!(Decoration::Block.word(), 2);
        assert_eq!(Decoration::ArrayStride.word(), 6);
        assert_eq!(Decoration::BuiltIn.word(), 11);
        assert_eq!(Decoration::Binding.word(), 33);
        assert_eq!(Decoration::DescriptorSet.word(), 34);
        assert_eq!(Decoration::Offset.word(), 35);
    }

    #[test]
    fn every_builtin_matches_the_khronos_grammar() {
        assert_eq!(BuiltIn::WorkgroupId.word(), 26);
        assert_eq!(BuiltIn::LocalInvocationId.word(), 27);
        assert_eq!(BuiltIn::GlobalInvocationId.word(), 28);
        assert_eq!(BuiltIn::SubgroupSize.word(), 36);
        assert_eq!(BuiltIn::NumSubgroups.word(), 38);
        assert_eq!(BuiltIn::SubgroupLocalInvocationId.word(), 41);
    }
}
