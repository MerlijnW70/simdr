use crate::encode::Word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scope {
    CrossDevice,
    Device,
    Workgroup,
    Subgroup,
    Invocation,
}

impl Scope {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::CrossDevice => 0,
            Self::Device => 1,
            Self::Workgroup => 2,
            Self::Subgroup => 3,
            Self::Invocation => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupOperation {
    Reduce,
    InclusiveScan,
    ExclusiveScan,
    ClusteredReduce,
}

impl GroupOperation {
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::Reduce => 0,
            Self::InclusiveScan => 1,
            Self::ExclusiveScan => 2,
            Self::ClusteredReduce => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scope_matches_the_khronos_grammar() {
        assert_eq!(Scope::CrossDevice.word(), 0);
        assert_eq!(Scope::Device.word(), 1);
        assert_eq!(Scope::Workgroup.word(), 2);
        assert_eq!(Scope::Subgroup.word(), 3);
        assert_eq!(Scope::Invocation.word(), 4);
    }

    #[test]
    fn every_group_operation_matches_the_khronos_grammar() {
        assert_eq!(GroupOperation::Reduce.word(), 0);
        assert_eq!(GroupOperation::InclusiveScan.word(), 1);
        assert_eq!(GroupOperation::ExclusiveScan.word(), 2);
        assert_eq!(GroupOperation::ClusteredReduce.word(), 3);
    }
}
