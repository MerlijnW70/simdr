mod arithmetic;
mod shuffle;
mod vote;

use super::{BuildError, Id, Module};
use crate::spec::{GroupOperation, Scope};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    Reduce,
    InclusiveScan,
    ExclusiveScan,
    Clustered { size: Id },
}

impl Reduction {
    pub(super) const fn operation(self) -> GroupOperation {
        match self {
            Self::Reduce => GroupOperation::Reduce,
            Self::InclusiveScan => GroupOperation::InclusiveScan,
            Self::ExclusiveScan => GroupOperation::ExclusiveScan,
            Self::Clustered { .. } => GroupOperation::ClusteredReduce,
        }
    }

    pub(super) const fn cluster_size(self) -> Option<Id> {
        match self {
            Self::Clustered { size } => Some(size),
            _ => None,
        }
    }
}

impl Module {
    pub fn scope(&mut self, scope: Scope) -> Result<Id, BuildError> {
        self.constant_u32(scope.word())
    }
}

#[cfg(test)]
pub(super) mod test_support {
    use crate::decode;
    use crate::encode::Word;

    #[expect(
        clippy::expect_used,
        reason = "a test helper reports a missing instruction by panicking, which is how a test \
                  reports at all"
    )]
    pub fn operands_of(words: &[Word], opcode: u16) -> Vec<Word> {
        decode::body(words)
            .find(|instruction| instruction.opcode() == opcode)
            .expect("the instruction was emitted")
            .operands()
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::test_support::operands_of;
    use super::*;
    use crate::module::{Version, op};

    #[test]
    fn the_scope_constant_holds_the_scopes_value_rather_than_being_it() {
        let mut module = Module::new(Version::V1_3);

        let scope = module.scope(Scope::Subgroup).expect("scope");

        let declaration = operands_of(&module.finish(), op::CONSTANT);
        assert_eq!(declaration[1], scope.word(), "the constant is our scope");
        assert_eq!(declaration[2], 3, "and it holds Subgroup's value");
    }

    #[test]
    fn asking_for_one_scope_twice_declares_one_constant() {
        let mut module = Module::new(Version::V1_3);

        let first = module.scope(Scope::Subgroup).expect("scope");
        let second = module.scope(Scope::Subgroup).expect("scope again");

        assert_eq!(first, second);
    }

    #[test]
    fn two_different_scopes_are_two_constants() {
        let mut module = Module::new(Version::V1_3);

        let subgroup = module.scope(Scope::Subgroup).expect("subgroup");
        let workgroup = module.scope(Scope::Workgroup).expect("workgroup");

        assert_ne!(subgroup, workgroup);
    }

    #[test]
    fn each_reduction_shape_maps_to_its_own_literal() {
        let size = Module::new(Version::V1_3)
            .alloc_id()
            .expect("an id to stand in for the cluster size");
        let cluster = Reduction::Clustered { size };

        assert_eq!(Reduction::Reduce.operation(), GroupOperation::Reduce);
        assert_eq!(
            Reduction::InclusiveScan.operation(),
            GroupOperation::InclusiveScan
        );
        assert_eq!(
            Reduction::ExclusiveScan.operation(),
            GroupOperation::ExclusiveScan
        );
        assert_eq!(cluster.operation(), GroupOperation::ClusteredReduce);
    }

    #[test]
    fn only_a_clustered_reduction_carries_a_size() {
        let size = Module::new(Version::V1_3).alloc_id().expect("an id");

        assert!(Reduction::Reduce.cluster_size().is_none());
        assert!(Reduction::InclusiveScan.cluster_size().is_none());
        assert!(Reduction::ExclusiveScan.cluster_size().is_none());
        assert_eq!(Reduction::Clustered { size }.cluster_size(), Some(size));
    }
}
