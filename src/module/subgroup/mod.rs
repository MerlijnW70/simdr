//! Subgroup operations — the instructions this crate exists to emit.
//!
//! A subgroup is a hardware vector unit whose lanes are spelled as threads: 32 on NVIDIA, 32 or
//! 64 on AMD. Everything here treats it as the vector it is, which is what makes `Simd<T, N>`
//! lowerable onto it — an elementwise `+` is free because each lane already holds its own
//! element, and only the operations that *cross* lanes need an instruction.
//!
//! # Two operands, two encodings
//!
//! Every instruction here takes an execution scope, and the scope is **the id of a 32-bit integer
//! constant** rather than a literal (`IdScope` in the grammar). On an arithmetic instruction it
//! sits directly beside a `GroupOperation`, which *is* a literal. Getting them the wrong way
//! round produces a module that assembles and means something else, so [`Module::scope`] exists
//! to make the constant and nothing here takes a bare number.
//!
//! # Capabilities
//!
//! These are gated, and the split of this module into files follows the gates:
//! `GroupNonUniform` for `elect`, `GroupNonUniformVote` for the votes,
//! `GroupNonUniformBallot` for ballots and broadcasts, `GroupNonUniformShuffle` and
//! `ShuffleRelative` for [`shuffle`], `GroupNonUniformArithmetic` for [`arithmetic`], and
//! `GroupNonUniformClustered` on top of that for [`Reduction::Clustered`]. Nothing here declares
//! them for you; the validator is what notices.

mod arithmetic;
mod shuffle;
mod vote;

use super::{BuildError, Id, Module};
use crate::spec::{GroupOperation, Scope};

/// Which shape of reduction to perform, carrying whatever that shape needs.
///
/// `ClusterSize` is only allowed when the operation is `ClusteredReduce`, and is required then.
/// Pairing them in one type makes both halves of that rule unstateable rather than merely
/// documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// Combine every lane into one value, delivered to all of them.
    Reduce,
    /// Each lane receives the combination of every lane up to and including itself.
    InclusiveScan,
    /// Each lane receives the combination of every lane before it, itself excluded.
    ExclusiveScan,
    /// Combine within clusters of adjacent lanes.
    ///
    /// `size` is the id of a 32-bit integer constant, and must be a power of two no larger than
    /// the subgroup. This is how a `Simd<T, N>` narrower than the hardware avoids idling it: a
    /// 32-lane subgroup running `Clustered { size: 8 }` performs four independent 8-lane
    /// reductions in one instruction rather than wasting twenty-four lanes.
    Clustered {
        /// The id of the cluster-size constant.
        size: Id,
    },
}

impl Reduction {
    /// The literal operand this encodes to.
    pub(super) const fn operation(self) -> GroupOperation {
        match self {
            Self::Reduce => GroupOperation::Reduce,
            Self::InclusiveScan => GroupOperation::InclusiveScan,
            Self::ExclusiveScan => GroupOperation::ExclusiveScan,
            Self::Clustered { .. } => GroupOperation::ClusteredReduce,
        }
    }

    /// The trailing cluster-size operand, when there is one.
    pub(super) const fn cluster_size(self) -> Option<Id> {
        match self {
            Self::Clustered { size } => Some(size),
            _ => None,
        }
    }
}

impl Module {
    /// A constant holding `scope`, for the execution-scope operand every subgroup instruction
    /// takes.
    ///
    /// Deduplicated like any other `u32` constant, so asking repeatedly costs nothing.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the constant cannot be declared.
    pub fn scope(&mut self, scope: Scope) -> Result<Id, BuildError> {
        self.constant_u32(scope.word())
    }
}

/// Shared by the tests in this module's files.
#[cfg(test)]
pub(super) mod test_support {
    use crate::decode;
    use crate::encode::Word;

    /// The operands of the one instruction in `words` carrying `opcode`.
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
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::test_support::operands_of;
    use super::*;
    use crate::module::{Version, op};

    #[test]
    fn the_scope_constant_holds_the_scopes_value_rather_than_being_it() {
        let mut module = Module::new(Version::V1_3);

        let scope = module.scope(Scope::Subgroup).expect("scope");

        // The constant's own id is whatever the allocator handed out; its *value* is 3.
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
