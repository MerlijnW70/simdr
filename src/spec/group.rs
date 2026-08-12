//! How wide a group an instruction acts across, and what it does there.

use crate::encode::Word;

/// How wide a group of invocations an instruction acts across.
///
/// **A scope reaches SPIR-V as the id of a 32-bit integer constant, not as a literal**, which is
/// why every subgroup operation takes an [`crate::module::Id`] for it. The grammar spells this
/// `IdScope`, and it sits directly beside a [`GroupOperation`] that *is* a literal — so the two
/// operands of one instruction are encoded in opposite ways.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scope {
    /// Every invocation on every device.
    CrossDevice,
    /// Every invocation on this device.
    Device,
    /// Every invocation in the workgroup.
    Workgroup,
    /// Every invocation in the subgroup — 32 lanes on NVIDIA, 32 or 64 on AMD.
    ///
    /// This is the scope this crate exists for: it is the hardware vector unit a `Simd<T, N>`
    /// maps onto.
    Subgroup,
    /// This invocation alone.
    Invocation,
}

impl Scope {
    /// The value a constant of this scope holds.
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

/// Which shape of combination a group arithmetic instruction performs.
///
/// Unlike [`Scope`], this one *is* a literal operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupOperation {
    /// Combine every lane into one value, delivered to all of them.
    Reduce,
    /// Each lane receives the combination of every lane up to and including itself.
    InclusiveScan,
    /// Each lane receives the combination of every lane before it, itself excluded.
    ExclusiveScan,
    /// Combine within fixed-size clusters of adjacent lanes.
    ///
    /// This is how a lane count *below* the subgroup width avoids wasting hardware: rather than
    /// idling the spare lanes, a 32-wide subgroup runs four independent 8-lane reductions at once.
    ClusteredReduce,
}

impl GroupOperation {
    /// The word this encodes to.
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
