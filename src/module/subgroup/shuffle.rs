//! Lane exchanges — what `simd_swizzle!` and the rotates lower to.
//!
//! Needs `GroupNonUniformShuffle` for the indexed forms and `GroupNonUniformShuffleRelative` for
//! the up and down ones. Note that no instruction here takes a `GroupOperation`: a shuffle moves
//! values between lanes rather than combining them, so it has no reduction shape to name.

use crate::module::{BuildError, Id, Module, op};

impl Module {
    /// Read `value` from the lane `lane` names.
    ///
    /// `simd_swizzle!`'s general case: an arbitrary permutation in one instruction.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_shuffle(
        &mut self,
        result_type: Id,
        scope: Id,
        value: Id,
        lane: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::GROUP_NON_UNIFORM_SHUFFLE,
            result_type,
            &[scope.word(), value.word(), lane.word()],
        )
    }

    /// Read `value` from the lane whose index is ours exclusive-or `mask` — the butterfly.
    ///
    /// The exchange a hand-written tree reduction is built from, and the reason such a reduction
    /// needs no shared memory.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_shuffle_xor(
        &mut self,
        result_type: Id,
        scope: Id,
        value: Id,
        mask: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::GROUP_NON_UNIFORM_SHUFFLE_XOR,
            result_type,
            &[scope.word(), value.word(), mask.word()],
        )
    }

    /// Read `value` from the lane `delta` below ours.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_shuffle_up(
        &mut self,
        result_type: Id,
        scope: Id,
        value: Id,
        delta: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::GROUP_NON_UNIFORM_SHUFFLE_UP,
            result_type,
            &[scope.word(), value.word(), delta.word()],
        )
    }

    /// Read `value` from the lane `delta` above ours.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_shuffle_down(
        &mut self,
        result_type: Id,
        scope: Id,
        value: Id,
        delta: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::GROUP_NON_UNIFORM_SHUFFLE_DOWN,
            result_type,
            &[scope.word(), value.word(), delta.word()],
        )
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::module::Version;
    use crate::module::subgroup::test_support::operands_of;
    use crate::spec::Scope;

    #[test]
    fn a_shuffle_names_the_lane_it_reads_from_and_no_group_operation() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let value = module.constant_f32(1.0).expect("1.0");
        let lane = module.constant_u32(7).expect("7u32");

        let read = module
            .subgroup_shuffle(float, scope, value, lane)
            .expect("shuffle");

        assert_eq!(
            operands_of(&module.finish(), op::GROUP_NON_UNIFORM_SHUFFLE),
            vec![
                float.word(),
                read.word(),
                scope.word(),
                value.word(),
                lane.word()
            ],
            "scope, value, lane — a shuffle has no reduction shape to name"
        );
    }

    #[test]
    fn the_four_shuffles_are_four_instructions() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let value = module.constant_f32(1.0).expect("1.0");
        let one = module.constant_u32(1).expect("1u32");

        module
            .subgroup_shuffle(float, scope, value, one)
            .expect("shuffle");
        module
            .subgroup_shuffle_xor(float, scope, value, one)
            .expect("xor");
        module
            .subgroup_shuffle_up(float, scope, value, one)
            .expect("up");
        module
            .subgroup_shuffle_down(float, scope, value, one)
            .expect("down");

        let words = module.finish();
        let emitted: Vec<u16> = crate::decode::body(&words)
            .map(|instruction| instruction.opcode())
            .filter(|opcode| *opcode >= op::GROUP_NON_UNIFORM_ELECT)
            .collect();

        assert_eq!(
            emitted,
            vec![
                op::GROUP_NON_UNIFORM_SHUFFLE,
                op::GROUP_NON_UNIFORM_SHUFFLE_XOR,
                op::GROUP_NON_UNIFORM_SHUFFLE_UP,
                op::GROUP_NON_UNIFORM_SHUFFLE_DOWN
            ]
        );
    }

    #[test]
    fn every_shuffle_has_the_same_operand_count() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let value = module.constant_f32(1.0).expect("1.0");
        let one = module.constant_u32(1).expect("1u32");

        module
            .subgroup_shuffle_xor(float, scope, value, one)
            .expect("xor");

        assert_eq!(
            operands_of(&module.finish(), op::GROUP_NON_UNIFORM_SHUFFLE_XOR).len(),
            5
        );
    }
}
