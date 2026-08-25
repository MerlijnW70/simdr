//! Votes, ballots and broadcasts — what a `Mask<T, N>` lowers to.
//!
//! Needs `GroupNonUniform` for [`Module::subgroup_elect`], `GroupNonUniformVote` for the votes,
//! and `GroupNonUniformBallot` for the rest.

use crate::module::{BuildError, Id, Module, op};

impl Module {
    /// True in exactly one lane of the group, and false in the rest.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_elect(&mut self, bool_type: Id, scope: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::GROUP_NON_UNIFORM_ELECT, bool_type, &[scope.word()])
    }

    /// True when `predicate` holds in every active lane — `Mask::all`.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_all(
        &mut self,
        bool_type: Id,
        scope: Id,
        predicate: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::GROUP_NON_UNIFORM_ALL,
            bool_type,
            &[scope.word(), predicate.word()],
        )
    }

    /// True when `predicate` holds in any active lane — `Mask::any`.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_any(
        &mut self,
        bool_type: Id,
        scope: Id,
        predicate: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::GROUP_NON_UNIFORM_ANY,
            bool_type,
            &[scope.word(), predicate.word()],
        )
    }

    /// True when every active lane holds the same `value`.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_all_equal(
        &mut self,
        bool_type: Id,
        scope: Id,
        value: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::GROUP_NON_UNIFORM_ALL_EQUAL,
            bool_type,
            &[scope.word(), value.word()],
        )
    }

    /// Every lane's `predicate`, gathered into a bitmask.
    ///
    /// `result_type` must be a four-component vector of `u32` — 128 bits, enough for the widest
    /// subgroup any implementation reports. This is a `Mask<T, N>` made explicit.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_ballot(
        &mut self,
        result_type: Id,
        scope: Id,
        predicate: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::GROUP_NON_UNIFORM_BALLOT,
            result_type,
            &[scope.word(), predicate.word()],
        )
    }

    /// The value held by the lane `lane` names, delivered to every lane.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_broadcast(
        &mut self,
        result_type: Id,
        scope: Id,
        value: Id,
        lane: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::GROUP_NON_UNIFORM_BROADCAST,
            result_type,
            &[scope.word(), value.word(), lane.word()],
        )
    }

    /// The value held by the lowest-numbered active lane, delivered to every lane.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn subgroup_broadcast_first(
        &mut self,
        result_type: Id,
        scope: Id,
        value: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::GROUP_NON_UNIFORM_BROADCAST_FIRST,
            result_type,
            &[scope.word(), value.word()],
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
    fn elect_takes_only_a_scope() {
        let mut module = Module::new(Version::V1_3);
        let boolean = module.type_bool().expect("bool");
        let scope = module.scope(Scope::Subgroup).expect("scope");

        let elected = module.subgroup_elect(boolean, scope).expect("elect");

        assert_eq!(
            operands_of(&module.finish(), op::GROUP_NON_UNIFORM_ELECT),
            vec![boolean.word(), elected.word(), scope.word()]
        );
    }

    #[test]
    fn a_vote_takes_a_scope_and_a_predicate() {
        let mut module = Module::new(Version::V1_3);
        let boolean = module.type_bool().expect("bool");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let predicate = module.constant_bool(true).expect("true");

        let verdict = module.subgroup_all(boolean, scope, predicate).expect("all");

        assert_eq!(
            operands_of(&module.finish(), op::GROUP_NON_UNIFORM_ALL),
            vec![
                boolean.word(),
                verdict.word(),
                scope.word(),
                predicate.word()
            ]
        );
    }

    #[test]
    fn all_and_any_are_different_instructions() {
        let mut module = Module::new(Version::V1_3);
        let boolean = module.type_bool().expect("bool");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let predicate = module.constant_bool(true).expect("true");

        module.subgroup_all(boolean, scope, predicate).expect("all");
        module.subgroup_any(boolean, scope, predicate).expect("any");

        let words = module.finish();
        let emitted: Vec<u16> = crate::decode::body(&words)
            .map(|instruction| instruction.opcode())
            .filter(|opcode| *opcode >= op::GROUP_NON_UNIFORM_ELECT)
            .collect();

        assert_eq!(
            emitted,
            vec![op::GROUP_NON_UNIFORM_ALL, op::GROUP_NON_UNIFORM_ANY]
        );
    }

    #[test]
    fn a_ballot_reduces_a_predicate_to_a_mask() {
        let mut module = Module::new(Version::V1_3);
        let uint = module.type_int(32, false).expect("u32");
        let uint4 = module.type_vector(uint, 4).expect("uvec4");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let predicate = module.constant_bool(true).expect("true");

        let mask = module
            .subgroup_ballot(uint4, scope, predicate)
            .expect("ballot");

        assert_eq!(
            operands_of(&module.finish(), op::GROUP_NON_UNIFORM_BALLOT),
            vec![uint4.word(), mask.word(), scope.word(), predicate.word()]
        );
    }

    #[test]
    fn a_broadcast_names_the_lane_and_broadcast_first_does_not() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let value = module.constant_f32(1.0).expect("1.0");
        let lane = module.constant_u32(3).expect("3u32");

        module
            .subgroup_broadcast(float, scope, value, lane)
            .expect("broadcast");
        module
            .subgroup_broadcast_first(float, scope, value)
            .expect("broadcast first");

        let words = module.finish();

        assert_eq!(
            operands_of(&words, op::GROUP_NON_UNIFORM_BROADCAST).len(),
            5
        );
        assert_eq!(
            operands_of(&words, op::GROUP_NON_UNIFORM_BROADCAST_FIRST).len(),
            4
        );
    }

    #[test]
    fn a_vote_on_a_value_takes_the_value_itself_and_not_a_predicate_about_it() {
        // `OpGroupNonUniformAllEqual` is shaped like the votes above and asks a different question:
        // its operand is the value being compared across lanes rather than a boolean, and only the
        // result is a boolean. It sat in this crate with no caller until an audit found it, which
        // is why it gets its own check rather than sharing one with `all` and `any`.
        let mut module = Module::new(Version::V1_3);
        let boolean = module.type_bool().expect("bool");
        let value = module.constant_f32(1.5).expect("1.5");
        let scope = module.scope(Scope::Subgroup).expect("subgroup");

        let agreed = module
            .subgroup_all_equal(boolean, scope, value)
            .expect("all_equal");

        assert_eq!(
            operands_of(&module.finish(), op::GROUP_NON_UNIFORM_ALL_EQUAL),
            vec![boolean.word(), agreed.word(), scope.word(), value.word()],
            "the result type is the boolean and the operand is the value, not the other way round"
        );
    }
}
