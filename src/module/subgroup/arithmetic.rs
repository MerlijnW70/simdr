use super::Reduction;
use crate::module::{BuildError, Id, Module, op};

impl Module {
    pub fn subgroup_reduce(
        &mut self,
        opcode: u16,
        result_type: Id,
        scope: Id,
        reduction: Reduction,
        value: Id,
    ) -> Result<Id, BuildError> {
        self.group_arithmetic(opcode, result_type, scope, reduction, value)
    }

    pub fn subgroup_f_add(
        &mut self,
        result_type: Id,
        scope: Id,
        reduction: Reduction,
        value: Id,
    ) -> Result<Id, BuildError> {
        self.group_arithmetic(
            op::GROUP_NON_UNIFORM_F_ADD,
            result_type,
            scope,
            reduction,
            value,
        )
    }

    pub fn subgroup_i_add(
        &mut self,
        result_type: Id,
        scope: Id,
        reduction: Reduction,
        value: Id,
    ) -> Result<Id, BuildError> {
        self.group_arithmetic(
            op::GROUP_NON_UNIFORM_I_ADD,
            result_type,
            scope,
            reduction,
            value,
        )
    }

    pub fn subgroup_f_max(
        &mut self,
        result_type: Id,
        scope: Id,
        reduction: Reduction,
        value: Id,
    ) -> Result<Id, BuildError> {
        self.group_arithmetic(
            op::GROUP_NON_UNIFORM_F_MAX,
            result_type,
            scope,
            reduction,
            value,
        )
    }

    pub fn subgroup_f_min(
        &mut self,
        result_type: Id,
        scope: Id,
        reduction: Reduction,
        value: Id,
    ) -> Result<Id, BuildError> {
        self.group_arithmetic(
            op::GROUP_NON_UNIFORM_F_MIN,
            result_type,
            scope,
            reduction,
            value,
        )
    }

    fn group_arithmetic(
        &mut self,
        opcode: u16,
        result_type: Id,
        scope: Id,
        reduction: Reduction,
        value: Id,
    ) -> Result<Id, BuildError> {
        let mut operands = vec![scope.word(), reduction.operation().word(), value.word()];
        if let Some(size) = reduction.cluster_size() {
            operands.push(size.word());
        }
        self.result_instruction(opcode, result_type, &operands)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::module::Version;
    use crate::module::subgroup::test_support::operands_of;
    use crate::spec::{GroupOperation, Scope};

    #[test]
    fn the_scope_is_a_constants_id_and_the_operation_beside_it_is_a_literal() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let value = module.constant_f32(1.0).expect("1.0");

        let sum = module
            .subgroup_f_add(float, scope, Reduction::Reduce, value)
            .expect("reduction");

        assert_eq!(
            operands_of(&module.finish(), op::GROUP_NON_UNIFORM_F_ADD),
            vec![
                float.word(),
                sum.word(),
                scope.word(),
                GroupOperation::Reduce.word(),
                value.word()
            ],
            "the scope is an id while the operation next to it is a bare 0 — one instruction, \
             two encodings"
        );
    }

    #[test]
    fn a_plain_reduce_has_no_sixth_operand() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let value = module.constant_f32(1.0).expect("1.0");

        module
            .subgroup_f_add(float, scope, Reduction::Reduce, value)
            .expect("reduce");

        assert_eq!(
            operands_of(&module.finish(), op::GROUP_NON_UNIFORM_F_ADD).len(),
            5
        );
    }

    #[test]
    fn a_clustered_reduction_carries_its_size_as_an_id() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let value = module.constant_f32(1.0).expect("1.0");
        let eight = module.constant_u32(8).expect("8u32");

        module
            .subgroup_f_add(float, scope, Reduction::Clustered { size: eight }, value)
            .expect("clustered");

        let operands = operands_of(&module.finish(), op::GROUP_NON_UNIFORM_F_ADD);

        assert_eq!(operands[3], GroupOperation::ClusteredReduce.word());
        assert_eq!(operands.len(), 6, "the cluster size is a sixth operand");
        assert_eq!(operands[5], eight.word(), "an id, not the literal 8");
    }

    #[test]
    fn a_scan_differs_from_a_reduce_only_in_its_literal() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let value = module.constant_f32(1.0).expect("1.0");

        module
            .subgroup_f_add(float, scope, Reduction::InclusiveScan, value)
            .expect("scan");

        assert_eq!(
            operands_of(&module.finish(), op::GROUP_NON_UNIFORM_F_ADD)[3],
            GroupOperation::InclusiveScan.word()
        );
    }

    #[test]
    fn each_reduction_uses_its_own_opcode() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let uint = module.type_int(32, false).expect("u32");
        let scope = module.scope(Scope::Subgroup).expect("scope");
        let value = module.constant_f32(1.0).expect("1.0");
        let count = module.constant_u32(1).expect("1u32");

        module
            .subgroup_f_add(float, scope, Reduction::Reduce, value)
            .expect("f add");
        module
            .subgroup_i_add(uint, scope, Reduction::Reduce, count)
            .expect("i add");
        module
            .subgroup_f_max(float, scope, Reduction::Reduce, value)
            .expect("f max");
        module
            .subgroup_f_min(float, scope, Reduction::Reduce, value)
            .expect("f min");

        let words = module.finish();
        let emitted: Vec<u16> = crate::decode::body(&words)
            .map(|instruction| instruction.opcode())
            .filter(|opcode| *opcode >= op::GROUP_NON_UNIFORM_ELECT)
            .collect();

        assert_eq!(
            emitted,
            vec![
                op::GROUP_NON_UNIFORM_F_ADD,
                op::GROUP_NON_UNIFORM_I_ADD,
                op::GROUP_NON_UNIFORM_F_MAX,
                op::GROUP_NON_UNIFORM_F_MIN
            ]
        );
    }
}
