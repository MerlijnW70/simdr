use super::{BuildError, Id, Module, op};

impl Module {
    pub fn binary(
        &mut self,
        opcode: u16,
        result_type: Id,
        left: Id,
        right: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(opcode, result_type, &[left.word(), right.word()])
    }

    pub fn unary(&mut self, opcode: u16, result_type: Id, operand: Id) -> Result<Id, BuildError> {
        self.result_instruction(opcode, result_type, &[operand.word()])
    }

    pub fn f_mul(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::F_MUL, result_type, &[left.word(), right.word()])
    }

    pub fn f_add(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::F_ADD, result_type, &[left.word(), right.word()])
    }

    pub fn f_sub(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::F_SUB, result_type, &[left.word(), right.word()])
    }

    pub fn f_div(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::F_DIV, result_type, &[left.word(), right.word()])
    }

    pub fn f_negate(&mut self, result_type: Id, value: Id) -> Result<Id, BuildError> {
        self.unary(op::F_NEGATE, result_type, value)
    }

    pub fn i_sub(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::I_SUB, result_type, &[left.word(), right.word()])
    }

    pub fn u_div(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::U_DIV, result_type, &[left.word(), right.word()])
    }

    pub fn i_add(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::I_ADD, result_type, &[left.word(), right.word()])
    }

    pub fn i_mul(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::I_MUL, result_type, &[left.word(), right.word()])
    }

    pub fn logical_or(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::LOGICAL_OR, result_type, &[left.word(), right.word()])
    }

    pub fn logical_and(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::LOGICAL_AND, result_type, &[left.word(), right.word()])
    }

    pub fn select(
        &mut self,
        result_type: Id,
        condition: Id,
        when_true: Id,
        when_false: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::SELECT,
            result_type,
            &[condition.word(), when_true.word(), when_false.word()],
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::module::Version;

    type Binary = fn(&mut Module, Id, Id, Id) -> Result<Id, BuildError>;

    fn scratch() -> (Module, Id, Id, Id) {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let left = module.alloc_id().expect("%left");
        let right = module.alloc_id().expect("%right");
        (module, float, left, right)
    }

    fn last(words: &[u32]) -> (u16, Vec<u32>) {
        let instruction = decode::body(words)
            .last()
            .expect("an instruction was emitted");
        (instruction.opcode(), instruction.operands().to_vec())
    }

    #[test]
    fn every_named_wrapper_emits_the_opcode_it_is_named_for() {
        let cases: [(&str, u16, Binary); 10] = [
            ("f_mul", op::F_MUL, |m, t, l, r| m.f_mul(t, l, r)),
            ("f_add", op::F_ADD, |m, t, l, r| m.f_add(t, l, r)),
            ("f_sub", op::F_SUB, |m, t, l, r| m.f_sub(t, l, r)),
            ("f_div", op::F_DIV, |m, t, l, r| m.f_div(t, l, r)),
            ("i_sub", op::I_SUB, |m, t, l, r| m.i_sub(t, l, r)),
            ("u_div", op::U_DIV, |m, t, l, r| m.u_div(t, l, r)),
            ("i_add", op::I_ADD, |m, t, l, r| m.i_add(t, l, r)),
            ("i_mul", op::I_MUL, |m, t, l, r| m.i_mul(t, l, r)),
            ("logical_or", op::LOGICAL_OR, |m, t, l, r| {
                m.logical_or(t, l, r)
            }),
            ("logical_and", op::LOGICAL_AND, |m, t, l, r| {
                m.logical_and(t, l, r)
            }),
        ];

        for (name, expected, emit) in cases {
            let (mut module, kind, left, right) = scratch();
            let result = emit(&mut module, kind, left, right).expect(name);

            let (opcode, operands) = last(&module.finish());
            assert_eq!(opcode, expected, "{name} emitted the wrong instruction");
            assert_eq!(
                operands,
                vec![kind.word(), result.word(), left.word(), right.word()],
                "{name} did not lay out <type> <result> <left> <right>"
            );
        }
    }

    #[test]
    fn a_subtraction_keeps_the_operands_in_the_order_it_was_handed_them() {
        let crossing: [(&str, Binary); 4] = [
            ("f_sub", |m, t, l, r| m.f_sub(t, l, r)),
            ("i_sub", |m, t, l, r| m.i_sub(t, l, r)),
            ("f_div", |m, t, l, r| m.f_div(t, l, r)),
            ("u_div", |m, t, l, r| m.u_div(t, l, r)),
        ];

        for (name, emit) in crossing {
            let (mut module, kind, left, right) = scratch();
            assert_ne!(left, right, "the two operands have to be distinguishable");
            emit(&mut module, kind, left, right).expect(name);

            let (_, operands) = last(&module.finish());
            assert_eq!(
                operands[2],
                left.word(),
                "{name} put the right operand where the left one belongs"
            );
            assert_eq!(operands[3], right.word(), "{name} crossed its operands");
        }
    }

    #[test]
    fn the_general_forms_emit_what_the_named_ones_do() {
        let (mut named, kind, left, right) = scratch();
        named.i_add(kind, left, right).expect("i_add");

        let (mut general, kind, left, right) = scratch();
        general
            .binary(op::I_ADD, kind, left, right)
            .expect("binary I_ADD");

        assert_eq!(last(&named.finish()), last(&general.finish()));

        let (mut named, kind, value, _) = scratch();
        named.f_negate(kind, value).expect("f_negate");

        let (mut general, kind, value, _) = scratch();
        general
            .unary(op::F_NEGATE, kind, value)
            .expect("unary F_NEGATE");

        assert_eq!(last(&named.finish()), last(&general.finish()));
    }

    #[test]
    fn a_selection_names_its_condition_first_and_then_the_two_arms() {
        let (mut module, kind, when_true, when_false) = scratch();
        let condition = module.alloc_id().expect("%condition");

        let result = module
            .select(kind, condition, when_true, when_false)
            .expect("select");

        let (opcode, operands) = last(&module.finish());
        assert_eq!(opcode, op::SELECT);
        assert_eq!(
            operands,
            vec![
                kind.word(),
                result.word(),
                condition.word(),
                when_true.word(),
                when_false.word()
            ]
        );
    }
}
