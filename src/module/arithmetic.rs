//! Arithmetic, comparison and selection — the instructions that produce a value from two others.
//!
//! Split out of [`super::body`] because they are all the same shape and the shape is the point:
//! `<result type> <result id> <left> <right>`, emitted by `result_instruction`. What differs
//! between them is one opcode, which is why [`Module::binary`] exists alongside the named
//! spellings — a layer above may hold the opcode as data, and [`crate::lanes::Element`] does.

use super::{BuildError, Id, Module, op};

impl Module {
    /// Any two-operand instruction shaped `<type> <result> <left> <right>`.
    ///
    /// The named wrappers below are the readable spelling; this one exists because a layer above
    /// may hold the opcode as data — [`crate::lanes::Element`] carries one per element type, so
    /// that `add` is a single function rather than one per type.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn binary(
        &mut self,
        opcode: u16,
        result_type: Id,
        left: Id,
        right: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(opcode, result_type, &[left.word(), right.word()])
    }

    /// Any one-operand instruction shaped `<type> <result> <operand>`.
    ///
    /// The conversions and `OpCopyObject` all have it, which is what lets
    /// [`crate::lanes::Element`] carry one as data and a caller convert a value without naming an
    /// opcode — including the case where the conversion is nothing at all.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn unary(&mut self, opcode: u16, result_type: Id, operand: Id) -> Result<Id, BuildError> {
        self.result_instruction(opcode, result_type, &[operand.word()])
    }

    /// Floating-point multiply.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn f_mul(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::F_MUL, result_type, &[left.word(), right.word()])
    }

    /// Floating-point add.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn f_add(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::F_ADD, result_type, &[left.word(), right.word()])
    }

    /// Floating-point subtract.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn f_sub(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::F_SUB, result_type, &[left.word(), right.word()])
    }

    /// Floating-point divide.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn f_div(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::F_DIV, result_type, &[left.word(), right.word()])
    }

    /// A float with its sign flipped.
    ///
    /// Not a multiply by −1.0: this flips the sign bit and touches nothing else, including on a
    /// zero and a NaN, and it is one instruction the implementation may not reassociate.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn f_negate(&mut self, result_type: Id, value: Id) -> Result<Id, BuildError> {
        self.unary(op::F_NEGATE, result_type, value)
    }

    /// Integer subtract.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn i_sub(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::I_SUB, result_type, &[left.word(), right.word()])
    }

    /// Unsigned integer divide.
    ///
    /// A division by a constant is one an implementation turns into a multiply and a shift, so this
    /// is not the expensive instruction it looks like where the divisor is known when the kernel is
    /// built — which is the only way anything here uses it.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn u_div(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::U_DIV, result_type, &[left.word(), right.word()])
    }

    /// Integer add.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn i_add(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::I_ADD, result_type, &[left.word(), right.word()])
    }

    /// Integer multiply.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn i_mul(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::I_MUL, result_type, &[left.word(), right.word()])
    }

    /// Boolean or.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn logical_or(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::LOGICAL_OR, result_type, &[left.word(), right.word()])
    }

    /// Boolean and.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn logical_and(&mut self, result_type: Id, left: Id, right: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::LOGICAL_AND, result_type, &[left.word(), right.word()])
    }

    /// Pick `when_true` or `when_false` according to `condition`.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::module::Version;

    /// A two-operand instruction, as the tables below hold one: the module to emit into, the
    /// result type, and the left and right operands.
    type Binary = fn(&mut Module, Id, Id, Id) -> Result<Id, BuildError>;

    /// A module holding a type and two distinct operand ids, which is all any instruction here
    /// needs to be emitted.
    ///
    /// The ids name nothing. Every test below reads the words back rather than running them, and
    /// what it is checking is that the opcode and the operand *order* are the ones the caller
    /// asked for — neither of which depends on the operands standing for anything.
    fn scratch() -> (Module, Id, Id, Id) {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let left = module.alloc_id().expect("%left");
        let right = module.alloc_id().expect("%right");
        (module, float, left, right)
    }

    /// The last instruction in `words` — the one the test just emitted.
    fn last(words: &[u32]) -> (u16, Vec<u32>) {
        let instruction = decode::body(words)
            .last()
            .expect("an instruction was emitted");
        (instruction.opcode(), instruction.operands().to_vec())
    }

    #[test]
    fn every_named_wrapper_emits_the_opcode_it_is_named_for() {
        // **The `dot_unsigned` shape, which `tests/integrity.rs` opens with.** A public wrapper
        // that names one instruction and emits another is invisible to everything that checks the
        // answer: an `OpFAdd` where an `OpFSub` belongs still returns a number, and a kernel built
        // from it still validates. Only the words say which one it was.
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
        // Crossed operands are the half of the previous test that an add or a multiply cannot
        // catch: `a + b` and `b + a` are the same instruction with the same answer, and `a - b` is
        // not. So the four instructions where order decides the value get their own check.
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
        // `binary` and `unary` are what a layer holding an opcode as data reaches for —
        // `crate::lanes::Element` carries one per element type — so they are the path every typed
        // `add` in the crate actually takes, and the named spellings are the ones a reader sees.
        // The two must not be able to drift apart.
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
        // Three operands and no types to tell them apart: a `select` that emitted them in any
        // other order would still be a valid `OpSelect` and would pick the wrong arm.
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
