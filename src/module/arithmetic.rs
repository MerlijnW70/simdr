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
