//! Instructions inside a function.
//!
//! Every one of these appends to [`Section::Function`], so the caller's call order *is* the
//! program order. There is no block or dominance tracking here yet: the validator is what says a
//! sequence is well-formed, and until there is a reason to duplicate its judgement, it can keep
//! saying so.

use super::{BuildError, Id, Module, Section, op};
use crate::spec::FunctionControl;

impl Module {
    /// Open a function definition.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn begin_function(
        &mut self,
        returns: Id,
        function: Id,
        control: FunctionControl,
        signature: Id,
    ) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::FUNCTION,
            &[
                returns.word(),
                function.word(),
                control.word(),
                signature.word(),
            ],
        )
    }

    /// Open a block and yield its label.
    ///
    /// Every block starts with one of these, and the first block of a function is its entry.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn label(&mut self) -> Result<Id, BuildError> {
        let id = self.alloc_id()?;
        self.emit(Section::Function, op::LABEL, &[id.word()])?;
        self.enter_block(id);
        Ok(id)
    }

    /// Return from a function whose return type is void.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn return_void(&mut self) -> Result<(), BuildError> {
        self.emit(Section::Function, op::RETURN, &[])?;
        self.leave_block();
        Ok(())
    }

    /// Close the open function definition.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn end_function(&mut self) -> Result<(), BuildError> {
        self.emit(Section::Function, op::FUNCTION_END, &[])
    }

    /// Read through `pointer`.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn load(&mut self, result_type: Id, pointer: Id) -> Result<Id, BuildError> {
        self.result_instruction(op::LOAD, result_type, &[pointer.word()])
    }

    /// Write `value` through `pointer`.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn store(&mut self, pointer: Id, value: Id) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::STORE,
            &[pointer.word(), value.word()],
        )
    }

    /// Walk into an aggregate and yield a pointer to the part `indices` names.
    ///
    /// The indices are *ids of constants*, not literals — which is the usual first surprise, and
    /// the reason a buffer access needs a `constant_u32(0)` for the struct member before the
    /// index that varies per invocation.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn access_chain(
        &mut self,
        result_type: Id,
        base: Id,
        indices: &[Id],
    ) -> Result<Id, BuildError> {
        let mut operands = vec![base.word()];
        operands.extend(indices.iter().map(|index| index.word()));
        self.result_instruction(op::ACCESS_CHAIN, result_type, &operands)
    }

    /// Pull one component out of a composite.
    ///
    /// Here the indices *are* literals, unlike [`Module::access_chain`] — the specification is
    /// inconsistent about this and the validator is unforgiving about it.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn composite_extract(
        &mut self,
        result_type: Id,
        composite: Id,
        indices: &[u32],
    ) -> Result<Id, BuildError> {
        let mut operands = vec![composite.word()];
        operands.extend_from_slice(indices);
        self.result_instruction(op::COMPOSITE_EXTRACT, result_type, &operands)
    }

    /// Emit an instruction shaped `<result type> <result id> <operands…>` and yield its id.
    ///
    /// Nearly every value-producing instruction has that shape, which is why it is worth naming
    /// once rather than repeating the two-word prefix at every call site.
    ///
    /// Visible to the sibling modules because the subgroup instructions have it too.
    pub(super) fn result_instruction(
        &mut self,
        opcode: u16,
        result_type: Id,
        tail: &[crate::encode::Word],
    ) -> Result<Id, BuildError> {
        let id = self.alloc_id()?;
        let mut operands = vec![result_type.word(), id.word()];
        operands.extend_from_slice(tail);
        self.emit(Section::Function, opcode, &operands)?;
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::encode::Word;
    use crate::module::Version;

    /// The operands of the one instruction in `words` carrying `opcode`.
    ///
    /// Assertions used to index the word stream by hand, and two of them were wrong about how
    /// long an instruction is rather than about what it held. Offsets are the encoder's business.
    fn operands_of(words: &[Word], opcode: u16) -> Vec<Word> {
        decode::body(words)
            .find(|instruction| instruction.opcode() == opcode)
            .expect("the instruction was emitted")
            .operands()
            .to_vec()
    }

    #[test]
    fn a_value_producing_instruction_names_its_type_then_itself() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let left = module.constant_f32(1.0).expect("1.0");
        let right = module.constant_f32(2.0).expect("2.0");

        let product = module.f_mul(float, left, right).expect("product");

        assert_eq!(
            operands_of(&module.finish(), op::F_MUL),
            vec![float.word(), product.word(), left.word(), right.word()],
            "result type, then result id, then the arguments"
        );
    }

    #[test]
    fn a_store_produces_no_id_and_so_names_no_type() {
        let mut module = Module::new(Version::V1_3);
        let pointer = module.alloc_id().expect("%1");
        let value = module.alloc_id().expect("%2");

        module.store(pointer, value).expect("stored");

        assert_eq!(
            operands_of(&module.finish(), op::STORE),
            vec![pointer.word(), value.word()]
        );
    }

    #[test]
    fn an_access_chain_takes_ids_as_indices() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let base = module.alloc_id().expect("%base");
        let zero = module.constant_u32(0).expect("0u32");
        let one = module.constant_u32(1).expect("1u32");

        let chain = module
            .access_chain(float, base, &[zero, one])
            .expect("chain");

        assert_eq!(
            operands_of(&module.finish(), op::ACCESS_CHAIN),
            vec![
                float.word(),
                chain.word(),
                base.word(),
                zero.word(),
                one.word()
            ],
            "the indices are constants' ids, not the literals 0 and 1"
        );
    }

    #[test]
    fn a_composite_extract_takes_literals_as_indices() {
        let mut module = Module::new(Version::V1_3);
        let uint = module.type_int(32, false).expect("u32");
        let composite = module.alloc_id().expect("%vec");

        let component = module
            .composite_extract(uint, composite, &[0])
            .expect("extract");

        assert_eq!(
            operands_of(&module.finish(), op::COMPOSITE_EXTRACT),
            vec![uint.word(), component.word(), composite.word(), 0],
            "the trailing 0 is the literal index, not a constant's id"
        );
    }

    #[test]
    fn a_label_yields_the_id_it_just_declared() {
        let mut module = Module::new(Version::V1_3);

        let block = module.label().expect("block");

        assert_eq!(operands_of(&module.finish(), op::LABEL), vec![block.word()]);
    }

    #[test]
    fn instructions_appear_in_the_order_they_were_called() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let one = module.constant_f32(1.0).expect("1.0");

        module.label().expect("block");
        module.f_add(float, one, one).expect("sum");
        module.return_void().expect("return");
        module.end_function().expect("end");

        assert_eq!(
            decode::opcodes(&module.finish()),
            vec![
                op::TYPE_FLOAT,
                op::CONSTANT,
                op::LABEL,
                op::F_ADD,
                op::RETURN,
                op::FUNCTION_END
            ],
            "types first because they are their own section, then the body in call order"
        );
    }
}
