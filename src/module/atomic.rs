use super::{BuildError, Id, Module, Section, op};
use crate::spec::MemorySemantics;

impl Module {
    pub fn memory_semantics(&mut self, semantics: MemorySemantics) -> Result<Id, BuildError> {
        self.constant_u32(semantics.word())
    }

    pub fn atomic(
        &mut self,
        opcode: u16,
        result_type: Id,
        pointer: Id,
        scope: Id,
        semantics: Id,
        value: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            opcode,
            result_type,
            &[pointer.word(), scope.word(), semantics.word(), value.word()],
        )
    }

    pub fn atomic_i_add(
        &mut self,
        result_type: Id,
        pointer: Id,
        scope: Id,
        semantics: Id,
        value: Id,
    ) -> Result<Id, BuildError> {
        self.atomic(
            op::ATOMIC_I_ADD,
            result_type,
            pointer,
            scope,
            semantics,
            value,
        )
    }

    pub fn atomic_increment(
        &mut self,
        result_type: Id,
        pointer: Id,
        scope: Id,
        semantics: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::ATOMIC_I_INCREMENT,
            result_type,
            &[pointer.word(), scope.word(), semantics.word()],
        )
    }

    pub fn atomic_exchange(
        &mut self,
        result_type: Id,
        pointer: Id,
        scope: Id,
        semantics: Id,
        value: Id,
    ) -> Result<Id, BuildError> {
        self.atomic(
            op::ATOMIC_EXCHANGE,
            result_type,
            pointer,
            scope,
            semantics,
            value,
        )
    }

    pub fn atomic_load(
        &mut self,
        result_type: Id,
        pointer: Id,
        scope: Id,
        semantics: Id,
    ) -> Result<Id, BuildError> {
        self.result_instruction(
            op::ATOMIC_LOAD,
            result_type,
            &[pointer.word(), scope.word(), semantics.word()],
        )
    }

    pub fn atomic_store(
        &mut self,
        pointer: Id,
        scope: Id,
        semantics: Id,
        value: Id,
    ) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::ATOMIC_STORE,
            &[pointer.word(), scope.word(), semantics.word(), value.word()],
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::encode::Word;
    use crate::module::Version;
    use crate::spec::Scope;

    fn module() -> Module {
        Module::new(Version::V1_3)
    }

    fn operands_of(words: &[Word], opcode: u16) -> Vec<Word> {
        decode::body(words)
            .find(|instruction| instruction.opcode() == opcode)
            .expect("the instruction was emitted")
            .operands()
            .to_vec()
    }

    #[test]
    fn an_atomic_add_names_its_pointer_scope_semantics_and_value_in_that_order() {
        let mut module = module();
        let uint = module.type_int(32, false).expect("u32");
        let pointer = module.alloc_id().expect("%pointer");
        let scope = module.scope(Scope::Device).expect("device");
        let semantics = module
            .memory_semantics(MemorySemantics::None)
            .expect("no ordering");
        let one = module.constant_u32(1).expect("1");

        let previous = module
            .atomic_i_add(uint, pointer, scope, semantics, one)
            .expect("added");

        assert_eq!(
            operands_of(&module.finish(), op::ATOMIC_I_ADD),
            vec![
                uint.word(),
                previous.word(),
                pointer.word(),
                scope.word(),
                semantics.word(),
                one.word()
            ]
        );
    }

    #[test]
    fn the_scope_and_the_semantics_are_ids_and_not_literals() {
        let mut module = module();
        let uint = module.type_int(32, false).expect("u32");
        let pointer = module.alloc_id().expect("%pointer");
        let scope = module.scope(Scope::Device).expect("device");
        let semantics = module
            .memory_semantics(MemorySemantics::None)
            .expect("no ordering");
        let one = module.constant_u32(1).expect("1");

        module
            .atomic_i_add(uint, pointer, scope, semantics, one)
            .expect("added");

        let operands = operands_of(&module.finish(), op::ATOMIC_I_ADD);
        assert_ne!(operands[3], Scope::Device.word(), "an id, not the number 1");
        assert_eq!(operands[3], scope.word());
        assert_eq!(operands[4], semantics.word());
    }

    #[test]
    fn a_relaxed_semantics_constant_is_zero_and_still_has_to_exist() {
        let mut module = module();

        let semantics = module
            .memory_semantics(MemorySemantics::None)
            .expect("no ordering");

        assert_ne!(semantics.word(), 0);
        assert_eq!(
            operands_of(&module.finish(), op::CONSTANT),
            vec![
                module.type_int(32, false).expect("u32").word(),
                semantics.word(),
                0
            ]
        );
    }

    #[test]
    fn an_increment_takes_no_value_operand() {
        let mut module = module();
        let uint = module.type_int(32, false).expect("u32");
        let pointer = module.alloc_id().expect("%pointer");
        let scope = module.scope(Scope::Device).expect("device");
        let semantics = module
            .memory_semantics(MemorySemantics::None)
            .expect("no ordering");

        module
            .atomic_increment(uint, pointer, scope, semantics)
            .expect("incremented");

        assert_eq!(
            operands_of(&module.finish(), op::ATOMIC_I_INCREMENT).len(),
            5,
            "type, result, pointer, scope, semantics — and no value"
        );
    }

    #[test]
    fn an_atomic_store_produces_no_id() {
        let mut module = module();
        let pointer = module.alloc_id().expect("%pointer");
        let scope = module.scope(Scope::Device).expect("device");
        let semantics = module
            .memory_semantics(MemorySemantics::None)
            .expect("no ordering");
        let seven = module.constant_u32(7).expect("7");

        module
            .atomic_store(pointer, scope, semantics, seven)
            .expect("stored");

        assert_eq!(
            operands_of(&module.finish(), op::ATOMIC_STORE),
            vec![pointer.word(), scope.word(), semantics.word(), seven.word()],
            "no result type and no result id, as with an ordinary store"
        );
    }

    #[test]
    fn the_semantics_constant_is_shared_with_every_other_use_of_that_mask() {
        let mut module = module();

        let first = module
            .memory_semantics(MemorySemantics::None)
            .expect("no ordering");
        let second = module
            .memory_semantics(MemorySemantics::None)
            .expect("again");
        let other = module
            .memory_semantics(MemorySemantics::AcquireReleaseBuffer)
            .expect("acquire-release");

        assert_eq!(first, second);
        assert_ne!(first, other);
    }

    #[test]
    fn an_exchange_carries_a_value_and_a_load_is_the_same_instruction_without_one() {
        let mut module = module();
        let uint = module.type_int(32, false).expect("u32");
        let pointer = module.alloc_id().expect("%pointer");
        let scope = module.scope(Scope::Device).expect("device");
        let semantics = module
            .memory_semantics(MemorySemantics::None)
            .expect("no ordering");
        let replacement = module.constant_u32(9).expect("9");

        let swapped = module
            .atomic_exchange(uint, pointer, scope, semantics, replacement)
            .expect("exchanged");
        let read = module
            .atomic_load(uint, pointer, scope, semantics)
            .expect("loaded");

        let words = module.finish();
        assert_eq!(
            operands_of(&words, op::ATOMIC_EXCHANGE),
            vec![
                uint.word(),
                swapped.word(),
                pointer.word(),
                scope.word(),
                semantics.word(),
                replacement.word()
            ]
        );
        assert_eq!(
            operands_of(&words, op::ATOMIC_LOAD),
            vec![
                uint.word(),
                read.word(),
                pointer.word(),
                scope.word(),
                semantics.word()
            ],
            "a load has no value to write and must not name one"
        );
    }
}
