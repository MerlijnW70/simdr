//! Read-modify-write on one memory location, without another invocation getting in between.
//!
//! Every other write this crate emits goes to an address derived from the invocation index, so no
//! two invocations ever touch the same element and there is nothing to race. That is what made a
//! bounds-test-free dispatch possible and it is also a real limit: a histogram, a scatter, an
//! append buffer and any reduction whose *output slot* depends on the data all need two
//! invocations to reach one location and come out with both contributions.
//!
//! # Three operands that are not what they look like
//!
//! `OpAtomicIAdd %type %result %pointer %scope %semantics %value`. The scope and the semantics are
//! **ids of integer constants**, not literals — the same trap `IdScope` sets on every subgroup
//! instruction, and the same one `OpControlBarrier` sets on all three of its. A number written
//! where a constant's id belongs assembles into a well-formed instruction that means something
//! else. [`Module::scope`] and [`Module::memory_semantics`] make them.
//!
//! # What the semantics have to say
//!
//! [`MemorySemantics::None`] — SPIR-V spells it `Relaxed` — is enough for a counter whose value
//! nobody reads until the dispatch is over, which is the histogram case and the only case the
//! kernels here need. Anything that publishes *other* memory and
//! expects a reader to see it needs `Release` on the write and `Acquire` on the read, plus the
//! storage class named in the mask. This layer takes the semantics as an operand rather than
//! choosing them, because the choice belongs to the algorithm.

use super::{BuildError, Id, Module, Section, op};
use crate::spec::MemorySemantics;

impl Module {
    /// A constant holding a memory-semantics mask, for the operand that takes one as an id.
    ///
    /// Deduplicated like any other `u32` constant, so asking repeatedly costs nothing.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the constant cannot be declared.
    pub fn memory_semantics(&mut self, semantics: MemorySemantics) -> Result<Id, BuildError> {
        self.constant_u32(semantics.word())
    }

    /// Any atomic shaped `<type> <result> <pointer> <scope> <semantics> <value>`.
    ///
    /// Which is all of them except the load, the store and the compare-exchange. The opcode is a
    /// parameter for the same reason [`Module::binary`] takes one: a layer above holds it as data,
    /// so that `atomic_add` is one function rather than one per element type.
    ///
    /// The result is the value the location held **before** the operation, which is what makes an
    /// atomic add usable as an allocator: every invocation gets a different answer and the
    /// answers are consecutive.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// Add `value` to what `pointer` names, atomically, and yield the previous value.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// Add one, atomically, and yield the previous value.
    ///
    /// Its own instruction rather than an add of a constant one, and worth having for that reason:
    /// it takes no value operand at all, so the shape differs from every other atomic here.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// Replace what `pointer` names with `value`, atomically, and yield the previous value.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// Read what `pointer` names, atomically.
    ///
    /// Takes no value, and is not the same thing as an `OpLoad`: a plain load of a location
    /// another invocation may be writing is a race, and this is the read that is not.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// Write `value` to what `pointer` names, atomically.
    ///
    /// Produces no id — the only atomic here that does not.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::encode::Word;
    use crate::module::Version;
    use crate::spec::Scope;

    fn module() -> Module {
        Module::new(Version::V1_3)
    }

    /// The operands of the one instruction carrying `opcode`.
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
        // The mistake that assembles: `Scope::Device` is 1 and `Relaxed` is 0, and writing those
        // numbers straight into the instruction gives an atomic naming ids %1 and %0 — the first
        // of which is some other declaration and the second of which is not an id at all.
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
        // The one that would be easiest to leave out: `Relaxed` is the empty mask, so the constant
        // holds zero — and a zero *operand* would name no id, which is a different thing from an
        // id naming a constant zero.
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
        // Its shape is one word shorter than every other atomic here, which is the reason it is
        // not simply an add of one.
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
}
