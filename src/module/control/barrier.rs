//! The two barriers, and what they order.
//!
//! Split from [`super`] because they answer a different question. Everything there is about where
//! control *goes* — blocks, branches, merges, phis — and these are about when other invocations
//! are allowed to see what this one wrote.
//!
//! # Both scopes are ids
//!
//! `OpControlBarrier` takes an execution scope, a memory scope and a semantics mask, and all three
//! arrive as ids of integer constants rather than as literals. That is the same trap `IdScope`
//! sets on every subgroup instruction: putting the number where the constant's id belongs
//! assembles into a well-formed instruction that means something else.
//! [`Module::scope`] and [`Module::constant_u32`] make them.

use super::super::{BuildError, Id, Module, Section, op};

impl Module {
    /// Every invocation in `execution` waits here, and the named memory is made coherent.
    ///
    /// The handover instruction. Without it, one invocation writing shared memory and another
    /// reading it is a race the specification says nothing kind about — and on real hardware it
    /// usually *works* at small workgroup sizes, which is worse than failing.
    ///
    /// **It must be reached by every invocation in the scope.** A barrier inside a branch that only
    /// some of them take is undefined behaviour rather than a slow path. Nothing here can check
    /// that; `decisions/DR-0003` refusing divergent branches is what makes it hard to get wrong by
    /// accident, which that record now says explicitly.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn control_barrier(
        &mut self,
        execution: Id,
        memory: Id,
        semantics: Id,
    ) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::CONTROL_BARRIER,
            &[execution.word(), memory.word(), semantics.word()],
        )
    }

    /// Order memory accesses without making anyone wait.
    ///
    /// Rarely what a caller wants on its own — [`Module::control_barrier`] is the one that also
    /// synchronises execution, and a handover needs both halves.
    ///
    /// # The semantics may not be `Relaxed`, and this cannot check it
    ///
    /// [`crate::spec::MemorySemantics::None`] encodes to `Relaxed`, and
    /// `VUID-StandaloneSpirv-MemorySemantics-10869` forbids it **here specifically**: a barrier
    /// that orders nothing is not a cheaper barrier, it is an invalid module. An atomic with the
    /// same mask is perfectly legal, which is what makes this easy to get wrong — and the
    /// documentation on `MemorySemantics::None` recommended exactly that mask without saying where
    /// it does not apply.
    ///
    /// Nothing here can refuse it: the operand is the *id* of a constant by the time it arrives,
    /// and this layer cannot ask what value that constant holds. So it is stated, and
    /// `tests/instructions.rs` carries both halves — a barrier with `AcquireRelease` that the
    /// validator accepts, and one with `Relaxed` that it rejects.
    ///
    /// This was the last operation in the crate with no caller and no validator behind it, and the
    /// first time one was pointed at it, it was rejected.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn memory_barrier(&mut self, memory: Id, semantics: Id) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::MEMORY_BARRIER,
            &[memory.word(), semantics.word()],
        )
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use crate::decode;
    use crate::module::{Module, Version, op};
    use crate::spec::{MemorySemantics, Scope};

    /// The operands of the sole instruction with `opcode`.
    fn operands_of(words: &[u32], opcode: u16) -> Vec<u32> {
        decode::body(words)
            .find(|instruction| instruction.opcode() == opcode)
            .expect("the instruction was emitted")
            .operands()
            .to_vec()
    }

    #[test]
    fn a_control_barrier_carries_three_operands_in_order() {
        let mut module = Module::new(Version::V1_3);
        let execution = module.scope(Scope::Workgroup).expect("scope");
        let memory = module.scope(Scope::Subgroup).expect("scope");
        let semantics = module
            .constant_u32(MemorySemantics::AcquireReleaseWorkgroup.word())
            .expect("semantics");

        module
            .control_barrier(execution, memory, semantics)
            .expect("emitted");

        let operands = operands_of(&module.finish(), op::CONTROL_BARRIER);
        assert_eq!(operands.len(), 3);
        assert_eq!(operands[0], execution.word());
        assert_eq!(operands[1], memory.word());
        assert_eq!(operands[2], semantics.word());
    }

    #[test]
    fn the_two_scopes_are_kept_apart() {
        // The specification allows them to differ and the obvious implementation passes one twice.
        // A caller synchronising a workgroup while ordering only subgroup memory would then be
        // silently given workgroup semantics.
        let mut module = Module::new(Version::V1_3);
        let workgroup = module.scope(Scope::Workgroup).expect("scope");
        let subgroup = module.scope(Scope::Subgroup).expect("scope");
        let semantics = module.constant_u32(0).expect("none");

        module
            .control_barrier(workgroup, subgroup, semantics)
            .expect("emitted");

        let operands = operands_of(&module.finish(), op::CONTROL_BARRIER);
        assert_ne!(
            operands[0], operands[1],
            "the execution and memory scopes were collapsed into one"
        );
    }

    #[test]
    fn a_memory_barrier_carries_two() {
        let mut module = Module::new(Version::V1_3);
        let memory = module.scope(Scope::Workgroup).expect("scope");
        let semantics = module
            .constant_u32(MemorySemantics::AcquireReleaseWorkgroup.word())
            .expect("semantics");

        module.memory_barrier(memory, semantics).expect("emitted");

        let operands = operands_of(&module.finish(), op::MEMORY_BARRIER);
        assert_eq!(operands.len(), 2, "no execution scope on this one");
        assert_eq!(operands[0], memory.word());
        assert_eq!(operands[1], semantics.word());
    }

    #[test]
    fn a_barrier_does_not_close_the_block_it_is_in() {
        // It is not a terminator. Treating it as one would clear `current_block` and every phi
        // built after a barrier would then name no predecessor.
        let mut module = Module::new(Version::V1_3);
        let block = module.label().expect("opened");
        let scope = module.scope(Scope::Workgroup).expect("scope");
        let semantics = module.constant_u32(0).expect("none");

        module
            .control_barrier(scope, scope, semantics)
            .expect("emitted");

        assert_eq!(module.current_block(), Some(block));
    }
}
