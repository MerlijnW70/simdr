use super::super::{BuildError, Id, Module, Section, op};

impl Module {
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
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use crate::decode;
    use crate::module::{Module, Version, op};
    use crate::spec::{MemorySemantics, Scope};

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
