use super::Kernel;
use crate::lanes::{Element, LaneError};
use crate::module::Id;
use crate::spec::{MemorySemantics, Scope};

impl<T: Element> Kernel<T> {
    pub fn element_pointer_to(&mut self, binding: u32, index: Id) -> Result<Id, LaneError> {
        let buffer = self.buffer(binding)?;
        let element_pointer = self.element_pointer();
        let zero = self.zero();
        Ok(self
            .module()
            .access_chain(element_pointer, buffer, &[zero, index])?)
    }

    pub fn atomic_add_at(&mut self, binding: u32, index: Id, value: Id) -> Result<Id, LaneError> {
        let pointer = self.element_pointer_to(binding, index)?;
        let element = self.element();
        let scope = self.module().scope(Scope::Device)?;
        let semantics = self.module().memory_semantics(MemorySemantics::None)?;

        Ok(self
            .module()
            .atomic_i_add(element, pointer, scope, semantics, value)?)
    }

    pub fn atomic_exchange_at(
        &mut self,
        binding: u32,
        index: Id,
        value: Id,
    ) -> Result<Id, LaneError> {
        let pointer = self.element_pointer_to(binding, index)?;
        let element = self.element();
        let scope = self.module().scope(Scope::Device)?;
        let semantics = self.module().memory_semantics(MemorySemantics::None)?;

        Ok(self
            .module()
            .atomic_exchange(element, pointer, scope, semantics, value)?)
    }

    pub fn atomic_load_at(&mut self, binding: u32, index: Id) -> Result<Id, LaneError> {
        let pointer = self.element_pointer_to(binding, index)?;
        let element = self.element();
        let scope = self.module().scope(Scope::Device)?;
        let semantics = self.module().memory_semantics(MemorySemantics::None)?;

        Ok(self
            .module()
            .atomic_load(element, pointer, scope, semantics)?)
    }

    pub fn atomic_increment_at(&mut self, binding: u32, index: Id) -> Result<Id, LaneError> {
        let pointer = self.element_pointer_to(binding, index)?;
        let element = self.element();
        let scope = self.module().scope(Scope::Device)?;
        let semantics = self.module().memory_semantics(MemorySemantics::None)?;

        Ok(self
            .module()
            .atomic_increment(element, pointer, scope, semantics)?)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use crate::decode;
    use crate::kernel::{Kernel, Shape};
    use crate::lanes::{LaneError, U32};
    use crate::module::op;

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    #[test]
    fn an_atomic_add_walks_to_the_index_it_was_given_rather_than_this_invocations() {
        let mut kernel = Kernel::<U32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");
        let one = kernel.module().constant_u32(1).expect("1");

        kernel.atomic_add_at(1, value.id(), one).expect("scattered");

        let words = kernel.finish().expect("finished");
        assert_eq!(count(&words, op::ATOMIC_I_ADD), 1);

        let chains: Vec<Vec<u32>> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::ACCESS_CHAIN)
            .map(|instruction| instruction.operands().to_vec())
            .collect();
        assert_eq!(chains.len(), 2, "one for the load, one for the scatter");
        assert_eq!(
            chains[1].last().copied(),
            Some(value.id().word()),
            "the scatter's index is the data, not the invocation index"
        );
    }

    #[test]
    fn an_exchange_takes_a_value_and_a_load_does_not() {
        let mut kernel = Kernel::<U32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");
        let seven = kernel.module().constant_u32(7).expect("7");

        let displaced = kernel
            .atomic_exchange_at(1, value.id(), seven)
            .expect("exchanged");
        kernel.atomic_load_at(1, displaced).expect("read back");

        let words = kernel.finish().expect("finished");
        assert_eq!(count(&words, op::ATOMIC_EXCHANGE), 1);
        assert_eq!(count(&words, op::ATOMIC_LOAD), 1);

        let operands = |opcode: u16| {
            decode::body(&words)
                .find(|instruction| instruction.opcode() == opcode)
                .expect("emitted")
                .operands()
                .to_vec()
        };
        assert_eq!(operands(op::ATOMIC_EXCHANGE).len(), 6);
        assert_eq!(
            operands(op::ATOMIC_LOAD).len(),
            5,
            "a load has nothing to write"
        );
    }

    #[test]
    fn an_atomic_load_reads_the_index_the_data_chose() {
        let mut kernel = Kernel::<U32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");

        kernel.atomic_load_at(0, value.id()).expect("read");

        let words = kernel.finish().expect("finished");
        let chains: Vec<Vec<u32>> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::ACCESS_CHAIN)
            .map(|instruction| instruction.operands().to_vec())
            .collect();
        assert_eq!(chains.len(), 2, "one for the load, one for the atomic");
        assert_eq!(chains[1].last().copied(), Some(value.id().word()));
    }

    #[test]
    fn an_increment_emits_the_instruction_that_takes_no_value() {
        let mut kernel = Kernel::<U32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");

        kernel
            .atomic_increment_at(1, value.id())
            .expect("incremented");

        let words = kernel.finish().expect("finished");
        assert_eq!(count(&words, op::ATOMIC_I_INCREMENT), 1);
        assert_eq!(count(&words, op::ATOMIC_I_ADD), 0);
    }

    #[test]
    fn the_scope_is_the_device_and_not_the_workgroup() {
        use crate::spec::Scope;

        let mut kernel = Kernel::<U32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");
        kernel.atomic_increment_at(1, value.id()).expect("counted");

        let words = kernel.finish().expect("finished");
        let scope_operand = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::ATOMIC_I_INCREMENT)
            .expect("emitted")
            .operands()[3];

        let scope_value = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CONSTANT)
            .find(|instruction| instruction.operands()[1] == scope_operand)
            .expect("the scope is a declared constant")
            .operands()[2];

        assert_eq!(scope_value, Scope::Device.word());
        assert_ne!(scope_value, Scope::Workgroup.word());
    }

    #[test]
    fn scattering_into_a_buffer_that_was_never_bound_is_refused() {
        let mut kernel = Kernel::<U32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");

        assert_eq!(
            kernel.atomic_add_at(7, value.id(), value.id()).err(),
            Some(LaneError::NoSuchBuffer { index: 7, bound: 2 })
        );
    }
}
