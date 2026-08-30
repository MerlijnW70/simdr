//! ```text
//!   total = reduce_sum(value)      every lane of a subgroup holds its subgroup's total
//!   shared[local_id] = total       every invocation writes; no two write the same slot
//!   barrier                        every invocation reaches it, so it is well defined
//!   answer = shared[0] + shared[w] + …      w, 2w, … are build-time constants
//! ```

use super::Kernel;
use crate::lanes::{Element, LaneError};
use crate::module::Id;
use crate::spec::{Capability, MemorySemantics, Scope, StorageClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shared {
    variable: Id,
    element_pointer: Id,
    length: u32,
}

impl Shared {
    #[must_use]
    pub const fn len(self) -> u32 {
        self.length
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.length == 0
    }
}

impl<T: Element> Kernel<T> {
    pub fn shared(&mut self, length: u32) -> Result<Shared, LaneError> {
        if length == 0 {
            return Err(LaneError::EmptyShared);
        }

        let element = self.element();
        let count = self.module().constant_u32(length)?;
        let array = self.module().type_array(element, count)?;

        let array_pointer = self.module().type_pointer(StorageClass::Workgroup, array)?;
        let element_pointer = self
            .module()
            .type_pointer(StorageClass::Workgroup, element)?;
        let variable = self
            .module()
            .global_variable(array_pointer, StorageClass::Workgroup)?;

        Ok(Shared {
            variable,
            element_pointer,
            length,
        })
    }

    /// Writes `value` into the slot `slot` names.
    ///
    /// The slot is a value rather than a number known here, so unlike
    /// [`Kernel::load_shared`] this cannot check it against the length. A write
    /// past the end is what the device makes of it, and in workgroup storage
    /// that is another allocation rather than a wasted read -- so this is the
    /// side to keep inside the bounds, not the read.
    pub fn store_shared(&mut self, shared: Shared, slot: Id, value: Id) -> Result<(), LaneError> {
        let pointer =
            self.module()
                .access_chain(shared.element_pointer, shared.variable, &[slot])?;
        Ok(self.module().store(pointer, value)?)
    }

    /// Reads the slot `index` names, which is a number known here and so is
    /// checked against the length rather than trusted.
    pub fn load_shared(&mut self, shared: Shared, index: u32) -> Result<Id, LaneError> {
        if index >= shared.length {
            return Err(LaneError::NoSuchBuffer {
                index,
                bound: shared.length,
            });
        }

        let at = self.module().constant_u32(index)?;
        let element = self.element();
        let pointer = self
            .module()
            .access_chain(shared.element_pointer, shared.variable, &[at])?;
        Ok(self.module().load(element, pointer)?)
    }

    pub fn barrier(&mut self) -> Result<(), LaneError> {
        let scope = self.module().scope(Scope::Workgroup)?;
        let semantics = self
            .module()
            .constant_u32(MemorySemantics::AcquireReleaseWorkgroup.word())?;

        self.module().require_capability(Capability::Shader)?;
        Ok(self.module().control_barrier(scope, scope, semantics)?)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use crate::decode;
    use crate::kernel::{Kernel, Shape};
    use crate::lanes::{F32, LaneError};
    use crate::module::op;
    use crate::spec::StorageClass;

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    fn opcodes(words: &[u32]) -> Vec<u16> {
        decode::body(words)
            .map(|instruction| instruction.opcode())
            .collect()
    }

    #[test]
    fn a_shared_array_declares_its_length_as_a_constant_and_not_a_literal() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let shared = kernel.shared(64).expect("declared");
        assert_eq!(shared.len(), 64);
        assert!(!shared.is_empty());

        let words = kernel.finish().expect("finished");
        let array = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::TYPE_ARRAY)
            .expect("an array type")
            .operands()
            .to_vec();

        let length_id = array[2];
        let declared = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CONSTANT)
            .find(|instruction| instruction.operands().get(1) == Some(&length_id))
            .and_then(|instruction| instruction.operands().get(2).copied())
            .expect("the length is a declared constant");

        assert_eq!(declared, 64);
    }

    #[test]
    fn the_variable_lands_in_workgroup_storage() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        kernel.shared(64).expect("declared");

        let words = kernel.finish().expect("finished");
        let storages: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::VARIABLE)
            .filter_map(|instruction| instruction.operands().get(2).copied())
            .collect();

        assert!(
            storages.contains(&StorageClass::Workgroup.word()),
            "no workgroup variable among {storages:?}"
        );
    }

    #[test]
    fn a_shared_array_of_nothing_is_refused() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        assert_eq!(kernel.shared(0).err(), Some(LaneError::EmptyShared));
    }

    #[test]
    fn reading_past_the_end_is_refused_by_index() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let shared = kernel.shared(4).expect("declared");

        assert!(kernel.load_shared(shared, 3).is_ok());
        assert_eq!(
            kernel.load_shared(shared, 4).err(),
            Some(LaneError::NoSuchBuffer { index: 4, bound: 4 })
        );
    }

    #[test]
    fn a_barrier_names_the_workgroup_scope_twice_and_orders_workgroup_memory() {
        use crate::spec::{MemorySemantics, Scope};

        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        kernel.barrier().expect("emitted");

        let words = kernel.finish().expect("finished");
        let barrier = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::CONTROL_BARRIER)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(barrier.len(), 3, "execution scope, memory scope, semantics");
        assert_eq!(barrier[0], barrier[1], "both scopes are the workgroup");

        let value_of = |id: u32| {
            decode::body(&words)
                .filter(|instruction| instruction.opcode() == op::CONSTANT)
                .find(|instruction| instruction.operands().get(1) == Some(&id))
                .and_then(|instruction| instruction.operands().get(2).copied())
        };

        assert_eq!(value_of(barrier[0]), Some(Scope::Workgroup.word()));
        assert_eq!(
            value_of(barrier[2]),
            Some(MemorySemantics::AcquireReleaseWorkgroup.word())
        );
    }

    #[test]
    fn a_store_and_a_load_reach_the_same_variable() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let shared = kernel.shared(64).expect("declared");
        let value = kernel.load::<32>(0).expect("loaded");
        let slot = kernel.local_index();

        kernel
            .store_shared(shared, slot, value.id())
            .expect("stored");
        kernel.barrier().expect("barrier");
        kernel.load_shared(shared, 0).expect("read back");

        let words = kernel.finish().expect("finished");
        let seen = opcodes(&words);

        assert_eq!(count(&words, op::CONTROL_BARRIER), 1);
        assert!(count(&words, op::ACCESS_CHAIN) >= 3);
        assert!(seen.contains(&op::TYPE_ARRAY));
    }
}
