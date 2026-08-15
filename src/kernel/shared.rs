//! Workgroup shared memory, and the barrier that makes a handover through it legal.
//!
//! Every subgroup instruction this crate emits stops at the subgroup. A workgroup holds several of
//! them — 64 invocations over 32-wide subgroups is two — and nothing in SPIR-V lets one subgroup
//! read another's registers. Shared memory is the only route, and it needs three things the
//! emitter did not have: a fixed-size array type, a variable in the `Workgroup` storage class, and
//! `OpControlBarrier`.
//!
//! # What this makes possible
//!
//! `Gpu::sum` used to end with *"the last two floats come home"*: the final workgroup produced one
//! total per subgroup and the host added them, because there was no way to combine two subgroups
//! on the device. [`Shared`] removes that boundary.
//!
//! # The pattern, and why it needs no divergence
//!
//! ```text
//!   total = reduce_sum(value)      every lane of a subgroup holds its subgroup's total
//!   shared[local_id] = total       every invocation writes; no two write the same slot
//!   barrier                        every invocation reaches it, so it is well defined
//!   answer = shared[0] + shared[w] + …      w, 2w, … are build-time constants
//! ```
//!
//! The reads are at *constant* indices — the subgroup width is fixed when the module is built — so
//! every invocation executes the identical instruction sequence and computes the identical answer.
//! No `elect`, no per-lane branch, nothing `decisions/DR-0003` refuses. The cost is that every
//! invocation redundantly computes the final sum, which is a handful of adds.
//!
//! # The barrier must be unconditional
//!
//! A barrier some invocations reach and others do not is undefined behaviour, not a slow path.
//! Nothing here can enforce that — it is a property of where the caller puts it — so
//! [`Kernel::barrier`] says so and DR-0003's refusal of divergent branches is what makes it hard
//! to get wrong by accident.

use super::Kernel;
use crate::lanes::{Element, LaneError};
use crate::module::Id;
use crate::spec::{Capability, MemorySemantics, Scope, StorageClass};

/// An array in workgroup memory, shared by every invocation in the workgroup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shared {
    /// The `OpVariable` in `Workgroup` storage.
    variable: Id,
    /// A pointer to one element, for `OpAccessChain`.
    element_pointer: Id,
    /// How many elements it holds.
    length: u32,
}

impl Shared {
    /// How many elements it holds.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.length
    }

    /// Whether it holds nothing, which [`Kernel::shared`] refuses to produce.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.length == 0
    }
}

impl<T: Element> Kernel<T> {
    /// Declare an array of `length` elements in workgroup memory.
    ///
    /// Allocated for the whole workgroup, not per invocation: 64 invocations sharing a
    /// `shared::<F32>(64)` see one 64-element array between them.
    ///
    /// # Errors
    ///
    /// [`LaneError::EmptyShared`] if `length` is zero — an array of nothing is not a smaller array,
    /// it is a mistake — otherwise [`LaneError::Build`].
    ///
    /// It reported `BadShape { workgroup, buffers: 0 }` until an audit read the message it prints:
    /// *"a kernel of 64 invocations over 0 buffers describes nothing"*, about a kernel whose
    /// buffers are fine and one of whose arrays is empty.
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

    /// Write `value` to `slot` of `shared`.
    ///
    /// `slot` is an id, so it may be this invocation's index — which is the normal case and the
    /// one that guarantees no two invocations collide.
    ///
    /// # Errors
    ///
    /// [`LaneError::Build`] if the instruction cannot be emitted.
    pub fn store_shared(&mut self, shared: Shared, slot: Id, value: Id) -> Result<(), LaneError> {
        let pointer =
            self.module()
                .access_chain(shared.element_pointer, shared.variable, &[slot])?;
        Ok(self.module().store(pointer, value)?)
    }

    /// Read slot `index` of `shared`, where `index` is known when the kernel is built.
    ///
    /// A constant index rather than an id, deliberately. Reading at a *runtime* index is legal and
    /// is not what the workgroup-reduction pattern needs: there the interesting slots are 0, `w`,
    /// `2w`, … for a subgroup width fixed at build time, so every invocation reads the same places
    /// and none of them diverges.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchBuffer`] if `index` is past the end — reported with the array's own
    /// length, because reading off the end of shared memory is undefined rather than zero.
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

    /// Every invocation in the workgroup waits here, and shared memory becomes coherent.
    ///
    /// **Must be reached by every invocation of the workgroup.** A barrier inside a branch some of
    /// them skip is undefined behaviour rather than a slow path, and on real hardware it often
    /// appears to work — which is worse than failing.
    ///
    /// # Errors
    ///
    /// [`LaneError::Build`] if the instruction cannot be emitted.
    pub fn barrier(&mut self) -> Result<(), LaneError> {
        let scope = self.module().scope(Scope::Workgroup)?;
        let semantics = self
            .module()
            .constant_u32(MemorySemantics::AcquireReleaseWorkgroup.word())?;

        // `Shader` is already declared by every kernel; a workgroup barrier needs nothing beyond
        // it, which is why this is the one synchronisation primitive available without asking for
        // a capability a device might not offer.
        self.module().require_capability(Capability::Shader)?;
        Ok(self.module().control_barrier(scope, scope, semantics)?)
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
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

    /// Every opcode in the whole module — `decode::body` skips only the header, not the sections.
    fn opcodes(words: &[u32]) -> Vec<u16> {
        decode::body(words)
            .map(|instruction| instruction.opcode())
            .collect()
    }

    #[test]
    fn a_shared_array_declares_its_length_as_a_constant_and_not_a_literal() {
        // `OpTypeArray` takes the *id* of a constant. A literal there declares an array as long as
        // whatever that id names, and it assembles.
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
        // By name, and by its *own* name. This asserted `BadShape` while the message said the
        // kernel had no buffers — it has two, and the thing with nothing in it is the array.
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

        // Every operand is an id. Reading the constants back is the only way to check the *values*
        // rather than that three ids arrived.
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
        // Two chains into shared memory — one to store, one to load — plus the buffer load's own.
        assert!(count(&words, op::ACCESS_CHAIN) >= 3);
        assert!(seen.contains(&op::TYPE_ARRAY));
    }
}
