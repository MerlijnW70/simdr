//! Writing to a slot the *data* chooses, rather than one the invocation index chooses.
//!
//! Every other access in [`super::access`] computes its address from the invocation index, so no
//! two invocations ever touch the same element. That is what makes a dispatch need no bounds test
//! and no synchronisation, and it is also the limit: a histogram's slot comes from the value being
//! counted, and two invocations counting the same value must both be counted.
//!
//! # The index is a value, and that changes two things
//!
//! It has to be **in range**, and nothing here can check it. A `Kernel` binds a runtime array
//! whose length is whatever buffer the caller supplies, and an out-of-range index into a storage
//! buffer is undefined behaviour rather than an error — robust access may clamp it, or it may not.
//! So the caller keeps the index inside the buffer, and the kernels in `runner/src/kernels` do it
//! by clamping with `Lanes::min` before scattering.
//!
//! And the write has to be **atomic**, or two invocations reading, adding and writing back lose
//! one of the two contributions. That is what the rest of this module is.

use super::Kernel;
use crate::lanes::{Element, LaneError};
use crate::module::Id;
use crate::spec::{MemorySemantics, Scope};

impl<T: Element> Kernel<T> {
    /// A pointer to element `index` of buffer `binding`, where `index` is a value.
    ///
    /// The escape hatch from the invocation-derived addressing the rest of the kernel uses. What
    /// the caller does with the pointer is its own business — an atomic, usually, because a slot
    /// the data chose is a slot another invocation may also have chosen.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchBuffer`] if `binding` was not bound.
    pub fn element_pointer_to(&mut self, binding: u32, index: Id) -> Result<Id, LaneError> {
        let buffer = self.buffer(binding)?;
        let element_pointer = self.element_pointer();
        let zero = self.zero();
        Ok(self
            .module()
            .access_chain(element_pointer, buffer, &[zero, index])?)
    }

    /// Add `value` to element `index` of buffer `binding`, atomically.
    ///
    /// The histogram primitive: several invocations may name the same `index` and every one of
    /// their contributions lands. Yields what the slot held before, which is what makes the same
    /// instruction serve as an allocator — each invocation gets a different answer and the
    /// answers are consecutive.
    ///
    /// The scope is the **device**, not the workgroup: invocations in different workgroups reach
    /// the same buffer, and a workgroup-scoped atomic orders only the ones that share a workgroup.
    /// The semantics are `None`, which is right when nothing is published besides the counter
    /// itself — see [`crate::module::Module::atomic`] for when that is not enough.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchBuffer`] if `binding` was not bound.
    pub fn atomic_add_at(&mut self, binding: u32, index: Id, value: Id) -> Result<Id, LaneError> {
        let pointer = self.element_pointer_to(binding, index)?;
        let element = self.element();
        let scope = self.module().scope(Scope::Device)?;
        let semantics = self.module().memory_semantics(MemorySemantics::None)?;

        Ok(self
            .module()
            .atomic_i_add(element, pointer, scope, semantics, value)?)
    }

    /// Put `value` in element `index` of buffer `binding`, atomically, and yield what was there.
    ///
    /// **The primitive that publishes and learns in one instruction.** An add accumulates and a
    /// store overwrites without saying what it replaced; this replaces *and* reports, which is what
    /// a claim needs — the invocation that gets back the empty marker is the one that won the slot,
    /// and every other gets the winner's value rather than a second chance.
    ///
    /// `Module::atomic_exchange` had been in this crate since the atomics landed with nothing able
    /// to reach it: no `Kernel` path, no test, no validator. An audit of the public surface found
    /// it beside [`Kernel::atomic_load_at`] and `Lanes::all_equal`.
    ///
    /// Same scope and semantics as [`Kernel::atomic_add_at`], and the same warning applies to
    /// both: `MemorySemantics::None` orders nothing but the location itself. An exchange that
    /// publishes a *pointer to* data another invocation then reads needs
    /// `MemorySemantics::AcquireReleaseBuffer`, which this does not use because nothing here
    /// publishes anything but the value.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchBuffer`] if `binding` was not bound.
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

    /// Read element `index` of buffer `binding`, atomically.
    ///
    /// **Not the same instruction as a load, and not the same claim.** `Kernel::load` reads an
    /// address derived from the invocation index, which no other invocation touches; this reads an
    /// address the *data* chose, which another invocation may be writing at the same moment. A
    /// plain `OpLoad` there is a data race — undefined, not merely stale — and this is the read
    /// that is not.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchBuffer`] if `binding` was not bound.
    pub fn atomic_load_at(&mut self, binding: u32, index: Id) -> Result<Id, LaneError> {
        let pointer = self.element_pointer_to(binding, index)?;
        let element = self.element();
        let scope = self.module().scope(Scope::Device)?;
        let semantics = self.module().memory_semantics(MemorySemantics::None)?;

        Ok(self
            .module()
            .atomic_load(element, pointer, scope, semantics)?)
    }

    /// Add one to element `index` of buffer `binding`, atomically.
    ///
    /// `OpAtomicIIncrement` rather than an add of a constant one: it is a different instruction
    /// with no value operand, and a counter is common enough to be worth reaching directly.
    ///
    /// # Errors
    ///
    /// As [`Kernel::atomic_add_at`].
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
    // A test may panic — that is how it reports.
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

        // The chain the atomic uses names the *loaded value* as its index, which is the whole
        // difference from every other write in this kernel.
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
        // Two instructions with the same shape as the add and the increment, and the same
        // distinction between them. Both had been emittable and unreachable since the atomics
        // landed: no `Kernel` path, no test, no validator.
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
        // Result type, pointer, scope, semantics — and the exchange's value after them.
        assert_eq!(operands(op::ATOMIC_EXCHANGE).len(), 6);
        assert_eq!(
            operands(op::ATOMIC_LOAD).len(),
            5,
            "a load has nothing to write"
        );
    }

    #[test]
    fn an_atomic_load_reads_the_index_the_data_chose() {
        // The distinction that makes it worth having at all: `Kernel::load` addresses by
        // invocation, and this addresses by value. A version that read the invocation's own slot
        // would agree with a plain load and be a different instruction for no reason.
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
        // Invocations in different workgroups reach the same buffer. A workgroup-scoped atomic
        // orders only the ones sharing a workgroup, which is a histogram that is right whenever
        // the dispatch happens to be one workgroup — the size every test here uses.
        use crate::spec::Scope;

        let mut kernel = Kernel::<U32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");
        kernel.atomic_increment_at(1, value.id()).expect("counted");

        let words = kernel.finish().expect("finished");
        let scope_operand = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::ATOMIC_I_INCREMENT)
            .expect("emitted")
            .operands()[3];

        // The operand is an id; find the constant it names and read its value.
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
