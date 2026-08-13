//! Reading and writing a kernel's buffers, and where each element lives.
//!
//! # The address
//!
//! `group × workgroup × strips + local + strip × workgroup`.
//!
//! Blocked by workgroup and strided within it: workgroup `w` owns the contiguous run of
//! `workgroup × strips` elements starting at `w × workgroup × strips`, and inside that run
//! invocation `l` reads `l`, `l + workgroup`, and so on.
//!
//! Both halves are load-bearing. Striding within the run keeps each strip coalesced across the
//! subgroup. Blocking by workgroup keeps one invocation's elements near each other, which a
//! stride spanning the whole dispatch does not.
//!
//! **`strips` belongs to the access, not to the kernel.** A buffer read four elements at a time
//! has four times the run per workgroup that a buffer written one at a time does, so a kernel
//! reducing `Simd<f32,128>` into a scalar has two differently shaped buffers and both are laid
//! out correctly. Sizing them is the caller's job; agreeing with this arithmetic is all that is
//! required.
//!
//! **A grid kernel's address is this one plus a row.** [`super::plane`] computes `row × pitch` and
//! adds the index below to it, calling the same [`Kernel::run_start`] and [`Kernel::address`]
//! rather than writing a second arithmetic that would have to keep agreeing with this one.

use super::Kernel;
use crate::lanes::{Element, LaneError, Vector};
use crate::module::Id;

impl<T: Element> Kernel<T> {
    /// How many elements per invocation a vector of `LANES` needs.
    ///
    /// A caller sizing a buffer wants this: the kernel reads `workgroup × strips` elements per
    /// workgroup.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if `LANES` has no mapping onto this subgroup.
    pub fn strips<const LANES: u32>(&mut self) -> Result<usize, LaneError> {
        self.lanes()?.strips_for::<LANES>()
    }

    /// Read a vector of `LANES` from buffer `index`.
    ///
    /// One load per strip, at the addresses this module describes.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchBuffer`] if `index` was not bound, otherwise as [`Kernel::lanes`].
    pub fn load<const LANES: u32>(&mut self, index: u32) -> Result<Vector<T, LANES>, LaneError> {
        self.load_offset(index, 0)
    }

    /// Read a vector of `LANES` from buffer `index`, `offset` elements further along.
    ///
    /// The same addresses as [`Kernel::load`] with a constant added to each. What a pairwise fold
    /// needs: `in[i]` and `in[i + half]` are one load and one offset load, and the dispatch is
    /// sized so `i + half` is always in range rather than guarded by a branch that would diverge.
    ///
    /// `offset` counts *elements*, not bytes, and is a build-time constant — a runtime offset
    /// would be a different address expression and is not what this is for.
    ///
    /// # Errors
    ///
    /// As [`Kernel::load`].
    pub fn load_offset<const LANES: u32>(
        &mut self,
        index: u32,
        offset: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let buffer = self.buffer(index)?;
        let strips = self.strips::<LANES>()?;
        let base = self.run_start(strips)?;

        let element = self.element();
        let mut loaded = Vec::with_capacity(strips);
        for strip in 0..strips {
            let pointer = self.element_pointer_at(buffer, base, strip, offset)?;
            loaded.push(self.module().load(element, pointer)?);
        }

        self.lanes()?.from_strips(&loaded)
    }

    /// Write one value per invocation to buffer `index`, at this invocation's own slot.
    ///
    /// What a reduction produces: every lane holds the same total, and each writes it once.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchBuffer`] if `index` was not bound.
    pub fn store_scalar(&mut self, index: u32, value: Id) -> Result<(), LaneError> {
        let buffer = self.buffer(index)?;
        let base = self.run_start(1)?;
        let pointer = self.element_pointer_at(buffer, base, 0, 0)?;
        Ok(self.module().store(pointer, value)?)
    }

    /// Write `value` to buffer `binding` at `index`, which is decided while the kernel runs.
    ///
    /// The counterpart to [`Kernel::store_scalar`], which writes at this *invocation's* slot. Here
    /// the caller names the slot, and the reason to want that is a per-workgroup result:
    /// [`Kernel::workgroup_index`] is the index a block's total belongs at, and no
    /// invocation-derived address reaches it.
    ///
    /// # Several invocations may name the same slot
    ///
    /// **Then they must be writing the same value.** The whole workgroup runs this, so a block
    /// total written at `workgroup_index` is written 64 times to one address. Identical writes to
    /// one location are the case this is for, and the reason it is a plain store: the value does
    /// not depend on the order they land in, so there is nothing for an atomic to order.
    ///
    /// Writing *different* values to one slot from several invocations is a race whatever this
    /// returns, and the last writer is not defined. [`Kernel::atomic_add_at`] is for the case where
    /// each invocation has something of its own to contribute.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchBuffer`] if `binding` was not bound.
    pub fn store_at(&mut self, binding: u32, index: Id, value: Id) -> Result<(), LaneError> {
        let pointer = self.element_pointer_to(binding, index)?;
        Ok(self.module().store(pointer, value)?)
    }

    /// Read one element of buffer `binding` at `index`, which is decided while the kernel runs.
    ///
    /// The reading counterpart to [`Kernel::store_at`], and it exists for the same reason: a value
    /// held **per workgroup** rather than per invocation. The second half of a long scan adds each
    /// block's offset to every element of that block, and the offset lives at
    /// [`Kernel::workgroup_index`] of a buffer — one number that all 64 invocations read.
    ///
    /// **The bound is not checked and cannot be.** A constant index is compared against the buffer
    /// when the kernel is built; an id is a number this function never sees. Reading past the end
    /// of a storage buffer is undefined rather than zero, so the caller owes an argument that
    /// `index` is inside it — usually because it came from `workgroup_index` and the buffer was
    /// sized to the dispatch.
    ///
    /// Every invocation reading the same slot is the normal case here and costs nothing to say:
    /// concurrent *reads* need no ordering.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchBuffer`] if `binding` was not bound.
    pub fn load_at(&mut self, binding: u32, index: Id) -> Result<Id, LaneError> {
        let pointer = self.element_pointer_to(binding, index)?;
        let element = self.element();
        Ok(self.module().load(element, pointer)?)
    }

    /// Write a whole vector to buffer `index`, one element per strip.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchBuffer`] if `index` was not bound.
    pub fn store<const LANES: u32>(
        &mut self,
        index: u32,
        value: Vector<T, LANES>,
    ) -> Result<(), LaneError> {
        let buffer = self.buffer(index)?;
        let strips = value.strip_count();
        let base = self.run_start(strips)?;

        for (strip, &id) in value.strips().iter().enumerate() {
            let pointer = self.element_pointer_at(buffer, base, strip, 0)?;
            self.module().store(pointer, id)?;
        }
        Ok(())
    }

    /// Read a vector of `LANES` from buffer `index`, offset by a value rather than a constant.
    ///
    /// What a specialization constant needs. [`Kernel::load_offset`] takes a `u32` and folds it
    /// into the address at build time, which costs nothing and means a different module per
    /// offset. This takes an [`Id`] — a specialization constant, or anything else uniform — and
    /// pays one `OpIAdd` per strip for it.
    ///
    /// The caller vouches that `offset` names a `u32`. Nothing here can check that, and a
    /// mismatch is a validation failure rather than a wrong number.
    ///
    /// # Errors
    ///
    /// As [`Kernel::load`].
    pub fn load_offset_by<const LANES: u32>(
        &mut self,
        index: u32,
        offset: Id,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let buffer = self.buffer(index)?;
        let strips = self.strips::<LANES>()?;
        let base = self.run_start(strips)?;

        let element = self.element();
        let uint = self.uint();
        let mut loaded = Vec::with_capacity(strips);
        for strip in 0..strips {
            // The constant part of the address first, then the value on top of it — so a strip's
            // own stride still folds at build time and only the open offset costs an instruction.
            let at = self.address(base, strip, 0)?;
            let shifted = self.module().i_add(uint, at, offset)?;

            let element_pointer = self.element_pointer();
            let zero = self.zero();
            let pointer = self
                .module()
                .access_chain(element_pointer, buffer, &[zero, shifted])?;
            loaded.push(self.module().load(element, pointer)?);
        }

        self.lanes()?.from_strips(&loaded)
    }

    /// A pointer to this invocation's element on `strip` of the run starting at `base`.
    fn element_pointer_at(
        &mut self,
        buffer: Id,
        base: Id,
        strip: usize,
        offset: u32,
    ) -> Result<Id, LaneError> {
        let at = self.address(base, strip, offset)?;
        let element_pointer = self.element_pointer();
        let zero = self.zero();
        Ok(self
            .module()
            .access_chain(element_pointer, buffer, &[zero, at])?)
    }

    /// Where this workgroup's run begins: `group × workgroup × strips`.
    ///
    /// Hoisted out of [`Kernel::address`] rather than computed inside it, because every strip of
    /// one access shares it. It was inside, and a four-strip load emitted four identical
    /// multiplies — which any driver folds back to one, and which made the module say something
    /// the arithmetic does not.
    ///
    /// [`super::plane`] wants it for the same reason, and shares its row the same way.
    pub(super) fn run_start(&mut self, strips: usize) -> Result<Id, LaneError> {
        let uint = self.uint();
        let workgroup = self.shape().workgroup;
        let (_, group) = self.position();

        // Both factors are invocation counts below `MAX_STRIPS` times a workgroup size, so this
        // saturates only for a shape no device would accept.
        let run = self
            .module()
            .constant_u32(workgroup.saturating_mul(strips as u32))?;
        Ok(self.module().i_mul(uint, group, run)?)
    }

    /// The element index this invocation touches on `strip` of the run starting at `base`.
    ///
    /// With one strip and no offset it collapses to `group × workgroup + local`, the plain global
    /// invocation index. Strip zero skips one addition, which is the common case and one
    /// instruction fewer; `offset` folds into the same addition rather than costing another.
    ///
    /// [`super::plane`] uses the same expression as its *column*, which is why this is visible
    /// there: a grid's address is this index within a row, plus the row's own offset. Two
    /// arithmetics that agreed by being written twice would not stay agreed.
    pub(super) fn address(&mut self, base: Id, strip: usize, offset: u32) -> Result<Id, LaneError> {
        let uint = self.uint();
        let workgroup = self.shape().workgroup;
        let (local, _) = self.position();

        // The strip's stride and the caller's offset are both constants, so they add at build time
        // and cost one instruction between them rather than two.
        let shift = (strip as u32)
            .saturating_mul(workgroup)
            .saturating_add(offset);
        let within = if shift == 0 {
            local
        } else {
            let shift = self.module().constant_u32(shift)?;
            self.module().i_add(uint, local, shift)?
        };

        Ok(self.module().i_add(uint, base, within)?)
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used)]

    use crate::decode;
    use crate::kernel::{Kernel, Shape};
    use crate::lanes::{F32, LaneError};
    use crate::module::op;

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    #[test]
    fn a_store_at_writes_once_wherever_it_was_pointed() {
        // One store, and its address comes from the caller's id rather than from this invocation.
        // A version that fell back to `store_scalar`'s addressing would write 64 different slots
        // instead of one, which is a plausible-looking module and the wrong answer.
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 3)).expect("built");
        let slot = kernel.workgroup_index();
        let value = kernel.local_index();
        kernel.store_at(2, slot, value).expect("stored");

        let words = kernel.finish().expect("finished");
        assert_eq!(count(&words, op::STORE), 1);

        let chains: Vec<Vec<u32>> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::ACCESS_CHAIN)
            .map(|instruction| instruction.operands().to_vec())
            .collect();
        let chain = chains.last().expect("an access chain");

        // result type, result id, base, then the struct member and the element index.
        assert_eq!(
            chain.get(4).copied(),
            Some(slot.word()),
            "the index is not the slot the caller named"
        );
    }

    #[test]
    fn a_load_at_reads_once_from_where_it_was_pointed() {
        // One load from the buffer, at the caller's index. The hazard it guards against is the
        // same as `store_at`'s: falling back to invocation-derived addressing gives every lane a
        // different element of a buffer holding one value per workgroup.
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 3)).expect("built");
        let slot = kernel.workgroup_index();
        kernel.load_at(2, slot).expect("loaded");

        let words = kernel.finish().expect("finished");

        // Against a kernel that did nothing, because the prologue loads the two built-in vectors
        // and those are `OpLoad`s too. The difference is the one this call added.
        let bare = Kernel::<F32>::new(Shape::new(32, 64, 3))
            .expect("built")
            .finish()
            .expect("finished");
        assert_eq!(
            count(&words, op::LOAD),
            count(&bare, op::LOAD) + 1,
            "one load, and no address to build"
        );

        let chains: Vec<Vec<u32>> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::ACCESS_CHAIN)
            .map(|instruction| instruction.operands().to_vec())
            .collect();
        let chain = chains.last().expect("an access chain");

        assert_eq!(chain.get(4).copied(), Some(slot.word()));
    }

    #[test]
    fn reading_from_a_buffer_that_was_never_bound_is_refused() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let slot = kernel.workgroup_index();

        assert_eq!(
            kernel.load_at(2, slot).err(),
            Some(LaneError::NoSuchBuffer { index: 2, bound: 2 })
        );
    }

    #[test]
    fn store_at_and_store_scalar_reach_different_addresses() {
        // The distinction the pair exists for. `store_scalar` derives its slot from the invocation
        // and so has arithmetic behind it; `store_at` has none, because the caller did it. If the
        // two ever emit the same instruction count, one of them has stopped being itself.
        let scalar = {
            let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 3)).expect("built");
            let value = kernel.local_index();
            kernel.store_scalar(2, value).expect("stored");
            kernel.finish().expect("finished")
        };
        let at = {
            let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 3)).expect("built");
            let slot = kernel.workgroup_index();
            let value = kernel.local_index();
            kernel.store_at(2, slot, value).expect("stored");
            kernel.finish().expect("finished")
        };

        assert_eq!(count(&scalar, op::STORE), count(&at, op::STORE));
        assert!(
            count(&scalar, op::I_ADD) > count(&at, op::I_ADD),
            "store_at should compute no address of its own"
        );
    }

    #[test]
    fn storing_into_a_buffer_that_was_never_bound_is_refused() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let slot = kernel.workgroup_index();

        assert_eq!(
            kernel.store_at(2, slot, slot).err(),
            Some(LaneError::NoSuchBuffer { index: 2, bound: 2 })
        );
    }

    #[test]
    fn a_strip_mined_load_reads_once_per_strip() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<128>(0).expect("loaded");

        assert_eq!(value.strip_count(), 4);
        assert_eq!(
            count(&kernel.finish().expect("finished"), op::ACCESS_CHAIN),
            4
        );
    }

    #[test]
    fn the_workgroups_own_run_is_multiplied_out_once_per_access() {
        // It used to be once per *strip*, so a four-strip load emitted four identical multiplies.
        // Every driver folds those back into one, which is exactly why nothing caught it: the
        // module said something the arithmetic does not, and the answer was right anyway.
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<128>(0).expect("loaded");
        kernel.store(1, value).expect("stored");

        let words = kernel.finish().expect("finished");
        assert_eq!(count(&words, op::ACCESS_CHAIN), 8, "four strips each way");
        assert_eq!(count(&words, op::I_MUL), 2, "one per access");
    }

    #[test]
    fn strip_zero_costs_one_addition_fewer_than_the_rest() {
        // `base + local` for the first, `base + (local + s*workgroup)` for the others.
        let mut one = Kernel::<F32>::new(Shape::new(32, 64, 1)).expect("built");
        one.load::<32>(0).expect("loaded");
        let single = count(&one.finish().expect("finished"), op::I_ADD);

        let mut two = Kernel::<F32>::new(Shape::new(32, 64, 1)).expect("built");
        two.load::<64>(0).expect("loaded");
        let double = count(&two.finish().expect("finished"), op::I_ADD);

        assert_eq!(single, 1);
        assert_eq!(double, 3, "one for strip zero, two for strip one");
    }

    #[test]
    fn a_buffer_that_was_never_bound_is_refused_by_index() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");

        assert_eq!(
            kernel.load::<32>(2).err(),
            Some(LaneError::NoSuchBuffer { index: 2, bound: 2 })
        );
    }

    #[test]
    fn a_scalar_store_uses_the_one_strip_layout_whatever_was_loaded() {
        // The point of `strips` belonging to the access: a kernel that reads four at a time and
        // writes one at a time has two differently shaped buffers.
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<128>(0).expect("loaded");
        let total = kernel
            .lanes()
            .expect("lanes")
            .reduce_sum(value)
            .expect("sum");
        kernel.store_scalar(1, total).expect("stored");

        let words = kernel.finish().expect("finished");
        // Four chains for the load, one for the store.
        assert_eq!(count(&words, op::ACCESS_CHAIN), 5);
        assert_eq!(count(&words, op::STORE), 1);
    }

    #[test]
    fn an_offset_load_folds_into_the_addition_strip_zero_already_had() {
        // `base + (local + offset)`, not `base + local + offset`: the offset is a constant and the
        // stride is a constant, so they meet before any instruction is emitted.
        let mut plain = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        plain.load::<32>(0).expect("loaded");
        let without = count(&plain.finish().expect("finished"), op::I_ADD);

        let mut shifted = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        shifted.load_offset::<32>(0, 4096).expect("loaded");
        let with = count(&shifted.finish().expect("finished"), op::I_ADD);

        assert_eq!(without, 1, "base + local");
        assert_eq!(with, 2, "base + (local + offset)");
    }

    #[test]
    fn an_offset_of_zero_is_the_plain_load_instruction_for_instruction() {
        let mut plain = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        plain.load::<64>(0).expect("loaded");
        let expected = plain.finish().expect("finished");

        let mut shifted = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        shifted.load_offset::<64>(0, 0).expect("loaded");

        assert_eq!(shifted.finish().expect("finished"), expected);
    }

    #[test]
    fn the_offset_reaches_the_constant_the_address_adds() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        kernel.load_offset::<32>(0, 512).expect("loaded");

        let words = kernel.finish().expect("finished");
        let declared: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CONSTANT)
            .filter_map(|instruction| instruction.operands().get(2).copied())
            .collect();

        assert!(
            declared.contains(&512),
            "the offset never became a constant: {declared:?}"
        );
    }

    #[test]
    fn each_strip_of_an_offset_load_keeps_its_own_stride() {
        // Two strips at an offset of 100 on a workgroup of 64: `local + 100` and `local + 164`.
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        kernel.load_offset::<64>(0, 100).expect("loaded");

        let words = kernel.finish().expect("finished");
        let declared: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CONSTANT)
            .filter_map(|instruction| instruction.operands().get(2).copied())
            .collect();

        assert!(declared.contains(&100), "strip zero");
        assert!(declared.contains(&164), "strip one keeps its stride");
    }

    #[test]
    fn an_offset_that_is_a_value_costs_one_addition_per_strip() {
        // The whole difference between `load_offset` and `load_offset_by`: a constant folds into
        // the address arithmetic and a value cannot.
        //
        // The comparison is against a load with **no** offset, not against one with a constant
        // offset — because a constant offset of zero is free on strip zero and a non-zero one is
        // not, so `load_offset` already costs a variable amount. Two strips, two extra adds.
        let mut plain = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        plain.load::<64>(0).expect("loaded");
        let without = count(&plain.finish().expect("finished"), op::I_ADD);

        let mut open = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let offset = open.module().constant_u32(128).expect("128");
        open.load_offset_by::<64>(0, offset).expect("loaded");
        let paid = count(&open.finish().expect("finished"), op::I_ADD);

        assert_eq!(paid, without + 2, "one addition per strip, and no more");
    }

    #[test]
    fn an_offset_by_value_reads_as_many_places_as_the_constant_one() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let offset = kernel.module().constant_u32(64).expect("64");
        let value = kernel.load_offset_by::<128>(0, offset).expect("loaded");

        assert_eq!(value.strip_count(), 4);
        // Chains rather than loads: the interface itself reads the two built-in vectors, so an
        // `OpLoad` count answers a question about `Kernel::new` as well as about this.
        assert_eq!(
            count(&kernel.finish().expect("finished"), op::ACCESS_CHAIN),
            4
        );
    }

    #[test]
    fn the_open_offset_is_added_to_the_address_and_not_used_as_one() {
        // The mistake that would still validate: using the offset *as* the index rather than
        // adding it to the address gives every invocation the same element, which looks like a
        // broadcast and is a wrong answer in the shape of a plausible one.
        use crate::decode;

        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let offset = kernel.module().constant_u32(64).expect("64");
        kernel.load_offset_by::<32>(0, offset).expect("loaded");

        let words = kernel.finish().expect("finished");
        let index = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::ACCESS_CHAIN)
            .expect("emitted")
            .operands()
            .last()
            .copied()
            .expect("an index");

        assert_ne!(
            index,
            offset.word(),
            "the chain indexes by the offset itself rather than by the address plus it"
        );
    }

    #[test]
    fn storing_a_vector_writes_once_per_strip() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<64>(0).expect("loaded");
        kernel.store(1, value).expect("stored");

        assert_eq!(count(&kernel.finish().expect("finished"), op::STORE), 2);
    }
}
