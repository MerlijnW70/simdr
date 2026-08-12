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

        let element = self.element();
        let mut loaded = Vec::with_capacity(strips);
        for strip in 0..strips {
            let pointer = self.element_pointer_at(buffer, strip, strips, offset)?;
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
        let pointer = self.element_pointer_at(buffer, 0, 1, 0)?;
        Ok(self.module().store(pointer, value)?)
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

        for (strip, &id) in value.strips().iter().enumerate() {
            let pointer = self.element_pointer_at(buffer, strip, strips, 0)?;
            self.module().store(pointer, id)?;
        }
        Ok(())
    }

    /// A pointer to this invocation's element on `strip` of an access that has `strips` of them.
    fn element_pointer_at(
        &mut self,
        buffer: Id,
        strip: usize,
        strips: usize,
        offset: u32,
    ) -> Result<Id, LaneError> {
        let at = self.address(strip, strips, offset)?;
        let element_pointer = self.element_pointer();
        let zero = self.zero();
        Ok(self
            .module()
            .access_chain(element_pointer, buffer, &[zero, at])?)
    }

    /// The element index this invocation touches on `strip`.
    ///
    /// With one strip and no offset it collapses to `group × workgroup + local`, the plain global
    /// invocation index. Strip zero skips one addition, which is the common case and one
    /// instruction fewer; `offset` folds into the same addition rather than costing another.
    fn address(&mut self, strip: usize, strips: usize, offset: u32) -> Result<Id, LaneError> {
        let uint = self.uint();
        let workgroup = self.shape().workgroup;
        let (local, group) = self.position();

        // Both factors are invocation counts below `MAX_STRIPS` times a workgroup size, so these
        // saturate only for a shape no device would accept.
        let run = self
            .module()
            .constant_u32(workgroup.saturating_mul(strips as u32))?;
        let base = self.module().i_mul(uint, group, run)?;

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
    fn storing_a_vector_writes_once_per_strip() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<64>(0).expect("loaded");
        kernel.store(1, value).expect("stored");

        assert_eq!(count(&kernel.finish().expect("finished"), op::STORE), 2);
    }
}
