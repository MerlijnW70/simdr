use super::Kernel;
use crate::lanes::{Element, LaneError, Vector};
use crate::module::Id;
use crate::spec::{Decoration, StorageClass};
use core::marker::PhantomData;

impl<T: Element> Kernel<T> {
    pub fn strips<const LANES: u32>(&mut self) -> Result<usize, LaneError> {
        self.lanes()?.strips_for::<LANES>()
    }

    pub fn load<const LANES: u32>(&mut self, index: u32) -> Result<Vector<T, LANES>, LaneError> {
        self.load_offset(index, 0)
    }

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

    /// A binding whose elements are `E` rather than the kernel's own type.
    ///
    /// The bindings a [`Shape`](super::Shape) asks for all hold what the kernel
    /// holds, which is what a kernel usually wants and is not what a look-up
    /// table is: the indices are `u32` and the table is whatever is being
    /// looked up. This declares one more descriptor beyond those, at the next
    /// index, holding `E`.
    pub fn bind<E: Element>(&mut self) -> Result<Binding<E>, LaneError> {
        let at = self.bound();

        let element = E::type_id(self.module())?;
        E::require_in_storage_buffer(self.module())?;

        let elements = self.module().type_runtime_array(element)?;
        let block = self.module().type_struct(&[elements])?;
        self.module()
            .decorate(elements, Decoration::ArrayStride, &[E::STRIDE])?;
        self.module().decorate(block, Decoration::Block, &[])?;
        self.module()
            .member_decorate(block, 0, Decoration::Offset, &[0])?;

        let pointer = self
            .module()
            .type_pointer(StorageClass::StorageBuffer, block)?;
        let variable = self
            .module()
            .global_variable(pointer, StorageClass::StorageBuffer)?;
        self.module()
            .decorate(variable, Decoration::DescriptorSet, &[0])?;
        self.module()
            .decorate(variable, Decoration::Binding, &[at])?;

        let element_pointer = self
            .module()
            .type_pointer(StorageClass::StorageBuffer, element)?;

        self.remember(variable);
        Ok(Binding {
            variable,
            element,
            element_pointer,
            at,
            held: PhantomData,
        })
    }

    /// One element per lane out of a binding that holds `E`, laid out the way
    /// [`Kernel::load`] lays out the kernel's own.
    pub fn load_from<E: Element, const LANES: u32>(
        &mut self,
        binding: Binding<E>,
    ) -> Result<Vector<E, LANES>, LaneError> {
        let strips = self.strips::<LANES>()?;
        let base = self.run_start(strips)?;
        let zero = self.zero();

        let mut loaded = Vec::with_capacity(strips);
        for strip in 0..strips {
            let at = self.address(base, strip, 0)?;
            let pointer = self.module().access_chain(
                binding.element_pointer,
                binding.variable,
                &[zero, at],
            )?;
            loaded.push(self.module().load(binding.element, pointer)?);
        }

        self.lanes()?.from_strips(&loaded)
    }

    pub fn store_into<E: Element, const LANES: u32>(
        &mut self,
        binding: Binding<E>,
        value: Vector<E, LANES>,
    ) -> Result<(), LaneError> {
        let strips = value.strip_count();
        let base = self.run_start(strips)?;
        let zero = self.zero();

        for (strip, &id) in value.strips().iter().enumerate() {
            let at = self.address(base, strip, 0)?;
            let pointer = self.module().access_chain(
                binding.element_pointer,
                binding.variable,
                &[zero, at],
            )?;
            self.module().store(pointer, id)?;
        }
        Ok(())
    }

    pub fn store_scalar(&mut self, index: u32, value: Id) -> Result<(), LaneError> {
        let buffer = self.buffer(index)?;
        let base = self.run_start(1)?;
        let pointer = self.element_pointer_at(buffer, base, 0, 0)?;
        Ok(self.module().store(pointer, value)?)
    }

    pub fn store_at(&mut self, binding: u32, index: Id, value: Id) -> Result<(), LaneError> {
        let pointer = self.element_pointer_to(binding, index)?;
        Ok(self.module().store(pointer, value)?)
    }

    pub fn load_at(&mut self, binding: u32, index: Id) -> Result<Id, LaneError> {
        let pointer = self.element_pointer_to(binding, index)?;
        let element = self.element();
        Ok(self.module().load(element, pointer)?)
    }

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

    pub(super) fn run_start(&mut self, strips: usize) -> Result<Id, LaneError> {
        let uint = self.uint();
        let workgroup = self.shape().workgroup;
        let (_, group) = self.position();

        let run = checked("workgroup × strips", u64::from(workgroup) * strips as u64)?;

        let run = self.module().constant_u32(run)?;
        Ok(self.module().i_mul(uint, group, run)?)
    }

    pub(super) fn address(&mut self, base: Id, strip: usize, offset: u32) -> Result<Id, LaneError> {
        let uint = self.uint();
        let workgroup = self.shape().workgroup;
        let (local, _) = self.position();

        let shift = checked(
            "strip × workgroup + offset",
            strip as u64 * u64::from(workgroup) + u64::from(offset),
        )?;

        let within = if shift == 0 {
            local
        } else {
            let shift = self.module().constant_u32(shift)?;
            self.module().i_add(uint, local, shift)?
        };

        Ok(self.module().i_add(uint, base, within)?)
    }
}

fn checked(term: &'static str, needed: u64) -> Result<u32, LaneError> {
    u32::try_from(needed).map_err(|_| LaneError::AddressOverflow { term, needed })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use crate::decode;
    use crate::kernel::{Kernel, Shape};
    use crate::lanes::{F32, LaneError, U8, U32};
    use crate::module::op;

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    #[test]
    fn an_address_that_does_not_fit_a_word_is_refused_rather_than_saturated() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        assert_eq!(
            kernel.load_offset::<128>(0, u32::MAX).err(),
            Some(LaneError::AddressOverflow {
                term: "strip × workgroup + offset",
                needed: u64::from(u32::MAX) + 64,
            }),
            "strip 1 of a 64-invocation workgroup, plus every bit of offset"
        );

        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        assert!(kernel.load_offset::<32>(0, u32::MAX).is_ok());

        let mut wide = Kernel::<F32>::new(Shape::new(32, 1 << 30, 2)).expect("built");
        assert_eq!(
            wide.load::<128>(0).err(),
            Some(LaneError::AddressOverflow {
                term: "workgroup × strips",
                needed: 1 << 32,
            }),
            "four strips of a workgroup of 2³⁰"
        );
    }

    #[test]
    fn a_store_at_writes_once_wherever_it_was_pointed() {
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

        assert_eq!(
            chain.get(4).copied(),
            Some(slot.word()),
            "the index is not the slot the caller named"
        );
    }

    #[test]
    fn a_load_at_reads_once_from_where_it_was_pointed() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 3)).expect("built");
        let slot = kernel.workgroup_index();
        kernel.load_at(2, slot).expect("loaded");

        let words = kernel.finish().expect("finished");

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
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<128>(0).expect("loaded");
        kernel.store(1, value).expect("stored");

        let words = kernel.finish().expect("finished");
        assert_eq!(count(&words, op::ACCESS_CHAIN), 8, "four strips each way");
        assert_eq!(count(&words, op::I_MUL), 2, "one per access");
    }

    #[test]
    fn strip_zero_costs_one_addition_fewer_than_the_rest() {
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
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<128>(0).expect("loaded");
        let total = kernel
            .lanes()
            .expect("lanes")
            .reduce_sum(value)
            .expect("sum");
        kernel.store_scalar(1, total).expect("stored");

        let words = kernel.finish().expect("finished");
        assert_eq!(count(&words, op::ACCESS_CHAIN), 5);
        assert_eq!(count(&words, op::STORE), 1);
    }

    #[test]
    fn an_offset_load_folds_into_the_addition_strip_zero_already_had() {
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
        assert_eq!(
            count(&kernel.finish().expect("finished"), op::ACCESS_CHAIN),
            4
        );
    }

    #[test]
    fn the_open_offset_is_added_to_the_address_and_not_used_as_one() {
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

    #[test]
    fn a_bound_descriptor_lands_after_the_ones_the_shape_asked_for() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");

        let first = kernel.bind::<U32>().expect("bound");
        let second = kernel.bind::<U32>().expect("bound");

        assert_eq!(first.at(), 2, "the shape asked for two, so the next is two");
        assert_eq!(second.at(), 3, "and one more lands beside it, not on it");
    }

    #[test]
    fn a_bound_descriptor_declares_the_stride_of_its_own_element_and_not_the_kernels() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 1)).expect("built");
        let narrow = kernel.bind::<U8>().expect("bound");

        let held = kernel.load_from::<U8, 32>(narrow).expect("loaded");
        kernel.store_into(narrow, held).expect("stored");

        let words = kernel.finish().expect("finished");
        let strides: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::DECORATE)
            .filter(|instruction| {
                instruction.operands().get(1).copied()
                    == Some(crate::spec::Decoration::ArrayStride.word())
            })
            .filter_map(|instruction| instruction.operands().get(2).copied())
            .collect();

        assert!(
            strides.contains(&4),
            "the kernel holds f32, which is four wide"
        );
        assert!(
            strides.contains(&1),
            "and the binding holds u8, which is one -- a binding taking the kernel's stride would \
             read every fourth byte"
        );
    }

    #[test]
    fn a_kernel_reads_indices_as_one_type_and_the_table_as_another() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let slots = kernel.bind::<U32>().expect("bound");

        let indices = kernel.load_from::<U32, 32>(slots).expect("loaded");
        let picked = kernel.gather::<32>(0, indices).expect("gathered");
        kernel.store(1, picked).expect("stored");

        let words = kernel.finish().expect("finished");
        let elements: std::collections::BTreeSet<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::TYPE_RUNTIME_ARRAY)
            .filter_map(|instruction| instruction.operands().get(1).copied())
            .collect();

        assert_eq!(
            elements.len(),
            2,
            "the arrays behind the three descriptors are of two elements: the table and the              answer hold what the kernel holds, and the indices hold their own type. A binding              that took the kernel's would leave one"
        );
    }
}

/// A descriptor binding holding `E`, which is not the type the kernel that
/// declared it holds.
#[derive(Debug, PartialEq, Eq)]
pub struct Binding<E> {
    variable: Id,
    element: Id,
    element_pointer: Id,
    at: u32,
    held: PhantomData<E>,
}

impl<E> Clone for Binding<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E> Copy for Binding<E> {}

impl<E> Binding<E> {
    /// Which descriptor this is, counting from the ones the shape asked for.
    #[must_use]
    pub const fn at(self) -> u32 {
        self.at
    }
}
