use super::Kernel;
use crate::lanes::{Element, LaneError, Vector};
use crate::module::Id;

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
    use crate::lanes::{F32, LaneError};
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
}
