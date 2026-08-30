use super::Kernel;
use crate::lanes::{Element, LaneError, Vector};
use crate::module::Id;

impl<T: Element> Kernel<T> {
    pub fn row(&self) -> Result<Id, LaneError> {
        self.row_index().ok_or(LaneError::NotAGrid)
    }

    pub fn load_row<const LANES: u32>(
        &mut self,
        index: u32,
        pitch: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let row = self.row()?;
        self.load_row_at(index, pitch, row)
    }

    pub fn load_row_at<const LANES: u32>(
        &mut self,
        index: u32,
        pitch: u32,
        row: Id,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let buffer = self.buffer(index)?;
        let strips = self.strips::<LANES>()?;
        let start = self.start_of(pitch, row, strips)?;

        let element = self.element();
        let mut loaded = Vec::with_capacity(strips);
        for strip in 0..strips {
            let pointer = self.cell_pointer(buffer, start, strip)?;
            loaded.push(self.module().load(element, pointer)?);
        }

        self.lanes()?.from_strips(&loaded)
    }

    pub fn store_row<const LANES: u32>(
        &mut self,
        index: u32,
        pitch: u32,
        value: Vector<T, LANES>,
    ) -> Result<(), LaneError> {
        let row = self.row()?;
        self.store_row_at(index, pitch, row, value)
    }

    pub fn store_row_scalar(&mut self, index: u32, pitch: u32, value: Id) -> Result<(), LaneError> {
        let row = self.row()?;
        let buffer = self.buffer(index)?;
        let start = self.start_of(pitch, row, 1)?;
        let pointer = self.cell_pointer(buffer, start, 0)?;
        Ok(self.module().store(pointer, value)?)
    }

    pub fn store_row_at<const LANES: u32>(
        &mut self,
        index: u32,
        pitch: u32,
        row: Id,
        value: Vector<T, LANES>,
    ) -> Result<(), LaneError> {
        let buffer = self.buffer(index)?;
        let strips = value.strip_count();
        let start = self.start_of(pitch, row, strips)?;

        for (strip, &id) in value.strips().iter().enumerate() {
            let pointer = self.cell_pointer(buffer, start, strip)?;
            self.module().store(pointer, id)?;
        }
        Ok(())
    }

    fn start_of(&mut self, pitch: u32, row: Id, strips: usize) -> Result<Id, LaneError> {
        if pitch == 0 {
            return Err(LaneError::BadPitch);
        }

        let uint = self.uint();
        let pitch = self.module().constant_u32(pitch)?;
        let above = self.module().i_mul(uint, row, pitch)?;
        let run = self.run_start(strips)?;
        Ok(self.module().i_add(uint, above, run)?)
    }

    fn cell_pointer(&mut self, buffer: Id, start: Id, strip: usize) -> Result<Id, LaneError> {
        let at = self.address(start, strip, 0)?;
        let element_pointer = self.element_pointer();
        let zero = self.zero();
        Ok(self
            .module()
            .access_chain(element_pointer, buffer, &[zero, at])?)
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

    fn local_size(words: &[u32]) -> [u32; 3] {
        let operands = decode::body(words)
            .find(|instruction| instruction.opcode() == op::EXECUTION_MODE)
            .expect("every kernel declares one")
            .operands()
            .to_vec();
        [operands[2], operands[3], operands[4]]
    }

    #[test]
    fn a_linear_kernel_has_no_row_and_refuses_to_invent_one() {
        let mut kernel = Kernel::<U32>::new(Shape::new(32, 64, 2)).expect("built");

        assert_eq!(kernel.row().err(), Some(LaneError::NotAGrid));
        assert_eq!(
            kernel.load_row::<32>(0, 64).err(),
            Some(LaneError::NotAGrid)
        );
    }

    #[test]
    fn a_grid_one_row_deep_declares_the_same_local_size_as_a_linear_kernel() {
        let linear = Kernel::<U32>::new(Shape::new(32, 64, 2))
            .expect("built")
            .finish()
            .expect("finished");
        let grid = Kernel::<U32>::new(Shape::grid(32, 64, 1, 2))
            .expect("built")
            .finish()
            .expect("finished");

        assert_eq!(local_size(&linear), [64, 1, 1]);
        assert_eq!(local_size(&grid), [64, 1, 1]);
        assert_ne!(linear, grid, "the grid still extracts a row");
    }

    #[test]
    fn a_deeper_workgroup_reaches_the_local_size() {
        let words = Kernel::<U32>::new(Shape::grid(32, 32, 8, 2))
            .expect("built")
            .finish()
            .expect("finished");

        assert_eq!(local_size(&words), [32, 8, 1]);
    }

    #[test]
    fn one_row_per_workgroup_costs_no_arithmetic_because_local_y_can_only_be_zero() {
        let shallow = Kernel::<U32>::new(Shape::grid(32, 32, 1, 2))
            .expect("built")
            .finish()
            .expect("finished");
        let deep = Kernel::<U32>::new(Shape::grid(32, 32, 2, 2))
            .expect("built")
            .finish()
            .expect("finished");

        assert_eq!(
            count(&shallow, op::COMPOSITE_EXTRACT),
            3,
            "x, x and group.y"
        );
        assert_eq!(count(&shallow, op::I_MUL), 0);
        assert_eq!(count(&shallow, op::I_ADD), 0);

        assert_eq!(count(&deep, op::COMPOSITE_EXTRACT), 4, "and local.y");
        assert_eq!(count(&deep, op::I_MUL), 1, "group.y * rows");
        assert_eq!(count(&deep, op::I_ADD), 1, "+ local.y");
    }

    #[test]
    fn a_grid_with_no_rows_in_it_is_refused_by_name() {
        assert_eq!(
            Kernel::<U32>::new(Shape::grid(32, 64, 0, 2)).err(),
            Some(LaneError::BadRows { rows: 0 })
        );
    }

    #[test]
    fn a_pitch_of_zero_is_refused_rather_than_treated_as_one_row() {
        let mut kernel = Kernel::<U32>::new(Shape::grid(32, 64, 1, 2)).expect("built");

        assert_eq!(kernel.load_row::<32>(0, 0).err(), Some(LaneError::BadPitch));
    }

    #[test]
    fn a_row_load_multiplies_twice_however_many_strips_it_reads() {
        let mut one = Kernel::<U32>::new(Shape::grid(32, 32, 1, 2)).expect("built");
        one.load_row::<32>(0, 1024).expect("loaded");
        let single = one.finish().expect("finished");

        let mut four = Kernel::<U32>::new(Shape::grid(32, 32, 1, 2)).expect("built");
        four.load_row::<128>(0, 1024).expect("loaded");
        let quadruple = four.finish().expect("finished");

        assert_eq!(count(&single, op::I_MUL), 2, "the row and the run");
        assert_eq!(count(&quadruple, op::I_MUL), 2, "still two");
        assert_eq!(count(&single, op::ACCESS_CHAIN), 1);
        assert_eq!(count(&quadruple, op::ACCESS_CHAIN), 4);
    }

    #[test]
    fn the_pitch_reaches_the_constant_the_address_multiplies_by() {
        let mut kernel = Kernel::<U32>::new(Shape::grid(32, 32, 1, 2)).expect("built");
        kernel.load_row::<32>(0, 4096).expect("loaded");

        let words = kernel.finish().expect("finished");
        let declared: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CONSTANT)
            .filter_map(|instruction| instruction.operands().get(2).copied())
            .collect();

        assert!(
            declared.contains(&4096),
            "the pitch never became a constant: {declared:?}"
        );
    }

    #[test]
    fn a_named_row_is_the_one_multiplied_and_not_this_invocations() {
        let mut kernel = Kernel::<U32>::new(Shape::grid(32, 32, 1, 2)).expect("built");
        let first = kernel.module().constant_u32(0).expect("0");
        kernel.load_row_at::<32>(0, 1024, first).expect("loaded");

        let words = kernel.finish().expect("finished");
        let multiplied = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::I_MUL)
            .expect("emitted")
            .operands()[2];

        assert_eq!(multiplied, first.word(), "the row that was passed in");
    }

    #[test]
    fn storing_a_row_writes_once_per_strip_and_multiplies_per_access() {
        let mut kernel = Kernel::<U32>::new(Shape::grid(32, 32, 1, 2)).expect("built");
        let value = kernel.load_row::<64>(0, 512).expect("loaded");
        kernel.store_row(1, 512, value).expect("stored");

        let words = kernel.finish().expect("finished");
        assert_eq!(count(&words, op::STORE), 2, "one per strip");
        assert_eq!(count(&words, op::I_MUL), 4, "two per access, not per strip");
    }

    #[test]
    fn a_row_access_to_a_buffer_that_was_never_bound_is_refused_by_index() {
        let mut kernel = Kernel::<U32>::new(Shape::grid(32, 32, 1, 2)).expect("built");

        assert_eq!(
            kernel.load_row::<32>(4, 512).err(),
            Some(LaneError::NoSuchBuffer { index: 4, bound: 2 })
        );
    }
}
