use super::{I32, Integer, LaneError, Lanes, U32, Vector};
use crate::module::op;

impl Lanes<'_> {
    /// ```compile_fail
    /// use simdr::kernel::{Kernel, Shape};
    /// use simdr::lanes::{F32, U32};
    ///
    /// let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2))?;
    /// let value = kernel.load::<32>(0)?;
    /// let mut lanes = kernel.lanes()?;
    /// let by = lanes.splat_bits::<U32, 32>(3)?;
    ///
    /// let shifted = lanes.shift_left(value, by)?;
    /// # Ok::<(), simdr::lanes::LaneError>(())
    /// ```
    pub fn shift_left<T: Integer, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
        amount: Vector<U32, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.shift(op::SHIFT_LEFT_LOGICAL, value, amount)
    }

    pub fn shift_right_logical<T: Integer, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
        amount: Vector<U32, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.shift(op::SHIFT_RIGHT_LOGICAL, value, amount)
    }

    pub fn shift_right_arithmetic<T: Integer, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
        amount: Vector<U32, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.shift(op::SHIFT_RIGHT_ARITHMETIC, value, amount)
    }

    fn shift<T: Integer, const LANES: u32>(
        &mut self,
        opcode: u16,
        value: Vector<T, LANES>,
        amount: Vector<U32, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let mut ids = Vec::with_capacity(value.strip_count());

        for (&base, &by) in value.strips().iter().zip(amount.strips()) {
            ids.push(self.module().binary(opcode, element, base, by)?);
        }

        self.from_strips(&ids)
    }

    pub fn reinterpret_unsigned<const LANES: u32>(
        &mut self,
        value: Vector<U32, LANES>,
    ) -> Result<Vector<I32, LANES>, LaneError> {
        let signed = self.type_of::<I32>()?;
        let mut ids = Vec::with_capacity(value.strip_count());

        for &strip in value.strips() {
            ids.push(self.module().unary(op::BITCAST, signed, strip)?);
        }

        self.from_strips(&ids)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::module::{Module, Version};

    fn built() -> Module {
        Module::new(Version::V1_3)
    }

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    #[test]
    fn the_two_right_shifts_are_two_instructions() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 32>(0xffff_0000).expect("splat");
        let by = lanes.splat_bits::<U32, 32>(8).expect("eight");

        lanes.shift_right_logical(value, by).expect("logical");
        lanes.shift_right_arithmetic(value, by).expect("arithmetic");

        let words = module.finish();
        assert_eq!(count(&words, op::SHIFT_RIGHT_LOGICAL), 1);
        assert_eq!(count(&words, op::SHIFT_RIGHT_ARITHMETIC), 1);
    }

    #[test]
    fn a_shift_is_one_instruction_per_strip() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 128>(1).expect("splat");
        let by = lanes.splat_bits::<U32, 128>(3).expect("three");

        let shifted = lanes.shift_left(value, by).expect("shifted");

        assert_eq!(shifted.strip_count(), 4);
        assert_eq!(count(&module.finish(), op::SHIFT_LEFT_LOGICAL), 4);
    }

    #[test]
    fn the_amount_is_the_second_operand_and_the_value_the_first() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 32>(1).expect("one");
        let by = lanes.splat_bits::<U32, 32>(24).expect("twenty-four");

        lanes.shift_left(value, by).expect("shifted");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::SHIFT_LEFT_LOGICAL)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(operands[2], value.id().word(), "the value shifts");
        assert_eq!(operands[3], by.id().word(), "the amount is how far");
    }

    #[test]
    fn a_shift_keeps_the_values_element_type_and_not_the_amounts() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let signed = lanes.splat_bits::<I32, 32>(0xffff_ff00).expect("negative");
        let by = lanes.splat_bits::<U32, 32>(4).expect("four");

        let shifted = lanes.shift_right_arithmetic(signed, by).expect("shifted");

        let words = module.finish();
        let element = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::SHIFT_RIGHT_ARITHMETIC)
            .expect("emitted")
            .operands()[0];

        assert_eq!(shifted.strip_count(), 1);
        assert_ne!(element, 0);
    }

    #[test]
    fn reinterpreting_is_a_bitcast_in_both_directions() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let unsigned = lanes.splat_bits::<U32, 32>(0xffff_ffff).expect("splat");

        let signed = lanes.reinterpret_unsigned(unsigned).expect("to i32");
        lanes.reinterpret(signed).expect("and back");

        assert_eq!(count(&module.finish(), op::BITCAST), 2);
    }
}
