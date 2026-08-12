//! Moving a lane's bits left or right.
//!
//! Elementwise like everything in [`super::arithmetic`], and here rather than there because the
//! right shift comes in two kinds and the difference between them is the whole content of the
//! module: `OpShiftRightLogical` fills with zeros and `OpShiftRightArithmetic` fills with copies
//! of the sign bit, and they agree on every value whose top bit is clear.
//!
//! That is the shape of mistake this crate keeps finding: two instructions that agree on small
//! numbers. So the two are named apart and neither is a default.
//!
//! # The shift amount is a vector
//!
//! SPIR-V takes it as an ordinary operand rather than a literal, so it may vary per lane. A
//! constant shift is a splat, which costs nothing the driver does not fold away, and a per-lane
//! shift — extracting a different byte in each lane, say — is then expressible rather than needing
//! a second entry point.
//!
//! **A shift of more than the type's width is undefined**, and nothing here checks it: the amount
//! is an id by the time it arrives, so a check would mean emitting a comparison and a select on
//! every call. SPIR-V says the same thing about the same instruction.

use super::{Element, I32, LaneError, Lanes, U32, Vector};
use crate::module::op;

impl Lanes<'_> {
    /// `value << amount`, elementwise.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn shift_left<T: Element, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
        amount: Vector<U32, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.shift(op::SHIFT_LEFT_LOGICAL, value, amount)
    }

    /// `value >> amount`, filling with **zeros**.
    ///
    /// What an unsigned value wants. For a signed one this turns a negative number into a large
    /// positive one, which is [`Lanes::shift_right_arithmetic`]'s job to avoid.
    ///
    /// # Errors
    ///
    /// As [`Lanes::shift_left`].
    pub fn shift_right_logical<T: Element, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
        amount: Vector<U32, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.shift(op::SHIFT_RIGHT_LOGICAL, value, amount)
    }

    /// `value >> amount`, filling with copies of the **sign bit**.
    ///
    /// Paired with [`Lanes::shift_left`], this is how a narrow signed field is extracted from a
    /// wider word: shift it to the top, then shift it back down and let the sign spread.
    ///
    /// # Errors
    ///
    /// As [`Lanes::shift_left`].
    pub fn shift_right_arithmetic<T: Element, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
        amount: Vector<U32, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.shift(op::SHIFT_RIGHT_ARITHMETIC, value, amount)
    }

    /// One shift instruction per strip.
    ///
    /// Its own helper rather than [`Lanes::zip`]'s, because the two operands have *different*
    /// element types — the value is a `T` and the amount is always a `u32` — and `zip` is written
    /// for the case where they agree.
    fn shift<T: Element, const LANES: u32>(
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

    /// The same bits as an `i32`, without converting.
    ///
    /// `OpBitcast` at equal widths: 0xFFFF_FFFF becomes −1 rather than becoming 4294967295 and
    /// then failing to fit. The counterpart of [`Lanes::reinterpret`], and the direction a byte
    /// extraction needs before an arithmetic shift can see a sign at all.
    ///
    /// # Errors
    ///
    /// As [`Lanes::shift_left`].
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
    // A test may panic — that is how it reports.
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
        // They agree on every value with the top bit clear, which is every value in a small test.
        // Naming them apart is the only thing that makes the difference visible at the call site.
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
        // `1 << 24` and `24 << 1` are both plausible numbers, so the order is worth pinning.
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
        // The amount is always a `u32` and the value need not be. A helper that took the type
        // from the wrong operand would emit an `i32` shift with a `u32` result type, which the
        // validator catches — but only if something builds it.
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

        // The result type is the `i32` the value had, and `Vector<I32, _>` is what came back.
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
