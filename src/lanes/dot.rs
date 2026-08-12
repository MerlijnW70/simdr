//! Four 8-bit products per lane, summed in one instruction.
//!
//! Every other elementwise operation here takes a `Vector<T, N>` and gives one back of the same
//! type. These do not: they read a `Vector<U32, N>` as four bytes per lane and give a
//! `Vector<I32, N>` of the four products' sum. The types differ because the *widths* do, and
//! saying so in the signature is the only thing that stops a caller adding the result back into
//! the operands.
//!
//! # Where the packing is
//!
//! In the instruction, not in the mapping. `decisions/DR-0004` says a narrow element is one
//! element per lane and that is unchanged: a `Vector<U32, 32>` here is thirty-two lanes each
//! holding one `u32`, exactly as everywhere else. `OpSDot` is an operation that reads each of
//! those `u32`s as four bytes — which is a fact about the operands, not about the vector.
//!
//! A caller with a buffer of `i8` therefore reads it as `U32` and gets four elements per lane for
//! free, and a caller who wants `i8` arithmetic reads it as [`crate::lanes::I8`] and gets one. The
//! two are different kernels over the same bytes, and both are available.
//!
//! # What the device has to offer
//!
//! `VK_KHR_shader_integer_dot_product`, and `shaderIntegerDotProduct` enabled. Whether the packed
//! form is *accelerated* is a separate property a device reports and neither the module nor the
//! validator can see — an implementation may support the instruction and lower it to the four
//! multiplies it replaces. `simdr probe` reports what this machine says.

use super::{F32, I32, LaneError, Lanes, U32, Vector};
use crate::spec::PackedVectorFormat;

impl Lanes<'_> {
    /// Four signed 8-bit products per lane, summed into an `i32`.
    ///
    /// Each lane's `u32` holds four `i8`, least significant byte first. The sum wraps; use
    /// [`Lanes::dot_signed_saturating`] for the version that clamps.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn dot_signed<const LANES: u32>(
        &mut self,
        left: Vector<U32, LANES>,
        right: Vector<U32, LANES>,
    ) -> Result<Vector<I32, LANES>, LaneError> {
        let result = self.type_of::<I32>()?;
        let mut ids = Vec::with_capacity(left.strip_count());

        for (&a, &b) in left.strips().iter().zip(right.strips()) {
            ids.push(
                self.module()
                    .s_dot(result, a, b, PackedVectorFormat::FourEightBit)?,
            );
        }

        self.from_strips(&ids)
    }

    /// The same over unsigned bytes.
    ///
    /// # Errors
    ///
    /// As [`Lanes::dot_signed`].
    pub fn dot_unsigned<const LANES: u32>(
        &mut self,
        left: Vector<U32, LANES>,
        right: Vector<U32, LANES>,
    ) -> Result<Vector<I32, LANES>, LaneError> {
        let result = self.type_of::<I32>()?;
        let mut ids = Vec::with_capacity(left.strip_count());

        for (&a, &b) in left.strips().iter().zip(right.strips()) {
            ids.push(
                self.module()
                    .u_dot(result, a, b, PackedVectorFormat::FourEightBit)?,
            );
        }

        self.from_strips(&ids)
    }

    /// Signed bytes against unsigned ones — a quantised layer's usual pairing.
    ///
    /// **The two arguments are not interchangeable.** `signed` is read as four `i8` and `unsigned`
    /// as four `u8`; swapping them computes a different number from the same bits.
    ///
    /// # Errors
    ///
    /// As [`Lanes::dot_signed`].
    pub fn dot_mixed<const LANES: u32>(
        &mut self,
        signed: Vector<U32, LANES>,
        unsigned: Vector<U32, LANES>,
    ) -> Result<Vector<I32, LANES>, LaneError> {
        let result = self.type_of::<I32>()?;
        let mut ids = Vec::with_capacity(signed.strip_count());

        for (&a, &b) in signed.strips().iter().zip(unsigned.strips()) {
            ids.push(
                self.module()
                    .su_dot(result, a, b, PackedVectorFormat::FourEightBit)?,
            );
        }

        self.from_strips(&ids)
    }

    /// [`Lanes::dot_signed`] added into a running total, clamped rather than wrapped.
    ///
    /// The accumulator is a `Vector<I32, LANES>` because that is what the previous call returned,
    /// so a chain of these reads as a chain. Saturation makes it a different arithmetic from
    /// adding the results yourself — deliberately, and it is the one a long quantised sum wants.
    ///
    /// # Errors
    ///
    /// As [`Lanes::dot_signed`].
    pub fn dot_signed_saturating<const LANES: u32>(
        &mut self,
        left: Vector<U32, LANES>,
        right: Vector<U32, LANES>,
        accumulator: Vector<I32, LANES>,
    ) -> Result<Vector<I32, LANES>, LaneError> {
        let result = self.type_of::<I32>()?;
        let mut ids = Vec::with_capacity(left.strip_count());

        for ((&a, &b), &carried) in left
            .strips()
            .iter()
            .zip(right.strips())
            .zip(accumulator.strips())
        {
            ids.push(self.module().s_dot_acc_sat(
                result,
                a,
                b,
                carried,
                PackedVectorFormat::FourEightBit,
            )?);
        }

        self.from_strips(&ids)
    }

    /// Reinterpret a vector's bits as another element type, without converting.
    ///
    /// What a dot product's caller needs: a buffer of packed bytes is loaded as `U32` and its
    /// *result* is `I32`, and the two have to meet somewhere. `OpBitcast` at equal widths is the
    /// instruction that says "the same bits, read differently" — as opposed to `convert_u32`,
    /// which says "the same number".
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn reinterpret<const LANES: u32>(
        &mut self,
        value: Vector<I32, LANES>,
    ) -> Result<Vector<U32, LANES>, LaneError> {
        let unsigned = self.type_of::<U32>()?;
        let mut ids = Vec::with_capacity(value.strip_count());

        for &strip in value.strips() {
            ids.push(
                self.module()
                    .unary(crate::module::op::BITCAST, unsigned, strip)?,
            );
        }

        self.from_strips(&ids)
    }

    /// The `f32` a lane's `i32` denotes, as a number rather than as bits.
    ///
    /// A dot product produces integers and a network's next layer usually wants floats. Separate
    /// from [`Lanes::reinterpret`] for the reason [`Lanes::convert_u32`] is separate from a
    /// bitcast: reading 7 as a float gives a denormal near zero, and converting gives 7.0.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn to_f32<const LANES: u32>(
        &mut self,
        value: Vector<I32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        let float = self.type_of::<F32>()?;
        let mut ids = Vec::with_capacity(value.strip_count());

        for &strip in value.strips() {
            ids.push(
                self.module()
                    .unary(crate::module::op::CONVERT_S_TO_F, float, strip)?,
            );
        }

        self.from_strips(&ids)
    }
}

/// The four signed bytes of a packed word, as a CPU reference would read them.
///
/// Not used by the emitter — it is here so that a caller building test data and a caller reading
/// results agree about which byte is which, rather than each writing the shift by hand.
#[must_use]
pub const fn signed_bytes(packed: u32) -> [i32; 4] {
    [
        (packed as u8) as i8 as i32,
        ((packed >> 8) as u8) as i8 as i32,
        ((packed >> 16) as u8) as i8 as i32,
        ((packed >> 24) as u8) as i8 as i32,
    ]
}

/// The four unsigned bytes of a packed word.
#[must_use]
pub const fn unsigned_bytes(packed: u32) -> [i32; 4] {
    [
        (packed & 0xff) as i32,
        ((packed >> 8) & 0xff) as i32,
        ((packed >> 16) & 0xff) as i32,
        ((packed >> 24) & 0xff) as i32,
    ]
}

/// Four bytes packed into a word, least significant first.
#[must_use]
pub const fn pack(bytes: [i32; 4]) -> u32 {
    (bytes[0] as u32 & 0xff)
        | ((bytes[1] as u32 & 0xff) << 8)
        | ((bytes[2] as u32 & 0xff) << 16)
        | ((bytes[3] as u32 & 0xff) << 24)
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::module::{Module, Version, op};

    fn built() -> Module {
        Module::new(Version::V1_3)
    }

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    #[test]
    fn a_dot_over_a_whole_subgroup_is_one_instruction() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 32>(0x0102_0304).expect("splat");

        lanes.dot_signed(value, value).expect("dot");

        assert_eq!(count(&module.finish(), op::S_DOT), 1);
    }

    #[test]
    fn a_strip_mined_dot_is_one_instruction_per_strip() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 128>(0x0102_0304).expect("splat");

        let total = lanes.dot_signed(value, value).expect("dot");

        assert_eq!(total.strip_count(), 4);
        assert_eq!(count(&module.finish(), op::S_DOT), 4);
    }

    #[test]
    fn the_result_is_a_signed_vector_and_not_the_operands_type() {
        // The signature is the whole safety net here: four bytes multiplied and summed do not fit
        // in a byte, and a result typed like its operands would invite a caller to add it back in.
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let packed = lanes.splat_bits::<U32, 32>(0x0102_0304).expect("splat");

        let total = lanes.dot_signed(packed, packed).expect("dot");
        // Compiles only because `total` is `Vector<I32, 32>`: adding it to `packed` would not.
        let doubled = lanes.add(total, total).expect("added");

        assert_eq!(doubled.strip_count(), 1);
    }

    #[test]
    fn the_three_sign_combinations_reach_three_instructions() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 32>(0x0102_0304).expect("splat");

        lanes.dot_signed(value, value).expect("signed");
        lanes.dot_unsigned(value, value).expect("unsigned");
        lanes.dot_mixed(value, value).expect("mixed");

        let words = module.finish();
        assert_eq!(count(&words, op::S_DOT), 1);
        assert_eq!(count(&words, op::U_DOT), 1);
        assert_eq!(count(&words, op::SU_DOT), 1);
    }

    #[test]
    fn the_saturating_form_carries_its_accumulator_through() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 32>(0x0102_0304).expect("splat");
        let zero = lanes.splat_bits::<I32, 32>(0).expect("zero");

        let first = lanes
            .dot_signed_saturating(value, value, zero)
            .expect("first");
        lanes
            .dot_signed_saturating(value, value, first)
            .expect("second");

        assert_eq!(count(&module.finish(), op::S_DOT_ACC_SAT), 2);
    }

    #[test]
    fn a_reinterpretation_is_a_bitcast_and_a_conversion_is_not() {
        // The same distinction `convert_u32` makes, in the other direction. Reading an `i32` of 7
        // as a float gives a denormal; converting gives 7.0.
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<I32, 32>(7).expect("seven");

        lanes.reinterpret(value).expect("bitcast");
        lanes.to_f32(value).expect("converted");

        let words = module.finish();
        assert_eq!(count(&words, op::BITCAST), 1);
        assert_eq!(count(&words, op::CONVERT_S_TO_F), 1);
    }

    #[test]
    fn packing_and_unpacking_are_each_others_opposite() {
        for bytes in [[0, 0, 0, 0], [1, 2, 3, 4], [-1, -128, 127, 0], [-1; 4]] {
            assert_eq!(signed_bytes(pack(bytes)), bytes, "{bytes:?}");
        }
    }

    #[test]
    fn the_unsigned_reading_differs_from_the_signed_one_above_127() {
        // The reason `SDot` and `UDot` are two instructions, stated on the host so a test can use
        // either reference without deriving the shifts again.
        let packed = pack([-1, -128, 127, 0]);

        assert_eq!(signed_bytes(packed), [-1, -128, 127, 0]);
        assert_eq!(unsigned_bytes(packed), [255, 128, 127, 0]);
    }

    #[test]
    fn the_least_significant_byte_is_the_first_component() {
        // Which end is component zero is not something a caller can guess, and getting it wrong
        // reverses every vector while still producing a plausible dot product.
        assert_eq!(pack([1, 0, 0, 0]), 1);
        assert_eq!(pack([0, 1, 0, 0]), 0x100);
        assert_eq!(pack([0, 0, 0, 1]), 0x0100_0000);
    }
}
