use super::{Element, F32, I32, LaneError, Lanes, U32, Vector};
use crate::spec::PackedVectorFormat;

impl Lanes<'_> {
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

    pub fn dot_unsigned<const LANES: u32>(
        &mut self,
        left: Vector<U32, LANES>,
        right: Vector<U32, LANES>,
    ) -> Result<Vector<U32, LANES>, LaneError> {
        let result = self.type_of::<U32>()?;
        let mut ids = Vec::with_capacity(left.strip_count());

        for (&a, &b) in left.strips().iter().zip(right.strips()) {
            ids.push(
                self.module()
                    .u_dot(result, a, b, PackedVectorFormat::FourEightBit)?,
            );
        }

        self.from_strips(&ids)
    }

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

    pub fn to_u32<const LANES: u32>(
        &mut self,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<U32, LANES>, LaneError> {
        self.convert(crate::module::op::CONVERT_F_TO_U, value)
    }

    pub fn to_i32<const LANES: u32>(
        &mut self,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<I32, LANES>, LaneError> {
        self.convert(crate::module::op::CONVERT_F_TO_S, value)
    }

    fn convert<T: Element, const LANES: u32>(
        &mut self,
        opcode: u16,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let mut ids = Vec::with_capacity(value.strip_count());

        for &strip in value.strips() {
            ids.push(self.module().unary(opcode, element, strip)?);
        }

        self.from_strips(&ids)
    }
}

#[must_use]
pub const fn signed_bytes(packed: u32) -> [i32; 4] {
    [
        (packed as u8) as i8 as i32,
        ((packed >> 8) as u8) as i8 as i32,
        ((packed >> 16) as u8) as i8 as i32,
        ((packed >> 24) as u8) as i8 as i32,
    ]
}

#[must_use]
pub const fn unsigned_bytes(packed: u32) -> [i32; 4] {
    [
        (packed & 0xff) as i32,
        ((packed >> 8) & 0xff) as i32,
        ((packed >> 16) & 0xff) as i32,
        ((packed >> 24) & 0xff) as i32,
    ]
}

#[must_use]
pub const fn pack(bytes: [i32; 4]) -> u32 {
    (bytes[0] as u32 & 0xff)
        | ((bytes[1] as u32 & 0xff) << 8)
        | ((bytes[2] as u32 & 0xff) << 16)
        | ((bytes[3] as u32 & 0xff) << 24)
}

#[cfg(test)]
mod tests {
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
    fn the_unsigned_dot_declares_an_unsigned_result_because_the_instruction_demands_one() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let packed = lanes.splat_bits::<U32, 32>(0x0102_0304).expect("splat");

        let total = lanes.dot_unsigned(packed, packed).expect("dot");
        lanes.add(total, packed).expect("same type");

        let words = module.finish();
        assert_eq!(count(&words, op::U_DOT), 1);

        let integers: Vec<Vec<u32>> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::TYPE_INT)
            .map(|instruction| instruction.operands().to_vec())
            .collect();
        let unsigned = integers
            .iter()
            .find(|operands| operands[2] == 0)
            .expect("an unsigned 32-bit integer type");
        let result = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::U_DOT)
            .expect("emitted")
            .operands()[0];

        assert_eq!(result, unsigned[0], "OpUDot named a signed result type");
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
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let packed = lanes.splat_bits::<U32, 32>(0x0102_0304).expect("splat");

        let total = lanes.dot_signed(packed, packed).expect("dot");
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
        let packed = pack([-1, -128, 127, 0]);

        assert_eq!(signed_bytes(packed), [-1, -128, 127, 0]);
        assert_eq!(unsigned_bytes(packed), [255, 128, 127, 0]);
    }

    #[test]
    fn the_least_significant_byte_is_the_first_component() {
        assert_eq!(pack([1, 0, 0, 0]), 1);
        assert_eq!(pack([0, 1, 0, 0]), 0x100);
        assert_eq!(pack([0, 0, 0, 1]), 0x0100_0000);
    }
}
