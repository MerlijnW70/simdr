use super::vector::Strips;
use super::{Element, Integer, LaneError, Lanes, Signed, Vector};
use crate::module::{Id, op};

impl Lanes<'_> {
    pub fn add<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip(T::ADD, left, right)
    }

    pub fn mul<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip(T::MUL, left, right)
    }

    pub fn sub<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip(T::SUB, left, right)
    }

    /// Integer division truncates and, on a zero divisor, is undefined rather
    /// than a trap: SPIR-V leaves the result unspecified and the device is free
    /// to return anything at all. Guard the divisor if a lane can hold zero.
    pub fn div<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip(T::DIV, left, right)
    }

    pub fn neg<T: Signed, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let mut ids = Vec::with_capacity(value.strip_count());

        for &strip in value.strips() {
            ids.push(self.module().unary(T::NEGATE, element, strip)?);
        }

        self.from_strips(&ids)
    }

    pub fn greater_than<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Predicate<LANES>, LaneError> {
        self.compare(T::GREATER_THAN, left, right)
    }

    pub fn equal<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Predicate<LANES>, LaneError> {
        self.compare(T::EQUAL, left, right)
    }

    pub fn greater_equal<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Predicate<LANES>, LaneError> {
        self.compare(T::GREATER_THAN_EQUAL, left, right)
    }

    pub fn less_than<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Predicate<LANES>, LaneError> {
        self.compare(T::LESS_THAN, left, right)
    }

    pub fn less_equal<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Predicate<LANES>, LaneError> {
        self.compare(T::LESS_THAN_EQUAL, left, right)
    }

    /// Ordered, like every other comparison here, so this is not the negation
    /// of [`Lanes::equal`]: a NaN operand answers false to both.
    pub fn not_equal<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Predicate<LANES>, LaneError> {
        self.compare(T::NOT_EQUAL, left, right)
    }

    pub fn xor<T: Integer, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip(op::BITWISE_XOR, left, right)
    }

    pub fn select<T: Element, const LANES: u32>(
        &mut self,
        predicate: Predicate<LANES>,
        when_true: Vector<T, LANES>,
        when_false: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let mut ids = Vec::with_capacity(when_true.strip_count());

        for ((&condition, &yes), &no) in predicate
            .strips()
            .iter()
            .zip(when_true.strips())
            .zip(when_false.strips())
        {
            ids.push(self.module().select(element, condition, yes, no)?);
        }

        self.from_strips(&ids)
    }

    fn compare<T: Element, const LANES: u32>(
        &mut self,
        opcode: u16,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Predicate<LANES>, LaneError> {
        let boolean = self.module().type_bool()?;
        let mut ids = Vec::with_capacity(left.strip_count());

        for (&a, &b) in left.strips().iter().zip(right.strips()) {
            ids.push(self.module().binary(opcode, boolean, a, b)?);
        }

        Strips::new(&ids)
            .map(|strips| Predicate { strips })
            .ok_or(LaneError::TooManyStrips {
                strips: ids.len(),
                limit: super::MAX_STRIPS,
            })
    }

    fn zip<T: Element, const LANES: u32>(
        &mut self,
        opcode: u16,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let mut ids = Vec::with_capacity(left.strip_count());

        for (&a, &b) in left.strips().iter().zip(right.strips()) {
            ids.push(self.module().binary(opcode, element, a, b)?);
        }

        self.from_strips(&ids)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Predicate<const LANES: u32> {
    strips: Strips,
}

impl<const LANES: u32> Predicate<LANES> {
    #[must_use]
    pub fn strips(&self) -> &[Id] {
        self.strips.as_slice()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{F32, I32, U32};
    use crate::module::{Module, Version, op};

    type Build = fn(&mut Lanes<'_>);

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    fn adds_emitted<const LANES: u32>() -> usize {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, LANES>(1.0_f32.to_bits())
            .expect("splat");
        lanes.add(value, value).expect("added");

        count(&module.finish(), op::F_ADD)
    }

    #[test]
    fn equality_is_one_instruction_for_both_integer_families_and_its_own_for_floats() {
        let emitted = |compare: fn(&mut Lanes<'_>) -> ()| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            compare(&mut lanes);
            let words = module.finish();
            (
                count(&words, op::I_EQUAL),
                count(&words, op::F_ORD_EQUAL),
                count(&words, op::S_GREATER_THAN),
                count(&words, op::U_GREATER_THAN),
            )
        };

        let signed = emitted(|lanes| {
            let value = lanes.splat_bits::<I32, 32>(7).expect("splat");
            lanes.equal(value, value).expect("compared");
        });
        let unsigned = emitted(|lanes| {
            let value = lanes.splat_bits::<U32, 32>(7).expect("splat");
            lanes.equal(value, value).expect("compared");
        });
        let float = emitted(|lanes| {
            let value = lanes
                .splat_bits::<F32, 32>(1.0_f32.to_bits())
                .expect("splat");
            lanes.equal(value, value).expect("compared");
        });

        assert_eq!(signed, (1, 0, 0, 0), "the signed integers use OpIEqual");
        assert_eq!(unsigned, (1, 0, 0, 0), "and so do the unsigned ones");
        assert_eq!(float, (0, 1, 0, 0), "the floats keep an ordered comparison");
    }

    #[test]
    fn a_strip_mined_equality_compares_every_strip() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let wide = lanes
            .splat_bits::<F32, 128>(1.0_f32.to_bits())
            .expect("splat");

        let same = lanes.equal(wide, wide).expect("compared");

        assert_eq!(same.strips().len(), 4);
        assert_eq!(count(&module.finish(), op::F_ORD_EQUAL), 4);
    }

    #[test]
    fn an_add_within_the_subgroup_is_one_instruction_whatever_the_lane_count() {
        assert_eq!(adds_emitted::<4>(), 1);
        assert_eq!(adds_emitted::<8>(), 1);
        assert_eq!(adds_emitted::<32>(), 1);
    }

    #[test]
    fn a_strip_mined_add_is_one_instruction_per_strip_and_no_more() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 128>(1.0_f32.to_bits())
            .expect("splat");

        let sum = lanes.add(value, value).expect("added");

        assert_eq!(sum.strip_count(), 4);
        assert_eq!(
            count(&module.finish(), op::F_ADD),
            4,
            "four elements per lane, four adds — what a hand-written loop would emit"
        );
    }

    #[test]
    fn an_add_names_the_element_type_and_both_operands() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let float = lanes.type_of::<F32>().expect("f32");
        let two = lanes.splat_bits::<F32, 32>(2.0_f32.to_bits()).expect("two");
        let three = lanes
            .splat_bits::<F32, 32>(3.0_f32.to_bits())
            .expect("three");

        let sum = lanes.add(two, three).expect("added");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::F_ADD)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(
            operands,
            vec![
                float.word(),
                sum.id().word(),
                two.id().word(),
                three.id().word()
            ]
        );
    }

    #[test]
    fn integers_add_with_their_own_opcode() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<I32, 32>(7).expect("seven");

        lanes.add(value, value).expect("added");

        let words = module.finish();
        assert_eq!(count(&words, op::I_ADD), 1);
        assert_eq!(count(&words, op::F_ADD), 0, "no float instruction anywhere");
    }

    #[test]
    fn signed_and_unsigned_compare_with_different_instructions() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let signed = lanes.splat_bits::<I32, 32>(1).expect("1i32");
        let unsigned = lanes.splat_bits::<U32, 32>(1).expect("1u32");

        lanes.greater_than(signed, signed).expect("signed");
        lanes.greater_than(unsigned, unsigned).expect("unsigned");

        let words = module.finish();
        assert_eq!(count(&words, op::S_GREATER_THAN), 1);
        assert_eq!(count(&words, op::U_GREATER_THAN), 1);
    }

    #[test]
    fn a_comparison_yields_bools_rather_than_elements() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let zero = lanes
            .splat_bits::<F32, 32>(0.0_f32.to_bits())
            .expect("zero");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        lanes.greater_than(one, zero).expect("compared");

        let words = module.finish();
        assert_eq!(count(&words, op::TYPE_BOOL), 1);
        assert_eq!(count(&words, op::F_ORD_GREATER_THAN), 1);
    }

    #[test]
    fn a_select_is_a_per_element_pick_and_not_a_branch() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let zero = lanes
            .splat_bits::<F32, 32>(0.0_f32.to_bits())
            .expect("zero");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");
        let positive = lanes.greater_than(one, zero).expect("compared");

        lanes.select(positive, one, zero).expect("selected");

        let words = module.finish();
        assert_eq!(count(&words, op::SELECT), 1);
        assert_eq!(count(&words, op::LABEL), 0, "nothing diverged");
    }

    #[test]
    fn a_strip_mined_comparison_and_select_keep_their_strips_aligned() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let zero = lanes
            .splat_bits::<F32, 64>(0.0_f32.to_bits())
            .expect("zero");
        let one = lanes.splat_bits::<F32, 64>(1.0_f32.to_bits()).expect("one");

        let positive = lanes.greater_than(one, zero).expect("compared");
        let picked = lanes.select(positive, one, zero).expect("selected");

        assert_eq!(positive.strips().len(), 2);
        assert_eq!(picked.strip_count(), 2);

        let words = module.finish();
        assert_eq!(count(&words, op::F_ORD_GREATER_THAN), 2);
        assert_eq!(count(&words, op::SELECT), 2);
    }

    #[test]
    fn each_new_binary_operation_emits_the_instruction_of_its_own_family() {
        let emitted = |build: Build| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            build(&mut lanes);
            module.finish()
        };

        let cases: [(&str, u16, Build); 9] = [
            ("f32 sub", op::F_SUB, |lanes| {
                let v = lanes.splat_bits::<F32, 32>(0).expect("splat");
                lanes.sub(v, v).expect("sub");
            }),
            ("i32 sub", op::I_SUB, |lanes| {
                let v = lanes.splat_bits::<I32, 32>(0).expect("splat");
                lanes.sub(v, v).expect("sub");
            }),
            ("u32 sub", op::I_SUB, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(0).expect("splat");
                lanes.sub(v, v).expect("sub");
            }),
            ("f32 div", op::F_DIV, |lanes| {
                let v = lanes.splat_bits::<F32, 32>(0).expect("splat");
                lanes.div(v, v).expect("div");
            }),
            ("i32 div", op::S_DIV, |lanes| {
                let v = lanes.splat_bits::<I32, 32>(0).expect("splat");
                lanes.div(v, v).expect("div");
            }),
            ("u32 div", op::U_DIV, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(0).expect("splat");
                lanes.div(v, v).expect("div");
            }),
            ("f32 neg", op::F_NEGATE, |lanes| {
                let v = lanes.splat_bits::<F32, 32>(0).expect("splat");
                lanes.neg(v).expect("neg");
            }),
            ("i32 neg", op::S_NEGATE, |lanes| {
                let v = lanes.splat_bits::<I32, 32>(0).expect("splat");
                lanes.neg(v).expect("neg");
            }),
            ("i32 not_equal", op::I_NOT_EQUAL, |lanes| {
                let v = lanes.splat_bits::<I32, 32>(0).expect("splat");
                lanes.not_equal(v, v).expect("compared");
            }),
        ];

        for (name, expected, build) in cases {
            assert_eq!(count(&emitted(build), expected), 1, "{name}");
        }
    }

    #[test]
    fn the_three_families_reach_three_different_divisions() {
        assert_ne!(F32::DIV, I32::DIV);
        assert_ne!(I32::DIV, U32::DIV, "signedness picks the division too");
        assert_ne!(F32::DIV, U32::DIV);
    }

    #[test]
    fn signed_and_unsigned_order_themselves_with_different_instructions() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let signed = lanes.splat_bits::<I32, 32>(1).expect("1i32");
        let unsigned = lanes.splat_bits::<U32, 32>(1).expect("1u32");

        lanes.less_than(signed, signed).expect("signed <");
        lanes.less_than(unsigned, unsigned).expect("unsigned <");
        lanes.less_equal(signed, signed).expect("signed <=");
        lanes.less_equal(unsigned, unsigned).expect("unsigned <=");
        lanes.greater_equal(signed, signed).expect("signed >=");
        lanes
            .greater_equal(unsigned, unsigned)
            .expect("unsigned >=");

        let words = module.finish();
        for opcode in [
            op::S_LESS_THAN,
            op::U_LESS_THAN,
            op::S_LESS_THAN_EQUAL,
            op::U_LESS_THAN_EQUAL,
            op::S_GREATER_THAN_EQUAL,
            op::U_GREATER_THAN_EQUAL,
        ] {
            assert_eq!(count(&words, opcode), 1, "opcode {opcode} was not emitted");
        }
    }

    #[test]
    fn a_subtraction_does_not_cross_its_operands() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let two = lanes.splat_bits::<F32, 32>(2.0_f32.to_bits()).expect("two");
        let three = lanes
            .splat_bits::<F32, 32>(3.0_f32.to_bits())
            .expect("three");

        let difference = lanes.sub(two, three).expect("subtracted");

        let words = module.finish();
        let float = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::F_SUB)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(
            float[2],
            two.id().word(),
            "the left operand is the one it was handed first"
        );
        assert_eq!(float[3], three.id().word());
        assert_eq!(float[1], difference.id().word());
    }

    #[test]
    fn the_comparisons_are_the_ordered_family_so_not_equal_is_not_a_negated_equal() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.not_equal(value, value).expect("compared");
        lanes.less_than(value, value).expect("compared");
        lanes.less_equal(value, value).expect("compared");
        lanes.greater_equal(value, value).expect("compared");

        let words = module.finish();
        for ordered in [
            op::F_ORD_NOT_EQUAL,
            op::F_ORD_LESS_THAN,
            op::F_ORD_LESS_THAN_EQUAL,
            op::F_ORD_GREATER_THAN_EQUAL,
        ] {
            assert_eq!(count(&words, ordered), 1);
        }
        assert_eq!(
            count(&words, op::F_ORD_EQUAL),
            0,
            "not_equal is its own instruction and not an equality anyone negates"
        );
    }

    #[test]
    fn a_strip_mined_negation_is_one_instruction_per_strip() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let wide = lanes
            .splat_bits::<F32, 128>(1.0_f32.to_bits())
            .expect("splat");

        let negated = lanes.neg(wide).expect("negated");

        assert_eq!(negated.strip_count(), 4);
        assert_eq!(count(&module.finish(), op::F_NEGATE), 4);
    }

    #[test]
    fn a_negation_names_the_element_type_and_its_one_operand() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let float = lanes.type_of::<F32>().expect("f32");
        let value = lanes.splat_bits::<F32, 32>(2.0_f32.to_bits()).expect("two");

        let negated = lanes.neg(value).expect("negated");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::F_NEGATE)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(
            operands,
            vec![float.word(), negated.id().word(), value.id().word()],
            "a unary instruction names one operand and no more"
        );
    }
}
