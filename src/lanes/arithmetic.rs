//! Elementwise operations — the ones that cost nothing across lanes.
//!
//! Each lane already holds its own element, so `a + b` over a subgroup-wide vector is one scalar
//! instruction. A strip-mined vector costs one *per strip*, which is the same count a hand-written
//! loop would emit and no more.
//!
//! The lane count never reaches an instruction. It lives in the type, so `Vector<F32, 8>` and
//! `Vector<F32, 32>` do not combine, and neither do `Vector<F32, 8>` and `Vector<I32, 8>`.

use super::vector::Strips;
use super::{Element, LaneError, Lanes, Vector};
use crate::module::Id;

impl Lanes<'_> {
    /// `a + b`, elementwise.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn add<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip(T::ADD, left, right)
    }

    /// `a * b`, elementwise.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn mul<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip(T::MUL, left, right)
    }

    /// `a > b`, elementwise, yielding a boolean per element.
    ///
    /// For floats the comparison is *ordered*: a NaN on either side gives false. For integers the
    /// signed and unsigned forms are different instructions, which [`Element`] carries.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn greater_than<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Predicate<LANES>, LaneError> {
        self.compare(T::GREATER_THAN, left, right)
    }

    /// `a == b`, elementwise, yielding a boolean per element — `Simd::simd_eq`.
    ///
    /// **Ordered for floats, which is the whole of what the word "ordered" buys**: a NaN is equal
    /// to nothing, itself included, so `equal(x, x)` is a NaN test written backwards. The integers
    /// have no such case and both families share one instruction — see [`Element::EQUAL`], which
    /// is the only place in this trait where the signed and unsigned paths do.
    ///
    /// Added because the lane API had `greater_than` and no equality at all, which is the
    /// comparison a `Simd` layer is asked for first — and because a strip-mined
    /// [`Lanes::all_equal`] cannot be built without one: one vote per strip says each strip is
    /// uniform, and saying the strips agree with *each other* is exactly this.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn equal<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Predicate<LANES>, LaneError> {
        self.compare(T::EQUAL, left, right)
    }

    /// Pick `when_true` or `when_false` per element, according to `predicate`.
    ///
    /// A per-element select, not a branch: nothing diverges, so there is no reconvergence for the
    /// scheduler to arrange.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
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

    /// Any comparison, strip by strip: the same shape as [`Lanes::zip`] with a boolean result.
    ///
    /// One function rather than one per comparison, for the reason `Module::binary` takes an
    /// opcode: what differs between `>` and `==` is a number the [`Element`] holds, and writing the
    /// loop twice would be two places for the result type to stop being `bool`.
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

        // `ids` came from a vector's own strips, so it is already between one and `MAX_STRIPS`.
        Strips::new(&ids)
            .map(|strips| Predicate { strips })
            .ok_or(LaneError::TooManyStrips {
                strips: ids.len(),
                limit: super::MAX_STRIPS,
            })
    }

    /// Apply a two-operand instruction strip by strip.
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

/// A boolean per element — what a comparison yields, and what `Mask<T, N>` is.
///
/// Strip-shaped like a [`Vector`], because a comparison of a strip-mined vector produces one
/// boolean per element and not one per lane. It shares its strip storage with `Vector` rather
/// than keeping a second copy: the duplicate had a duplicate bounds check, only one of the two
/// was reachable, and the prober noticed before a reader would have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Predicate<const LANES: u32> {
    strips: Strips,
}

impl<const LANES: u32> Predicate<LANES> {
    /// The booleans this is made of, one per strip.
    #[must_use]
    pub fn strips(&self) -> &[Id] {
        self.strips.as_slice()
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{F32, I32, U32};
    use crate::module::{Module, Version, op};

    /// How many instructions carrying `opcode` a freshly built module ends up with.
    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    /// How many `OpFAdd` an add of `LANES` lanes emits on a 32-wide subgroup.
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
        // The claim `Element::EQUAL` makes, checked. `greater_than` is three instructions across
        // the three types because `OpSGreaterThan` and `OpUGreaterThan` disagree above 2³¹;
        // equality is two, because two bit patterns are equal or they are not.
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
        // One instruction per strip, the same as every other elementwise operation — and a
        // `Predicate` with a boolean per element rather than per lane.
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
        // The claim that makes this mapping worth having: four lanes, eight, and the full
        // thirty-two all emit the same single scalar add, because each lane already holds its own
        // element and the lane count never reaches an instruction.
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
}
