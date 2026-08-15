//! Elementwise operations that core SPIR-V does not have.
//!
//! `min`, `max`, `abs`, `sqrt` and the rest are not opcodes. They live in the GLSL.std.450
//! extended instruction set, reached through `OpExtInst` — one instruction, exactly like an
//! `OpFAdd`, with the set's id and an instruction number in front of the operands. So everything
//! here costs one instruction per strip, the same as [`super::arithmetic`], and the lane count
//! still never reaches an instruction.
//!
//! # Why these and not the other seventy
//!
//! The set has transcendentals, packing helpers, matrix operations and geometry. What is exposed
//! here is what a lane program has been observed to want: the two extremes and the clamp between
//! them, magnitude, and the four float functions that a normalisation or an activation reaches
//! for. The rest is a larger surface with no caller, and [`crate::module::Module::ext_inst`] is
//! public for anyone who has one.
//!
//! # What this did not buy
//!
//! Speed, on the kernel that motivated it. `runner/examples/nnue.rs` timed a clamped and an
//! unclamped kernel at 6.50 µs and 6.47 µs, either side wobbling by more than the difference,
//! because that kernel waits on memory rather than on arithmetic. Four instructions per element
//! became one and nothing moved. This is an expressiveness change and should be read as one.

use super::element::Signed;
use super::{Element, F32, LaneError, Lanes, Vector};
use crate::module::Id;
use crate::spec::Glsl;

impl Lanes<'_> {
    /// The smaller of two elements — `Simd::simd_min`.
    ///
    /// Signed and unsigned integers reach different instructions, and for floats the choice with
    /// a NaN in it is left undefined by the specification. `runner/tests/floats.rs` records what
    /// this machine does with one.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn min<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip_extended(T::MIN, left, right)
    }

    /// The larger of two elements — `Simd::simd_max`.
    ///
    /// # Errors
    ///
    /// As [`Lanes::min`].
    pub fn max<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip_extended(T::MAX, left, right)
    }

    /// `value` held between `low` and `high` — `Simd::simd_clamp`.
    ///
    /// One instruction, against the two comparisons and two selects the core spelling costs. The
    /// bounds are vectors rather than scalars because a clamp whose bounds vary per lane is the
    /// general case and a splat is how the constant one is written.
    ///
    /// **`low` above `high` is undefined**, and deliberately not checked: the bounds are ids by
    /// the time they reach here, so there is nothing to compare without emitting instructions that
    /// would then be paid for on every call. GLSL.std.450 says the same thing about `FClamp`.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn clamp<T: Element, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
        low: Vector<T, LANES>,
        high: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip3_extended(T::CLAMP, value, low, high)
    }

    /// Magnitude — `Simd::abs`.
    ///
    /// Only for the types that have a sign: `abs` of a `u32` is the value, and the set has no
    /// `UAbs` to emit for it. See [`Signed`].
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn abs<T: Signed, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.map_extended(T::ABS, value)
    }

    /// Square root.
    ///
    /// Float-only, and concretely so: there is no integer square root in the set, and a generic
    /// signature would have to refuse two of the three element types at runtime.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn sqrt<const LANES: u32>(
        &mut self,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        self.map_extended(Glsl::Sqrt, value)
    }

    /// One over the square root, in one instruction rather than a divide after a root.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn inverse_sqrt<const LANES: u32>(
        &mut self,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        self.map_extended(Glsl::InverseSqrt, value)
    }

    /// e raised to each element.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn exp<const LANES: u32>(
        &mut self,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        self.map_extended(Glsl::Exp, value)
    }

    /// The natural logarithm of each element.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn log<const LANES: u32>(
        &mut self,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        self.map_extended(Glsl::Log, value)
    }

    /// `a * b + c`, rounded once.
    ///
    /// Not the same value as [`Lanes::mul`] followed by [`Lanes::add`], which rounds twice. It is
    /// usually the more accurate of the two and it is never bit-identical **in general**, so a
    /// kernel that must agree with a CPU reference exactly has to make the same choice on both
    /// sides.
    ///
    /// **That sentence used to end "which is why the fuzzer's vocabulary has `min`, `max` and
    /// `clamp` in it and not this", and it outlived its own scope.** The fuzzer's float corpus is
    /// small integers below 2²⁴ by construction — that is what lets any float comparison there be
    /// exact — and in that range a product and a sum are both exact, so the fused and unfused
    /// spellings give the *same bits*. `Op::FusedMulAdd` generates it now and holds the pair to
    /// agreeing, which is what `Op::RepeatAdd` and `Op::RolledAdd` are for one level down.
    ///
    /// The general claim stands and the conclusion drawn from it did not.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the instructions cannot be emitted.
    pub fn fma<const LANES: u32>(
        &mut self,
        a: Vector<F32, LANES>,
        b: Vector<F32, LANES>,
        c: Vector<F32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        self.zip3_extended(Glsl::Fma, a, b, c)
    }

    /// One extended instruction per strip, over one operand.
    fn map_extended<T: Element, const LANES: u32>(
        &mut self,
        instruction: Glsl,
        value: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let set = self.glsl()?;
        let mut ids = Vec::with_capacity(value.strip_count());

        for &strip in value.strips() {
            ids.push(
                self.module()
                    .ext_inst(element, set, instruction.word(), &[strip])?,
            );
        }

        self.from_strips(&ids)
    }

    /// One extended instruction per strip, over two operands.
    fn zip_extended<T: Element, const LANES: u32>(
        &mut self,
        instruction: Glsl,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let set = self.glsl()?;
        let mut ids = Vec::with_capacity(left.strip_count());

        for (&a, &b) in left.strips().iter().zip(right.strips()) {
            ids.push(
                self.module()
                    .ext_inst(element, set, instruction.word(), &[a, b])?,
            );
        }

        self.from_strips(&ids)
    }

    /// One extended instruction per strip, over three operands.
    ///
    /// The three-operand shape is its own function rather than a slice of vectors because a slice
    /// would need an index per strip, and an index needs a bound to check that the types have
    /// already made true. Three arities, three signatures, no unreachable branch in any of them.
    fn zip3_extended<T: Element, const LANES: u32>(
        &mut self,
        instruction: Glsl,
        first: Vector<T, LANES>,
        second: Vector<T, LANES>,
        third: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let set = self.glsl()?;
        let mut ids = Vec::with_capacity(first.strip_count());

        for ((&a, &b), &c) in first
            .strips()
            .iter()
            .zip(second.strips())
            .zip(third.strips())
        {
            ids.push(
                self.module()
                    .ext_inst(element, set, instruction.word(), &[a, b, c])?,
            );
        }

        self.from_strips(&ids)
    }

    /// The imported GLSL.std.450 set, importing it if this is the first call.
    ///
    /// Asked for per instruction rather than held on [`Lanes`]: the import is interned, so this is
    /// a hash lookup, and a module that ends up emitting no extended instruction ends up with no
    /// import either. A field would have imported the set for every kernel that built a `Lanes`.
    fn glsl(&mut self) -> Result<Id, LaneError> {
        Ok(self.module().ext_inst_import(Glsl::SET_NAME)?)
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{I32, U32};
    use crate::module::{Module, Version, op};

    fn built() -> Module {
        Module::new(Version::V1_3)
    }

    /// The instruction number of every `OpExtInst` in `words`, in order.
    fn extended_calls(words: &[u32]) -> Vec<u32> {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == op::EXT_INST)
            // Result type, result id, set, then the instruction number.
            .filter_map(|instruction| instruction.operands().get(3).copied())
            .collect()
    }

    #[test]
    fn a_min_within_the_subgroup_is_one_instruction_whatever_the_lane_count() {
        for emitted in [one_min::<4>(), one_min::<8>(), one_min::<32>()] {
            assert_eq!(emitted, vec![Glsl::FMin.word()]);
        }
    }

    /// The extended calls a single `min` of `LANES` float lanes emits.
    fn one_min<const LANES: u32>() -> Vec<u32> {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, LANES>(1.0_f32.to_bits())
            .expect("splat");

        lanes.min(value, value).expect("min");
        extended_calls(&module.finish())
    }

    #[test]
    fn a_strip_mined_max_is_one_instruction_per_strip_and_no_more() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 128>(1.0_f32.to_bits())
            .expect("splat");

        let largest = lanes.max(value, value).expect("max");

        assert_eq!(largest.strip_count(), 4);
        assert_eq!(extended_calls(&module.finish()), vec![Glsl::FMax.word(); 4]);
    }

    #[test]
    fn the_three_element_types_reach_three_different_instructions() {
        // The one mistake this layer can make that nothing downstream would catch: a `u32` maximum
        // emitted as `SMax` is right for every value below 2³¹ and wrong above it.
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let float = lanes.splat_bits::<F32, 32>(0).expect("f32");
        let signed = lanes.splat_bits::<I32, 32>(0).expect("i32");
        let unsigned = lanes.splat_bits::<U32, 32>(0).expect("u32");

        lanes.max(float, float).expect("float max");
        lanes.max(signed, signed).expect("signed max");
        lanes.max(unsigned, unsigned).expect("unsigned max");

        assert_eq!(
            extended_calls(&module.finish()),
            vec![Glsl::FMax.word(), Glsl::SMax.word(), Glsl::UMax.word()]
        );
    }

    #[test]
    fn a_clamp_is_one_instruction_and_not_four() {
        // The core spelling is two comparisons and two selects. This is the whole reason the set
        // is imported, so it is worth asserting the absence as well as the presence.
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<I32, 32>(7).expect("7");
        let low = lanes.splat_bits::<I32, 32>(0).expect("0");
        let high = lanes.splat_bits::<I32, 32>(3).expect("3");

        lanes.clamp(value, low, high).expect("clamped");

        let words = module.finish();
        assert_eq!(extended_calls(&words), vec![Glsl::SClamp.word()]);
        assert_eq!(
            decode::body(&words)
                .filter(|instruction| instruction.opcode() == op::SELECT)
                .count(),
            0
        );
    }

    #[test]
    fn a_clamp_names_its_value_then_its_bounds_in_that_order() {
        // `SClamp x minVal maxVal`. Transposing the last two gives an instruction that assembles,
        // validates, and returns the *low* bound for every input in range.
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<I32, 32>(7).expect("7");
        let low = lanes.splat_bits::<I32, 32>(1).expect("1");
        let high = lanes.splat_bits::<I32, 32>(3).expect("3");

        lanes.clamp(value, low, high).expect("clamped");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::EXT_INST)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(
            &operands[4..],
            &[value.id().word(), low.id().word(), high.id().word()]
        );
    }

    #[test]
    fn every_call_carries_the_arity_its_instruction_declares() {
        // The structural check that a helper of the wrong arity would fail: `Glsl::operands` says
        // how many an instruction takes, and the emitted call has to hold exactly that many after
        // the result type, result id, set and number.
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(2.0_f32.to_bits())
            .expect("splat");

        lanes.abs(value).expect("abs");
        lanes.sqrt(value).expect("sqrt");
        lanes.min(value, value).expect("min");
        lanes.max(value, value).expect("max");
        lanes.clamp(value, value, value).expect("clamp");
        lanes.fma(value, value, value).expect("fma");

        let words = module.finish();
        let arities: Vec<(u32, usize)> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::EXT_INST)
            .map(|instruction| {
                let operands = instruction.operands();
                (operands[3], operands.len() - 4)
            })
            .collect();

        assert_eq!(arities.len(), 6);
        for (number, emitted) in arities {
            let declared = [
                Glsl::FAbs,
                Glsl::Sqrt,
                Glsl::FMin,
                Glsl::FMax,
                Glsl::FClamp,
                Glsl::Fma,
            ]
            .into_iter()
            .find(|instruction| instruction.word() == number)
            .expect("a number this module emitted");

            assert_eq!(emitted, declared.operands(), "{declared:?}");
        }
    }

    #[test]
    fn the_set_is_imported_once_however_many_calls_reach_it() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 64>(1.0_f32.to_bits())
            .expect("splat");

        lanes.min(value, value).expect("min");
        lanes.max(value, value).expect("max");
        lanes.sqrt(value).expect("sqrt");

        let words = module.finish();
        assert_eq!(
            decode::body(&words)
                .filter(|instruction| instruction.opcode() == op::EXT_INST_IMPORT)
                .count(),
            1
        );
        assert_eq!(extended_calls(&words).len(), 6, "two strips, three calls");
    }

    #[test]
    fn a_module_that_calls_nothing_extended_imports_nothing() {
        // The reason the import is asked for per call rather than held on `Lanes`: every kernel
        // builds one of these, and most of them never reach the set.
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.add(value, value).expect("added");

        assert_eq!(
            decode::body(&module.finish())
                .filter(|instruction| instruction.opcode() == op::EXT_INST_IMPORT)
                .count(),
            0
        );
    }

    #[test]
    fn the_float_only_functions_are_four_different_instructions() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(4.0_f32.to_bits())
            .expect("splat");

        lanes.sqrt(value).expect("sqrt");
        lanes.inverse_sqrt(value).expect("inverse sqrt");
        lanes.exp(value).expect("exp");
        lanes.log(value).expect("log");

        assert_eq!(
            extended_calls(&module.finish()),
            vec![
                Glsl::Sqrt.word(),
                Glsl::InverseSqrt.word(),
                Glsl::Exp.word(),
                Glsl::Log.word()
            ]
        );
    }

    #[test]
    fn a_signed_integer_takes_its_magnitude_with_the_integer_instruction() {
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<I32, 32>(u32::from_ne_bytes((-7_i32).to_ne_bytes()))
            .expect("-7");

        lanes.abs(value).expect("abs");

        assert_eq!(extended_calls(&module.finish()), vec![Glsl::SAbs.word()]);
    }

    #[test]
    fn an_extended_result_is_an_ordinary_vector_that_keeps_composing() {
        // `OpExtInst` yields a value of the result type, so nothing downstream needs to know it
        // came from a set. This is the assertion that the strip bookkeeping survived the trip.
        let mut module = built();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 64>(3.0_f32.to_bits())
            .expect("splat");

        let bounded = lanes.min(value, value).expect("min");
        let summed = lanes.add(bounded, value).expect("added");
        let total = lanes.reduce_sum(summed).expect("reduced");

        assert_eq!(bounded.strip_count(), 2);
        assert_ne!(total, summed.id());
    }
}
