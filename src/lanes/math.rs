use super::element::Signed;
use super::{Element, F32, LaneError, Lanes, Vector};
use crate::module::Id;
use crate::spec::Glsl;

impl Lanes<'_> {
    pub fn min<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip_extended(T::MIN, left, right)
    }

    pub fn max<T: Element, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip_extended(T::MAX, left, right)
    }

    pub fn clamp<T: Element, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
        low: Vector<T, LANES>,
        high: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.zip3_extended(T::CLAMP, value, low, high)
    }

    pub fn abs<T: Signed, const LANES: u32>(
        &mut self,
        value: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.map_extended(T::ABS, value)
    }

    pub fn sqrt<const LANES: u32>(
        &mut self,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        self.map_extended(Glsl::Sqrt, value)
    }

    pub fn inverse_sqrt<const LANES: u32>(
        &mut self,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        self.map_extended(Glsl::InverseSqrt, value)
    }

    pub fn exp<const LANES: u32>(
        &mut self,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        self.map_extended(Glsl::Exp, value)
    }

    pub fn log<const LANES: u32>(
        &mut self,
        value: Vector<F32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        self.map_extended(Glsl::Log, value)
    }

    pub fn fma<const LANES: u32>(
        &mut self,
        a: Vector<F32, LANES>,
        b: Vector<F32, LANES>,
        c: Vector<F32, LANES>,
    ) -> Result<Vector<F32, LANES>, LaneError> {
        self.zip3_extended(Glsl::Fma, a, b, c)
    }

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

    fn glsl(&mut self) -> Result<Id, LaneError> {
        Ok(self.module().ext_inst_import(Glsl::SET_NAME)?)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{I32, U32};
    use crate::module::{Module, Version, op};

    fn built() -> Module {
        Module::new(Version::V1_3)
    }

    fn extended_calls(words: &[u32]) -> Vec<u32> {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == op::EXT_INST)
            .filter_map(|instruction| instruction.operands().get(3).copied())
            .collect()
    }

    #[test]
    fn a_min_within_the_subgroup_is_one_instruction_whatever_the_lane_count() {
        for emitted in [one_min::<4>(), one_min::<8>(), one_min::<32>()] {
            assert_eq!(emitted, vec![Glsl::FMin.word()]);
        }
    }

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
