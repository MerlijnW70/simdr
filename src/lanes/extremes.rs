use super::{Element, LaneError, Lanes, Vector};
use crate::module::Id;

impl Lanes<'_> {
    pub fn reduce_max<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_extreme::<T, LANES>(T::GROUP_MAX, Extreme::Max, vector)
    }

    pub fn reduce_min<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_extreme::<T, LANES>(T::GROUP_MIN, Extreme::Min, vector)
    }

    fn reduce_extreme<T: Element, const LANES: u32>(
        &mut self,
        group: u16,
        extreme: Extreme,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        let element = self.type_of::<T>()?;
        let scope = self.scope();
        let reduction = self.reduction::<LANES>()?;

        let mut partial = vector
            .strips()
            .first()
            .copied()
            .ok_or_else(LaneError::no_strips)?;
        let boolean = self.module().type_bool()?;

        for &next in vector.strips().iter().skip(1) {
            let keeps_partial = self
                .module()
                .binary(T::GREATER_THAN, boolean, partial, next)?;
            let (when_true, when_false) = match extreme {
                Extreme::Max => (partial, next),
                Extreme::Min => (next, partial),
            };
            partial = self
                .module()
                .select(element, keeps_partial, when_true, when_false)?;
        }

        Ok(self
            .module()
            .subgroup_reduce(group, element, scope, reduction, partial)?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Extreme {
    Max,
    Min,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{F32, I32, U32};
    use crate::module::{Module, Version, op};

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    #[test]
    fn a_full_width_max_needs_no_local_fold_at_all() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.reduce_max(value).expect("max");

        let words = module.finish();
        assert_eq!(count(&words, op::SELECT), 0, "one strip, nothing to fold");
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_MAX), 1);
    }

    fn fold_arms(words: &[u32]) -> (u32, u32, u32) {
        let operands = decode::body(words)
            .find(|instruction| instruction.opcode() == op::SELECT)
            .expect("a strip fold selects")
            .operands()
            .to_vec();
        (operands[2], operands[3], operands[4])
    }

    fn comparison_operands(words: &[u32], opcode: u16) -> (u32, u32) {
        let operands = decode::body(words)
            .find(|instruction| instruction.opcode() == opcode)
            .expect("a strip fold compares")
            .operands()
            .to_vec();
        (operands[2], operands[3])
    }

    fn two_strips(minimum: bool) -> Vec<u32> {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");

        let low = U32::constant_from_bits(lanes.module(), 3).expect("3");
        let high = U32::constant_from_bits(lanes.module(), 9).expect("9");
        let value = lanes
            .from_strips::<U32, 64>(&[low, high])
            .expect("two strips");

        if minimum {
            lanes.reduce_min(value).expect("min");
        } else {
            lanes.reduce_max(value).expect("max");
        }
        module.finish()
    }

    #[test]
    fn a_minimum_folds_its_strips_to_the_smaller_and_not_the_larger() {
        let words = two_strips(true);
        let (condition, when_true, when_false) = fold_arms(&words);
        let (left, right) = comparison_operands(&words, op::U_GREATER_THAN);

        assert_eq!(left, when_false, "the larger operand is the one discarded");
        assert_eq!(right, when_true, "and the smaller is the one kept");
        assert_ne!(condition, 0);
    }

    #[test]
    fn a_maximum_folds_its_strips_the_other_way_round() {
        let words = two_strips(false);
        let (_, when_true, when_false) = fold_arms(&words);
        let (left, right) = comparison_operands(&words, op::U_GREATER_THAN);

        assert_eq!(left, when_true, "the larger operand is the one kept");
        assert_eq!(right, when_false);
    }

    #[test]
    fn the_two_extremes_do_not_emit_the_same_module() {
        let minimum = two_strips(true);
        let maximum = two_strips(false);

        assert_ne!(
            decode::body(&minimum)
                .map(|instruction| instruction.operands().to_vec())
                .collect::<Vec<_>>(),
            decode::body(&maximum)
                .map(|instruction| instruction.operands().to_vec())
                .collect::<Vec<_>>(),
            "a minimum and a maximum that fold identically are not both right"
        );
    }

    #[test]
    fn a_strip_mined_max_folds_with_a_comparison_and_a_select() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 64>(1.0_f32.to_bits())
            .expect("splat");

        lanes.reduce_max(value).expect("max");

        let words = module.finish();
        assert_eq!(count(&words, op::F_ORD_GREATER_THAN), 1);
        assert_eq!(count(&words, op::SELECT), 1);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_MAX), 1);
    }

    #[test]
    fn four_strips_cost_three_folds() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 128>(1.0_f32.to_bits())
            .expect("splat");

        lanes.reduce_min(value).expect("min");

        assert_eq!(count(&module.finish(), op::SELECT), 3);
    }

    #[test]
    fn signed_and_unsigned_maxima_are_different_instructions() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let signed = lanes.splat_bits::<I32, 32>(1).expect("splat");
        let unsigned = lanes.splat_bits::<U32, 32>(1).expect("splat");

        lanes.reduce_max(signed).expect("signed max");
        lanes.reduce_max(unsigned).expect("unsigned max");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_S_MAX), 1);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_U_MAX), 1);
    }

    #[test]
    fn max_and_min_swap_the_operands_rather_than_the_comparison() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 64>(1.0_f32.to_bits())
            .expect("splat");
        let other = lanes
            .splat_bits::<F32, 64>(2.0_f32.to_bits())
            .expect("splat");
        let mixed = lanes.add(value, other).expect("distinct strips");

        lanes.reduce_max(mixed).expect("max");
        let after_max = module.finish();

        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 64>(1.0_f32.to_bits())
            .expect("splat");
        let other = lanes
            .splat_bits::<F32, 64>(2.0_f32.to_bits())
            .expect("splat");
        let mixed = lanes.add(value, other).expect("distinct strips");

        lanes.reduce_min(mixed).expect("min");
        let after_min = module.finish();

        assert_ne!(
            after_max, after_min,
            "a maximum and a minimum must not emit the same module"
        );
    }
}
