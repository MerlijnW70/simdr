//! Maximum and minimum across a group.
//!
//! Separated from the other reductions because their strip fold is a different shape. **There is
//! no core scalar max opcode.** `OpFMax` and friends live in the GLSL.std.450 extended
//! instruction set; a comparison and a select say the same thing in core SPIR-V, at two
//! instructions instead of one.
//!
//! They also agree with the group instruction on NaN, which is the part that could have gone
//! wrong. The comparison is *ordered*, so `max(NaN, x)` selects `x` — and that is what
//! `OpGroupNonUniformFMax` gives on the hardware this has been run against. See
//! `runner/tests/floats.rs`, which observes it rather than asserting it: the specification
//! declines to pin the case.
//!
//! # Why the fold did not become an `FMax` when the set was imported
//!
//! [`Lanes::max`] emits one now, so the fold *could* — one instruction per extra strip instead of
//! two, on a fold that `notes/NEXT.md` measured as buying no time at all.
//!
//! What it would cost is the paragraph above. Compare-and-select is **defined** for NaN: an
//! ordered comparison against one is false, so the fold keeps the other operand by a rule written
//! down in the specification. `FMax` with a NaN is explicitly **undefined** — GLSL.std.450 says
//! which operand comes back "is undefined if one of the operands is a NaN". This machine returns
//! the non-NaN operand either way round (`runner/tests/extended.rs` observes it), so the two agree
//! *here*, and agreeing on one device is not the same claim as being defined.
//!
//! Trading a defined behaviour for an undefined one, to save an instruction that was measured not
//! to matter, is the wrong side of that trade.

use super::{Element, LaneError, Lanes, Vector};
use crate::module::Id;

impl Lanes<'_> {
    /// The largest element, delivered to every lane — `Simd::reduce_max`.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoMapping`] if `LANES` cannot sit on this subgroup, [`LaneError::Build`] if an
    /// instruction cannot be emitted.
    pub fn reduce_max<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_extreme::<T, LANES>(T::GROUP_MAX, Extreme::Max, vector)
    }

    /// The smallest element, delivered to every lane — `Simd::reduce_min`.
    ///
    /// # Errors
    ///
    /// As [`Lanes::reduce_max`].
    pub fn reduce_min<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_extreme::<T, LANES>(T::GROUP_MIN, Extreme::Min, vector)
    }

    /// Fold the strips with compare-and-select, then reduce across the group.
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
            // One comparison, always the same way round, and the extreme decides which arm the
            // select keeps. Swapping the *operands* instead — as this did until 2026-08-11 —
            // cancels against swapping which arm is taken, so both ends folded to the maximum.
            // `reduce_min` was therefore wrong for exactly the strip-mined case and right
            // everywhere else, because with one strip this loop does not run at all.
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

/// Which end of the range a fold is looking for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Extreme {
    Max,
    Min,
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
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

    /// The operands of the sole `OpSelect` in a strip fold, as `(condition, when_true, when_false)`.
    fn fold_arms(words: &[u32]) -> (u32, u32, u32) {
        let operands = decode::body(words)
            .find(|instruction| instruction.opcode() == op::SELECT)
            .expect("a strip fold selects")
            .operands()
            .to_vec();
        // result type, result id, condition, true, false.
        (operands[2], operands[3], operands[4])
    }

    /// The `(left, right)` of the sole comparison.
    fn comparison_operands(words: &[u32], opcode: u16) -> (u32, u32) {
        let operands = decode::body(words)
            .find(|instruction| instruction.opcode() == opcode)
            .expect("a strip fold compares")
            .operands()
            .to_vec();
        (operands[2], operands[3])
    }

    /// A two-strip module whose strips are two *different* constants.
    ///
    /// The distinctness is the point. Every earlier test here splatted one value across both
    /// strips, so a fold that returned the wrong end returned the same number anyway — which is
    /// how `reduce_min` folded strips with a maximum for weeks under a green suite.
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
        // The bug the fuzzer found. `partial > next` is the same comparison for both extremes;
        // what differs is which arm the select keeps, and getting *that* backwards is invisible
        // until two strips hold different values.
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
        // The cheapest possible statement of the bug: they differed only in operand order before,
        // which cancelled, and the two modules were identical apart from ids.
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
        // Max has no core scalar opcode, so the local fold is two instructions per extra strip.
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
        // Both use `GREATER_THAN`; what differs is which side is asked about. A test that only
        // checked the group opcode would miss a fold that always kept the same end.
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
