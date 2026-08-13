//! Reductions — the operations that cross lanes.
//!
//! Which instruction comes out depends on how `LANES` sits on the subgroup, and that is decided in
//! one place: [`Lanes::mapping`].
//!
//! - As wide as the subgroup: one plain `Reduce`.
//! - Narrower: one `ClusteredReduce` whose cluster size is the vector's own width, so the lanes
//!   that would otherwise idle are running other copies of the same vector.
//! - Wider: fold the strips together inside each lane first — `strips - 1` scalar operations —
//!   then one subgroup instruction over the partials.
//!
//! Needs `GroupNonUniform` and `GroupNonUniformArithmetic`, plus `GroupNonUniformClustered`
//! whenever a vector is narrower than the subgroup. The caller declares them; nothing here does.

use super::{Element, LaneError, Lanes, Mapping, Vector};
use crate::module::{Id, Reduction};
use crate::spec::Capability;

impl Lanes<'_> {
    /// The sum of every element, delivered to every lane — `Simd::reduce_sum`.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoMapping`] if `LANES` cannot sit on this subgroup, [`LaneError::Build`] if an
    /// instruction cannot be emitted.
    pub fn reduce_sum<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_with::<T, LANES>(T::GROUP_ADD, T::ADD, vector)
    }

    /// Running totals: each lane receives the sum of itself and every lane before it.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchForm`] unless the vector is exactly the subgroup's width. SPIR-V's
    /// clustered form is a *reduce*, so a narrower vector has no clustered scan; and a
    /// strip-mined scan would have to carry a running total between strips, which is a different
    /// algorithm rather than a different operand.
    pub fn prefix_sum<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Reduction::InclusiveScan, "prefix_sum", vector)
    }

    /// The same, with each lane's own element left out: lane 0 receives the additive identity.
    ///
    /// **The form a multi-block scan needs, and the reason it is a separate instruction rather than
    /// a subtraction.** Block `b` of a long scan owes the total of every block before it and not
    /// its own, which is an exclusive scan of the block totals. Computing it as `inclusive - own`
    /// costs an operation and, in floating point, is not the same number — subtracting a large
    /// running total back off itself loses precisely the low bits the scan just accumulated.
    ///
    /// SPIR-V has the operation, so this asks for it. `GroupOperation::ExclusiveScan` was in
    /// `spec::group` from the beginning and nothing had ever emitted one.
    ///
    /// # Errors
    ///
    /// As [`Lanes::prefix_sum`].
    pub fn prefix_sum_exclusive<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Reduction::ExclusiveScan, "prefix_sum_exclusive", vector)
    }

    /// Both scans, which differ only in the group operation they name.
    ///
    /// `operation` is carried through to the error rather than hard-coded, so a caller told why its
    /// vector has no scan is told which scan it asked for.
    fn scan_with<T: Element, const LANES: u32>(
        &mut self,
        reduction: Reduction,
        operation: &'static str,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        match self.mapping::<LANES>()? {
            Mapping::WholeSubgroup => {}
            Mapping::Clusters { .. } => {
                return Err(LaneError::NoSuchForm {
                    operation,
                    because: "SPIR-V's clustered form is a reduce, so a scan would run across                               lanes belonging to a different vector",
                });
            }
            Mapping::Strips { .. } => {
                return Err(LaneError::NoSuchForm {
                    operation,
                    because: "a strip-mined scan must carry a running total between strips,                               which is not built",
                });
            }
        }

        let element = self.type_of::<T>()?;
        let scope = self.scope();
        self.module()
            .require_capability(Capability::GroupNonUniform)?;
        self.module()
            .require_capability(Capability::GroupNonUniformArithmetic)?;

        let id =
            self.module()
                .subgroup_reduce(T::GROUP_ADD, element, scope, reduction, vector.id())?;
        self.from_lane_value(id)
    }

    /// Fold the strips inside each lane with `local`, then reduce across the subgroup with
    /// `group`.
    fn reduce_with<T: Element, const LANES: u32>(
        &mut self,
        group: u16,
        local: u16,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        let element = self.type_of::<T>()?;
        let scope = self.scope();
        let (reduction, partial) = self.fold_strips::<T, LANES>(local, vector)?;

        Ok(self
            .module()
            .subgroup_reduce(group, element, scope, reduction, partial)?)
    }

    /// Combine a vector's strips into one per-lane value, and say which reduction shape the
    /// subgroup step then needs.
    fn fold_strips<T: Element, const LANES: u32>(
        &mut self,
        local: u16,
        vector: Vector<T, LANES>,
    ) -> Result<(Reduction, Id), LaneError> {
        let element = self.type_of::<T>()?;
        let reduction = self.reduction::<LANES>()?;

        let mut partial = vector
            .strips()
            .first()
            .copied()
            .ok_or(LaneError::no_strips())?;
        for &next in vector.strips().iter().skip(1) {
            partial = self.module().binary(local, element, partial, next)?;
        }

        Ok((reduction, partial))
    }

    /// The reduction shape this vector's mapping implies, declaring what it needs on the way.
    ///
    /// A strip-mined vector reduces over the *whole* subgroup once its strips are folded — the
    /// strips are within a lane, so they never needed a cluster. Only the clustered case asks for
    /// `GroupNonUniformClustered`, which is why a kernel that never uses one stays runnable on a
    /// device that does not offer it.
    pub(super) fn reduction<const LANES: u32>(&mut self) -> Result<Reduction, LaneError> {
        let mapping = self.mapping::<LANES>()?;

        self.module()
            .require_capability(Capability::GroupNonUniform)?;
        self.module()
            .require_capability(Capability::GroupNonUniformArithmetic)?;

        match mapping {
            Mapping::WholeSubgroup | Mapping::Strips { .. } => Ok(Reduction::Reduce),
            Mapping::Clusters { size } => {
                self.module()
                    .require_capability(Capability::GroupNonUniformClustered)?;
                let size = self.module().constant_u32(size)?;
                Ok(Reduction::Clustered { size })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{F32, U32};
    use crate::module::{Module, Version, op};
    use crate::spec::GroupOperation;

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    /// The operands of the group reduction in a module built at `width` for `LANES`.
    fn reduce_operands<const LANES: u32>(width: u32) -> Vec<u32> {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, width).expect("built");
        let value = lanes
            .splat_bits::<F32, LANES>(1.0_f32.to_bits())
            .expect("splat");
        lanes.reduce_sum(value).expect("reduced");

        let words = module.finish();
        decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .expect("emitted")
            .operands()
            .to_vec()
    }

    #[test]
    fn a_full_width_vector_reduces_over_the_whole_subgroup() {
        let operands = reduce_operands::<32>(32);

        assert_eq!(operands[3], GroupOperation::Reduce.word());
        assert_eq!(operands.len(), 5, "a plain reduce carries no cluster size");
    }

    #[test]
    fn a_narrow_vector_reduces_in_clusters_of_its_own_width() {
        let operands = reduce_operands::<8>(32);

        assert_eq!(operands[3], GroupOperation::ClusteredReduce.word());
        assert_eq!(operands.len(), 6);
    }

    #[test]
    fn a_strip_mined_vector_folds_locally_then_reduces_over_the_whole_subgroup() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 128>(1.0_f32.to_bits())
            .expect("splat");

        lanes.reduce_sum(value).expect("reduced");

        let words = module.finish();

        // Four elements per lane: three scalar adds to fold them, then one subgroup instruction.
        assert_eq!(count(&words, op::F_ADD), 3);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_ADD), 1);

        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .expect("emitted")
            .operands()
            .to_vec();
        assert_eq!(
            operands[3],
            GroupOperation::Reduce.word(),
            "the strips are within a lane, so the subgroup step never needed a cluster"
        );
    }

    #[test]
    fn the_cluster_size_that_is_emitted_is_the_vectors_own_width() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 4>(1.0_f32.to_bits())
            .expect("splat");
        lanes.reduce_sum(value).expect("reduced");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .expect("emitted")
            .operands()
            .to_vec();

        let cluster_id = operands[5];
        let size = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CONSTANT)
            .find(|instruction| instruction.operands().get(1) == Some(&cluster_id))
            .and_then(|instruction| instruction.operands().get(2).copied())
            .expect("the cluster size is a declared constant");

        assert_eq!(size, 4);
    }

    #[test]
    fn one_lane_count_reduces_three_different_ways_across_two_devices() {
        // DR-0002, read off the emitted instruction rather than the mapping.
        assert_eq!(reduce_operands::<32>(32)[3], GroupOperation::Reduce.word());
        assert_eq!(
            reduce_operands::<32>(64)[3],
            GroupOperation::ClusteredReduce.word()
        );
        assert_eq!(
            reduce_operands::<64>(32)[3],
            GroupOperation::Reduce.word(),
            "two strips folded, then the whole subgroup"
        );
    }

    #[test]
    fn a_lane_count_with_no_mapping_is_refused() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");

        assert_eq!(
            lanes.splat_bits::<F32, 12>(0).err(),
            Some(LaneError::NoMapping {
                lanes: 12,
                width: 32
            })
        );
    }

    #[test]
    fn integers_reduce_with_their_own_opcode() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 32>(1).expect("splat");

        lanes.reduce_sum(value).expect("reduced");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_I_ADD), 1);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_ADD), 0);
    }

    #[test]
    fn a_prefix_sum_scans_when_the_vector_is_the_whole_subgroup() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.prefix_sum(value).expect("scanned");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(operands[3], GroupOperation::InclusiveScan.word());
    }

    #[test]
    fn an_exclusive_prefix_sum_names_the_other_group_operation() {
        // The whole difference between the two scans is this one literal. Both emit the same
        // opcode with the same operands otherwise, so a version that ignored the argument and
        // always scanned inclusively would look right everywhere except here — and would give a
        // multi-block scan every block its own total twice over.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.prefix_sum_exclusive(value).expect("scanned");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(operands[3], GroupOperation::ExclusiveScan.word());
        assert_ne!(operands[3], GroupOperation::InclusiveScan.word());
    }

    #[test]
    fn the_two_scans_differ_in_nothing_but_that_literal() {
        // They share a builder, and this is what says the sharing did not quietly make them the
        // same instruction — or leave one of them declaring a capability the other does not.
        let scan = |exclusive: bool| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            let value = lanes
                .splat_bits::<F32, 32>(1.0_f32.to_bits())
                .expect("splat");
            if exclusive {
                lanes.prefix_sum_exclusive(value).expect("scanned");
            } else {
                lanes.prefix_sum(value).expect("scanned");
            }
            module.finish()
        };

        let inclusive = scan(false);
        let exclusive = scan(true);

        assert_eq!(inclusive.len(), exclusive.len(), "same instruction count");
        assert_eq!(
            decode::opcodes(&inclusive),
            decode::opcodes(&exclusive),
            "same instructions, in the same order"
        );
        assert_ne!(inclusive, exclusive, "and not the same words");
    }

    #[test]
    fn the_exclusive_scan_is_refused_for_the_same_shapes_as_the_inclusive_one() {
        // Both go through one builder, so a caller must not find that one of them accepts a shape
        // the other refuses — and the error must name the operation the caller actually asked for.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let narrow = lanes.splat_bits::<F32, 8>(0).expect("splat");
        let wide = lanes.splat_bits::<F32, 64>(0).expect("splat");

        for refused in [
            lanes.prefix_sum_exclusive(narrow).err(),
            lanes.prefix_sum_exclusive(wide).err(),
        ] {
            assert!(matches!(
                refused,
                Some(LaneError::NoSuchForm {
                    operation: "prefix_sum_exclusive",
                    ..
                })
            ));
        }
    }

    #[test]
    fn a_scan_is_refused_for_both_shapes_that_have_no_form_of_it() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let narrow = lanes.splat_bits::<F32, 8>(0).expect("splat");
        let wide = lanes.splat_bits::<F32, 64>(0).expect("splat");

        assert!(matches!(
            lanes.prefix_sum(narrow).err(),
            Some(LaneError::NoSuchForm { .. })
        ));
        assert!(matches!(
            lanes.prefix_sum(wide).err(),
            Some(LaneError::NoSuchForm { .. })
        ));
    }
}
