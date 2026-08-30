use super::{Element, LaneError, Lanes, Mapping, U32, Vector};
use crate::module::{Id, Reduction, op};
use crate::spec::Capability;

impl Lanes<'_> {
    pub fn prefix_sum<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Reduction::InclusiveScan, vector)
    }

    pub fn prefix_sum_exclusive<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Reduction::ExclusiveScan, vector)
    }

    fn scan_with<T: Element, const LANES: u32>(
        &mut self,
        reduction: Reduction,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        match self.mapping::<LANES>()? {
            Mapping::WholeSubgroup => {}
            Mapping::Clusters { size } => {
                let exclusive = matches!(reduction, Reduction::ExclusiveScan);
                return self.scan_clusters::<T, LANES>(size, exclusive, vector);
            }
            Mapping::Strips { .. } => {
                return self.scan_strips::<T, LANES>(reduction, vector);
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

    fn scan_strips<T: Element, const LANES: u32>(
        &mut self,
        reduction: Reduction,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let scope = self.scope();
        self.module()
            .require_capability(Capability::GroupNonUniform)?;
        self.module()
            .require_capability(Capability::GroupNonUniformArithmetic)?;

        let strips = vector.strips().to_vec();
        let last = strips.len().saturating_sub(1);
        let mut scanned = Vec::with_capacity(strips.len());
        let mut carried: Option<Id> = None;

        for (index, &strip) in strips.iter().enumerate() {
            let within =
                self.module()
                    .subgroup_reduce(T::GROUP_ADD, element, scope, reduction, strip)?;
            scanned.push(match carried {
                None => within,
                Some(carry) => self.module().binary(T::ADD, element, within, carry)?,
            });

            if index == last {
                continue;
            }
            let total = self.module().subgroup_reduce(
                T::GROUP_ADD,
                element,
                scope,
                Reduction::Reduce,
                strip,
            )?;
            carried = Some(match carried {
                None => total,
                Some(carry) => self.module().binary(T::ADD, element, carry, total)?,
            });
        }

        self.from_strips(&scanned)
    }

    /// ```text
    ///   for distance in 1, 2, 4, … < size:
    ///       value += (lane % size) > distance - 1  ?  the value `distance` lanes below  :  nothing
    /// ```
    fn scan_clusters<T: Element, const LANES: u32>(
        &mut self,
        size: u32,
        exclusive: bool,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        if size == 1 {
            return if exclusive {
                self.splat_bits::<T, LANES>(0)
            } else {
                Ok(vector)
            };
        }

        let lane = self.lane_index()?;
        let uint = self.type_of::<U32>()?;
        let wrap = self.module().constant_u32(size.saturating_sub(1))?;
        let within = self.module().binary(op::BITWISE_AND, uint, lane, wrap)?;

        let mut value = vector;
        let mut distance = 1;
        while distance < size {
            let below = self.shift_up_across_clusters(value, distance)?;
            let raised = self.add(value, below)?;
            let inside = self.beyond::<LANES>(within, distance.saturating_sub(1))?;
            value = self.select(inside, raised, value)?;
            distance = distance.saturating_mul(2);
        }

        if !exclusive {
            return Ok(value);
        }

        let shifted = self.shift_up_across_clusters(value, 1)?;
        let identity = self.splat_bits::<T, LANES>(0)?;
        let inside = self.beyond::<LANES>(within, 0)?;
        self.select(inside, shifted, identity)
    }

    fn beyond<const LANES: u32>(
        &mut self,
        within: Id,
        edge: u32,
    ) -> Result<crate::lanes::Predicate<LANES>, LaneError> {
        let position = self.from_lane_value::<U32, LANES>(within)?;
        let edge = self.splat_bits::<U32, LANES>(edge)?;
        self.greater_than(position, edge)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use crate::decode;
    use crate::lanes::{F32, Lanes};
    use crate::module::{Module, Version, op};
    use crate::spec::{Capability, GroupOperation};

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
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
    fn the_two_scans_accept_exactly_the_same_shapes() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let narrow = lanes.splat_bits::<F32, 8>(0).expect("splat");
        let whole = lanes.splat_bits::<F32, 32>(0).expect("splat");
        let wide = lanes.splat_bits::<F32, 128>(0).expect("splat");

        assert!(lanes.prefix_sum(narrow).is_ok());
        assert!(lanes.prefix_sum_exclusive(narrow).is_ok());
        assert!(lanes.prefix_sum(whole).is_ok());
        assert!(lanes.prefix_sum_exclusive(whole).is_ok());
        assert!(lanes.prefix_sum(wide).is_ok());
        assert!(lanes.prefix_sum_exclusive(wide).is_ok());
    }

    #[test]
    fn a_clustered_scan_is_a_ladder_of_one_step_per_doubling() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<F32, 8>(0).expect("splat");

        lanes.prefix_sum(value).expect("scanned");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE_UP), 3);
        assert_eq!(count(&words, op::SELECT), 3);
        assert_eq!(
            count(&words, op::GROUP_NON_UNIFORM_F_ADD),
            0,
            "there is no clustered scan instruction to have emitted"
        );
    }

    #[test]
    fn a_cluster_of_one_lane_scans_nothing_and_emits_nothing() {
        for exclusive in [false, true] {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            let value = lanes.splat_bits::<F32, 1>(0).expect("splat");

            let scanned = if exclusive {
                lanes.prefix_sum_exclusive(value).expect("scanned")
            } else {
                lanes.prefix_sum(value).expect("scanned")
            };
            if !exclusive {
                assert_eq!(scanned.id(), value.id(), "the same value, untouched");
            }

            let words = module.finish();
            assert_eq!(count(&words, op::SELECT), 0);
            assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE_UP), 0);
            assert_eq!(count(&words, op::BITWISE_AND), 0, "no mask was needed");
            assert_eq!(
                count(&words, op::VARIABLE),
                0,
                "and `SubgroupLocalInvocationId` was never asked for"
            );
        }
    }

    #[test]
    fn the_exclusive_clustered_scan_shifts_rather_than_subtracting() {
        let ladder = |exclusive: bool| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            let value = lanes.splat_bits::<F32, 8>(0).expect("splat");
            if exclusive {
                lanes.prefix_sum_exclusive(value).expect("scanned");
            } else {
                lanes.prefix_sum(value).expect("scanned");
            }
            module.finish()
        };

        let inclusive = ladder(false);
        let exclusive = ladder(true);

        assert_eq!(
            count(&exclusive, op::GROUP_NON_UNIFORM_SHUFFLE_UP),
            count(&inclusive, op::GROUP_NON_UNIFORM_SHUFFLE_UP) + 1
        );
        assert_eq!(
            count(&exclusive, op::SELECT),
            count(&inclusive, op::SELECT) + 1
        );
        assert_eq!(
            count(&exclusive, op::F_ADD),
            count(&inclusive, op::F_ADD),
            "the exclusive form adds nothing the inclusive one does not"
        );
    }

    #[test]
    fn the_clustered_ladder_masks_with_the_lane_the_specification_defines() {
        use crate::spec::{BuiltIn, Decoration};

        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<F32, 8>(0).expect("splat");
        lanes.prefix_sum(value).expect("scanned");

        let words = module.finish();
        let built_ins: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::DECORATE)
            .filter_map(|instruction| match instruction.operands() {
                [_target, decoration, built_in] if *decoration == Decoration::BuiltIn.word() => {
                    Some(*built_in)
                }
                _ => None,
            })
            .collect();

        assert_eq!(
            built_ins,
            vec![BuiltIn::SubgroupLocalInvocationId.word()],
            "one built-in, and it is the lane's own position"
        );
        assert_eq!(count(&words, op::BITWISE_AND), 1, "masked into its cluster");
    }

    #[test]
    fn a_clustered_scan_declares_the_shuffles_it_uses_and_no_arithmetic_it_does_not() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<F32, 8>(0).expect("splat");

        lanes.prefix_sum(value).expect("scanned");

        let words = module.finish();
        let declared: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CAPABILITY)
            .filter_map(|instruction| instruction.operands().first().copied())
            .collect();

        assert!(declared.contains(&Capability::GroupNonUniform.word()));
        assert!(declared.contains(&Capability::GroupNonUniformShuffleRelative.word()));
        assert!(!declared.contains(&Capability::GroupNonUniformArithmetic.word()));
        assert!(!declared.contains(&Capability::GroupNonUniformClustered.word()));
    }

    #[test]
    fn a_strip_mined_scan_is_one_scan_per_strip_and_one_reduce_fewer() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let wide = lanes.splat_bits::<F32, 128>(0).expect("splat");

        let scanned = lanes.prefix_sum(wide).expect("scanned");
        assert_eq!(scanned.strip_count(), 4);

        let words = module.finish();
        let operations: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .filter_map(|instruction| instruction.operands().get(3).copied())
            .collect();

        assert_eq!(
            operations
                .iter()
                .filter(|&&op| op == GroupOperation::InclusiveScan.word())
                .count(),
            4,
            "one scan per strip"
        );
        assert_eq!(
            operations
                .iter()
                .filter(|&&op| op == GroupOperation::Reduce.word())
                .count(),
            3,
            "a carry for every strip but the last"
        );
    }

    #[test]
    fn the_strip_mined_exclusive_scan_carries_the_same_way() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let wide = lanes.splat_bits::<F32, 64>(0).expect("splat");

        lanes.prefix_sum_exclusive(wide).expect("scanned");

        let words = module.finish();
        let operations: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .filter_map(|instruction| instruction.operands().get(3).copied())
            .collect();

        assert_eq!(
            operations,
            vec![
                GroupOperation::ExclusiveScan.word(),
                GroupOperation::Reduce.word(),
                GroupOperation::ExclusiveScan.word(),
            ],
            "scan, carry, scan — and the last strip takes no carry"
        );
    }
}
