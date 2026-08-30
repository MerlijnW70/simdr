use super::{Element, Integer, LaneError, Lanes, Mapping, U32, Vector};
use crate::module::{Id, Reduction, op};
use crate::spec::{Capability, Glsl};

impl Lanes<'_> {
    pub fn prefix_sum<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::sum::<T>(), Reduction::InclusiveScan, vector)
    }

    pub fn prefix_sum_exclusive<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::sum::<T>(), Reduction::ExclusiveScan, vector)
    }

    /// The running product. Over floats it is not associative, so the answer
    /// depends on the order the device folds in.
    pub fn prefix_product<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::product::<T>(), Reduction::InclusiveScan, vector)
    }

    pub fn prefix_product_exclusive<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::product::<T>(), Reduction::ExclusiveScan, vector)
    }

    /// The running minimum, so each lane holds the smallest element at or
    /// before it.
    pub fn prefix_min<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::min::<T>(), Reduction::InclusiveScan, vector)
    }

    pub fn prefix_min_exclusive<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::min::<T>(), Reduction::ExclusiveScan, vector)
    }

    pub fn prefix_max<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::max::<T>(), Reduction::InclusiveScan, vector)
    }

    pub fn prefix_max_exclusive<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::max::<T>(), Reduction::ExclusiveScan, vector)
    }

    pub fn prefix_and<T: Integer, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::AND, Reduction::InclusiveScan, vector)
    }

    pub fn prefix_and_exclusive<T: Integer, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::AND, Reduction::ExclusiveScan, vector)
    }

    pub fn prefix_or<T: Integer, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::OR, Reduction::InclusiveScan, vector)
    }

    pub fn prefix_or_exclusive<T: Integer, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::OR, Reduction::ExclusiveScan, vector)
    }

    /// The running parity: each lane holds the exclusive-or of everything at or
    /// before it.
    pub fn prefix_xor<T: Integer, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::XOR, Reduction::InclusiveScan, vector)
    }

    pub fn prefix_xor_exclusive<T: Integer, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Fold::XOR, Reduction::ExclusiveScan, vector)
    }

    fn scan_with<T: Element, const LANES: u32>(
        &mut self,
        fold: Fold,
        reduction: Reduction,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        match self.mapping::<LANES>()? {
            Mapping::WholeSubgroup => {}
            Mapping::Clusters { size } => {
                let exclusive = matches!(reduction, Reduction::ExclusiveScan);
                return self.scan_clusters::<T, LANES>(fold, size, exclusive, vector);
            }
            Mapping::Strips { .. } => {
                return self.scan_strips::<T, LANES>(fold, reduction, vector);
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
                .subgroup_reduce(fold.group, element, scope, reduction, vector.id())?;
        self.from_lane_value(id)
    }

    fn scan_strips<T: Element, const LANES: u32>(
        &mut self,
        fold: Fold,
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
            let within = self
                .module()
                .subgroup_reduce(fold.group, element, scope, reduction, strip)?;
            scanned.push(match carried {
                None => within,
                Some(carry) => self.join(fold, element, within, carry)?,
            });

            if index == last {
                continue;
            }
            let total = self.module().subgroup_reduce(
                fold.group,
                element,
                scope,
                Reduction::Reduce,
                strip,
            )?;
            carried = Some(match carried {
                None => total,
                Some(carry) => self.join(fold, element, carry, total)?,
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
        fold: Fold,
        size: u32,
        exclusive: bool,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        if size == 1 {
            return if exclusive {
                self.splat_bits::<T, LANES>(fold.identity)
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
            let raised = self.join_vectors::<T, LANES>(fold, value, below)?;
            let inside = self.beyond::<LANES>(within, distance.saturating_sub(1))?;
            value = self.select(inside, raised, value)?;
            distance = distance.saturating_mul(2);
        }

        if !exclusive {
            return Ok(value);
        }

        let shifted = self.shift_up_across_clusters(value, 1)?;
        let identity = self.splat_bits::<T, LANES>(fold.identity)?;
        let inside = self.beyond::<LANES>(within, 0)?;
        self.select(inside, shifted, identity)
    }

    /// The operation this fold carries, over two ids of the element type. Most
    /// are one instruction; a minimum and a maximum are one of the extended set.
    fn join(&mut self, fold: Fold, element: Id, left: Id, right: Id) -> Result<Id, LaneError> {
        match fold.combine {
            Combine::Instruction(opcode) => {
                Ok(self.module().binary(opcode, element, left, right)?)
            }
            Combine::Extended(instruction) => {
                let set = self.glsl()?;
                Ok(self
                    .module()
                    .ext_inst(element, set, instruction.word(), &[left, right])?)
            }
        }
    }

    fn join_vectors<T: Element, const LANES: u32>(
        &mut self,
        fold: Fold,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let mut ids = Vec::with_capacity(left.strip_count());

        for (&a, &b) in left.strips().iter().zip(right.strips()) {
            ids.push(self.join(fold, element, a, b)?);
        }

        self.from_strips(&ids)
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

/// What a scan folds with: the group instruction that does it natively, the
/// same operation elementwise for the paths this crate emulates, and the value
/// a lane with nothing before it takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fold {
    group: u16,
    combine: Combine,
    identity: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Combine {
    Instruction(u16),
    Extended(Glsl),
}

impl Fold {
    const AND: Self = Self {
        group: op::GROUP_NON_UNIFORM_BITWISE_AND,
        combine: Combine::Instruction(op::BITWISE_AND),
        identity: u32::MAX,
    };

    const OR: Self = Self {
        group: op::GROUP_NON_UNIFORM_BITWISE_OR,
        combine: Combine::Instruction(op::BITWISE_OR),
        identity: 0,
    };

    const XOR: Self = Self {
        group: op::GROUP_NON_UNIFORM_BITWISE_XOR,
        combine: Combine::Instruction(op::BITWISE_XOR),
        identity: 0,
    };

    fn sum<T: Element>() -> Self {
        Self {
            group: T::GROUP_ADD,
            combine: Combine::Instruction(T::ADD),
            identity: 0,
        }
    }

    fn product<T: Element>() -> Self {
        Self {
            group: T::GROUP_MUL,
            combine: Combine::Instruction(T::MUL),
            identity: T::ONE,
        }
    }

    fn min<T: Element>() -> Self {
        Self {
            group: T::GROUP_MIN,
            combine: Combine::Extended(T::MIN),
            identity: T::HIGHEST,
        }
    }

    fn max<T: Element>() -> Self {
        Self {
            group: T::GROUP_MAX,
            combine: Combine::Extended(T::MAX),
            identity: T::LOWEST,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::Fold;
    use crate::decode;
    use crate::lanes::{F32, I32, Lanes, U32};
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

    type Scan = fn(&mut Lanes<'_>);

    #[test]
    fn each_scan_reaches_the_group_instruction_of_its_own_operation() {
        let emitted = |build: Scan| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            build(&mut lanes);
            module.finish()
        };

        let cases: [(&str, u16, Scan); 6] = [
            ("product", op::GROUP_NON_UNIFORM_I_MUL, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(2).expect("splat");
                lanes.prefix_product(v).expect("scanned");
            }),
            ("min", op::GROUP_NON_UNIFORM_U_MIN, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(2).expect("splat");
                lanes.prefix_min(v).expect("scanned");
            }),
            ("max", op::GROUP_NON_UNIFORM_U_MAX, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(2).expect("splat");
                lanes.prefix_max(v).expect("scanned");
            }),
            ("and", op::GROUP_NON_UNIFORM_BITWISE_AND, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(2).expect("splat");
                lanes.prefix_and(v).expect("scanned");
            }),
            ("or", op::GROUP_NON_UNIFORM_BITWISE_OR, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(2).expect("splat");
                lanes.prefix_or(v).expect("scanned");
            }),
            ("xor", op::GROUP_NON_UNIFORM_BITWISE_XOR, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(2).expect("splat");
                lanes.prefix_xor(v).expect("scanned");
            }),
        ];

        for (name, expected, build) in cases {
            let words = emitted(build);
            assert_eq!(count(&words, expected), 1, "{name}");
            assert_eq!(
                count(&words, op::GROUP_NON_UNIFORM_I_ADD),
                0,
                "{name} reached the addition it was not asked for"
            );
        }
    }

    #[test]
    fn every_scan_carries_the_scan_literal_and_not_a_plain_reduce() {
        for exclusive in [false, true] {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            let value = lanes.splat_bits::<U32, 32>(3).expect("splat");

            if exclusive {
                lanes.prefix_max_exclusive(value).expect("scanned");
            } else {
                lanes.prefix_max(value).expect("scanned");
            }

            let words = module.finish();
            let operands = decode::body(&words)
                .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_U_MAX)
                .expect("emitted")
                .operands()
                .to_vec();

            let wanted = if exclusive {
                GroupOperation::ExclusiveScan
            } else {
                GroupOperation::InclusiveScan
            };
            assert_eq!(operands[3], wanted.word(), "exclusive: {exclusive}");
        }
    }

    #[test]
    fn a_strip_mined_scan_carries_between_strips_with_its_own_operation() {
        let folds = |build: Scan| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            build(&mut lanes);
            let words = module.finish();
            (
                count(&words, op::I_ADD),
                count(&words, op::I_MUL),
                count(&words, op::BITWISE_XOR),
                count(&words, op::EXT_INST),
            )
        };

        assert_eq!(
            folds(|lanes| {
                let v = lanes.splat_bits::<U32, 64>(2).expect("splat");
                lanes.prefix_product(v).expect("scanned");
            }),
            (0, 1, 0, 0),
            "two strips meet once, and they meet by multiplying rather than adding"
        );
        assert_eq!(
            folds(|lanes| {
                let v = lanes.splat_bits::<U32, 64>(2).expect("splat");
                lanes.prefix_xor(v).expect("scanned");
            }),
            (0, 0, 1, 0)
        );
        assert_eq!(
            folds(|lanes| {
                let v = lanes.splat_bits::<U32, 64>(2).expect("splat");
                lanes.prefix_max(v).expect("scanned");
            }),
            (0, 0, 0, 1),
            "a maximum carries through the extended set, which has no plain opcode"
        );
    }

    #[test]
    fn an_exclusive_scan_in_clusters_starts_each_one_from_its_own_identity() {
        let edge = |build: Scan| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            build(&mut lanes);
            let words = module.finish();
            decode::body(&words)
                .filter(|instruction| instruction.opcode() == op::CONSTANT)
                .filter_map(|instruction| instruction.operands().get(2).copied())
                .collect::<Vec<u32>>()
        };

        let summed = edge(|lanes| {
            let v = lanes.splat_bits::<U32, 8>(3).expect("splat");
            lanes.prefix_sum_exclusive(v).expect("scanned");
        });
        let multiplied = edge(|lanes| {
            let v = lanes.splat_bits::<U32, 8>(3).expect("splat");
            lanes.prefix_product_exclusive(v).expect("scanned");
        });
        let anded = edge(|lanes| {
            let v = lanes.splat_bits::<U32, 8>(3).expect("splat");
            lanes.prefix_and_exclusive(v).expect("scanned");
        });

        assert!(summed.contains(&0), "a sum starts from nought");
        assert!(multiplied.contains(&1), "a product starts from one");
        assert!(
            anded.contains(&u32::MAX),
            "an intersection starts from every bit set, and starting it from nought would leave \
             the first lane of every cluster empty"
        );
    }

    #[test]
    fn the_identity_of_each_operation_is_the_one_that_leaves_a_value_alone() {
        assert_eq!(Fold::sum::<U32>().identity, 0);
        assert_eq!(Fold::product::<U32>().identity, 1);
        assert_eq!(Fold::min::<U32>().identity, u32::MAX);
        assert_eq!(Fold::max::<U32>().identity, 0);
        assert_eq!(Fold::AND.identity, u32::MAX);
        assert_eq!(Fold::OR.identity, 0);
        assert_eq!(Fold::XOR.identity, 0);

        assert_eq!(f32::from_bits(Fold::product::<F32>().identity), 1.0);
        assert_eq!(f32::from_bits(Fold::min::<F32>().identity), f32::INFINITY);
        assert_eq!(
            f32::from_bits(Fold::max::<F32>().identity),
            f32::NEG_INFINITY
        );
        assert_eq!(Fold::min::<I32>().identity, i32::MAX as u32);
    }

    #[test]
    fn no_two_operations_fold_the_same_way() {
        let every = [
            Fold::sum::<U32>(),
            Fold::product::<U32>(),
            Fold::min::<U32>(),
            Fold::max::<U32>(),
            Fold::AND,
            Fold::OR,
            Fold::XOR,
        ];

        for (index, fold) in every.iter().enumerate() {
            for other in every.iter().skip(index + 1) {
                assert_ne!(
                    fold.group, other.group,
                    "two operations share a group instruction: {fold:?} and {other:?}"
                );
            }
        }
    }
}
