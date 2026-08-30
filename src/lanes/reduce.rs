use super::{Element, Integer, LaneError, Lanes, Mapping, Vector};
use crate::module::{Id, Reduction, op};
use crate::spec::Capability;

impl Lanes<'_> {
    pub fn reduce_sum<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_with::<T, LANES>(T::GROUP_ADD, T::ADD, vector)
    }

    /// The product over the lanes. Over floats this is not associative, so the
    /// answer depends on the order the device folds in and need not equal a
    /// left-to-right product on the host.
    pub fn reduce_product<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_with::<T, LANES>(T::GROUP_MUL, T::MUL, vector)
    }

    pub fn reduce_and<T: Integer, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_with::<T, LANES>(op::GROUP_NON_UNIFORM_BITWISE_AND, op::BITWISE_AND, vector)
    }

    pub fn reduce_or<T: Integer, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_with::<T, LANES>(op::GROUP_NON_UNIFORM_BITWISE_OR, op::BITWISE_OR, vector)
    }

    pub fn reduce_xor<T: Integer, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_with::<T, LANES>(op::GROUP_NON_UNIFORM_BITWISE_XOR, op::BITWISE_XOR, vector)
    }

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
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{F32, I32, U32};
    use crate::module::{Module, Version, op};
    use crate::spec::GroupOperation;

    type Build = fn(&mut Lanes<'_>);

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

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
    fn each_reduction_reaches_the_group_instruction_it_is_named_for() {
        let emitted = |build: Build| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            build(&mut lanes);
            module.finish()
        };

        let cases: [(&str, u16, Build); 5] = [
            ("f32 product", op::GROUP_NON_UNIFORM_F_MUL, |lanes| {
                let v = lanes.splat_bits::<F32, 32>(0).expect("splat");
                lanes.reduce_product(v).expect("product");
            }),
            ("u32 product", op::GROUP_NON_UNIFORM_I_MUL, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(0).expect("splat");
                lanes.reduce_product(v).expect("product");
            }),
            ("and", op::GROUP_NON_UNIFORM_BITWISE_AND, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(0).expect("splat");
                lanes.reduce_and(v).expect("and");
            }),
            ("or", op::GROUP_NON_UNIFORM_BITWISE_OR, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(0).expect("splat");
                lanes.reduce_or(v).expect("or");
            }),
            ("xor", op::GROUP_NON_UNIFORM_BITWISE_XOR, |lanes| {
                let v = lanes.splat_bits::<U32, 32>(0).expect("splat");
                lanes.reduce_xor(v).expect("xor");
            }),
        ];

        for (name, expected, build) in cases {
            let words = emitted(build);
            assert_eq!(count(&words, expected), 1, "{name}");
            assert_eq!(
                count(&words, op::GROUP_NON_UNIFORM_I_ADD)
                    + count(&words, op::GROUP_NON_UNIFORM_F_ADD),
                0,
                "{name} reached the addition it was not asked for"
            );
        }
    }

    #[test]
    fn a_strip_mined_reduction_folds_with_the_operation_it_is_reducing_by() {
        let folded = |build: Build| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            build(&mut lanes);
            let words = module.finish();
            (
                count(&words, op::BITWISE_AND),
                count(&words, op::BITWISE_OR),
                count(&words, op::BITWISE_XOR),
                count(&words, op::I_MUL),
                count(&words, op::I_ADD),
            )
        };

        assert_eq!(
            folded(|lanes| {
                let v = lanes.splat_bits::<U32, 128>(1).expect("splat");
                lanes.reduce_and(v).expect("and");
            }),
            (3, 0, 0, 0, 0),
            "four strips fold with three ANDs and nothing else"
        );
        assert_eq!(
            folded(|lanes| {
                let v = lanes.splat_bits::<U32, 128>(1).expect("splat");
                lanes.reduce_or(v).expect("or");
            }),
            (0, 3, 0, 0, 0)
        );
        assert_eq!(
            folded(|lanes| {
                let v = lanes.splat_bits::<U32, 128>(1).expect("splat");
                lanes.reduce_xor(v).expect("xor");
            }),
            (0, 0, 3, 0, 0)
        );
        assert_eq!(
            folded(|lanes| {
                let v = lanes.splat_bits::<U32, 128>(1).expect("splat");
                lanes.reduce_product(v).expect("product");
            }),
            (0, 0, 0, 3, 0),
            "a product folds by multiplying, not by adding"
        );
    }

    #[test]
    fn the_bitwise_reductions_are_one_instruction_for_both_integer_families() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let signed = lanes.splat_bits::<I32, 32>(1).expect("i32");
        let unsigned = lanes.splat_bits::<U32, 32>(1).expect("u32");

        lanes.reduce_and(signed).expect("signed");
        lanes.reduce_and(unsigned).expect("unsigned");

        assert_eq!(
            count(&module.finish(), op::GROUP_NON_UNIFORM_BITWISE_AND),
            2,
            "signedness does not reach a bitwise reduction"
        );
    }

    #[test]
    fn a_narrow_bitwise_reduction_clusters_like_every_other_one() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 8>(1).expect("splat");

        lanes.reduce_xor(value).expect("reduced");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_BITWISE_XOR)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(operands[3], GroupOperation::ClusteredReduce.word());
        assert_eq!(operands.len(), 6, "a cluster carries its size");
    }
}
