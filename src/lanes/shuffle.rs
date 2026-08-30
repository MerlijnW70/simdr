use super::{Element, LaneError, Lanes, Mapping, U32, Vector};
use crate::module::op;
use crate::spec::Capability;

impl Lanes<'_> {
    pub fn shift_down<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        delta: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.exchange::<T, LANES>(
            "shift_down",
            vector,
            Exchange::Down,
            delta,
            Capability::GroupNonUniformShuffleRelative,
        )
    }

    pub fn shift_up<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        delta: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.exchange::<T, LANES>(
            "shift_up",
            vector,
            Exchange::Up,
            delta,
            Capability::GroupNonUniformShuffleRelative,
        )
    }

    pub fn butterfly<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        mask: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.within_group::<LANES>("butterfly", mask)?;

        let mask = self.module().constant_u32(mask)?;
        self.shuffle::<T, LANES>(
            vector,
            Exchange::Xor,
            mask,
            Capability::GroupNonUniformShuffle,
        )
    }

    pub fn broadcast<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        source: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.within_group::<LANES>("broadcast", source)?;

        if let Mapping::Clusters { size } = self.mapping::<LANES>()? {
            return self.broadcast_within_cluster::<T, LANES>(vector, size, source);
        }

        self.exchange::<T, LANES>(
            "broadcast",
            vector,
            Exchange::Index,
            source,
            Capability::GroupNonUniformShuffle,
        )
    }

    pub fn rotate_up<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        delta: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let size = match self.mapping::<LANES>()? {
            Mapping::WholeSubgroup => self.width(),
            Mapping::Clusters { size } => size,
            Mapping::Strips { .. } => {
                return Err(LaneError::NoSuchForm {
                    operation: "rotate_up",
                    because: "a rotate of a strip-mined vector moves elements between strips, \
                              which is a shuffle per strip and a rotation of the strips as well",
                });
            }
        };

        let delta = delta % size.max(1);
        if delta == 0 {
            return Ok(vector);
        }

        let lane = self.lane_index()?;
        let uint = self.type_of::<U32>()?;
        let wrap = self.module().constant_u32(size.saturating_sub(1))?;
        let round_down = self.module().constant_u32(!size.wrapping_sub(1))?;
        let back = self.module().constant_u32(size.saturating_sub(delta))?;

        let first = self
            .module()
            .binary(op::BITWISE_AND, uint, lane, round_down)?;
        let moved = self.module().i_add(uint, lane, back)?;
        let within = self.module().binary(op::BITWISE_AND, uint, moved, wrap)?;
        let read = self.module().binary(op::BITWISE_OR, uint, first, within)?;

        self.shuffle::<T, LANES>(
            vector,
            Exchange::Index,
            read,
            Capability::GroupNonUniformShuffle,
        )
    }

    fn broadcast_within_cluster<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        size: u32,
        source: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        if size == 1 {
            return Ok(vector);
        }

        let lane = self.lane_index()?;
        let uint = self.type_of::<U32>()?;
        let round_down = self.module().constant_u32(!size.wrapping_sub(1))?;
        let first = self
            .module()
            .binary(op::BITWISE_AND, uint, lane, round_down)?;
        let offset = self.module().constant_u32(source)?;
        let read = self.module().i_add(uint, first, offset)?;

        self.shuffle::<T, LANES>(
            vector,
            Exchange::Index,
            read,
            Capability::GroupNonUniformShuffle,
        )
    }

    pub(super) fn shift_up_across_clusters<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        delta: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let delta = self.module().constant_u32(delta)?;
        self.shuffle::<T, LANES>(
            vector,
            Exchange::Up,
            delta,
            Capability::GroupNonUniformShuffleRelative,
        )
    }

    fn exchange<T: Element, const LANES: u32>(
        &mut self,
        name: &'static str,
        vector: Vector<T, LANES>,
        kind: Exchange,
        operand: u32,
        capability: Capability,
    ) -> Result<Vector<T, LANES>, LaneError> {
        if let Mapping::Clusters { .. } = self.mapping::<LANES>()? {
            return Err(LaneError::NoSuchForm {
                operation: name,
                because: "a shuffle reads a lane of the subgroup, and a narrower vector shares \
                          those lanes with other vectors",
            });
        }
        self.within_group::<LANES>(name, operand)?;

        let operand = self.module().constant_u32(operand)?;
        self.shuffle::<T, LANES>(vector, kind, operand, capability)
    }

    fn within_group<const LANES: u32>(
        &self,
        operation: &'static str,
        operand: u32,
    ) -> Result<(), LaneError> {
        let lanes = match self.mapping::<LANES>()? {
            Mapping::WholeSubgroup | Mapping::Strips { .. } => self.width(),
            Mapping::Clusters { size } => size,
        };

        if operand >= lanes {
            return Err(LaneError::LaneOutOfRange {
                operation,
                operand,
                lanes,
            });
        }
        Ok(())
    }

    fn shuffle<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        kind: Exchange,
        operand: crate::module::Id,
        capability: Capability,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let scope = self.scope();
        self.module()
            .require_capability(Capability::GroupNonUniform)?;
        self.module().require_capability(capability)?;

        let mut ids = Vec::with_capacity(vector.strip_count());
        for &strip in vector.strips() {
            ids.push(match kind {
                Exchange::Down => self
                    .module()
                    .subgroup_shuffle_down(element, scope, strip, operand)?,
                Exchange::Up => self
                    .module()
                    .subgroup_shuffle_up(element, scope, strip, operand)?,
                Exchange::Xor => self
                    .module()
                    .subgroup_shuffle_xor(element, scope, strip, operand)?,
                Exchange::Index => self
                    .module()
                    .subgroup_shuffle(element, scope, strip, operand)?,
            });
        }

        self.from_strips(&ids)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Exchange {
    Down,
    Up,
    Xor,
    Index,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::F32;
    use crate::module::{Module, Version, op};

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    #[test]
    fn a_butterfly_is_one_instruction_for_a_full_width_vector() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.butterfly(value, 1).expect("butterfly");

        assert_eq!(
            count(&module.finish(), op::GROUP_NON_UNIFORM_SHUFFLE_XOR),
            1
        );
    }

    #[test]
    fn a_strip_mined_shuffle_is_one_instruction_per_strip() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 128>(1.0_f32.to_bits())
            .expect("splat");

        let shuffled = lanes.butterfly(value, 2).expect("butterfly");

        assert_eq!(shuffled.strip_count(), 4);
        assert_eq!(
            count(&module.finish(), op::GROUP_NON_UNIFORM_SHUFFLE_XOR),
            4,
            "every strip is a full subgroup's worth and shuffles on its own"
        );
    }

    #[test]
    fn each_exchange_uses_its_own_instruction() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.shift_down(value, 1).expect("down");
        lanes.shift_up(value, 1).expect("up");
        lanes.butterfly(value, 1).expect("xor");
        lanes.broadcast(value, 0).expect("broadcast");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE_DOWN), 1);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE_UP), 1);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE_XOR), 1);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE), 1);
    }

    #[test]
    fn a_rotate_wraps_inside_the_vector_and_leaves_no_lane_undefined() {
        for lanes_wide in [false, true] {
            let mut module = Module::new(Version::V1_3);
            let mut builder = Lanes::new(&mut module, 32).expect("built");
            if lanes_wide {
                let value = builder.splat_bits::<F32, 32>(0).expect("splat");
                builder.rotate_up(value, 3).expect("rotated");
            } else {
                let value = builder.splat_bits::<F32, 8>(0).expect("splat");
                builder.rotate_up(value, 3).expect("rotated");
            }

            let words = module.finish();
            assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE), 1);
            assert_eq!(
                count(&words, op::GROUP_NON_UNIFORM_SHUFFLE_UP),
                0,
                "not a shift"
            );
            assert_eq!(
                count(&words, op::BITWISE_AND),
                2,
                "the vector's base, and the wrap"
            );
            assert_eq!(count(&words, op::BITWISE_OR), 1);
            assert_eq!(count(&words, op::SELECT), 0, "nothing is masked away");
        }
    }

    #[test]
    fn a_rotate_by_the_vectors_own_width_emits_nothing() {
        for delta in [0_u32, 8, 16] {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            let value = lanes.splat_bits::<F32, 8>(0).expect("splat");

            let rotated = lanes.rotate_up(value, delta).expect("rotated");

            assert_eq!(
                rotated.id(),
                value.id(),
                "a rotate by {delta} is the vector"
            );
            assert_eq!(count(&module.finish(), op::GROUP_NON_UNIFORM_SHUFFLE), 0);
        }
    }

    #[test]
    fn a_rotate_of_a_strip_mined_vector_is_refused_by_name() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let wide = lanes.splat_bits::<F32, 128>(0).expect("splat");

        assert!(matches!(
            lanes.rotate_up(wide, 1).err(),
            Some(LaneError::NoSuchForm {
                operation: "rotate_up",
                ..
            })
        ));
    }

    #[test]
    fn a_clustered_shift_is_refused_because_the_lanes_belong_to_other_vectors() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 8>(1.0_f32.to_bits())
            .expect("splat");

        assert!(matches!(
            lanes.shift_down(value, 1).err(),
            Some(LaneError::NoSuchForm { .. })
        ));
        assert!(matches!(
            lanes.shift_up(value, 1).err(),
            Some(LaneError::NoSuchForm { .. })
        ));
    }

    #[test]
    fn a_clustered_broadcast_reads_a_lane_of_its_own_cluster() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 8>(1.0_f32.to_bits())
            .expect("splat");

        lanes.broadcast(value, 3).expect("broadcast");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE), 1);
        assert_eq!(count(&words, op::BITWISE_AND), 1, "rounded to the cluster");
        assert_eq!(count(&words, op::I_ADD), 1, "plus the source's position");

        let constants: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CONSTANT)
            .filter_map(|instruction| instruction.operands().get(2).copied())
            .collect();
        assert!(
            constants.contains(&!7_u32),
            "the rounding mask is missing: {constants:?}"
        );
    }

    #[test]
    fn a_broadcast_within_a_one_lane_vector_emits_nothing() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 1>(1.0_f32.to_bits())
            .expect("splat");

        let shared = lanes.broadcast(value, 0).expect("broadcast");

        assert_eq!(shared.id(), value.id(), "the same value, untouched");
        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE), 0);
        assert_eq!(count(&words, op::BITWISE_AND), 0);
        assert_eq!(count(&words, op::I_ADD), 0);
    }

    #[test]
    fn a_clustered_broadcast_of_a_lane_outside_the_vector_is_refused() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 8>(1.0_f32.to_bits())
            .expect("splat");

        assert!(lanes.broadcast(value, 7).is_ok(), "the last lane is inside");
        assert!(matches!(
            lanes.broadcast(value, 8).err(),
            Some(LaneError::LaneOutOfRange {
                operation: "broadcast",
                operand: 8,
                lanes: 8,
            })
        ));
    }

    #[test]
    fn a_full_width_broadcast_still_names_a_constant_lane() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.broadcast(value, 3).expect("broadcast");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE), 1);
        assert_eq!(count(&words, op::BITWISE_AND), 0);
        assert_eq!(count(&words, op::I_ADD), 0);
    }

    #[test]
    fn a_clustered_butterfly_is_allowed_up_to_the_vectors_own_width() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 8>(1.0_f32.to_bits())
            .expect("splat");

        for mask in [1, 2, 4, 7] {
            assert!(
                lanes.butterfly(value, mask).is_ok(),
                "a mask of {mask} stays inside an eight-lane cluster"
            );
        }
        for mask in [8, 9, 16] {
            assert!(
                matches!(
                    lanes.butterfly(value, mask).err(),
                    Some(LaneError::LaneOutOfRange {
                        operation: "butterfly",
                        lanes: 8,
                        ..
                    })
                ),
                "a mask of {mask} leaves an eight-lane cluster and must be refused"
            );
        }
    }

    #[test]
    fn a_shuffle_past_the_subgroup_is_refused_at_every_mapping() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let whole = lanes.splat_bits::<F32, 32>(0).expect("splat");
        let wide = lanes.splat_bits::<F32, 128>(0).expect("splat");

        assert!(
            lanes.butterfly(whole, 31).is_ok(),
            "the last lane is inside"
        );
        assert!(lanes.broadcast(whole, 31).is_ok());
        assert!(lanes.shift_up(whole, 31).is_ok());
        assert!(lanes.shift_down(whole, 31).is_ok());

        for operand in [32, 40, 4096] {
            for outcome in [
                lanes.butterfly(whole, operand).err(),
                lanes.broadcast(whole, operand).err(),
                lanes.shift_up(whole, operand).err(),
                lanes.shift_down(whole, operand).err(),
                lanes.butterfly(wide, operand).err(),
                lanes.broadcast(wide, operand).err(),
                lanes.shift_up(wide, operand).err(),
                lanes.shift_down(wide, operand).err(),
            ] {
                assert!(
                    matches!(outcome, Some(LaneError::LaneOutOfRange { lanes: 32, .. })),
                    "{operand} reaches outside a 32-wide subgroup and gave {outcome:?}"
                );
            }
        }
    }

    #[test]
    fn a_refused_shuffle_leaves_the_module_as_it_was() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<F32, 32>(0).expect("splat");
        let before = module.finish().len();

        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        assert!(lanes.shift_up(value, 99).is_err());
        assert!(lanes.butterfly(value, 99).is_err());
        assert!(lanes.broadcast(value, 99).is_err());

        assert_eq!(module.finish().len(), before, "nothing was emitted");
    }

    #[test]
    fn a_clustered_butterfly_is_the_same_instruction_as_a_full_width_one() {
        let build = |lanes: u32| {
            let mut module = Module::new(Version::V1_3);
            let mut builder = Lanes::new(&mut module, 32).expect("built");
            match lanes {
                8 => {
                    let value = builder.splat_bits::<F32, 8>(0).expect("splat");
                    builder.butterfly(value, 2).expect("butterfly");
                }
                _ => {
                    let value = builder.splat_bits::<F32, 32>(0).expect("splat");
                    builder.butterfly(value, 2).expect("butterfly");
                }
            }
            module.finish()
        };

        assert_eq!(
            decode::opcodes(&build(8)),
            decode::opcodes(&build(32)),
            "the clustered form emits the same instructions in the same order"
        );
        assert_eq!(count(&build(8), op::SELECT), 0, "and nothing is masked");
    }

    #[test]
    fn the_relative_shifts_ask_for_a_different_capability_than_the_indexed_ones() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");
        lanes.shift_down(value, 1).expect("down");

        let words = module.finish();
        let declared: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CAPABILITY)
            .filter_map(|instruction| instruction.operands().first().copied())
            .collect();

        assert!(declared.contains(&Capability::GroupNonUniformShuffleRelative.word()));
        assert!(!declared.contains(&Capability::GroupNonUniformShuffle.word()));
    }

    #[test]
    fn a_shuffle_keeps_the_vectors_type_and_width() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let float = lanes.type_of::<F32>().expect("f32");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        let shuffled = lanes.butterfly(value, 4).expect("butterfly");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_SHUFFLE_XOR)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(operands[0], float.word());
        assert_eq!(operands[1], shuffled.id().word());
    }
}
