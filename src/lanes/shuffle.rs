//! Lane exchanges — what `simd_swizzle!` and the rotates become.
//!
//! A shuffle moves values between lanes without combining them, so unlike a reduction it has no
//! shape to choose: one instruction per strip, whatever the mapping.
//!
//! # A shuffle is over the subgroup, not over the vector
//!
//! `OpGroupNonUniformShuffle` reads a lane of the *subgroup*. For a vector as wide as the
//! subgroup those are the same thing. For a narrower one they are not — a `Simd<f32, 8>` on a
//! 32-lane machine has four vectors sharing the lanes, and lane 9 belongs to the second of them.
//! There is no clustered shuffle, so [`Lanes::shift_down`] and [`Lanes::broadcast`] refuse a
//! clustered mapping rather than reading a neighbour's data.
//!
//! [`Lanes::butterfly`] is the exception, and it is arithmetic rather than a special case: a
//! cluster is an aligned run of `LANES` lanes, `LANES` is a power of two, and `l ^ mask` for
//! `mask < LANES` cannot leave it. The refusal used to cover this too, which made the one shuffle
//! a clustered vector can have unreachable — and a hand-rolled tree inside a cluster is built from
//! butterflies.
//!
//! Strip-mined vectors are fine: every strip is a full subgroup's worth, and shuffling each one
//! separately is exactly right.

use super::{Element, LaneError, Lanes, Mapping, Vector};
use crate::spec::Capability;

impl Lanes<'_> {
    /// Read each element from the lane `delta` above this one.
    ///
    /// Lanes near the top read past the end; SPIR-V leaves the result undefined there rather than
    /// wrapping, which is why this is `rotate` only in the loose sense and why a caller that
    /// needs the wrap has to mask it themselves.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchForm`] for a clustered mapping, otherwise [`LaneError::Build`].
    pub fn shift_down<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        delta: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let delta = self.module().constant_u32(delta)?;
        self.exchange::<T, LANES>(
            "shift_down",
            vector,
            Exchange::Down,
            delta,
            Capability::GroupNonUniformShuffleRelative,
        )
    }

    /// Read each element from the lane `delta` below this one.
    ///
    /// # Errors
    ///
    /// As [`Lanes::shift_down`].
    pub fn shift_up<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        delta: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let delta = self.module().constant_u32(delta)?;
        self.exchange::<T, LANES>(
            "shift_up",
            vector,
            Exchange::Up,
            delta,
            Capability::GroupNonUniformShuffleRelative,
        )
    }

    /// Read each element from the lane whose index is this one's exclusive-or `mask`.
    ///
    /// The butterfly: pairs lanes at distance `mask`, which is what a hand-rolled tree reduction
    /// is built from and what `simd_swizzle!` lowers to for the shapes that are not a reduce.
    /// Unlike the shifts, every lane reads a lane that exists, so nothing is undefined.
    ///
    /// **And unlike the shifts, a clustered vector is allowed** — as long as `mask` is inside it.
    /// Clusters are aligned runs of `LANES` lanes and `LANES` is a power of two, so `l ^ mask` for
    /// `mask < LANES` flips only bits below the cluster's own width and cannot leave it. There is
    /// nothing to mask off and no lane to leave undefined; the pairing is exactly the one the
    /// caller asked for, in every one of the vectors sharing the subgroup at once.
    ///
    /// A `mask` at or above `LANES` *would* leave, and is refused by name rather than clamped —
    /// the caller asking for it has a different vector in mind than the one they are holding.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchForm`] if `mask` reaches outside a clustered vector, otherwise as
    /// [`Lanes::shift_down`].
    pub fn butterfly<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        mask: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        if let Mapping::Clusters { size } = self.mapping::<LANES>()?
            && mask >= size
        {
            return Err(LaneError::NoSuchForm {
                operation: "butterfly",
                because: "a mask as wide as the vector pairs it with a lane belonging to the next \
                          vector along",
            });
        }

        let mask = self.module().constant_u32(mask)?;
        self.shuffle::<T, LANES>(
            vector,
            Exchange::Xor,
            mask,
            Capability::GroupNonUniformShuffle,
        )
    }

    /// Give every lane the value held by lane `source`.
    ///
    /// # Errors
    ///
    /// As [`Lanes::shift_down`].
    pub fn broadcast<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        source: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let source = self.module().constant_u32(source)?;
        self.exchange::<T, LANES>(
            "broadcast",
            vector,
            Exchange::Index,
            source,
            Capability::GroupNonUniformShuffle,
        )
    }

    /// [`Lanes::shift_up`] without the refusal, for a caller that means to cross the boundary.
    ///
    /// **One caller, and it is the clustered scan.** The ladder in [`super::reduce`] reads the
    /// lane `distance` below and then masks off every lane whose neighbour belongs to a different
    /// cluster, so the crossing is deliberate and the mask is what undoes it. Every *other* caller
    /// reaching into a neighbouring vector's lanes is a bug, which is why the public form refuses
    /// and this one is private.
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

    /// One shuffle per strip, once the mapping has been checked.
    fn exchange<T: Element, const LANES: u32>(
        &mut self,
        name: &'static str,
        vector: Vector<T, LANES>,
        kind: Exchange,
        operand: crate::module::Id,
        capability: Capability,
    ) -> Result<Vector<T, LANES>, LaneError> {
        if let Mapping::Clusters { .. } = self.mapping::<LANES>()? {
            return Err(LaneError::NoSuchForm {
                operation: name,
                because: "a shuffle reads a lane of the subgroup, and a narrower vector shares \
                          those lanes with other vectors",
            });
        }

        self.shuffle::<T, LANES>(vector, kind, operand, capability)
    }

    /// The instructions themselves, which the mapping does not change: one shuffle per strip.
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

/// Which shuffle instruction an exchange lowers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Exchange {
    Down,
    Up,
    Xor,
    Index,
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
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
        assert!(matches!(
            lanes.broadcast(value, 0).err(),
            Some(LaneError::NoSuchForm { .. })
        ));
    }

    #[test]
    fn a_clustered_butterfly_is_allowed_up_to_the_vectors_own_width() {
        // The arithmetic the exception rests on: a cluster is an aligned run of `LANES` lanes and
        // `LANES` is a power of two, so `l ^ mask` for `mask < LANES` flips only bits below the
        // cluster's width. 7 is the largest mask an eight-lane vector can pair inside itself, and
        // 8 is the first that reaches the vector next door.
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
                    Some(LaneError::NoSuchForm {
                        operation: "butterfly",
                        ..
                    })
                ),
                "a mask of {mask} leaves an eight-lane cluster and must be refused"
            );
        }
    }

    #[test]
    fn a_clustered_butterfly_is_the_same_instruction_as_a_full_width_one() {
        // No mask, no lane index, no extra instruction — the mapping changes what the operand
        // *means* and not what is emitted. A version that quietly clamped the mask, or that added
        // a select for the lanes it thought were outside, would show up here.
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
        // `ShuffleRelative` is its own capability, and a device may offer one without the other.
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
