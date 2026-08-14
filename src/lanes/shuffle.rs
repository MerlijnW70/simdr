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
//! There is no clustered shuffle, and what follows from that is **not** that a narrower vector
//! cannot be shuffled. It is that each of these four has to be asked separately.
//!
//! | | a clustered vector | why |
//! | --- | --- | --- |
//! | [`Lanes::butterfly`] | yes, for `mask < LANES` | a cluster is an aligned run of a power-of-two size, so `l ^ mask` cannot leave it |
//! | [`Lanes::broadcast`] | yes, for `source < LANES` | the lane to read is `(l & !(LANES - 1)) + source`, and `OpGroupNonUniformShuffle` takes a dynamic id |
//! | [`Lanes::rotate_up`] | yes, for any `delta` | every lane reads a lane inside its own vector, so there is no edge at all |
//! | [`Lanes::shift_up`] | no | it really does read the vector next door, and the hardware would hand it over without a word |
//! | [`Lanes::shift_down`] | no | the same |
//!
//! The first three were refused as well until the refusal was read rather than trusted, which left
//! the mapping that exists to run four small vectors at once unable to swizzle any of them.
//!
//! **The shifts stay refused, and the rotate is why that is a decision rather than a gap.** What a
//! cluster's edge should mean has two bad answers — call it undefined, which promises less than the
//! hardware does and leaves a caller with a value it cannot use; or mask it to something, which
//! invents a semantics SPIR-V does not have and pays for it in every call. The third answer is that
//! the operation a caller wants at an edge is the one that has none.
//!
//! Strip-mined vectors are fine: every strip is a full subgroup's worth, and shuffling each one
//! separately is exactly right.

use super::{Element, LaneError, Lanes, Mapping, U32, Vector};
use crate::module::op;
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
    /// **A clustered vector is allowed, and costs two instructions for it.** `source` is a position
    /// in the *vector*, so for a narrower one the subgroup lane to read differs per invocation:
    /// this cluster's first lane, plus `source`. `OpGroupNonUniformShuffle` takes a **dynamic** id
    /// — unlike `OpGroupNonUniformBroadcast`, which requires a dynamically uniform one and is why
    /// this was ever emitted as a shuffle — so the whole difference is an `OpBitwiseAnd` to round
    /// the lane down to its cluster and an `OpIAdd`.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchForm`] if `source` is outside a clustered vector, otherwise as
    /// [`Lanes::shift_down`].
    pub fn broadcast<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        source: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        if let Mapping::Clusters { size } = self.mapping::<LANES>()? {
            return self.broadcast_within_cluster::<T, LANES>(vector, size, source);
        }

        let source = self.module().constant_u32(source)?;
        self.exchange::<T, LANES>(
            "broadcast",
            vector,
            Exchange::Index,
            source,
            Capability::GroupNonUniformShuffle,
        )
    }

    /// Read each element from the lane `delta` below this one, **wrapping inside the vector**.
    ///
    /// What `Simd::rotate_elements_right` means, and the answer to a question this API had left
    /// open. [`Lanes::shift_up`] leaves the bottom `delta` lanes undefined — SPIR-V says so — and
    /// refuses a clustered vector, because there the lanes it would read are another vector's and
    /// the hardware would hand them over without a word. A rotate has neither problem: **every lane
    /// reads a lane inside its own vector**, so there is no edge to define and nothing to mask.
    ///
    /// The source lane is `(l & !(size - 1)) | ((l + size - delta) & (size - 1))` — the vector's
    /// first lane, plus the position `delta` back from this one, wrapped. `size` is a power of two,
    /// which is what makes both halves a mask; and for a subgroup-wide vector the first half is
    /// zero and it collapses to the wrap alone.
    ///
    /// `delta` is reduced modulo the vector's width rather than refused: a rotate by the width is
    /// the identity, and by `width + 1` is a rotate by one. A rotate by zero emits **nothing**.
    ///
    /// The other direction is this one by `LANES - delta`, and is not a second method: a caller
    /// knows `LANES`, it is in the type, and a method with no caller is a method nothing verifies.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchForm`] for a strip-mined vector, where a rotate moves elements *between*
    /// strips — a shuffle per strip plus a rotation of the strips themselves, which is a different
    /// algorithm rather than a different operand. Otherwise [`LaneError::Build`].
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

        // Nothing to do, and nothing emitted: a rotate by a multiple of the width is the identity,
        // and a module that said otherwise would describe work the kernel does not do.
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

    /// [`Lanes::broadcast`] for a vector narrower than the subgroup.
    ///
    /// The lane read is `(lane & !(size - 1)) + source` — this cluster's first lane plus the
    /// position asked for. `size` is a power of two, which is what makes the rounding a mask.
    ///
    /// **A `size` of one is answered without emitting any of that.** The only `source` it accepts
    /// is 0, so every lane would read itself — three instructions computing the value they were
    /// given. The right answer is the vector, and a module that says so is the module that matches
    /// the kernel.
    fn broadcast_within_cluster<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        size: u32,
        source: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        if source >= size {
            return Err(LaneError::NoSuchForm {
                operation: "broadcast",
                because: "the lane named is past the end of this vector, and holds an element of \
                          the next one along",
            });
        }
        if size == 1 {
            return Ok(vector);
        }

        let lane = self.lane_index()?;
        // [`U32`]'s type rather than a second `type_int(32, false)`; see [`Lanes::lane_index`].
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
    fn a_rotate_wraps_inside_the_vector_and_leaves_no_lane_undefined() {
        // Four scalar instructions and one shuffle: round the lane down to its vector, step back,
        // wrap, put the two halves together. Every lane reads a lane of its own vector, which is
        // the whole difference from a shift.
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
        // The identity, and the module says so by containing none of it. `delta` is reduced rather
        // than refused, so a rotate by 8, 16 or 24 of an eight-lane vector is the same nothing.
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
        // It would have to move elements *between* strips, which is a shuffle per strip and a
        // rotation of the strips as well — a different algorithm, not a different operand.
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
        // The two that genuinely cross. Both were once four, and reading the refusal rather than
        // trusting it is what left these two here.
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
        // `source` names a position in the *vector*, so the subgroup lane to read differs per
        // invocation: this cluster's first, plus the source. That is one mask and one add more
        // than the full-width form, and the id operand is no longer a constant.
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

        // The mask is `!(8 - 1)`, which is the whole of the "clusters are aligned" argument. A
        // version that masked with `size - 1` instead would read the *offset* within the cluster
        // and broadcast a different lane to every one of them.
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
        // Every lane would read itself: a mask, an add and a shuffle computing the value they were
        // handed. The answer is the vector, and the module should say that by containing none of
        // them — the same call the clustered scan makes at the same width.
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
            Some(LaneError::NoSuchForm {
                operation: "broadcast",
                ..
            })
        ));
    }

    #[test]
    fn a_full_width_broadcast_still_names_a_constant_lane() {
        // The clustered form costs two instructions, and the full-width one must not pay them.
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
