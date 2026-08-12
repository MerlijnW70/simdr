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
//! There is no clustered shuffle, so [`Lanes::shift_down`] and friends refuse a clustered
//! mapping rather than reading a neighbour's data.
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
    /// # Errors
    ///
    /// As [`Lanes::shift_down`].
    pub fn butterfly<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
        mask: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let mask = self.module().constant_u32(mask)?;
        self.exchange::<T, LANES>(
            "butterfly",
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

    /// One shuffle per strip.
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
    fn a_clustered_shuffle_is_refused_because_the_lanes_belong_to_other_vectors() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 8>(1.0_f32.to_bits())
            .expect("splat");

        assert!(matches!(
            lanes.butterfly(value, 1).err(),
            Some(LaneError::NoSuchForm { .. })
        ));
        assert!(matches!(
            lanes.shift_down(value, 1).err(),
            Some(LaneError::NoSuchForm { .. })
        ));
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
