//! `Simd<T, N>` semantics, lowered onto subgroup lanes.
//!
//! This is the layer the project exists for. Below it, [`crate::module`] emits whatever SPIR-V it
//! is told to; here, a vector of `N` lanes is a *value*, an elementwise operation costs one scalar
//! instruction per strip whatever `N` is, and only the operations that cross lanes need a subgroup
//! instruction.
//!
//! # The width has to be known
//!
//! `N` is fixed when the kernel is written and the subgroup width is fixed by the implementation,
//! and the two only meet at build time: the three rows below are three different *instruction
//! sequences*, and no value arriving later can add instructions that were never emitted.
//! [`Lanes::new`] therefore takes the width, and the caller reads it off the device it is
//! targeting. See `decisions/DR-0002`, and `decisions/DR-0005` for the part of its reasoning that
//! turned out to be too strong — a `ClusterSize` *can* be deferred to pipeline creation; a choice
//! of mapping cannot.
//!
//! # Three ways a vector sits on a subgroup
//!
//! | `N` vs width | mapping | cost of a reduction |
//! | --- | --- | --- |
//! | equal | [`Mapping::WholeSubgroup`] | one subgroup instruction |
//! | a divisor | [`Mapping::Clusters`] | one clustered instruction, several vectors at once |
//! | a multiple | [`Mapping::Strips`] | `strips - 1` scalar ops, then one subgroup instruction |
//!
//! Anything else — 12 lanes on a 32-wide subgroup — has no mapping and is refused by name.
//!
//! # Branches
//!
//! A branch takes a [`Uniform`], which only a vote produces. `decisions/DR-0003` has the argument;
//! the short version is that a subgroup instruction inside a divergent branch answers for
//! whichever lanes happen to be running.

mod arithmetic;
mod branch;
mod dot;
mod element;
mod error;
mod extremes;
mod loops;
mod mapping;
mod math;
mod narrow;
mod reduce;
mod shift;
mod shuffle;
mod uniform;
mod vector;
mod vote;

pub use self::arithmetic::Predicate;
pub use self::dot::{pack, signed_bytes, unsigned_bytes};
pub use self::element::{Element, F32, I32, Signed, U32};
pub use self::error::LaneError;
pub use self::mapping::Mapping;
pub use self::narrow::{F16, I8, I16, U8, U16};
pub use self::uniform::Uniform;
pub use self::vector::{MAX_STRIPS, Vector};

use crate::module::{Id, Module};
use crate::spec::Scope;

/// Builds lane operations into a module, for a subgroup of a known width.
pub struct Lanes<'module> {
    module: &'module mut Module,
    width: u32,
    scope: Id,
}

impl<'module> Lanes<'module> {
    /// Start building for a subgroup of `width` lanes.
    ///
    /// `width` comes from the device — `runner`'s `Gpu::limits().subgroup_size`, or
    /// `VkPhysicalDeviceSubgroupProperties` directly. DR-0002 is why it cannot be discovered
    /// later.
    ///
    /// # Errors
    ///
    /// [`LaneError::BadWidth`] if `width` is not a power of two, [`LaneError::Build`] if the
    /// scope constant cannot be declared.
    pub fn new(module: &'module mut Module, width: u32) -> Result<Self, LaneError> {
        if width == 0 || !width.is_power_of_two() {
            return Err(LaneError::BadWidth { width });
        }

        let scope = module.scope(Scope::Subgroup)?;
        Ok(Self {
            module,
            width,
            scope,
        })
    }

    /// The subgroup width this is building for.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The module underneath, for anything this layer does not cover.
    pub const fn module(&mut self) -> &mut Module {
        self.module
    }

    /// The SPIR-V type of `T`.
    ///
    /// # Errors
    ///
    /// [`LaneError::Build`] if the type cannot be declared.
    pub fn type_of<T: Element>(&mut self) -> Result<Id, LaneError> {
        Ok(T::type_id(self.module)?)
    }

    /// Adopt per-lane values as a vector, one id per strip in strip order.
    ///
    /// No instruction: a value that differs per invocation is *already* one element per lane.
    /// That identity is why an elementwise operation costs one scalar instruction per strip
    /// rather than `LANES` of anything.
    ///
    /// # Errors
    ///
    /// [`LaneError::TooManyStrips`] if `ids` does not have exactly as many entries as the mapping
    /// calls for, [`LaneError::NoMapping`] if there is no mapping at all.
    pub fn from_strips<T: Element, const LANES: u32>(
        &self,
        ids: &[Id],
    ) -> Result<Vector<T, LANES>, LaneError> {
        let wanted = self.strips_for::<LANES>()?;
        if ids.len() != wanted {
            return Err(LaneError::TooManyStrips {
                strips: ids.len(),
                limit: wanted,
            });
        }

        Vector::from_strips(ids).ok_or(LaneError::TooManyStrips {
            strips: ids.len(),
            limit: MAX_STRIPS,
        })
    }

    /// Adopt a single per-lane value as a vector.
    ///
    /// Only valid where the mapping gives one element per lane; a strip-mined vector needs
    /// [`Lanes::from_strips`].
    ///
    /// # Errors
    ///
    /// As [`Lanes::from_strips`].
    pub fn from_lane_value<T: Element, const LANES: u32>(
        &self,
        id: Id,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.from_strips(&[id])
    }

    /// A vector holding `value` in every lane and every strip.
    ///
    /// The bits are the caller's to prepare — `1.5_f32.to_bits()`, `7_u32`, or
    /// `u32::from_ne_bytes((-7_i32).to_ne_bytes())`. One signature for all three element types,
    /// because the standard library has no numeric trait that would cover them.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the mapping does not exist or the constant cannot be declared.
    pub fn splat_bits<T: Element, const LANES: u32>(
        &mut self,
        bits: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let id = T::constant_from_bits(self.module, bits)?;
        self.splat_id(id)
    }

    /// A vector holding a value that already exists, in every lane and every strip.
    ///
    /// What a *specialization* constant needs: its value arrives at pipeline creation, so there
    /// are no bits here to hand to [`Lanes::splat_bits`] — there is an id, and it names something
    /// uniform across every invocation. Also the way to lift any other uniform value, such as one
    /// a broadcast produced.
    ///
    /// The caller vouches that `id` has type `T`. Nothing here can check it: an id carries no
    /// type at this layer, and a mismatch is a validation failure rather than a wrong number.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoMapping`] if `LANES` cannot sit on this subgroup.
    pub fn splat_id<T: Element, const LANES: u32>(
        &self,
        id: Id,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let strips = self.strips_for::<LANES>()?;
        // Every strip is the same id: a uniform value does not vary per lane, and it does not
        // vary per strip either.
        let ids = vec![id; strips];
        self.from_strips(&ids)
    }

    /// A `u32` value's *number*, as a value of `T`.
    ///
    /// What a loop counter needs. `Lanes::repeat_rolled` hands its body an iteration number, and
    /// that number is a `u32` whatever the vector's element type is — so a body wanting to add it,
    /// scale by it, or index with it has to convert first. Reinterpreting the bits instead would
    /// turn 7 into a denormal, silently.
    ///
    /// Costs nothing when `T` is already `u32`: the instruction is `OpCopyObject`, which the driver
    /// folds away, and keeping the shape uniform is worth more than the special case.
    ///
    /// # Errors
    ///
    /// [`LaneError::Build`] if the instruction cannot be emitted.
    pub fn convert_u32<T: Element>(&mut self, value: Id) -> Result<Id, LaneError> {
        let element = self.type_of::<T>()?;
        Ok(self.module().unary(T::FROM_U32, element, value)?)
    }

    /// The scope constant every subgroup instruction here uses.
    pub(crate) const fn scope(&self) -> Id {
        self.scope
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::decode;
    use crate::module::{Version, op};

    fn module() -> Module {
        Module::new(Version::V1_3)
    }

    #[test]
    fn a_width_that_is_not_a_power_of_two_did_not_come_from_a_device() {
        let mut module = module();

        assert_eq!(
            Lanes::new(&mut module, 24).err(),
            Some(LaneError::BadWidth { width: 24 })
        );
    }

    #[test]
    fn a_width_of_zero_is_refused() {
        let mut module = module();

        assert_eq!(
            Lanes::new(&mut module, 0).err(),
            Some(LaneError::BadWidth { width: 0 })
        );
    }

    #[test]
    fn every_width_an_implementation_reports_is_accepted() {
        // 32 on an NVIDIA part, 64 on an AMD one, 8 on Mesa's lavapipe — all measured here. 4 and
        // 16 are in the list because they are powers of two an implementation may report and this
        // layer has no reason to single them out; nothing here has run at either.
        for width in [4, 8, 16, 32, 64] {
            let mut module = module();
            assert_eq!(
                Lanes::new(&mut module, width)
                    .expect("a power of two")
                    .width(),
                width
            );
        }
    }

    #[test]
    fn a_splat_repeats_the_same_constant_across_every_strip() {
        let mut module = module();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");

        let wide = lanes
            .splat_bits::<F32, 128>(1.5_f32.to_bits())
            .expect("splat");

        assert_eq!(wide.strip_count(), 4);
        let strips = wide.strips().to_vec();
        assert!(
            strips.windows(2).all(|pair| pair.first() == pair.last()),
            "a constant does not vary per strip, and dedup makes that literal"
        );
    }

    #[test]
    fn adopting_the_wrong_number_of_strips_is_refused() {
        let mut module = module();
        let lanes = Lanes::new(&mut module, 32).expect("built");
        let id = lanes.scope();

        // 64 lanes on a 32-wide subgroup needs two ids, not one.
        assert!(lanes.from_lane_value::<F32, 64>(id).is_err());
        assert!(lanes.from_strips::<F32, 64>(&[id, id]).is_ok());
    }

    #[test]
    fn each_element_type_converts_a_u32_with_its_own_instruction() {
        // Three types, three instructions, and the float one must be a *conversion*: `OpBitcast`
        // to a float would make 7 a denormal near zero, which is a wrong answer that looks like a
        // numerical problem rather than a wrong opcode.
        for (expected, emitted) in [
            (op::CONVERT_U_TO_F, converted::<F32>()),
            (op::BITCAST, converted::<I32>()),
            (op::COPY_OBJECT, converted::<U32>()),
        ] {
            assert_eq!(emitted, Some(expected));
        }
    }

    /// The opcode `convert_u32::<T>` emits, in a module holding nothing else that could be it.
    fn converted<T: Element>() -> Option<u16> {
        let mut module = module();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let counter = lanes.module().constant_u32(7).expect("7");
        lanes.convert_u32::<T>(counter).expect("converted");

        let words = module.finish();
        decode::body(&words)
            .map(|instruction| instruction.opcode())
            .find(|opcode| matches!(*opcode, op::CONVERT_U_TO_F | op::BITCAST | op::COPY_OBJECT))
    }

    #[test]
    fn a_converted_counter_can_be_added_to_a_vector_of_its_new_type() {
        // The motivating use, spelled out: the counter arrives as a `u32` and has to join float
        // arithmetic without a cast at the call site.
        let mut module = module();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");

        let counter = lanes.module().constant_u32(3).expect("3");
        let as_float = lanes.convert_u32::<F32>(counter).expect("converted");
        let vector = lanes.from_lane_value::<F32, 32>(as_float).expect("adopted");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        assert!(lanes.add(vector, one).is_ok());
    }

    #[test]
    fn each_element_type_declares_its_own() {
        let mut module = module();
        let mut lanes = Lanes::new(&mut module, 32).expect("built");

        let float = lanes.type_of::<F32>().expect("f32");
        let signed = lanes.type_of::<I32>().expect("i32");
        let unsigned = lanes.type_of::<U32>().expect("u32");

        assert_ne!(float, signed);
        assert_ne!(signed, unsigned);
    }
}
