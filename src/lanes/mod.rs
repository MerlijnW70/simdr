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
mod scan;
mod shift;
mod shuffle;
mod uniform;
mod vector;
mod vote;

pub use self::arithmetic::Predicate;
pub use self::dot::{pack, signed_bytes, unsigned_bytes};
pub use self::element::{Element, F32, I32, Integer, Signed, U32};
pub use self::error::LaneError;
pub use self::mapping::Mapping;
pub use self::narrow::{F16, I8, I16, U8, U16};
pub use self::uniform::Uniform;
pub use self::vector::{MAX_STRIPS, Vector};

use crate::module::{Id, Module};
use crate::spec::{BuiltIn, Capability, Scope};

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

    /// A `u32` value's *number*, as a value of `T` — for every `T` a caller of this actually has.
    ///
    /// What a loop counter needs. `Lanes::repeat_rolled` hands its body an iteration number, and
    /// that number is a `u32` whatever the vector's element type is — so a body wanting to add it,
    /// scale by it, or index with it has to convert first. Reinterpreting the bits instead would
    /// turn 7 into a denormal, silently.
    ///
    /// # It is one method and five instructions
    ///
    /// `T::FROM_U32` is not one opcode, and the differences only show at the edges:
    ///
    /// | target | opcode | what it does |
    /// | --- | --- | --- |
    /// | `u32` | `OpCopyObject` | nothing, and the driver folds it away |
    /// | `i32` | `OpBitcast` | **the bits**, so `0xFFFF_FFFF` is −1 rather than 4 294 967 295 |
    /// | `i8`, `i16` | `OpSConvert` | truncate, and the top bit of what is left is a sign |
    /// | `u8`, `u16` | `OpUConvert` | truncate, zero-extended |
    /// | `f32`, `f16` | `OpConvertUToF` | the number, as a float |
    ///
    /// **The `i32` row is why the first sentence has a qualifier on it.** A bitcast and a numeric
    /// conversion agree on every value below `i32::MAX`, and a loop counter is one — so the two
    /// readings are indistinguishable for the caller this exists for, and part company above it,
    /// where there is no `i32` with that number to convert to anyway.
    ///
    /// Measured rather than reasoned, and by something that no longer exists: a sandbox tool ran
    /// all twelve boundary values into all six integer targets on four devices, against a reference
    /// written from the opcode table above rather than from this sentence. It agreed everywhere,
    /// and the `i32` row is the sentence it disagreed with. `notes/FINDINGS.md` has the account.
    ///
    /// **A measurement outlives the thing that took it and a check does not.** The table stands;
    /// nothing runs that sweep today, and this comment says so rather than implying otherwise.
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

    /// Where each lane sits **inside its own vector**: nought to `LANES - 1`, as a vector.
    ///
    /// # The one thing a butterfly network needs and could not ask for
    ///
    /// [`Lanes::butterfly`] hands a lane the value at `l ^ mask`, and every algorithm built on it —
    /// a Walsh–Hadamard or Fourier transform, a bitonic sort, a hand-rolled scan — then has to
    /// decide what to *do* with it, and the decision is always the same one: whether this lane is
    /// the low or the high half of the pair, which is one bit of its own position. Without that the
    /// exchange is symmetric and the algorithm cannot be written at all.
    ///
    /// It was missing until a workload needed it, and what a caller reached for instead was
    /// `local_index & (LANES - 1)`. That is the "cheaper wrong one" [`Lanes::lane_index`] is about:
    /// it is right only where subgroups are cut from consecutive workgroup indices, which Vulkan
    /// guarantees for a pipeline that asked for full subgroups — and `decisions/DR-0002` records
    /// this project deciding *not* to require that extension.
    ///
    /// # It is a position in the vector, not in the subgroup
    ///
    /// Which is the whole difference, and it is the mapping's:
    ///
    /// | mapping | what a lane holds | cost |
    /// | --- | --- | --- |
    /// | [`Mapping::WholeSubgroup`] | its subgroup lane | the built-in |
    /// | [`Mapping::Clusters`] | its lane within its cluster | one `OpBitwiseAnd` |
    /// | [`Mapping::Strips`] | `lane + strip × width`, per strip | one `OpIAdd` per strip past the first |
    ///
    /// A one-lane vector's only position is nought, and that is a constant rather than a mask by
    /// zero — the module says what the arithmetic does.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoMapping`] if `LANES` cannot sit on this subgroup, [`LaneError::Build`] if an
    /// instruction cannot be emitted.
    pub fn position<const LANES: u32>(&mut self) -> Result<Vector<U32, LANES>, LaneError> {
        let strips = match self.mapping::<LANES>()? {
            Mapping::WholeSubgroup => {
                let lane = self.lane_index()?;
                return self.from_strips(&[lane]);
            }
            Mapping::Clusters { size: 1 } => {
                let only = self.module().constant_u32(0)?;
                return self.from_strips(&[only]);
            }
            Mapping::Clusters { size } => {
                let lane = self.lane_index()?;
                let uint = self.type_of::<U32>()?;
                let wrap = self.module().constant_u32(size.wrapping_sub(1))?;
                let within = self
                    .module()
                    .binary(crate::module::op::BITWISE_AND, uint, lane, wrap)?;
                return self.from_strips(&[within]);
            }
            Mapping::Strips { count } => count,
        };

        let lane = self.lane_index()?;
        let uint = self.type_of::<U32>()?;
        let width = self.width();

        let mut ids = Vec::with_capacity(strips as usize);
        for strip in 0..strips {
            // Strip zero is the lane itself, and adding nought to it would be an instruction
            // describing no arithmetic — the same reason `Kernel::address` skips its own zero.
            if strip == 0 {
                ids.push(lane);
                continue;
            }
            let along = self.module().constant_u32(strip.wrapping_mul(width))?;
            ids.push(self.module().i_add(uint, lane, along)?);
        }
        self.from_strips(&ids)
    }

    /// This invocation's index within its subgroup, loaded from `SubgroupLocalInvocationId`.
    ///
    /// **The defined answer, and there is a cheaper wrong one.** A kernel already knows its index
    /// within the *workgroup*, and on the three implementations here `local & (width - 1)` gives
    /// the same number — but only because subgroups happen to be cut from consecutive local
    /// indices, which Vulkan guarantees for a pipeline that asked for full subgroups and not
    /// otherwise. This built-in is defined to be the lane's position, so the mask that keeps a
    /// clustered scan inside its cluster rests on the specification rather than on three devices
    /// agreeing.
    ///
    /// Declared on demand: nothing that does not ask for it pays the `Input` variable, and nothing
    /// that does not ask for it declares `GroupNonUniform` — which a kernel that only scales must
    /// not, or it stops running on devices that would have run it. Asking twice yields one
    /// variable and two loads; the second is `OpLoad` of a value the driver has in a register.
    ///
    /// The type is [`U32`]'s rather than a `type_int(32, false)` written here. The mutation gate
    /// found the `false`: flipping it changes the module and nothing observable, because SPIR-V's
    /// signedness is not what decides how `OpBitwiseAnd` or `OpUGreaterThan` behave. `Kernel` has
    /// the same note over `index_type` for the same reason — a decision written down twice, where
    /// only one copy can ever be load-bearing. This is the lane API's `u32`, so it says so.
    ///
    /// # Errors
    ///
    /// [`LaneError::Build`] if the variable or the load cannot be emitted.
    fn lane_index(&mut self) -> Result<Id, LaneError> {
        self.module()
            .require_capability(Capability::GroupNonUniform)?;
        let uint = self.type_of::<U32>()?;
        let variable = self
            .module()
            .builtin_input(BuiltIn::SubgroupLocalInvocationId, uint)?;
        Ok(self.module().load(uint, variable)?)
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

    /// How many instructions of one opcode a module holds after building nothing but a position.
    fn emitted<const LANES: u32>(width: u32, opcode: u16) -> usize {
        let mut module = module();
        let mut lanes = Lanes::new(&mut module, width).expect("built");
        lanes.position::<LANES>().expect("a mapping");

        let words = module.finish();
        decode::body(&words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    /// Each mapping's position costs what [`Lanes::position`]'s own table says it costs.
    ///
    /// **The mutation gate asked for this one, and it is the right kind of test to be asked for.**
    /// Skipping the add for strip zero changes the *module* and not the answer — `lane + 0` is
    /// `lane`, and every device agrees whether the branch is there or not — so the branch was
    /// "guarded but untested" and a mutant that deleted it survived every check in the repository.
    ///
    /// What it protects is a cost, so a cost is what is counted. The same argument the codebase
    /// already makes at `Kernel::run_start`, where four identical multiplies "made the module say
    /// something the arithmetic does not": a driver folds them, and the module is still wrong about
    /// itself.
    #[test]
    fn a_position_costs_what_its_mapping_says_it_costs() {
        // Strip-mined: one add per strip past the first, and the first is the lane itself.
        assert_eq!(emitted::<64>(32, op::I_ADD), 1, "two strips want one add");
        assert_eq!(emitted::<128>(32, op::I_ADD), 3, "four strips want three");
        assert_eq!(emitted::<32>(8, op::I_ADD), 3, "and four again on a narrow device");

        // A whole subgroup is the built-in and nothing else; a cluster is one mask and no add.
        assert_eq!(emitted::<32>(32, op::I_ADD), 0);
        assert_eq!(emitted::<32>(32, op::BITWISE_AND), 0);
        assert_eq!(emitted::<8>(32, op::I_ADD), 0);
        assert_eq!(emitted::<8>(32, op::BITWISE_AND), 1, "a cluster masks once");

        // And a vector of one is a constant rather than a mask by zero, which would answer the
        // same and describe arithmetic the kernel does not do.
        assert_eq!(emitted::<1>(32, op::BITWISE_AND), 0, "nought is not a mask");
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
