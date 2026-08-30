mod arithmetic;
mod branch;
mod dot;
mod element;
mod error;
mod extremes;
pub(crate) mod loops;
mod mapping;
mod math;
mod narrow;
mod reduce;
mod saturating;
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

pub struct Lanes<'module> {
    module: &'module mut Module,
    width: u32,
    scope: Id,
}

impl<'module> Lanes<'module> {
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

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn module(&mut self) -> &mut Module {
        self.module
    }

    pub fn type_of<T: Element>(&mut self) -> Result<Id, LaneError> {
        Ok(T::type_id(self.module)?)
    }

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

    pub fn from_lane_value<T: Element, const LANES: u32>(
        &self,
        id: Id,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.from_strips(&[id])
    }

    pub fn splat_bits<T: Element, const LANES: u32>(
        &mut self,
        bits: u32,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let id = T::constant_from_bits(self.module, bits)?;
        self.splat_id(id)
    }

    pub fn splat_id<T: Element, const LANES: u32>(
        &self,
        id: Id,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let strips = self.strips_for::<LANES>()?;
        let ids = vec![id; strips];
        self.from_strips(&ids)
    }

    pub fn convert_u32<T: Element>(&mut self, value: Id) -> Result<Id, LaneError> {
        let element = self.type_of::<T>()?;
        Ok(self.module().unary(T::FROM_U32, element, value)?)
    }

    pub(crate) const fn scope(&self) -> Id {
        self.scope
    }

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
                let within =
                    self.module()
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
            if strip == 0 {
                ids.push(lane);
                continue;
            }
            let along = self.module().constant_u32(strip.wrapping_mul(width))?;
            ids.push(self.module().i_add(uint, lane, along)?);
        }
        self.from_strips(&ids)
    }

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

        assert!(lanes.from_lane_value::<F32, 64>(id).is_err());
        assert!(lanes.from_strips::<F32, 64>(&[id, id]).is_ok());
    }

    #[test]
    fn each_element_type_converts_a_u32_with_its_own_instruction() {
        for (expected, emitted) in [
            (op::CONVERT_U_TO_F, converted::<F32>()),
            (op::BITCAST, converted::<I32>()),
            (op::COPY_OBJECT, converted::<U32>()),
        ] {
            assert_eq!(emitted, Some(expected));
        }
    }

    fn emitted<const LANES: u32>(width: u32, opcode: u16) -> usize {
        let mut module = module();
        let mut lanes = Lanes::new(&mut module, width).expect("built");
        lanes.position::<LANES>().expect("a mapping");

        let words = module.finish();
        decode::body(&words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    #[test]
    fn a_position_costs_what_its_mapping_says_it_costs() {
        assert_eq!(emitted::<64>(32, op::I_ADD), 1, "two strips want one add");
        assert_eq!(emitted::<128>(32, op::I_ADD), 3, "four strips want three");
        assert_eq!(
            emitted::<32>(8, op::I_ADD),
            3,
            "and four strips again, reached from the other side — a narrow device"
        );

        assert_eq!(emitted::<32>(32, op::I_ADD), 0);
        assert_eq!(emitted::<32>(32, op::BITWISE_AND), 0);
        assert_eq!(emitted::<8>(32, op::I_ADD), 0);
        assert_eq!(emitted::<8>(32, op::BITWISE_AND), 1, "a cluster masks once");

        assert_eq!(emitted::<1>(32, op::BITWISE_AND), 0, "nought is not a mask");
    }

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
