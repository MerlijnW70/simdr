use crate::module::{BuildError, Id, Module, op};
use crate::spec::Glsl;

pub trait Element: Copy + core::fmt::Debug + 'static {
    const NAME: &'static str;

    const STRIDE: u32;

    const ADD: u16;
    const SUB: u16;
    const MUL: u16;
    const DIV: u16;

    /// The comparisons are the ordered family throughout: a NaN operand answers
    /// false to every one of them, `NOT_EQUAL` included, so `not_equal` is not
    /// the negation of `equal` where NaN reaches it.
    const GREATER_THAN: u16;
    const GREATER_THAN_EQUAL: u16;
    const LESS_THAN: u16;
    const LESS_THAN_EQUAL: u16;
    const EQUAL: u16;
    const NOT_EQUAL: u16;

    const GROUP_ADD: u16;
    const GROUP_MUL: u16;
    const GROUP_MAX: u16;
    const GROUP_MIN: u16;

    const MIN: Glsl;
    const MAX: Glsl;
    const CLAMP: Glsl;

    const FROM_U32: u16;

    fn type_id(module: &mut Module) -> Result<Id, BuildError>;

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError>;

    fn require_in_storage_buffer(module: &mut Module) -> Result<(), BuildError> {
        let _ = module;
        Ok(())
    }
}

pub trait Signed: Element {
    const ABS: Glsl;
    const NEGATE: u16;
}

pub trait Integer: Element {
    /// Whether the top bit is a sign. It decides which sequence saturation
    /// takes: an unsigned one clamps with a complement, a signed one has to
    /// know which end it overflowed towards.
    const SIGNED: bool;

    /// The width of the type in bits, from the stride it occupies.
    const BITS: u32 = Self::STRIDE * 8;
}

impl Integer for I32 {
    const SIGNED: bool = true;
}

impl Integer for U32 {
    const SIGNED: bool = false;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32;

impl Element for F32 {
    const NAME: &'static str = "f32";
    const STRIDE: u32 = 4;
    const ADD: u16 = op::F_ADD;
    const SUB: u16 = op::F_SUB;
    const MUL: u16 = op::F_MUL;
    const DIV: u16 = op::F_DIV;
    const GREATER_THAN: u16 = op::F_ORD_GREATER_THAN;
    const GREATER_THAN_EQUAL: u16 = op::F_ORD_GREATER_THAN_EQUAL;
    const LESS_THAN: u16 = op::F_ORD_LESS_THAN;
    const LESS_THAN_EQUAL: u16 = op::F_ORD_LESS_THAN_EQUAL;
    const EQUAL: u16 = op::F_ORD_EQUAL;
    const NOT_EQUAL: u16 = op::F_ORD_NOT_EQUAL;
    const GROUP_ADD: u16 = op::GROUP_NON_UNIFORM_F_ADD;
    const GROUP_MUL: u16 = op::GROUP_NON_UNIFORM_F_MUL;
    const GROUP_MAX: u16 = op::GROUP_NON_UNIFORM_F_MAX;
    const GROUP_MIN: u16 = op::GROUP_NON_UNIFORM_F_MIN;
    const MIN: Glsl = Glsl::FMin;
    const MAX: Glsl = Glsl::FMax;
    const CLAMP: Glsl = Glsl::FClamp;
    const FROM_U32: u16 = op::CONVERT_U_TO_F;

    fn type_id(module: &mut Module) -> Result<Id, BuildError> {
        module.type_float(32)
    }

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError> {
        module.constant_f32(f32::from_bits(bits))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I32;

impl Element for I32 {
    const NAME: &'static str = "i32";
    const STRIDE: u32 = 4;
    const ADD: u16 = op::I_ADD;
    const SUB: u16 = op::I_SUB;
    const MUL: u16 = op::I_MUL;
    const DIV: u16 = op::S_DIV;
    const GREATER_THAN: u16 = op::S_GREATER_THAN;
    const GREATER_THAN_EQUAL: u16 = op::S_GREATER_THAN_EQUAL;
    const LESS_THAN: u16 = op::S_LESS_THAN;
    const LESS_THAN_EQUAL: u16 = op::S_LESS_THAN_EQUAL;
    const EQUAL: u16 = op::I_EQUAL;
    const NOT_EQUAL: u16 = op::I_NOT_EQUAL;
    const GROUP_ADD: u16 = op::GROUP_NON_UNIFORM_I_ADD;
    const GROUP_MUL: u16 = op::GROUP_NON_UNIFORM_I_MUL;
    const GROUP_MAX: u16 = op::GROUP_NON_UNIFORM_S_MAX;
    const GROUP_MIN: u16 = op::GROUP_NON_UNIFORM_S_MIN;
    const MIN: Glsl = Glsl::SMin;
    const MAX: Glsl = Glsl::SMax;
    const CLAMP: Glsl = Glsl::SClamp;
    const FROM_U32: u16 = op::BITCAST;

    fn type_id(module: &mut Module) -> Result<Id, BuildError> {
        module.type_int(32, true)
    }

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError> {
        module.constant_i32(i32::from_ne_bytes(bits.to_ne_bytes()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct U32;

impl Element for U32 {
    const NAME: &'static str = "u32";
    const STRIDE: u32 = 4;
    const ADD: u16 = op::I_ADD;
    const SUB: u16 = op::I_SUB;
    const MUL: u16 = op::I_MUL;
    const DIV: u16 = op::U_DIV;
    const GREATER_THAN: u16 = op::U_GREATER_THAN;
    const GREATER_THAN_EQUAL: u16 = op::U_GREATER_THAN_EQUAL;
    const LESS_THAN: u16 = op::U_LESS_THAN;
    const LESS_THAN_EQUAL: u16 = op::U_LESS_THAN_EQUAL;
    const EQUAL: u16 = op::I_EQUAL;
    const NOT_EQUAL: u16 = op::I_NOT_EQUAL;
    const GROUP_ADD: u16 = op::GROUP_NON_UNIFORM_I_ADD;
    const GROUP_MUL: u16 = op::GROUP_NON_UNIFORM_I_MUL;
    const GROUP_MAX: u16 = op::GROUP_NON_UNIFORM_U_MAX;
    const GROUP_MIN: u16 = op::GROUP_NON_UNIFORM_U_MIN;
    const MIN: Glsl = Glsl::UMin;
    const MAX: Glsl = Glsl::UMax;
    const CLAMP: Glsl = Glsl::UClamp;
    const FROM_U32: u16 = op::COPY_OBJECT;

    fn type_id(module: &mut Module) -> Result<Id, BuildError> {
        module.type_int(32, false)
    }

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError> {
        module.constant_u32(bits)
    }
}

impl Signed for F32 {
    const ABS: Glsl = Glsl::FAbs;
    const NEGATE: u16 = op::F_NEGATE;
}

impl Signed for I32 {
    const ABS: Glsl = Glsl::SAbs;
    const NEGATE: u16 = op::S_NEGATE;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::module::Version;

    #[test]
    fn the_two_integers_share_their_add_and_differ_at_max() {
        assert_eq!(I32::ADD, U32::ADD);
        assert_eq!(I32::MUL, U32::MUL);
        assert_ne!(I32::GROUP_MAX, U32::GROUP_MAX);
        assert_ne!(I32::GREATER_THAN, U32::GREATER_THAN);
    }

    #[test]
    fn floats_share_no_opcode_with_the_integers() {
        for integer in [I32::ADD, I32::MUL, I32::GROUP_ADD] {
            assert_ne!(F32::ADD, integer);
            assert_ne!(F32::MUL, integer);
            assert_ne!(F32::GROUP_ADD, integer);
        }
    }

    #[test]
    fn each_element_declares_its_own_type() {
        let mut module = Module::new(Version::V1_3);

        let float = F32::type_id(&mut module).expect("f32");
        let signed = I32::type_id(&mut module).expect("i32");
        let unsigned = U32::type_id(&mut module).expect("u32");

        assert_ne!(float, signed);
        assert_ne!(signed, unsigned, "signedness makes two types of one width");
    }

    #[test]
    fn a_constant_round_trips_through_its_bits() {
        let mut module = Module::new(Version::V1_3);

        let from_bits = F32::constant_from_bits(&mut module, 1.5_f32.to_bits()).expect("1.5");
        let directly = module.constant_f32(1.5).expect("1.5 again");

        assert_eq!(
            from_bits, directly,
            "the bits route reaches the same deduplicated constant"
        );
    }

    #[test]
    fn a_negative_signed_constant_survives_the_bit_route() {
        let mut module = Module::new(Version::V1_3);

        let bits = u32::from_ne_bytes((-7_i32).to_ne_bytes());
        let reinterpreted = I32::constant_from_bits(&mut module, bits).expect("-7");
        let directly = module.constant_i32(-7).expect("-7 again");

        assert_eq!(reinterpreted, directly);
    }

    #[test]
    fn each_element_names_itself() {
        assert_eq!(F32::NAME, "f32");
        assert_eq!(I32::NAME, "i32");
        assert_eq!(U32::NAME, "u32");
    }

    #[test]
    fn the_three_families_of_min_and_max_are_three_different_instructions() {
        for pair in [
            [F32::MIN, I32::MIN],
            [I32::MIN, U32::MIN],
            [F32::MAX, I32::MAX],
            [I32::MAX, U32::MAX],
            [F32::CLAMP, I32::CLAMP],
            [I32::CLAMP, U32::CLAMP],
        ] {
            assert_ne!(pair[0], pair[1]);
        }
    }

    #[test]
    fn min_and_max_are_never_the_same_instruction_within_a_type() {
        assert_ne!(F32::MIN, F32::MAX);
        assert_ne!(I32::MIN, I32::MAX);
        assert_ne!(U32::MIN, U32::MAX);
    }

    #[test]
    fn only_the_signed_types_have_a_magnitude() {
        assert_ne!(F32::ABS, I32::ABS);
        assert_eq!(F32::ABS.operands(), 1);
    }
}
