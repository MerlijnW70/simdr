use super::element::{Element, Integer, Signed};
use crate::module::{BuildError, Id, Module, op};
use crate::spec::{Capability, Glsl};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I8;

impl Element for I8 {
    const NAME: &'static str = "i8";
    const STRIDE: u32 = 1;
    const ADD: u16 = op::I_ADD;
    const MUL: u16 = op::I_MUL;
    const GREATER_THAN: u16 = op::S_GREATER_THAN;
    const EQUAL: u16 = op::I_EQUAL;
    const GROUP_ADD: u16 = op::GROUP_NON_UNIFORM_I_ADD;
    const GROUP_MAX: u16 = op::GROUP_NON_UNIFORM_S_MAX;
    const GROUP_MIN: u16 = op::GROUP_NON_UNIFORM_S_MIN;
    const MIN: Glsl = Glsl::SMin;
    const MAX: Glsl = Glsl::SMax;
    const CLAMP: Glsl = Glsl::SClamp;
    const FROM_U32: u16 = op::S_CONVERT;

    fn type_id(module: &mut Module) -> Result<Id, BuildError> {
        module.require_capability(Capability::Int8)?;
        module.type_int(8, true)
    }

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError> {
        let of_type = Self::type_id(module)?;
        module.constant_scalar(of_type, sign_extend(bits, 8))
    }

    fn require_in_storage_buffer(module: &mut Module) -> Result<(), BuildError> {
        module.require_capability(Capability::StorageBuffer8BitAccess)
    }
}

impl Signed for I8 {
    const ABS: Glsl = Glsl::SAbs;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct U8;

impl Element for U8 {
    const NAME: &'static str = "u8";
    const STRIDE: u32 = 1;
    const ADD: u16 = op::I_ADD;
    const MUL: u16 = op::I_MUL;
    const GREATER_THAN: u16 = op::U_GREATER_THAN;
    const EQUAL: u16 = op::I_EQUAL;
    const GROUP_ADD: u16 = op::GROUP_NON_UNIFORM_I_ADD;
    const GROUP_MAX: u16 = op::GROUP_NON_UNIFORM_U_MAX;
    const GROUP_MIN: u16 = op::GROUP_NON_UNIFORM_U_MIN;
    const MIN: Glsl = Glsl::UMin;
    const MAX: Glsl = Glsl::UMax;
    const CLAMP: Glsl = Glsl::UClamp;
    const FROM_U32: u16 = op::U_CONVERT;

    fn type_id(module: &mut Module) -> Result<Id, BuildError> {
        module.require_capability(Capability::Int8)?;
        module.type_int(8, false)
    }

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError> {
        let of_type = Self::type_id(module)?;
        module.constant_scalar(of_type, bits & 0xff)
    }

    fn require_in_storage_buffer(module: &mut Module) -> Result<(), BuildError> {
        module.require_capability(Capability::StorageBuffer8BitAccess)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I16;

impl Element for I16 {
    const NAME: &'static str = "i16";
    const STRIDE: u32 = 2;
    const ADD: u16 = op::I_ADD;
    const MUL: u16 = op::I_MUL;
    const GREATER_THAN: u16 = op::S_GREATER_THAN;
    const EQUAL: u16 = op::I_EQUAL;
    const GROUP_ADD: u16 = op::GROUP_NON_UNIFORM_I_ADD;
    const GROUP_MAX: u16 = op::GROUP_NON_UNIFORM_S_MAX;
    const GROUP_MIN: u16 = op::GROUP_NON_UNIFORM_S_MIN;
    const MIN: Glsl = Glsl::SMin;
    const MAX: Glsl = Glsl::SMax;
    const CLAMP: Glsl = Glsl::SClamp;
    const FROM_U32: u16 = op::S_CONVERT;

    fn type_id(module: &mut Module) -> Result<Id, BuildError> {
        module.require_capability(Capability::Int16)?;
        module.type_int(16, true)
    }

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError> {
        let of_type = Self::type_id(module)?;
        module.constant_scalar(of_type, sign_extend(bits, 16))
    }

    fn require_in_storage_buffer(module: &mut Module) -> Result<(), BuildError> {
        module.require_capability(Capability::StorageBuffer16BitAccess)
    }
}

impl Signed for I16 {
    const ABS: Glsl = Glsl::SAbs;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct U16;

impl Element for U16 {
    const NAME: &'static str = "u16";
    const STRIDE: u32 = 2;
    const ADD: u16 = op::I_ADD;
    const MUL: u16 = op::I_MUL;
    const GREATER_THAN: u16 = op::U_GREATER_THAN;
    const EQUAL: u16 = op::I_EQUAL;
    const GROUP_ADD: u16 = op::GROUP_NON_UNIFORM_I_ADD;
    const GROUP_MAX: u16 = op::GROUP_NON_UNIFORM_U_MAX;
    const GROUP_MIN: u16 = op::GROUP_NON_UNIFORM_U_MIN;
    const MIN: Glsl = Glsl::UMin;
    const MAX: Glsl = Glsl::UMax;
    const CLAMP: Glsl = Glsl::UClamp;
    const FROM_U32: u16 = op::U_CONVERT;

    fn type_id(module: &mut Module) -> Result<Id, BuildError> {
        module.require_capability(Capability::Int16)?;
        module.type_int(16, false)
    }

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError> {
        let of_type = Self::type_id(module)?;
        module.constant_scalar(of_type, bits & 0xffff)
    }

    fn require_in_storage_buffer(module: &mut Module) -> Result<(), BuildError> {
        module.require_capability(Capability::StorageBuffer16BitAccess)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F16;

impl Element for F16 {
    const NAME: &'static str = "f16";
    const STRIDE: u32 = 2;
    const ADD: u16 = op::F_ADD;
    const MUL: u16 = op::F_MUL;
    const GREATER_THAN: u16 = op::F_ORD_GREATER_THAN;
    const EQUAL: u16 = op::F_ORD_EQUAL;
    const GROUP_ADD: u16 = op::GROUP_NON_UNIFORM_F_ADD;
    const GROUP_MAX: u16 = op::GROUP_NON_UNIFORM_F_MAX;
    const GROUP_MIN: u16 = op::GROUP_NON_UNIFORM_F_MIN;
    const MIN: Glsl = Glsl::FMin;
    const MAX: Glsl = Glsl::FMax;
    const CLAMP: Glsl = Glsl::FClamp;
    const FROM_U32: u16 = op::CONVERT_U_TO_F;

    fn type_id(module: &mut Module) -> Result<Id, BuildError> {
        module.require_capability(Capability::Float16)?;
        module.type_float(16)
    }

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError> {
        let of_type = Self::type_id(module)?;
        module.constant_scalar(of_type, bits & 0xffff)
    }

    fn require_in_storage_buffer(module: &mut Module) -> Result<(), BuildError> {
        module.require_capability(Capability::StorageBuffer16BitAccess)
    }
}

impl Signed for F16 {
    const ABS: Glsl = Glsl::FAbs;
}

fn sign_extend(bits: u32, width: u32) -> u32 {
    let spare = 32 - width.clamp(1, 32);
    (((bits << spare) as i32) >> spare) as u32
}

impl Integer for I8 {}
impl Integer for U8 {}
impl Integer for I16 {}
impl Integer for U16 {}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::half;
    use crate::module::Version;

    fn module() -> Module {
        Module::new(Version::V1_3)
    }

    fn constant_literal(words: &[u32]) -> u32 {
        let operands = decode::body(words)
            .find(|instruction| instruction.opcode() == op::CONSTANT)
            .expect("a constant was declared")
            .operands()
            .to_vec();
        operands[2]
    }

    #[test]
    fn a_negative_narrow_constant_is_sign_extended_to_a_full_word() {
        let mut module = module();
        I8::constant_from_bits(&mut module, 0xff).expect("-1i8");

        assert_eq!(constant_literal(&module.finish()), 0xffff_ffff);
    }

    #[test]
    fn a_narrow_constant_keeps_the_low_bits_and_does_not_set_them() {
        let mut module = module();
        U8::constant_from_bits(&mut module, 0x1234).expect("a byte");

        assert_eq!(constant_literal(&module.finish()), 0x34);
    }

    #[test]
    fn a_sixteen_bit_constant_keeps_its_low_half_and_does_not_set_it() {
        let mut module = module();
        U16::constant_from_bits(&mut module, 0x1234_5678).expect("a half word");

        assert_eq!(constant_literal(&module.finish()), 0x5678);
    }

    #[test]
    fn sign_extending_by_a_width_no_caller_passes_still_returns_rather_than_panicking() {
        assert_eq!(sign_extend(0xff, 0), 0xffff_ffff, "one bit, all sign");
        assert_eq!(sign_extend(0x7f, 0), 0xffff_ffff, "still just the low bit");

        assert_eq!(sign_extend(0x8000_0000, 33), 0x8000_0000, "clamped to 32");
        assert_eq!(sign_extend(0x1234_5678, 33), 0x1234_5678);
        assert_eq!(sign_extend(0xffff_ffff, u32::MAX), 0xffff_ffff);
    }

    #[test]
    fn the_widths_the_element_impls_actually_pass_are_unchanged_by_the_clamp() {
        assert_eq!(sign_extend(0xff, 8), 0xffff_ffff, "-1i8");
        assert_eq!(sign_extend(0x7f, 8), 0x0000_007f, "127i8");
        assert_eq!(sign_extend(0xffff, 16), 0xffff_ffff, "-1i16");
        assert_eq!(sign_extend(0x7fff, 16), 0x0000_7fff, "32767i16");
    }

    #[test]
    fn an_unsigned_narrow_constant_is_zero_extended() {
        let mut module = module();
        U8::constant_from_bits(&mut module, 0xff).expect("255u8");

        assert_eq!(constant_literal(&module.finish()), 0x0000_00ff);
    }

    #[test]
    fn a_positive_signed_constant_is_not_disturbed_by_the_extension() {
        let mut module = module();
        I16::constant_from_bits(&mut module, 300).expect("300i16");

        assert_eq!(constant_literal(&module.finish()), 300);
    }

    #[test]
    fn the_extension_uses_the_types_own_width() {
        let mut narrow = module();
        I8::constant_from_bits(&mut narrow, 0xff).expect("i8");
        let mut wider = module();
        I16::constant_from_bits(&mut wider, 0xff).expect("i16");

        assert_eq!(constant_literal(&narrow.finish()), 0xffff_ffff);
        assert_eq!(constant_literal(&wider.finish()), 0x0000_00ff);
    }

    #[test]
    fn a_half_constant_carries_the_halfs_bits_and_not_a_floats() {
        let mut module = module();
        F16::constant_from_bits(&mut module, u32::from(half::from_f32(1.5))).expect("1.5h");

        assert_eq!(constant_literal(&module.finish()), 0x3e00);
    }

    #[test]
    fn each_narrow_type_declares_the_capability_its_width_needs() {
        for (declare, wanted) in [
            (
                I8::type_id as fn(&mut Module) -> Result<Id, BuildError>,
                Capability::Int8,
            ),
            (U8::type_id, Capability::Int8),
            (I16::type_id, Capability::Int16),
            (U16::type_id, Capability::Int16),
            (F16::type_id, Capability::Float16),
        ] {
            let mut module = module();
            declare(&mut module).expect("declared");

            let words = module.finish();
            let declared: Vec<u32> = decode::body(&words)
                .filter(|instruction| instruction.opcode() == op::CAPABILITY)
                .filter_map(|instruction| instruction.operands().first().copied())
                .collect();

            assert_eq!(declared, vec![wanted.word()]);
        }
    }

    #[test]
    fn declaring_the_type_does_not_declare_the_storage_capability() {
        let mut module = module();
        I8::type_id(&mut module).expect("i8");

        let words = module.finish();
        let declared: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CAPABILITY)
            .filter_map(|instruction| instruction.operands().first().copied())
            .collect();

        assert_eq!(declared, vec![Capability::Int8.word()]);
        assert!(!declared.contains(&Capability::StorageBuffer8BitAccess.word()));
    }

    #[test]
    fn a_buffer_of_8_bit_elements_declares_the_storage_capability_and_its_extension() {
        let mut module = module();
        I8::require_in_storage_buffer(&mut module).expect("declared");

        assert_eq!(
            decode::opcodes(&module.finish()),
            vec![op::CAPABILITY, op::EXTENSION],
            "8-bit storage is not core at SPIR-V 1.3 and needs SPV_KHR_8bit_storage"
        );
    }

    #[test]
    fn a_buffer_of_16_bit_elements_needs_no_extension() {
        let mut module = module();
        I16::require_in_storage_buffer(&mut module).expect("declared");

        assert_eq!(decode::opcodes(&module.finish()), vec![op::CAPABILITY]);
    }

    #[test]
    fn the_strides_are_the_widths_the_types_actually_occupy() {
        assert_eq!(I8::STRIDE, 1);
        assert_eq!(U8::STRIDE, 1);
        assert_eq!(I16::STRIDE, 2);
        assert_eq!(U16::STRIDE, 2);
        assert_eq!(F16::STRIDE, 2);
    }

    #[test]
    fn a_signed_narrowing_and_an_unsigned_one_are_different_instructions() {
        assert_eq!(I8::FROM_U32, op::S_CONVERT);
        assert_eq!(U8::FROM_U32, op::U_CONVERT);
        assert_ne!(I8::FROM_U32, U8::FROM_U32);
        assert_eq!(F16::FROM_U32, op::CONVERT_U_TO_F);
    }

    #[test]
    fn the_two_widths_of_one_signedness_are_two_types() {
        let mut module = module();

        let byte = I8::type_id(&mut module).expect("i8");
        let short = I16::type_id(&mut module).expect("i16");
        let unsigned_byte = U8::type_id(&mut module).expect("u8");
        let unsigned_short = U16::type_id(&mut module).expect("u16");

        assert_ne!(byte, short);
        assert_ne!(byte, unsigned_byte, "signedness is part of the type");
        assert_ne!(
            short, unsigned_short,
            "and at sixteen bits too — a `u16` declared signed is a type that orders the other \
             way round above 2¹⁵ and is otherwise indistinguishable"
        );
    }
}
