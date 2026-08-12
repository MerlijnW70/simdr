//! Elements narrower than a lane: `i8`, `u8`, `i16`, `u16` and `f16`.
//!
//! # One element per lane, still
//!
//! A subgroup lane is 32 bits wide and an `i8` is not, so there is an obvious thing to do — pack
//! four elements into each lane and process 128 of them per subgroup — and this crate does not do
//! it. `decisions/DR-0004` has the argument. The short version: the measured win is **memory
//! traffic**, a buffer of `i8` is a quarter the size whatever the lanes hold, and packing would
//! add a fourth mapping to the three `decisions/DR-0002` already has to explain.
//!
//! So `Simd<i8, 32>` is 32 lanes each holding one `i8`, exactly as `Simd<i32, 32>` is. What
//! changes is the type's width and the buffer's stride.
//!
//! # Two permissions, not one
//!
//! Vulkan splits these in a way that matters. `shaderInt8` says the *arithmetic* exists;
//! `storageBuffer8BitAccess` says a **buffer** may hold them. A device can offer the first and
//! not the second, and a module that declared only one of the two would be rejected for the
//! other. [`Element::type_id`] declares the first and [`Element::require_in_storage_buffer`] the
//! second, so a kernel that computes in `i8` but reads `i32` asks for only what it uses.
//!
//! There is a third permission with no SPIR-V capability at all: `shaderSubgroupExtendedTypes`.
//! Without it a device accepts `OpGroupNonUniformIAdd` on a 32-bit integer and refuses it on an
//! 8-bit one, and nothing in the module says so — the module is identical either way.
//! `simdr probe` reports it, and `runner` enables it when the device has it.
//!
//! # The conversions are not all the same instruction
//!
//! `OpUConvert` requires a result type whose signedness is 0 and `OpSConvert` does not, so
//! narrowing a `u32` reaches a different opcode depending on whether the target is signed — even
//! though both are the same truncation. That is the kind of asymmetry that assembles cleanly when
//! it is wrong, and `tests/kernels.rs` hands each of them to the validator.

use super::element::{Element, Signed};
use crate::module::{BuildError, Id, Module, op};
use crate::spec::{Capability, Glsl};

/// 8-bit signed integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I8;

impl Element for I8 {
    const NAME: &'static str = "i8";
    const STRIDE: u32 = 1;
    const ADD: u16 = op::I_ADD;
    const MUL: u16 = op::I_MUL;
    const GREATER_THAN: u16 = op::S_GREATER_THAN;
    const GROUP_ADD: u16 = op::GROUP_NON_UNIFORM_I_ADD;
    const GROUP_MAX: u16 = op::GROUP_NON_UNIFORM_S_MAX;
    const GROUP_MIN: u16 = op::GROUP_NON_UNIFORM_S_MIN;
    const MIN: Glsl = Glsl::SMin;
    const MAX: Glsl = Glsl::SMax;
    const CLAMP: Glsl = Glsl::SClamp;
    // Signed target, so `OpSConvert` — a truncation either way, and `OpUConvert` would be refused
    // for the result type's signedness rather than for the arithmetic.
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

/// 8-bit unsigned integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct U8;

impl Element for U8 {
    const NAME: &'static str = "u8";
    const STRIDE: u32 = 1;
    const ADD: u16 = op::I_ADD;
    const MUL: u16 = op::I_MUL;
    const GREATER_THAN: u16 = op::U_GREATER_THAN;
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

/// 16-bit signed integer.
///
/// What a quantised network's activations are, and the reason this exists: widening them to `i32`
/// doubled the bytes a bandwidth-bound kernel had to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I16;

impl Element for I16 {
    const NAME: &'static str = "i16";
    const STRIDE: u32 = 2;
    const ADD: u16 = op::I_ADD;
    const MUL: u16 = op::I_MUL;
    const GREATER_THAN: u16 = op::S_GREATER_THAN;
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

/// 16-bit unsigned integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct U16;

impl Element for U16 {
    const NAME: &'static str = "u16";
    const STRIDE: u32 = 2;
    const ADD: u16 = op::I_ADD;
    const MUL: u16 = op::I_MUL;
    const GREATER_THAN: u16 = op::U_GREATER_THAN;
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

/// 16-bit float.
///
/// A constant's bits are a *half's* bits, not an `f32`'s — [`crate::half::from_f32`] is what
/// produces them, and passing `1.5_f32.to_bits()` here would declare a number in the region of
/// 4×10³⁸ with no complaint from anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F16;

impl Element for F16 {
    const NAME: &'static str = "f16";
    const STRIDE: u32 = 2;
    const ADD: u16 = op::F_ADD;
    const MUL: u16 = op::F_MUL;
    const GREATER_THAN: u16 = op::F_ORD_GREATER_THAN;
    const GROUP_ADD: u16 = op::GROUP_NON_UNIFORM_F_ADD;
    const GROUP_MAX: u16 = op::GROUP_NON_UNIFORM_F_MAX;
    const GROUP_MIN: u16 = op::GROUP_NON_UNIFORM_F_MIN;
    const MIN: Glsl = Glsl::FMin;
    const MAX: Glsl = Glsl::FMax;
    const CLAMP: Glsl = Glsl::FClamp;
    // A conversion, as for `f32`: `OpConvertUToF` reaches any float width, so 7u32 becomes 7.0h
    // rather than the half whose bits are seven.
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

/// The low `width` bits of `bits`, sign-extended to 32.
///
/// §2.2.1's rule for the literal of a constant whose type is a signed integer narrower than a
/// word. Getting it wrong gives an `i8` constant of `-1` declared as `0x000000ff`, which the
/// validator rejects — and would otherwise have been a very quiet 255.
const fn sign_extend(bits: u32, width: u32) -> u32 {
    let spare = 32 - width;
    (((bits << spare) as i32) >> spare) as u32
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::half;
    use crate::module::Version;

    fn module() -> Module {
        Module::new(Version::V1_3)
    }

    /// The literal word of the one `OpConstant` in `words`.
    fn constant_literal(words: &[u32]) -> u32 {
        let operands = decode::body(words)
            .find(|instruction| instruction.opcode() == op::CONSTANT)
            .expect("a constant was declared")
            .operands()
            .to_vec();
        // Type, result id, then the literal.
        operands[2]
    }

    #[test]
    fn a_negative_narrow_constant_is_sign_extended_to_a_full_word() {
        // §2.2.1. `-1i8` is `0xffffffff` and not `0x000000ff`, and the difference is a validation
        // failure rather than a wrong number — which is the good case.
        let mut module = module();
        I8::constant_from_bits(&mut module, 0xff).expect("-1i8");

        assert_eq!(constant_literal(&module.finish()), 0xffff_ffff);
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
        // `0xff` is -1 as an `i8` and 255 as an `i16`. A shared width would make one of them wrong
        // and neither obviously so.
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

        // 1.5 as a half is 0x3e00. As an `f32` it would be 0x3fc00000, which is a different
        // number entirely once read as sixteen bits.
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
        // The two are separate on purpose: a kernel computing in `i8` from an `i32` buffer needs
        // `Int8` and not `StorageBuffer8BitAccess`, and asking for a capability the device lacks
        // fails at pipeline creation even when nothing uses it.
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
        // The number that makes the whole feature worth having: four times fewer bytes for the
        // same element count.
        assert_eq!(I8::STRIDE, 1);
        assert_eq!(U8::STRIDE, 1);
        assert_eq!(I16::STRIDE, 2);
        assert_eq!(U16::STRIDE, 2);
        assert_eq!(F16::STRIDE, 2);
    }

    #[test]
    fn a_signed_narrowing_and_an_unsigned_one_are_different_instructions() {
        // `OpUConvert` requires a result type whose signedness is 0, so it cannot produce an `i8`
        // however much it is the same truncation. This is the asymmetry that assembles cleanly
        // when it is wrong.
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
