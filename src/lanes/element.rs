//! What a lane can hold.
//!
//! `Simd<T, N>` is generic in `T`, and so is [`super::Vector`]. SPIR-V is not: `OpFAdd` and
//! `OpIAdd` are different instructions, and signed and unsigned integers differ again at `max`
//! and at `>`. This trait is where that difference lives, so every operation above it is written
//! once.

use crate::module::{BuildError, Id, Module, op};
use crate::spec::Glsl;

/// A type a lane can hold.
///
/// Sealed by construction rather than by a marker: implementing it requires naming SPIR-V opcodes
/// correctly, and the three implementations here cover what the emitter can declare.
pub trait Element: Copy + core::fmt::Debug + 'static {
    /// How this spells in a diagnostic.
    const NAME: &'static str;

    /// How many bytes one element occupies in a buffer.
    ///
    /// The buffer's `ArrayStride`, and the reason the narrow types are worth having at all: an
    /// `i8` kernel moves a quarter of the bytes an `i32` one does over the same element count.
    /// `decisions/DR-0004` is why that is the whole of the change — a narrow element is still one
    /// element per lane, not four packed into one.
    const STRIDE: u32;

    /// Elementwise add.
    const ADD: u16;
    /// Elementwise multiply.
    const MUL: u16;
    /// Elementwise `>`, yielding a boolean.
    const GREATER_THAN: u16;
    /// Elementwise `==`, yielding a boolean.
    ///
    /// **The one comparison where the two integer families agree.** `GREATER_THAN` is `OpSGreaterThan`
    /// for the signed types and `OpUGreaterThan` for the unsigned ones, and they disagree above
    /// 2³¹; equality is `OpIEqual` for both, because two bit patterns are equal or they are not and
    /// no interpretation of the sign bit changes that. The floats keep their own instruction — an
    /// *ordered* one, so a NaN equals nothing, itself included.
    const EQUAL: u16;

    /// Add across a group.
    const GROUP_ADD: u16;
    /// Maximum across a group.
    const GROUP_MAX: u16;
    /// Minimum across a group.
    const GROUP_MIN: u16;

    /// Elementwise minimum, from GLSL.std.450.
    ///
    /// Extended rather than core: SPIR-V has no scalar `min` opcode at all, for any type. The
    /// three families differ the way [`Element::GREATER_THAN`] does — `UMin` and `SMin` disagree
    /// above 2³¹ and nowhere else, which is not a difference a small test would see.
    const MIN: Glsl;
    /// Elementwise maximum, from GLSL.std.450.
    const MAX: Glsl;
    /// Elementwise clamp between two bounds, from GLSL.std.450.
    ///
    /// One instruction where the core spelling is two comparisons and two selects.
    const CLAMP: Glsl;

    /// Turn a `u32`'s numeric value into one of these.
    ///
    /// Every case has the shape `<type> <result> <operand>`, so this is one opcode rather than a
    /// branch — including `u32` itself, where the answer is `OpCopyObject` and the driver folds it
    /// away. A `None` here would buy nothing and would make every call site test for it.
    ///
    /// It matters that these are *conversions* and not reinterpretations. 7u32 must become 7.0f32,
    /// not the float whose bits are seven, which is a denormal near zero. The one place the two
    /// coincide is `i32`, where the widths are equal and there is nothing to convert.
    const FROM_U32: u16;

    /// Declare the SPIR-V type, or return the one already declared.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration cannot be emitted.
    fn type_id(module: &mut Module) -> Result<Id, BuildError>;

    /// Declare a constant of this type holding `value`, reinterpreted from its bits.
    ///
    /// Taking bits rather than a number is what lets one signature serve all three: the caller
    /// converts once, and nothing here has to be generic over a numeric trait the standard
    /// library does not have.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the constant cannot be declared.
    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError>;

    /// Declare whatever a *buffer* of this type needs, beyond the type itself.
    ///
    /// Two different permissions, and a device may grant one and not the other: `Int8` says the
    /// module computes in 8-bit integers, `StorageBuffer8BitAccess` says a storage buffer holds
    /// them. The 32-bit types need neither, which is why this has a body — every kernel calls it
    /// and only three of the eight implementations have anything to do.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if a declaration cannot be emitted.
    fn require_in_storage_buffer(module: &mut Module) -> Result<(), BuildError> {
        let _ = module;
        Ok(())
    }
}

/// An element whose values have a sign, so that `abs` means something.
///
/// [`F32`] and [`I32`] have one; [`U32`] does not, and GLSL.std.450 accordingly has `FAbs` and
/// `SAbs` and no `UAbs`. That absence is the whole reason this is a second trait rather than one
/// more constant on [`Element`]: `abs` of a `u32` is the value itself, and a caller who writes it
/// has misunderstood something. A `Option<Glsl>` would have made that a runtime error, and an
/// `OpCopyObject` would have made it silently fine. Refusing at the type is neither.
pub trait Signed: Element {
    /// Magnitude, from GLSL.std.450.
    const ABS: Glsl;
}

/// 32-bit float.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32;

impl Element for F32 {
    const NAME: &'static str = "f32";
    const STRIDE: u32 = 4;
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
    // A real conversion. Reinterpreting the bits of 7u32 as a float gives a denormal near zero.
    const FROM_U32: u16 = op::CONVERT_U_TO_F;

    fn type_id(module: &mut Module) -> Result<Id, BuildError> {
        module.type_float(32)
    }

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError> {
        module.constant_f32(f32::from_bits(bits))
    }
}

/// 32-bit signed integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I32;

impl Element for I32 {
    const NAME: &'static str = "i32";
    const STRIDE: u32 = 4;
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
    // Equal widths, so there is nothing to convert — only a different reading of the same bits.
    const FROM_U32: u16 = op::BITCAST;

    fn type_id(module: &mut Module) -> Result<Id, BuildError> {
        module.type_int(32, true)
    }

    fn constant_from_bits(module: &mut Module, bits: u32) -> Result<Id, BuildError> {
        module.constant_i32(i32::from_ne_bytes(bits.to_ne_bytes()))
    }
}

/// 32-bit unsigned integer.
///
/// Shares `OpIAdd` and `OpIMul` with [`I32`] — the specification makes addition and multiplication
/// sign-agnostic and only the comparisons and the min/max differ. That asymmetry is exactly why
/// this trait carries six opcodes rather than a single "integer" flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct U32;

impl Element for U32 {
    const NAME: &'static str = "u32";
    const STRIDE: u32 = 4;
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
    // Already a `u32`. `OpCopyObject` keeps the shape uniform and costs nothing at runtime.
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
}

impl Signed for I32 {
    const ABS: Glsl = Glsl::SAbs;
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::module::Version;

    #[test]
    fn the_two_integers_share_their_add_and_differ_at_max() {
        // §3 makes `OpIAdd` sign-agnostic and `OpGroupNonUniformSMax`/`UMax` distinct, which is
        // the whole reason this is a trait with six opcodes rather than one enum with three.
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
        // The same asymmetry as `GROUP_MAX`, one layer up: `UMin` and `SMin` agree on every value
        // below 2³¹ and disagree above it, so a transposition here is invisible to any test whose
        // numbers are small — which is every test in this file.
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
        // `U32: Signed` does not compile, which is the assertion — it cannot be written here, so
        // what is left is that the two that do have one do not share it.
        assert_ne!(F32::ABS, I32::ABS);
        assert_eq!(F32::ABS.operands(), 1);
    }
}
