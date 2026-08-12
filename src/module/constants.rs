//! Constant declarations, deduplicated by the bits they encode to.

use super::{BuildError, Id, Module, Section, op};
use crate::encode::Word;

/// What makes a constant the same constant as another.
///
/// The value is held as the bits it encodes to rather than as a number, which is both what the
/// instruction carries and the right notion of sameness: two `f32`s with identical bits are one
/// constant, and `0.0` and `-0.0` are two.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum ConstantKey {
    Bool { of_type: Id, value: bool },
    Scalar32 { of_type: Id, bits: u32 },
}

impl Module {
    /// Return the id of an identical constant, declaring it first if the module has not seen it.
    fn intern_constant(
        &mut self,
        key: ConstantKey,
        opcode: u16,
        of_type: Id,
        literal: &[Word],
    ) -> Result<Id, BuildError> {
        if let Some(&existing) = self.constants.get(&key) {
            return Ok(existing);
        }

        let id = self.alloc_id()?;
        // A constant names its type first and itself second — the reverse of a type declaration,
        // which has no type to name.
        let mut operands = vec![of_type.word(), id.word()];
        operands.extend_from_slice(literal);
        self.emit(Section::TypeConstantVariable, opcode, &operands)?;
        self.constants.insert(key, id);
        Ok(id)
    }

    /// A 32-bit unsigned constant, declaring `u32` if needed.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration cannot be emitted.
    pub fn constant_u32(&mut self, value: u32) -> Result<Id, BuildError> {
        let of_type = self.type_int(32, false)?;
        self.intern_constant(
            ConstantKey::Scalar32 {
                of_type,
                bits: value,
            },
            op::CONSTANT,
            of_type,
            &[value],
        )
    }

    /// A 32-bit signed constant, declaring `i32` if needed.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration cannot be emitted.
    pub fn constant_i32(&mut self, value: i32) -> Result<Id, BuildError> {
        let of_type = self.type_int(32, true)?;
        // Reinterpreted rather than converted: the instruction carries the bit pattern, and a
        // negative value must not be range-checked into something else on the way.
        let bits = u32::from_ne_bytes(value.to_ne_bytes());
        self.intern_constant(
            ConstantKey::Scalar32 { of_type, bits },
            op::CONSTANT,
            of_type,
            &[bits],
        )
    }

    /// A 32-bit float constant, declaring `f32` if needed.
    ///
    /// Sameness is by bit pattern, which is the specification's rule and also the useful one:
    /// `0.0` and `-0.0` stay two constants, and two NaNs with different payloads do too.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration cannot be emitted.
    pub fn constant_f32(&mut self, value: f32) -> Result<Id, BuildError> {
        let of_type = self.type_float(32)?;
        let bits = value.to_bits();
        self.intern_constant(
            ConstantKey::Scalar32 { of_type, bits },
            op::CONSTANT,
            of_type,
            &[bits],
        )
    }

    /// A scalar constant of a type this module has already declared.
    ///
    /// What the narrow element types need: `OpConstant` carries one literal word for any scalar
    /// type 32 bits or narrower, so an `i8` and an `f16` constant are the same instruction with a
    /// different type id.
    ///
    /// **The literal must already be extended to 32 bits the way the type requires** — sign-
    /// extended for a signed integer narrower than 32 bits, zero-extended otherwise. That is
    /// §2.2.1's rule and it is not checked here, because this layer holds an id and has no way to
    /// ask what the type behind it was. [`crate::lanes::Element::constant_from_bits`] is where
    /// each type does its own extending, and where the tests for it live.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration cannot be emitted.
    pub fn constant_scalar(&mut self, of_type: Id, literal: Word) -> Result<Id, BuildError> {
        self.intern_constant(
            ConstantKey::Scalar32 {
                of_type,
                bits: literal,
            },
            op::CONSTANT,
            of_type,
            &[literal],
        )
    }

    /// A boolean constant, declaring `bool` if needed.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration cannot be emitted.
    pub fn constant_bool(&mut self, value: bool) -> Result<Id, BuildError> {
        let of_type = self.type_bool()?;
        let opcode = if value {
            op::CONSTANT_TRUE
        } else {
            op::CONSTANT_FALSE
        };
        self.intern_constant(ConstantKey::Bool { of_type, value }, opcode, of_type, &[])
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::module::Version;

    #[test]
    fn the_same_constant_twice_is_declared_once() {
        let mut module = Module::new(Version::V1_3);

        let first = module.constant_u32(7).expect("7u32");
        let second = module.constant_u32(7).expect("7u32 again");

        assert_eq!(first, second);
    }

    #[test]
    fn different_values_of_one_type_are_different_constants() {
        let mut module = Module::new(Version::V1_3);

        let seven = module.constant_u32(7).expect("7u32");
        let eight = module.constant_u32(8).expect("8u32");

        assert_ne!(seven, eight);
    }

    #[test]
    fn constants_of_different_types_are_distinct_even_with_the_same_bits() {
        let mut module = Module::new(Version::V1_3);

        // Zero has the same 32 bits whether it is an unsigned, a signed or a float.
        let unsigned = module.constant_u32(0).expect("0u32");
        let signed = module.constant_i32(0).expect("0i32");
        let float = module.constant_f32(0.0).expect("0.0f32");

        assert_ne!(unsigned, signed);
        assert_ne!(signed, float);
        assert_ne!(unsigned, float);
    }

    #[test]
    fn negative_constants_keep_their_bit_pattern() {
        let mut module = Module::new(Version::V1_3);
        module.constant_i32(-1).expect("-1i32");

        let words = module.finish();

        // The literal is the last word: two's complement, not a range-checked conversion.
        assert_eq!(words.last(), Some(&0xffff_ffff));
    }

    #[test]
    fn positive_and_negative_zero_are_two_constants() {
        let mut module = Module::new(Version::V1_3);

        let positive = module.constant_f32(0.0).expect("0.0");
        let negative = module.constant_f32(-0.0).expect("-0.0");

        assert_ne!(
            positive, negative,
            "they compare equal as numbers and differ as bits, and SPIR-V follows the bits"
        );
    }

    #[test]
    fn two_nans_with_different_payloads_are_two_constants() {
        let mut module = Module::new(Version::V1_3);

        let quiet = module
            .constant_f32(f32::from_bits(0x7fc0_0000))
            .expect("NaN");
        let other = module
            .constant_f32(f32::from_bits(0x7fc0_0001))
            .expect("NaN");

        assert_ne!(quiet, other);
    }

    #[test]
    fn true_and_false_use_their_own_opcodes_and_share_a_type() {
        let mut module = Module::new(Version::V1_3);

        module.constant_bool(true).expect("true");
        module.constant_bool(false).expect("false");

        let words = module.finish();
        let body = &words[5..];

        // OpTypeBool, then OpConstantTrue, then OpConstantFalse — the type declared once.
        assert_eq!(body[0] & 0xffff, Word::from(op::TYPE_BOOL));
        assert_eq!(body[2] & 0xffff, Word::from(op::CONSTANT_TRUE));
        assert_eq!(body[5] & 0xffff, Word::from(op::CONSTANT_FALSE));
    }

    #[test]
    fn the_same_boolean_twice_is_declared_once() {
        let mut module = Module::new(Version::V1_3);

        let first = module.constant_bool(true).expect("true");
        let second = module.constant_bool(true).expect("true again");

        assert_eq!(first, second);
    }

    #[test]
    fn a_constant_declares_the_type_it_needs() {
        let mut module = Module::new(Version::V1_3);
        module.constant_f32(1.0).expect("1.0");

        let words = module.finish();
        let body = &words[5..];

        assert_eq!(body[0] & 0xffff, Word::from(op::TYPE_FLOAT));
        assert_eq!(body[3] & 0xffff, Word::from(op::CONSTANT));
    }
}
