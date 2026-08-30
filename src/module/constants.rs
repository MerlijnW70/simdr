use super::{BuildError, Id, Module, Section, op};
use crate::encode::Word;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum ConstantKey {
    Bool { of_type: Id, value: bool },
    Scalar32 { of_type: Id, bits: u32 },
}

impl Module {
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
        let mut operands = vec![of_type.word(), id.word()];
        operands.extend_from_slice(literal);
        self.emit(Section::TypeConstantVariable, opcode, &operands)?;
        self.constants.insert(key, id);
        Ok(id)
    }

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

    pub fn constant_i32(&mut self, value: i32) -> Result<Id, BuildError> {
        let of_type = self.type_int(32, true)?;
        let bits = u32::from_ne_bytes(value.to_ne_bytes());
        self.intern_constant(
            ConstantKey::Scalar32 { of_type, bits },
            op::CONSTANT,
            of_type,
            &[bits],
        )
    }

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

    #[test]
    fn the_general_scalar_form_interns_with_the_typed_one_it_generalises() {
        let mut module = Module::new(Version::V1_3);
        let uint = module.type_int(32, false).expect("u32");

        let typed = module.constant_u32(7).expect("7");
        let general = module.constant_scalar(uint, 7).expect("7 again");

        assert_eq!(typed, general);
        assert_eq!(
            crate::decode::opcodes(&module.finish())
                .iter()
                .filter(|opcode| **opcode == op::CONSTANT)
                .count(),
            1,
            "one constant, declared once"
        );
    }
}
