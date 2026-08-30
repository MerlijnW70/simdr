use super::{BuildError, Id, Module, Section, op};
use crate::encode::Word;
use crate::spec::Decoration;

impl Module {
    pub fn spec_constant(
        &mut self,
        of_type: Id,
        default: Word,
        spec_id: u32,
    ) -> Result<Id, BuildError> {
        let id = self.alloc_id()?;
        self.emit(
            Section::TypeConstantVariable,
            op::SPEC_CONSTANT,
            &[of_type.word(), id.word(), default],
        )?;
        self.decorate(id, Decoration::SpecId, &[spec_id])?;
        Ok(id)
    }

    pub fn spec_constant_bool(&mut self, default: bool, spec_id: u32) -> Result<Id, BuildError> {
        let of_type = self.type_bool()?;
        let id = self.alloc_id()?;
        let opcode = if default {
            op::SPEC_CONSTANT_TRUE
        } else {
            op::SPEC_CONSTANT_FALSE
        };
        self.emit(
            Section::TypeConstantVariable,
            opcode,
            &[of_type.word(), id.word()],
        )?;
        self.decorate(id, Decoration::SpecId, &[spec_id])?;
        Ok(id)
    }

    pub fn spec_constant_op(
        &mut self,
        result_type: Id,
        opcode: u16,
        operands: &[Id],
    ) -> Result<Id, BuildError> {
        let id = self.alloc_id()?;
        let mut words = vec![result_type.word(), id.word(), Word::from(opcode)];
        words.extend(operands.iter().map(|operand| operand.word()));
        self.emit(Section::TypeConstantVariable, op::SPEC_CONSTANT_OP, &words)?;
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::module::Version;

    fn module() -> Module {
        Module::new(Version::V1_3)
    }

    fn operands_of(words: &[Word], opcode: u16) -> Vec<Word> {
        decode::body(words)
            .find(|instruction| instruction.opcode() == opcode)
            .expect("the instruction was emitted")
            .operands()
            .to_vec()
    }

    #[test]
    fn a_specialization_constant_carries_its_default_and_is_given_its_id() {
        let mut module = module();
        let uint = module.type_int(32, false).expect("u32");

        let folded = module.spec_constant(uint, 7, 3).expect("declared");

        let words = module.finish();
        assert_eq!(
            operands_of(&words, op::SPEC_CONSTANT),
            vec![uint.word(), folded.word(), 7]
        );
        assert_eq!(
            operands_of(&words, op::DECORATE),
            vec![folded.word(), Decoration::SpecId.word(), 3],
            "without the SpecId nothing can replace it, and it is silently a plain constant"
        );
    }

    #[test]
    fn two_specialization_constants_with_the_same_default_stay_two_constants() {
        let mut module = module();
        let uint = module.type_int(32, false).expect("u32");

        let first = module.spec_constant(uint, 1, 0).expect("first");
        let second = module.spec_constant(uint, 1, 1).expect("second");

        assert_ne!(first, second);
        assert_eq!(
            decode::body(&module.finish())
                .filter(|instruction| instruction.opcode() == op::SPEC_CONSTANT)
                .count(),
            2
        );
    }

    #[test]
    fn a_specialization_constant_is_not_the_same_thing_as_an_ordinary_one() {
        let mut module = module();
        let uint = module.type_int(32, false).expect("u32");

        let ordinary = module.constant_u32(4).expect("4");
        let specialized = module.spec_constant(uint, 4, 0).expect("4, replaceable");

        assert_ne!(ordinary, specialized);
        assert_eq!(
            decode::opcodes(&module.finish()),
            vec![op::DECORATE, op::TYPE_INT, op::CONSTANT, op::SPEC_CONSTANT],
            "the decoration sorts into the annotations, ahead of what it decorates"
        );
    }

    #[test]
    fn a_boolean_specialization_constant_puts_its_default_in_the_opcode() {
        let mut module = module();

        let yes = module.spec_constant_bool(true, 0).expect("true");
        let no = module.spec_constant_bool(false, 1).expect("false");

        let words = module.finish();
        assert_eq!(
            operands_of(&words, op::SPEC_CONSTANT_TRUE),
            vec![module_bool(&words), yes.word()]
        );
        assert_eq!(
            operands_of(&words, op::SPEC_CONSTANT_FALSE),
            vec![module_bool(&words), no.word()]
        );
    }

    fn module_bool(words: &[Word]) -> Word {
        operands_of(words, op::TYPE_BOOL)[0]
    }

    #[test]
    fn a_derived_constant_carries_its_opcode_as_a_literal() {
        let mut module = module();
        let uint = module.type_int(32, false).expect("u32");
        let base = module.spec_constant(uint, 8, 0).expect("base");
        let two = module.constant_u32(2).expect("2");

        let doubled = module
            .spec_constant_op(uint, op::I_MUL, &[base, two])
            .expect("derived");

        assert_eq!(
            operands_of(&module.finish(), op::SPEC_CONSTANT_OP),
            vec![
                uint.word(),
                doubled.word(),
                Word::from(op::I_MUL),
                base.word(),
                two.word()
            ]
        );
    }

    #[test]
    fn a_derived_constant_is_not_decorated_with_a_spec_id() {
        let mut module = module();
        let uint = module.type_int(32, false).expect("u32");
        let base = module.spec_constant(uint, 8, 0).expect("base");

        module
            .spec_constant_op(uint, op::I_MUL, &[base, base])
            .expect("derived");

        assert_eq!(
            decode::body(&module.finish())
                .filter(|instruction| instruction.opcode() == op::DECORATE)
                .count(),
            1,
            "one decoration, on the constant that has a value of its own"
        );
    }
}
