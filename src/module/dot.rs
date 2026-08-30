use super::{BuildError, Id, Module, op};
use crate::spec::{Capability, PackedVectorFormat};

impl Module {
    fn require_dot_product(&mut self) -> Result<(), BuildError> {
        self.require_capability(Capability::DotProduct)?;
        self.require_capability(Capability::DotProductInput4x8BitPacked)
    }

    pub fn s_dot(
        &mut self,
        result_type: Id,
        left: Id,
        right: Id,
        format: PackedVectorFormat,
    ) -> Result<Id, BuildError> {
        self.require_dot_product()?;
        self.result_instruction(
            op::S_DOT,
            result_type,
            &[left.word(), right.word(), format.word()],
        )
    }

    pub fn u_dot(
        &mut self,
        result_type: Id,
        left: Id,
        right: Id,
        format: PackedVectorFormat,
    ) -> Result<Id, BuildError> {
        self.require_dot_product()?;
        self.result_instruction(
            op::U_DOT,
            result_type,
            &[left.word(), right.word(), format.word()],
        )
    }

    pub fn su_dot(
        &mut self,
        result_type: Id,
        signed: Id,
        unsigned: Id,
        format: PackedVectorFormat,
    ) -> Result<Id, BuildError> {
        self.require_dot_product()?;
        self.result_instruction(
            op::SU_DOT,
            result_type,
            &[signed.word(), unsigned.word(), format.word()],
        )
    }

    pub fn s_dot_acc_sat(
        &mut self,
        result_type: Id,
        left: Id,
        right: Id,
        accumulator: Id,
        format: PackedVectorFormat,
    ) -> Result<Id, BuildError> {
        self.require_dot_product()?;
        self.result_instruction(
            op::S_DOT_ACC_SAT,
            result_type,
            &[left.word(), right.word(), accumulator.word(), format.word()],
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::encode::Word;
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
    fn a_signed_dot_names_its_operands_then_its_format() {
        let mut module = module();
        let int = module.type_int(32, true).expect("i32");
        let left = module.constant_u32(0x0102_0304).expect("packed");
        let right = module.constant_u32(0x0506_0708).expect("packed");

        let total = module
            .s_dot(int, left, right, PackedVectorFormat::FourEightBit)
            .expect("dot");

        assert_eq!(
            operands_of(&module.finish(), op::S_DOT),
            vec![int.word(), total.word(), left.word(), right.word(), 0],
            "the trailing zero is the packed format, and leaving it off changes the instruction"
        );
    }

    #[test]
    fn the_format_operand_is_present_even_though_it_is_zero() {
        let mut module = module();
        let int = module.type_int(32, true).expect("i32");
        let value = module.constant_u32(1).expect("1");

        module
            .s_dot(int, value, value, PackedVectorFormat::FourEightBit)
            .expect("dot");

        let words = module.finish();
        let instruction = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::S_DOT)
            .expect("emitted");

        assert_eq!(
            instruction.operands().len(),
            5,
            "type, result, two operands and the format"
        );
    }

    #[test]
    fn every_dot_declares_both_capabilities_and_the_extension() {
        let mut module = module();
        let int = module.type_int(32, true).expect("i32");
        let value = module.constant_u32(1).expect("1");

        module
            .s_dot(int, value, value, PackedVectorFormat::FourEightBit)
            .expect("dot");

        let words = module.finish();
        let declared: Vec<Word> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CAPABILITY)
            .filter_map(|instruction| instruction.operands().first().copied())
            .collect();

        assert!(declared.contains(&Capability::DotProduct.word()));
        assert!(declared.contains(&Capability::DotProductInput4x8BitPacked.word()));
        assert_eq!(
            decode::body(&words)
                .filter(|instruction| instruction.opcode() == op::EXTENSION)
                .count(),
            1,
            "one extension, however many capabilities asked for it"
        );
    }

    #[test]
    fn the_three_sign_combinations_are_three_instructions() {
        let mut module = module();
        let int = module.type_int(32, true).expect("i32");
        let value = module.constant_u32(1).expect("1");
        let format = PackedVectorFormat::FourEightBit;

        module.s_dot(int, value, value, format).expect("signed");
        module.u_dot(int, value, value, format).expect("unsigned");
        module.su_dot(int, value, value, format).expect("mixed");

        let words = module.finish();
        for opcode in [op::S_DOT, op::U_DOT, op::SU_DOT] {
            assert_eq!(
                decode::body(&words)
                    .filter(|instruction| instruction.opcode() == opcode)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn the_accumulating_form_takes_a_third_operand_before_the_format() {
        let mut module = module();
        let int = module.type_int(32, true).expect("i32");
        let left = module.constant_u32(2).expect("2");
        let right = module.constant_u32(3).expect("3");
        let carried = module.constant_u32(10).expect("10");

        let total = module
            .s_dot_acc_sat(int, left, right, carried, PackedVectorFormat::FourEightBit)
            .expect("accumulated");

        assert_eq!(
            operands_of(&module.finish(), op::S_DOT_ACC_SAT),
            vec![
                int.word(),
                total.word(),
                left.word(),
                right.word(),
                carried.word(),
                0
            ]
        );
    }

    #[test]
    fn the_mixed_form_keeps_its_operands_in_the_order_it_was_given() {
        let mut module = module();
        let int = module.type_int(32, true).expect("i32");
        let signed = module.constant_u32(0x8080_8080).expect("negative bytes");
        let unsigned = module.constant_u32(0x0101_0101).expect("small bytes");

        module
            .su_dot(int, signed, unsigned, PackedVectorFormat::FourEightBit)
            .expect("mixed");

        let operands = operands_of(&module.finish(), op::SU_DOT);
        assert_eq!(operands[2], signed.word(), "the signed operand is first");
        assert_eq!(operands[3], unsigned.word());
    }
}
