//! Several narrow products summed into one wider accumulator, in a single instruction.
//!
//! `OpSDot` takes two 32-bit integers holding four `i8` each, multiplies them componentwise and
//! adds the four products into a 32-bit result. What a kernel would otherwise spell as four
//! shifts, four sign-extensions, four multiplies and three adds.
//!
//! # This is not a fourth mapping
//!
//! `decisions/DR-0004` says a narrow element is one element per lane, and that is unchanged. The
//! packing here is in the **instruction's operands**: a `Simd<u32, N>` is still one `u32` per
//! lane, and `OpSDot` is an operation that reads each of those `u32`s as four bytes. Nothing about
//! the vector, the lane count or the buffer changes.
//!
//! # Two capabilities, and both need the extension
//!
//! `DotProduct` says the instructions exist; `DotProductInput4x8BitPacked` says they accept the
//! packed form. Both are core only in SPIR-V 1.6, and this crate emits 1.3 — so both come with
//! `SPV_KHR_integer_dot_product`, which [`Module::require_capability`] declares for them.
//!
//! # The accumulating form saturates, and the plain one does not
//!
//! `OpSDotAccSat` adds a third operand and clamps the total instead of wrapping. That is a
//! different arithmetic, not a convenience, and it is offered under its own name for that reason:
//! a caller summing quantised weights usually wants the saturating one, and a caller checking
//! against a CPU reference that wraps definitely does not.

use super::{BuildError, Id, Module, op};
use crate::spec::{Capability, PackedVectorFormat};

impl Module {
    /// Declare what any of the dot-product instructions needs.
    ///
    /// Both capabilities and the extension behind them. Asked for by each of the instructions
    /// below rather than by a caller, because a module that emitted one without the other is
    /// rejected for the capability and the omission is the harder half to see.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if a declaration cannot be emitted.
    fn require_dot_product(&mut self) -> Result<(), BuildError> {
        self.require_capability(Capability::DotProduct)?;
        self.require_capability(Capability::DotProductInput4x8BitPacked)
    }

    /// The dot product of four **signed** 8-bit components in each operand.
    ///
    /// `Σ (a[i] as i32) × (b[i] as i32)` over the four bytes, into `result_type`, which must be a
    /// 32-bit integer. The sum wraps; [`Module::s_dot_acc_sat`] is the version that does not.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// The same over **unsigned** components.
    ///
    /// A different instruction rather than a flag, and it has to be: the two agree on every byte
    /// below 128 and disagree above it, which is exactly the half of the range a quantised weight
    /// lives in.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// Signed on the left, unsigned on the right.
    ///
    /// The mixed form, which exists because a quantised layer usually has signed weights and
    /// unsigned activations and would otherwise have to widen one of them.
    ///
    /// **The order is not symmetric.** `su_dot(a, b)` reads `a` as signed and `b` as unsigned;
    /// swapping the arguments computes something else.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// [`Module::s_dot`] plus an accumulator, clamped rather than wrapped.
    ///
    /// The total is `accumulator + Σ a[i] × b[i]`, saturating at the result type's bounds. What a
    /// running sum over many quantised terms wants, and a different answer from the wrapping form
    /// as soon as the sum leaves the range — so which one a kernel uses is a decision rather than
    /// a detail.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::encode::Word;
    use crate::module::Version;

    fn module() -> Module {
        Module::new(Version::V1_3)
    }

    /// The operands of the one instruction carrying `opcode`.
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
        // The grammar makes it optional and its only value is zero, so an emitter that treated
        // zero as absence would produce a *valid* instruction that reads its operands as vectors
        // rather than as packed scalars. The length is what says which.
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
        // `DotProduct` says the instructions exist and `DotProductInput4x8BitPacked` says they
        // take this input; a module with one and not the other is rejected for whichever it left
        // out, which reads as the wrong problem.
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
        // `SDot` and `UDot` agree on every byte below 128 and disagree above it, and `SUDot` is
        // neither. A shared code path that picked the wrong one would be right on small test data
        // and wrong on real quantised weights.
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
        // The order is operands, accumulator, format. Putting the format before the accumulator
        // would give an instruction of the right length whose accumulator is the literal zero —
        // an id nothing declares.
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
        // `SUDot` is not symmetric: the first operand is signed and the second is not. A wrapper
        // that reordered them would compute a different dot product from the same two values.
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
