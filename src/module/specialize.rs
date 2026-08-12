//! Specialization constants: values fixed when the pipeline is created, not when the module is.
//!
//! A specialization constant is declared with a default and a **SpecId**, and the driver may
//! replace its value at `vkCreateComputePipeline` from a `VkSpecializationInfo`. Everything
//! downstream still treats it as a constant, because by the time anything compiles it, it is one.
//!
//! # What this is for
//!
//! One module per parameter value. `Gpu::sum` built ten modules for ten fold sizes; every kernel
//! in `runner/src/kernels` bakes in the subgroup width. Modules are a few hundred words so the
//! bytes were never the problem — pipeline creation was, and `runner/examples/overhead.rs`
//! measured that at the expensive end of a round trip.
//!
//! # What it is not for
//!
//! **Anything the emitter has to reason about.** `Lanes` picks between a plain reduction and a
//! clustered one by comparing the lane count against the subgroup width, and it emits a different
//! instruction for each — that decision happens while the module is being built, and a value that
//! arrives later cannot inform it. `decisions/DR-0002` is the long form and it is unchanged by
//! anything here: a specialization constant defers a *number*, not a choice of instruction.
//!
//! `decisions/DR-0005` records what happened when that was tested rather than assumed.

use super::{BuildError, Id, Module, Section, op};
use crate::encode::Word;
use crate::spec::Decoration;

impl Module {
    /// A specialization constant of `of_type`, holding `default` unless the pipeline overrides it.
    ///
    /// `spec_id` is the number a `VkSpecializationInfo` entry names to replace it. Ids are the
    /// caller's to allocate and to keep unique — two constants sharing one would both be replaced
    /// by the same value, which validates and is almost never meant.
    ///
    /// **Not deduplicated**, unlike [`Module::constant_scalar`]. Two specialization constants with
    /// the same default are two different values as soon as a pipeline sets one of them, so
    /// merging them by their defaults would merge things that are only equal by coincidence.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration cannot be emitted.
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

    /// A boolean specialization constant.
    ///
    /// The default decides the *opcode* rather than an operand — `OpSpecConstantTrue` and
    /// `OpSpecConstantFalse` — which is the same shape [`Module::constant_bool`] has and the
    /// reason a boolean cannot go through [`Module::spec_constant`].
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration cannot be emitted.
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

    /// A constant computed from other constants, evaluated when the pipeline is created.
    ///
    /// `OpSpecConstantOp` carries an ordinary opcode as a *literal* and applies it to constant
    /// operands, so `width × 2` or `limit − 1` can be derived from a specialization constant and
    /// stay a constant — which matters because the places a constant is *required* (an array
    /// length, a cluster size) will not take a value computed in a function body.
    ///
    /// The opcode goes where an operand usually would, which is the one thing to get right here:
    /// this is a literal number in the instruction, not an instruction of its own.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration cannot be emitted.
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
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
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
        // The one place deduplication would be actively wrong: they are equal only until a
        // pipeline sets one of them.
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
        // They share a type and a value and differ in opcode, which is the difference that
        // decides whether a driver may replace it.
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

    /// The id of the module's `OpTypeBool`.
    fn module_bool(words: &[Word]) -> Word {
        operands_of(words, op::TYPE_BOOL)[0]
    }

    #[test]
    fn a_derived_constant_carries_its_opcode_as_a_literal() {
        // The shape that is easy to get wrong: the opcode sits where an operand would, so an
        // `OpSpecConstantOp` whose literal was left out is an instruction one word short with its
        // operands shifted — which decodes, and multiplies the wrong things.
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
        // It has no value of its own to replace — it is whatever its operands work out to. A
        // `SpecId` on one is a validation failure rather than a harmless extra.
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
