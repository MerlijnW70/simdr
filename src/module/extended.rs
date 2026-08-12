//! Extended instruction sets: importing one, and reaching an instruction in it.
//!
//! Two instructions do the whole of it. `OpExtInstImport` names a set and yields an id;
//! `OpExtInst` names that id, a literal instruction number, and the operands. Nothing else about
//! a module changes — no capability, no `OpExtension`, and the result is an ordinary value.
//!
//! The literal number is the part with no safety net. `OpExtInst %float %r %set 40 %a %b` is a
//! well-formed instruction whatever `40` means in the set that was imported, so the numbers come
//! from [`crate::spec::Glsl`], which read them out of that set's own grammar.

use super::{BuildError, Id, Module, Section, op};
use crate::encode::{self, Word};

impl Module {
    /// Import an extended instruction set, or return the id it already has.
    ///
    /// A module may import a set once. Importing twice is not two sets — it is two ids naming the
    /// same thing, which validates and then makes a reader wonder which one an instruction meant.
    /// So this interns, exactly as types and constants do.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn ext_inst_import(&mut self, name: &str) -> Result<Id, BuildError> {
        if let Some(&existing) = self.ext_imports.get(name) {
            return Ok(existing);
        }

        let id = self.alloc_id()?;
        let mut operands = vec![id.word()];
        encode::literal_string(&mut operands, name);
        self.emit(Section::ExtInstImport, op::EXT_INST_IMPORT, &operands)?;
        self.ext_imports.insert(name.to_owned(), id);
        Ok(id)
    }

    /// Call `instruction` from the set `set` names.
    ///
    /// `set` is the id [`Module::ext_inst_import`] returned, and `instruction` is a literal from
    /// that set's numbering — not from the core grammar's.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn ext_inst(
        &mut self,
        result_type: Id,
        set: Id,
        instruction: Word,
        operands: &[Id],
    ) -> Result<Id, BuildError> {
        let mut tail = vec![set.word(), instruction];
        tail.extend(operands.iter().map(|operand| operand.word()));
        self.result_instruction(op::EXT_INST, result_type, &tail)
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::module::Version;
    use crate::spec::Glsl;

    #[test]
    fn an_import_yields_an_id_and_carries_its_name() {
        let mut module = Module::new(Version::V1_3);

        let set = module.ext_inst_import(Glsl::SET_NAME).expect("imported");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::EXT_INST_IMPORT)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(operands[0], set.word());
        // "GLSL" as four bytes of one word, least significant first.
        assert_eq!(operands[1], u32::from_le_bytes(*b"GLSL"));
    }

    #[test]
    fn importing_the_same_set_twice_imports_it_once() {
        let mut module = Module::new(Version::V1_3);

        let first = module.ext_inst_import(Glsl::SET_NAME).expect("imported");
        let second = module.ext_inst_import(Glsl::SET_NAME).expect("again");

        assert_eq!(first, second);
        assert_eq!(
            decode::opcodes(&module.finish()),
            vec![op::EXT_INST_IMPORT],
            "one import, not two ids naming the same set"
        );
    }

    #[test]
    fn two_different_sets_are_two_imports() {
        let mut module = Module::new(Version::V1_3);

        let glsl = module.ext_inst_import(Glsl::SET_NAME).expect("glsl");
        let other = module
            .ext_inst_import("NonSemantic.DebugPrintf")
            .expect("another set");

        assert_ne!(glsl, other, "interning is by name, not by there being one");
    }

    #[test]
    fn a_call_names_the_set_then_the_instruction_then_the_operands() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let set = module.ext_inst_import(Glsl::SET_NAME).expect("imported");
        let left = module.constant_f32(1.0).expect("1.0");
        let right = module.constant_f32(2.0).expect("2.0");

        let largest = module
            .ext_inst(float, set, Glsl::FMax.word(), &[left, right])
            .expect("max");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::EXT_INST)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(
            operands,
            vec![
                float.word(),
                largest.word(),
                set.word(),
                40, // FMax, a literal in the set's own numbering
                left.word(),
                right.word()
            ]
        );
    }

    #[test]
    fn the_import_lands_in_its_own_section_ahead_of_the_memory_model() {
        // §2.4 puts `OpExtInstImport` after the capabilities and extensions and before everything
        // else. Emitting it from the middle of a function body — which is where the *call* is —
        // would put it in the wrong section, and a validator reads that as a module whose layout
        // is shuffled rather than as a misplaced instruction.
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let one = module.constant_f32(1.0).expect("1.0");

        module.label().expect("a block is already open");
        let set = module.ext_inst_import(Glsl::SET_NAME).expect("imported");
        module
            .ext_inst(float, set, Glsl::Sqrt.word(), &[one])
            .expect("sqrt");

        assert_eq!(
            decode::opcodes(&module.finish()),
            vec![
                op::EXT_INST_IMPORT,
                op::TYPE_FLOAT,
                op::CONSTANT,
                op::LABEL,
                op::EXT_INST
            ],
            "the import sorts to its section however late it was asked for"
        );
    }

    #[test]
    fn an_instruction_with_no_operands_at_all_still_encodes() {
        // Nothing in GLSL.std.450 takes none, and the shape is still worth pinning: the two-word
        // tail is the set and the number, and a length computed from the operand slice alone
        // would be two words short.
        let mut module = Module::new(Version::V1_3);
        let void = module.type_void().expect("void");
        let set = module.ext_inst_import(Glsl::SET_NAME).expect("imported");

        module.ext_inst(void, set, 1, &[]).expect("emitted");

        let words = module.finish();
        let instruction = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::EXT_INST)
            .expect("emitted");

        assert_eq!(instruction.operands().len(), 4);
    }
}
