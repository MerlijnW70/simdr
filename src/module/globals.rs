//! Decorations and global variables — the module's interface to the host.

use super::{BuildError, Id, Module, Section, op};
use crate::encode::Word;
use crate::spec::{BuiltIn, Decoration, StorageClass};

impl Module {
    /// Attach `decoration` to `target`.
    ///
    /// `extra` carries the decoration's own operands: the `0` in `Binding 0`, the built-in's word
    /// in `BuiltIn GlobalInvocationId`. Empty for the ones that stand alone, such as `Block`.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn decorate(
        &mut self,
        target: Id,
        decoration: Decoration,
        extra: &[Word],
    ) -> Result<(), BuildError> {
        let mut operands = vec![target.word(), decoration.word()];
        operands.extend_from_slice(extra);
        self.emit(Section::Annotation, op::DECORATE, &operands)
    }

    /// Attach `decoration` to one member of a struct, counted from zero.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn member_decorate(
        &mut self,
        structure: Id,
        member: u32,
        decoration: Decoration,
        extra: &[Word],
    ) -> Result<(), BuildError> {
        let mut operands = vec![structure.word(), member, decoration.word()];
        operands.extend_from_slice(extra);
        self.emit(Section::Annotation, op::MEMBER_DECORATE, &operands)
    }

    /// Declare a module-scope variable and yield the pointer to it.
    ///
    /// `pointer_type` must be an [`Module::type_pointer`] whose storage class matches `storage` —
    /// SPIR-V states it in both places and the validator checks that they agree.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration cannot be emitted.
    pub fn global_variable(
        &mut self,
        pointer_type: Id,
        storage: StorageClass,
    ) -> Result<Id, BuildError> {
        let id = self.alloc_id()?;
        self.emit(
            Section::TypeConstantVariable,
            op::VARIABLE,
            &[pointer_type.word(), id.word(), storage.word()],
        )?;
        Ok(id)
    }

    /// The `Input` variable carrying `built_in`, declaring it if this is the first ask.
    ///
    /// Three things have to happen together for a built-in to be usable, and forgetting the third
    /// is a module the validator rejects rather than a wrong number: the variable is declared, it
    /// is decorated, and it is named in the entry point's interface. This does all three, and does
    /// them **once** — a second caller asking for the same built-in gets the same variable, because
    /// two variables decorated with one built-in is not a duplicate, it is invalid.
    ///
    /// `value_type` is the type the specification gives the built-in — a scalar `u32` for
    /// `SubgroupLocalInvocationId`, a three-component vector for `LocalInvocationId`. It is the
    /// caller's because this layer has no table of them, and the cache is keyed on the built-in
    /// alone: asking twice with different types would be asking for something that does not exist.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the declaration, the decoration or the interface cannot be emitted.
    pub fn builtin_input(&mut self, built_in: BuiltIn, value_type: Id) -> Result<Id, BuildError> {
        if let Some(&variable) = self.builtins.get(&built_in.word()) {
            return Ok(variable);
        }

        let pointer = self.type_pointer(StorageClass::Input, value_type)?;
        let variable = self.global_variable(pointer, StorageClass::Input)?;
        self.decorate(variable, Decoration::BuiltIn, &[built_in.word()])?;
        self.require_interface(variable)?;
        self.builtins.insert(built_in.word(), variable);
        Ok(variable)
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::module::Version;
    use crate::spec::BuiltIn;

    #[test]
    fn a_decoration_without_operands_is_three_words() {
        let mut module = Module::new(Version::V1_3);
        let target = module.alloc_id().expect("%1");

        module
            .decorate(target, Decoration::Block, &[])
            .expect("fits");

        let words = module.finish();
        let body = &words[5..];

        assert_eq!(body[0] >> 16, 3, "opcode, target, decoration");
        assert_eq!(body[0] & 0xffff, Word::from(op::DECORATE));
        assert_eq!(body[1], target.word());
        assert_eq!(body[2], Decoration::Block.word());
    }

    #[test]
    fn a_decorations_own_operand_follows_it() {
        let mut module = Module::new(Version::V1_3);
        let target = module.alloc_id().expect("%1");

        module
            .decorate(target, Decoration::Binding, &[7])
            .expect("fits");

        let words = module.finish();
        let body = &words[5..];

        assert_eq!(body[0] >> 16, 4);
        assert_eq!(body[3], 7);
    }

    #[test]
    fn a_builtin_is_a_decoration_carrying_the_builtins_word() {
        let mut module = Module::new(Version::V1_3);
        let target = module.alloc_id().expect("%1");

        module
            .decorate(
                target,
                Decoration::BuiltIn,
                &[BuiltIn::GlobalInvocationId.word()],
            )
            .expect("fits");

        let words = module.finish();
        let body = &words[5..];

        assert_eq!(body[2], Decoration::BuiltIn.word());
        assert_eq!(body[3], BuiltIn::GlobalInvocationId.word());
    }

    #[test]
    fn a_member_decoration_names_the_member_between_target_and_decoration() {
        let mut module = Module::new(Version::V1_3);
        let structure = module.alloc_id().expect("%1");

        module
            .member_decorate(structure, 1, Decoration::Offset, &[4])
            .expect("fits");

        let words = module.finish();
        let body = &words[5..];

        assert_eq!(body[0] & 0xffff, Word::from(op::MEMBER_DECORATE));
        assert_eq!(body[1], structure.word());
        assert_eq!(body[2], 1, "the member index");
        assert_eq!(body[3], Decoration::Offset.word());
        assert_eq!(body[4], 4);
    }

    #[test]
    fn decorations_land_in_the_annotation_section_before_the_types() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        // Decorated after the type is declared, and must still be emitted before it.
        module
            .decorate(float, Decoration::ArrayStride, &[4])
            .expect("fits");

        let words = module.finish();
        let body = &words[5..];

        assert_eq!(body[0] & 0xffff, Word::from(op::DECORATE));
    }

    #[test]
    fn a_global_variable_names_its_pointer_type_and_its_storage_class() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let pointer = module
            .type_pointer(StorageClass::StorageBuffer, float)
            .expect("pointer");

        let variable = module
            .global_variable(pointer, StorageClass::StorageBuffer)
            .expect("variable");

        let words = module.finish();
        let declaration = crate::decode::body(&words)
            .find(|instruction| instruction.opcode() == op::VARIABLE)
            .expect("the variable was emitted");

        assert_eq!(
            declaration.operands(),
            &[
                pointer.word(),
                variable.word(),
                StorageClass::StorageBuffer.word()
            ]
        );
    }

    #[test]
    fn a_builtin_is_declared_decorated_and_named_in_the_interface_at_once() {
        // The three halves of a usable built-in. A module that declares and decorates one but
        // leaves it out of the interface is invalid — and it is invalid in a way that runs
        // correctly on the drivers here, which is the combination this crate exists to avoid.
        use crate::spec::ExecutionModel;

        let mut module = Module::new(Version::V1_3);
        let main = module.alloc_id().expect("%1");
        module
            .entry_point(ExecutionModel::GlCompute, main, "main")
            .expect("declared");
        let uint = module.type_int(32, false).expect("u32");

        let variable = module
            .builtin_input(BuiltIn::SubgroupLocalInvocationId, uint)
            .expect("declared");

        let words = module.finish();
        let decorated = crate::decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::DECORATE)
            .any(|instruction| {
                instruction.operands()
                    == [
                        variable.word(),
                        Decoration::BuiltIn.word(),
                        BuiltIn::SubgroupLocalInvocationId.word(),
                    ]
            });
        let interfaced = crate::decode::body(&words)
            .find(|instruction| instruction.opcode() == op::ENTRY_POINT)
            .is_some_and(|instruction| instruction.operands().contains(&variable.word()));

        assert!(decorated, "the built-in decoration is missing");
        assert!(interfaced, "the entry point does not name the variable");
    }

    #[test]
    fn the_same_builtin_asked_for_twice_is_one_variable() {
        // Two variables decorated with one built-in is not a duplicate declaration, it is an
        // invalid module — and the second caller is a different operation in a different file,
        // which is exactly the pair that cannot see each other.
        let mut module = Module::new(Version::V1_3);
        let uint = module.type_int(32, false).expect("u32");

        let first = module
            .builtin_input(BuiltIn::SubgroupLocalInvocationId, uint)
            .expect("declared");
        let second = module
            .builtin_input(BuiltIn::SubgroupLocalInvocationId, uint)
            .expect("again");

        assert_eq!(first, second);
        assert_eq!(
            crate::decode::body(&module.finish())
                .filter(|instruction| instruction.opcode() == op::VARIABLE)
                .count(),
            1
        );
    }
}
