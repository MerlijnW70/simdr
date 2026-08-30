pub mod op;

mod arithmetic;
mod atomic;
mod body;
mod constants;
mod control;
mod dot;
mod entry;
mod extended;
mod globals;
mod layout;
mod specialize;
mod subgroup;
mod types;

pub use self::layout::{BuildError, Id, Section, Version};
pub use self::subgroup::Reduction;

use self::constants::ConstantKey;
use self::entry::Entry;
use self::types::TypeKey;
use crate::encode::{self, Word};
use crate::spec::Capability;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Module {
    version: Version,
    next_id: u32,
    sections: [Vec<Word>; Self::SECTION_COUNT],
    types: HashMap<TypeKey, Id>,
    constants: HashMap<ConstantKey, Id>,
    capabilities: HashSet<Word>,
    extensions: HashSet<String>,
    ext_imports: HashMap<String, Id>,
    current_block: Option<Id>,
    entry: Option<Entry>,
    interface: Vec<Id>,
    builtins: HashMap<Word, Id>,
}

impl Module {
    const SECTION_COUNT: usize = 10;
    const HEADER_WORDS: usize = 5;

    const GENERATOR: Word = 0;

    #[must_use]
    pub fn new(version: Version) -> Self {
        Self {
            version,
            next_id: 1,
            sections: Default::default(),
            types: HashMap::new(),
            constants: HashMap::new(),
            capabilities: HashSet::new(),
            extensions: HashSet::new(),
            ext_imports: HashMap::new(),
            current_block: None,
            entry: None,
            interface: Vec::new(),
            builtins: HashMap::new(),
        }
    }

    pub(crate) const fn enter_block(&mut self, id: Id) {
        self.current_block = Some(id);
    }

    pub(crate) const fn leave_block(&mut self) {
        self.current_block = None;
    }

    pub fn require_capability(&mut self, capability: Capability) -> Result<(), BuildError> {
        if !self.capabilities.insert(capability.word()) {
            return Ok(());
        }
        self.emit(Section::Capability, op::CAPABILITY, &[capability.word()])?;

        if let Some(extension) = capability.extension() {
            self.require_extension(extension)?;
        }
        Ok(())
    }

    pub fn require_extension(&mut self, name: &str) -> Result<(), BuildError> {
        if !self.extensions.insert(name.to_owned()) {
            return Ok(());
        }
        let mut operands = Vec::new();
        encode::literal_string(&mut operands, name);
        self.emit(Section::Extension, op::EXTENSION, &operands)
    }

    pub fn alloc_id(&mut self) -> Result<Id, BuildError> {
        let id = Id::new(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(BuildError::IdSpaceExhausted)?;
        Ok(id)
    }

    pub fn emit(
        &mut self,
        section: Section,
        opcode: u16,
        operands: &[Word],
    ) -> Result<(), BuildError> {
        encode::instruction(self.section_mut(section), opcode, operands)?;
        Ok(())
    }

    fn section_mut(&mut self, section: Section) -> &mut Vec<Word> {
        let [
            capability,
            extension,
            ext_inst_import,
            memory_model,
            entry_point,
            execution_mode,
            debug,
            annotation,
            type_constant_variable,
            function,
        ] = &mut self.sections;

        match section {
            Section::Capability => capability,
            Section::Extension => extension,
            Section::ExtInstImport => ext_inst_import,
            Section::MemoryModel => memory_model,
            Section::EntryPoint => entry_point,
            Section::ExecutionMode => execution_mode,
            Section::Debug => debug,
            Section::Annotation => annotation,
            Section::TypeConstantVariable => type_constant_variable,
            Section::Function => function,
        }
    }

    pub fn name(&mut self, id: Id, text: &str) -> Result<(), BuildError> {
        let mut operands = vec![id.word()];
        encode::literal_string(&mut operands, text);
        self.emit(Section::Debug, op::NAME, &operands)
    }

    #[must_use]
    pub fn finish(&self) -> Vec<Word> {
        let body: usize = self.sections.iter().map(Vec::len).sum();
        let mut words = Vec::with_capacity(body.saturating_add(Self::HEADER_WORDS));

        words.push(encode::MAGIC);
        words.push(self.version.word());
        words.push(Self::GENERATOR);
        words.push(self.next_id);
        words.push(0);

        for section in &self.sections {
            words.extend_from_slice(section);
        }
        words
    }

    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let words = self.finish();
        let mut bytes = Vec::with_capacity(words.len().saturating_mul(4));
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;

    #[test]
    fn a_module_begins_with_the_magic_number() {
        let words = Module::new(Version::V1_0).finish();

        assert_eq!(words.first(), Some(&0x0723_0203));
    }

    #[test]
    fn an_empty_module_is_nothing_but_its_header() {
        assert_eq!(Module::new(Version::V1_0).finish().len(), 5);
    }

    #[test]
    fn ids_start_at_one_because_zero_means_no_id() {
        let mut module = Module::new(Version::V1_0);

        assert_eq!(module.alloc_id().expect("first id").word(), 1);
        assert_eq!(module.alloc_id().expect("second id").word(), 2);
    }

    #[test]
    fn the_bound_is_one_past_the_largest_id_issued() {
        let mut module = Module::new(Version::V1_0);
        for _ in 0..3 {
            module.alloc_id().expect("id space is not exhausted");
        }

        assert_eq!(module.finish()[3], 4);
    }

    #[test]
    fn an_empty_modules_bound_is_one_so_that_no_id_is_below_it() {
        assert_eq!(Module::new(Version::V1_0).finish()[3], 1);
    }

    #[test]
    fn sections_are_emitted_in_layout_order_whatever_order_they_were_filled_in() {
        let mut module = Module::new(Version::V1_0);

        module
            .emit(Section::Function, op::RETURN, &[])
            .expect("one word fits");
        module
            .emit(Section::Capability, op::CAPABILITY, &[1])
            .expect("two words fit");

        let words = module.finish();
        let body = &words[5..];

        assert_eq!(body[0] & 0xffff, Word::from(op::CAPABILITY));
        assert_eq!(body[2] & 0xffff, Word::from(op::RETURN));
    }

    #[test]
    fn a_refused_instruction_leaves_the_section_as_it_was() {
        let mut module = Module::new(Version::V1_0);
        let too_many = vec![0; usize::from(u16::MAX)];

        let refused = module.emit(Section::Debug, op::CAPABILITY, &too_many);

        assert!(matches!(refused, Err(BuildError::Encode(_))));
        assert_eq!(module.finish().len(), 5, "nothing was appended");
    }

    #[test]
    fn a_capability_asked_for_twice_is_declared_once() {
        let mut module = Module::new(Version::V1_3);

        module
            .require_capability(Capability::Shader)
            .expect("declared");
        module
            .require_capability(Capability::Shader)
            .expect("again");

        assert_eq!(module.finish().len(), 5 + 2);
    }

    #[test]
    fn a_capability_that_needs_an_extension_declares_it_too() {
        let mut module = Module::new(Version::V1_3);

        module
            .require_capability(Capability::StorageBuffer8BitAccess)
            .expect("declared");

        assert_eq!(
            decode::opcodes(&module.finish()),
            vec![op::CAPABILITY, op::EXTENSION],
            "the capability is unusable at this version without the extension beside it"
        );
    }

    #[test]
    fn a_capability_that_needs_no_extension_declares_none() {
        let mut module = Module::new(Version::V1_3);

        module
            .require_capability(Capability::StorageBuffer16BitAccess)
            .expect("declared");

        assert_eq!(
            decode::opcodes(&module.finish()),
            vec![op::CAPABILITY],
            "16-bit storage is core in SPIR-V 1.3, and a surplus extension is not free"
        );
    }

    #[test]
    fn an_extension_asked_for_twice_is_declared_once() {
        let mut module = Module::new(Version::V1_3);

        module
            .require_extension("SPV_KHR_8bit_storage")
            .expect("declared");
        module
            .require_extension("SPV_KHR_8bit_storage")
            .expect("again");

        assert_eq!(decode::opcodes(&module.finish()), vec![op::EXTENSION]);
    }

    #[test]
    fn an_extension_sorts_after_the_capabilities_and_before_everything_else() {
        let mut module = Module::new(Version::V1_3);
        module.type_float(32).expect("f32");
        module
            .require_capability(Capability::StorageBuffer8BitAccess)
            .expect("declared");
        module
            .require_capability(Capability::Shader)
            .expect("shader");

        assert_eq!(
            decode::opcodes(&module.finish()),
            vec![
                op::CAPABILITY,
                op::CAPABILITY,
                op::EXTENSION,
                op::TYPE_FLOAT
            ]
        );
    }

    #[test]
    fn a_name_lands_in_the_debug_section_and_carries_its_string() {
        let mut module = Module::new(Version::V1_3);
        let id = module.alloc_id().expect("%1");

        module.name(id, "main").expect("fits");

        let words = module.finish();
        let body = &words[5..];

        assert_eq!(body[0] & 0xffff, Word::from(op::NAME));
        assert_eq!(body[1], id.word());
        assert_eq!(body[2], 0x6e69_616d);
    }

    #[test]
    fn the_bytes_are_little_endian_whatever_the_host_is() {
        let bytes = Module::new(Version::V1_0).to_bytes();

        assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07]);
        assert_eq!(bytes.len(), 5 * 4);
    }

    #[test]
    fn the_minimal_compute_module_encodes_word_for_word() {
        let mut module = Module::new(Version::V1_0);
        let main = module.alloc_id().expect("%1");
        let void = module.alloc_id().expect("%2");
        let fn_type = module.alloc_id().expect("%3");
        let entry_block = module.alloc_id().expect("%4");

        module
            .emit(Section::Capability, op::CAPABILITY, &[1])
            .expect("fits");
        module
            .emit(Section::MemoryModel, op::MEMORY_MODEL, &[0, 1])
            .expect("fits");

        let mut entry = vec![5, main.word()];
        encode::literal_string(&mut entry, "main");
        module
            .emit(Section::EntryPoint, op::ENTRY_POINT, &entry)
            .expect("fits");

        module
            .emit(
                Section::ExecutionMode,
                op::EXECUTION_MODE,
                &[main.word(), 17, 1, 1, 1],
            )
            .expect("fits");

        module
            .emit(Section::TypeConstantVariable, op::TYPE_VOID, &[void.word()])
            .expect("fits");
        module
            .emit(
                Section::TypeConstantVariable,
                op::TYPE_FUNCTION,
                &[fn_type.word(), void.word()],
            )
            .expect("fits");

        module
            .emit(
                Section::Function,
                op::FUNCTION,
                &[void.word(), main.word(), 0, fn_type.word()],
            )
            .expect("fits");
        module
            .emit(Section::Function, op::LABEL, &[entry_block.word()])
            .expect("fits");
        module
            .emit(Section::Function, op::RETURN, &[])
            .expect("fits");
        module
            .emit(Section::Function, op::FUNCTION_END, &[])
            .expect("fits");

        #[rustfmt::skip]
        let expected = vec![
            0x0723_0203,
            0x0001_0000,
            0,
            5,
            0,
            (2 << 16) | 17, 1,
            (3 << 16) | 14, 0, 1,
            (5 << 16) | 15, 5, 1, 0x6e69_616d, 0x0000_0000,
            (6 << 16) | 16, 1, 17, 1, 1, 1,
            (2 << 16) | 19, 2,
            (3 << 16) | 33, 3, 2,
            (5 << 16) | 54, 2, 1, 0, 3,
            (2 << 16) | 248, 4,
            (1 << 16) | 253,
            (1 << 16) | 56,
        ];

        assert_eq!(module.finish(), expected);
    }
}
