//! The entry point, and the interface list that is not closed until the module is.
//!
//! `OpEntryPoint` names every `Input` and `Output` variable the entry point's call tree reaches,
//! and below SPIR-V 1.4 only those. Which variables those are is not known when the entry point is
//! declared: a kernel that turns out to need [`crate::spec::BuiltIn::SubgroupLocalInvocationId`]
//! learns that while its *body* is being built, long after the interface would have been written
//! out — so the list is held as data and the instruction is rendered from it every time it changes.
//!
//! The alternative was to declare every built-in a kernel might want up front. That costs a module
//! an `Input` variable and a capability it does not use, and a surplus capability is not free:
//! `GroupNonUniform` on a kernel that only scales makes it refuse to run on a device that could
//! have run it.

use super::{BuildError, Id, Module, Section, op};
use crate::encode::{self, Word};
use crate::spec::ExecutionModel;

/// The entry point a module declares, held as data rather than as emitted words.
#[derive(Debug, Clone)]
pub(super) struct Entry {
    model: Word,
    function: Id,
    name: String,
}

impl Module {
    /// Declare `function` as the module's entry point, under `name`.
    ///
    /// The instruction is rendered here and again whenever [`Module::require_interface`] adds to
    /// it, so a caller may declare the entry point before the variables it will end up naming.
    ///
    /// # Errors
    ///
    /// [`BuildError::Encode`] if the name is long enough to overrun the instruction length.
    pub fn entry_point(
        &mut self,
        model: ExecutionModel,
        function: Id,
        name: &str,
    ) -> Result<(), BuildError> {
        self.entry = Some(Entry {
            model: model.word(),
            function,
            name: name.to_owned(),
        });
        self.render_entry_point()
    }

    /// Name `variable` in the entry point's interface, unless it is named already.
    ///
    /// Ordered by first request rather than by id, so the list reads in the order the module
    /// declared the variables.
    ///
    /// **It records the id whether or not there is an entry point yet.** [`crate::lanes::Lanes`]
    /// is handed a `&mut Module` and may be building a fragment that has none — and a caller that
    /// declares its entry point afterwards must still get an interface naming what the fragment
    /// used.
    ///
    /// # Errors
    ///
    /// [`BuildError::Encode`] if the instruction can no longer be encoded.
    pub fn require_interface(&mut self, variable: Id) -> Result<(), BuildError> {
        if self.interface.contains(&variable) {
            return Ok(());
        }
        self.interface.push(variable);
        self.render_entry_point()
    }

    /// Write the entry point instruction out, replacing whatever was there.
    ///
    /// The section holds this one instruction and nothing else, so replacing it is the whole
    /// update. The words are built first and installed only once they encode, which keeps
    /// [`Module::emit`]'s promise that a refused instruction leaves the section as it was.
    fn render_entry_point(&mut self) -> Result<(), BuildError> {
        let Some(entry) = self.entry.clone() else {
            return Ok(());
        };

        let mut operands = vec![entry.model, entry.function.word()];
        encode::literal_string(&mut operands, &entry.name);
        operands.extend(self.interface.iter().map(|variable| variable.word()));

        let mut words = Vec::new();
        encode::instruction(&mut words, op::ENTRY_POINT, &operands)?;

        // `Section` has exactly as many variants as there are sections, so this cannot miss; the
        // `ok_or` keeps that structural rather than resting on it being kept true.
        let section = self
            .sections
            .get_mut(Section::EntryPoint as usize)
            .ok_or(BuildError::IdSpaceExhausted)?;
        *section = words;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::module::Version;

    /// The operands of the one `OpEntryPoint` in `module`.
    fn entry_operands(module: &Module) -> Vec<Word> {
        let words = module.finish();
        decode::body(&words)
            .find(|instruction| instruction.opcode() == op::ENTRY_POINT)
            .expect("an entry point was declared")
            .operands()
            .to_vec()
    }

    #[test]
    fn an_entry_point_names_its_execution_model_function_and_name() {
        let mut module = Module::new(Version::V1_3);
        let main = module.alloc_id().expect("%1");

        module
            .entry_point(ExecutionModel::GlCompute, main, "main")
            .expect("declared");

        let operands = entry_operands(&module);
        assert_eq!(operands[0], ExecutionModel::GlCompute.word());
        assert_eq!(operands[1], main.word());
        assert_eq!(operands[2], 0x6e69_616d, "\"main\", four bytes and a nul");
        assert_eq!(operands[3], 0);
        assert_eq!(operands.len(), 4, "and no interface");
    }

    #[test]
    fn a_variable_declared_after_the_entry_point_still_reaches_its_interface() {
        // The whole reason the instruction is rendered rather than emitted. A built-in a kernel
        // turns out to need is declared while its body is being built, which is long after the
        // entry point — and an interface missing a variable the body loads is what `spirv-val`
        // rejects.
        let mut module = Module::new(Version::V1_3);
        let main = module.alloc_id().expect("%1");
        module
            .entry_point(ExecutionModel::GlCompute, main, "main")
            .expect("declared");

        let variable = module.alloc_id().expect("%2");
        module.require_interface(variable).expect("named");

        let operands = entry_operands(&module);
        assert_eq!(operands.last(), Some(&variable.word()));
        assert_eq!(operands.len(), 5, "one instruction, one entry longer");
    }

    #[test]
    fn a_variable_declared_before_the_entry_point_reaches_it_too() {
        // `Lanes` is handed a module and may be building a fragment that has no entry point yet.
        let mut module = Module::new(Version::V1_3);
        let variable = module.alloc_id().expect("%1");
        module.require_interface(variable).expect("named");

        let main = module.alloc_id().expect("%2");
        module
            .entry_point(ExecutionModel::GlCompute, main, "main")
            .expect("declared");

        assert_eq!(entry_operands(&module).last(), Some(&variable.word()));
    }

    #[test]
    fn naming_a_variable_twice_names_it_once() {
        let mut module = Module::new(Version::V1_3);
        let main = module.alloc_id().expect("%1");
        let variable = module.alloc_id().expect("%2");
        module
            .entry_point(ExecutionModel::GlCompute, main, "main")
            .expect("declared");

        module.require_interface(variable).expect("named");
        module.require_interface(variable).expect("again");

        assert_eq!(entry_operands(&module).len(), 5);
    }

    #[test]
    fn the_interface_keeps_the_order_it_was_declared_in() {
        let mut module = Module::new(Version::V1_3);
        let main = module.alloc_id().expect("%1");
        let first = module.alloc_id().expect("%2");
        let second = module.alloc_id().expect("%3");
        module
            .entry_point(ExecutionModel::GlCompute, main, "main")
            .expect("declared");

        module.require_interface(second).expect("named");
        module.require_interface(first).expect("named");

        let operands = entry_operands(&module);
        assert_eq!(
            &operands[4..],
            &[second.word(), first.word()],
            "first requested, first listed"
        );
    }

    #[test]
    fn a_module_with_no_entry_point_emits_none_however_many_variables_it_names() {
        let mut module = Module::new(Version::V1_3);
        let variable = module.alloc_id().expect("%1");

        module.require_interface(variable).expect("recorded");

        assert_eq!(module.finish().len(), 5, "the header and nothing else");
    }

    #[test]
    fn re_rendering_replaces_the_instruction_rather_than_appending_one() {
        let mut module = Module::new(Version::V1_3);
        let main = module.alloc_id().expect("%1");
        let variable = module.alloc_id().expect("%2");

        module
            .entry_point(ExecutionModel::GlCompute, main, "main")
            .expect("declared");
        module.require_interface(variable).expect("named");

        let words = module.finish();
        assert_eq!(
            decode::body(&words)
                .filter(|instruction| instruction.opcode() == op::ENTRY_POINT)
                .count(),
            1
        );
    }
}
