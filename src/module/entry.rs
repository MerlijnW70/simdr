use super::{BuildError, Id, Module, Section, op};
use crate::encode::{self, Word};
use crate::spec::ExecutionModel;

#[derive(Debug, Clone)]
pub(super) struct Entry {
    model: Word,
    function: Id,
    name: String,
}

impl Module {
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

    pub fn require_interface(&mut self, variable: Id) -> Result<(), BuildError> {
        if self.interface.contains(&variable) {
            return Ok(());
        }
        self.interface.push(variable);
        self.render_entry_point()
    }

    fn render_entry_point(&mut self) -> Result<(), BuildError> {
        let Some(entry) = self.entry.clone() else {
            return Ok(());
        };

        let mut operands = vec![entry.model, entry.function.word()];
        encode::literal_string(&mut operands, &entry.name);
        operands.extend(self.interface.iter().map(|variable| variable.word()));

        let mut words = Vec::new();
        encode::instruction(&mut words, op::ENTRY_POINT, &operands)?;

        *self.section_mut(Section::EntryPoint) = words;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::module::Version;

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
