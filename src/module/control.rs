mod barrier;

use super::{BuildError, Id, Module, Section, op};
use crate::encode::Word;
use crate::spec::{LoopControl, SelectionControl};

impl Module {
    pub fn selection_merge(
        &mut self,
        merge: Id,
        control: SelectionControl,
    ) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::SELECTION_MERGE,
            &[merge.word(), control.word()],
        )
    }

    pub fn loop_merge(
        &mut self,
        merge: Id,
        continue_target: Id,
        control: LoopControl,
    ) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::LOOP_MERGE,
            &[merge.word(), continue_target.word(), control.word()],
        )
    }

    pub fn branch(&mut self, target: Id) -> Result<(), BuildError> {
        self.emit(Section::Function, op::BRANCH, &[target.word()])?;
        self.leave_block();
        Ok(())
    }

    pub fn branch_conditional(
        &mut self,
        condition: Id,
        when_true: Id,
        when_false: Id,
    ) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::BRANCH_CONDITIONAL,
            &[condition.word(), when_true.word(), when_false.word()],
        )?;
        self.leave_block();
        Ok(())
    }

    #[must_use]
    pub const fn current_block(&self) -> Option<Id> {
        self.current_block
    }

    pub fn label_at(&mut self, id: Id) -> Result<(), BuildError> {
        self.emit(Section::Function, op::LABEL, &[id.word()])?;
        self.enter_block(id);
        Ok(())
    }

    pub fn phi_at(
        &mut self,
        id: Id,
        result_type: Id,
        sources: &[(Id, Id)],
    ) -> Result<(), BuildError> {
        let mut operands = vec![result_type.word(), id.word()];
        for &(value, from) in sources {
            operands.push(value.word());
            operands.push(from.word());
        }
        self.emit(Section::Function, op::PHI, &operands)
    }

    pub fn copy_object_at(&mut self, id: Id, result_type: Id, value: Id) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::COPY_OBJECT,
            &[result_type.word(), id.word(), value.word()],
        )
    }

    pub fn i_add_at(
        &mut self,
        id: Id,
        result_type: Id,
        left: Id,
        right: Id,
    ) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::I_ADD,
            &[result_type.word(), id.word(), left.word(), right.word()],
        )
    }

    pub fn phi(&mut self, result_type: Id, sources: &[(Id, Id)]) -> Result<Id, BuildError> {
        let mut operands: Vec<Word> = Vec::new();
        for &(value, from) in sources {
            operands.push(value.word());
            operands.push(from.word());
        }
        self.result_instruction(op::PHI, result_type, &operands)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::module::Version;

    fn operands_of(words: &[Word], opcode: u16) -> Vec<Word> {
        decode::body(words)
            .find(|instruction| instruction.opcode() == opcode)
            .expect("the instruction was emitted")
            .operands()
            .to_vec()
    }

    #[test]
    fn a_selection_merge_names_its_block_and_its_control() {
        let mut module = Module::new(Version::V1_3);
        let merge = module.alloc_id().expect("%1");

        module
            .selection_merge(merge, SelectionControl::None)
            .expect("emitted");

        assert_eq!(
            operands_of(&module.finish(), op::SELECTION_MERGE),
            vec![merge.word(), 0]
        );
    }

    #[test]
    fn a_loop_merge_names_both_of_its_targets() {
        let mut module = Module::new(Version::V1_3);
        let merge = module.alloc_id().expect("%1");
        let carry_on = module.alloc_id().expect("%2");

        module
            .loop_merge(merge, carry_on, LoopControl::None)
            .expect("emitted");

        assert_eq!(
            operands_of(&module.finish(), op::LOOP_MERGE),
            vec![merge.word(), carry_on.word(), 0]
        );
    }

    #[test]
    fn a_conditional_branch_names_the_true_arm_before_the_false_one() {
        let mut module = Module::new(Version::V1_3);
        let condition = module.alloc_id().expect("%1");
        let yes = module.alloc_id().expect("%2");
        let no = module.alloc_id().expect("%3");

        module
            .branch_conditional(condition, yes, no)
            .expect("emitted");

        assert_eq!(
            operands_of(&module.finish(), op::BRANCH_CONDITIONAL),
            vec![condition.word(), yes.word(), no.word()],
            "swapping the arms would compile and invert the program"
        );
    }

    #[test]
    fn a_label_can_be_opened_at_an_id_allocated_earlier() {
        let mut module = Module::new(Version::V1_3);
        let target = module.alloc_id().expect("%1");

        module.branch(target).expect("branched");
        module.label_at(target).expect("opened");

        assert_eq!(
            operands_of(&module.finish(), op::LABEL),
            vec![target.word()]
        );
    }

    #[test]
    fn a_phi_interleaves_each_value_with_the_block_it_came_from() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let from_then = module.alloc_id().expect("%a");
        let then_block = module.alloc_id().expect("%b");
        let from_else = module.alloc_id().expect("%c");
        let else_block = module.alloc_id().expect("%d");

        let merged = module
            .phi(float, &[(from_then, then_block), (from_else, else_block)])
            .expect("emitted");

        assert_eq!(
            operands_of(&module.finish(), op::PHI),
            vec![
                float.word(),
                merged.word(),
                from_then.word(),
                then_block.word(),
                from_else.word(),
                else_block.word()
            ],
            "value then label, value then label — not both values and then both labels"
        );
    }

    #[test]
    fn a_phi_with_one_source_is_still_well_formed() {
        let mut module = Module::new(Version::V1_3);
        let float = module.type_float(32).expect("f32");
        let value = module.alloc_id().expect("%a");
        let from = module.alloc_id().expect("%b");

        module.phi(float, &[(value, from)]).expect("emitted");

        assert_eq!(operands_of(&module.finish(), op::PHI).len(), 4);
    }

    #[test]
    fn the_forms_that_write_into_an_id_allocated_earlier_name_that_id_as_their_result() {
        let mut module = Module::new(Version::V1_3);
        let uint = module.type_int(32, false).expect("u32");
        let left = module.constant_u32(1).expect("1");
        let right = module.constant_u32(2).expect("2");

        let copied = module.alloc_id().expect("%copied");
        let summed = module.alloc_id().expect("%summed");
        let merged = module.alloc_id().expect("%merged");
        let from_entry = module.alloc_id().expect("%entry");

        module
            .copy_object_at(copied, uint, left)
            .expect("copy_object_at");
        module
            .i_add_at(summed, uint, left, right)
            .expect("i_add_at");
        module
            .phi_at(merged, uint, &[(left, from_entry)])
            .expect("phi_at");

        let words = module.finish();
        assert_eq!(
            operands_of(&words, op::COPY_OBJECT),
            vec![uint.word(), copied.word(), left.word()]
        );
        assert_eq!(
            operands_of(&words, op::I_ADD),
            vec![uint.word(), summed.word(), left.word(), right.word()]
        );
        assert_eq!(
            operands_of(&words, op::PHI),
            vec![uint.word(), merged.word(), left.word(), from_entry.word()]
        );
    }
}
