//! Blocks, branches and the merge instructions that make them structured.
//!
//! SPIR-V does not accept arbitrary jumps. Every conditional has to declare, *before* it
//! branches, where its arms rejoin — the merge block — and every loop has to declare both its
//! merge and its continue target. That is what makes a module's control flow a tree the driver
//! can reason about, and it is why these instructions come in pairs rather than singly.
//!
//! Nothing here checks that the pairs are well formed; `spirv-val` does, and duplicating its
//! judgement would mean maintaining a second opinion about a specification that already has one.
//! What this layer offers is the vocabulary, in an order that makes the pairing hard to forget.

mod barrier;

use super::{BuildError, Id, Module, Section, op};
use crate::encode::Word;
use crate::spec::{LoopControl, SelectionControl};

impl Module {
    /// Declare where a selection's arms rejoin, immediately before branching.
    ///
    /// Must be the second-to-last instruction of its block, with the branch after it. SPIR-V
    /// spells the rule that way round because a consumer needs the merge target before it has
    /// walked either arm.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// Declare a loop's merge block and continue target, immediately before branching into it.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// Jump unconditionally to `target`.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn branch(&mut self, target: Id) -> Result<(), BuildError> {
        self.emit(Section::Function, op::BRANCH, &[target.word()])?;
        self.leave_block();
        Ok(())
    }

    /// Jump to one label or the other according to `condition`.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// Which block instructions are currently being emitted into, if any.
    ///
    /// An `OpPhi` names the block each of its values *arrived through*, and for a value computed
    /// inside a branch that is wherever the body happened to end — not the block the branch
    /// opened, because the body may have branched again. Tracking it here is the only way a
    /// nesting helper can get it right; asking the caller to remember would be asking them to
    /// re-derive what this file already knows.
    ///
    /// `None` after a branch or a return, when no block is open.
    #[must_use]
    pub const fn current_block(&self) -> Option<Id> {
        self.current_block
    }

    /// Open a block with a known label, rather than allocating a new one.
    ///
    /// A branch has to name its targets before they exist, so their labels are allocated first
    /// and opened later — which is what this is for. [`Module::label`] is the other way round and
    /// suits a block nothing jumps to by name.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn label_at(&mut self, id: Id) -> Result<(), BuildError> {
        self.emit(Section::Function, op::LABEL, &[id.word()])?;
        self.enter_block(id);
        Ok(())
    }

    /// A phi at an id allocated earlier.
    ///
    /// A loop's carried value has to be named in its own phi's incoming list — the value the body
    /// produces flows back to the header — so the id exists before either the phi or the value
    /// does. That is circular by nature and this is how SPIR-V expects it to be written.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// Copy `value` to an id allocated earlier.
    ///
    /// The other half of closing a loop's cycle: the body's result has to arrive under the name
    /// its phi was promised.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn copy_object_at(&mut self, id: Id, result_type: Id, value: Id) -> Result<(), BuildError> {
        self.emit(
            Section::Function,
            op::COPY_OBJECT,
            &[result_type.word(), id.word(), value.word()],
        )
    }

    /// Integer add into an id allocated earlier.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
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

    /// A value chosen by which block control arrived from.
    ///
    /// `sources` pairs each incoming value with the label it came through, and every predecessor
    /// of the current block must appear exactly once. It is the only way a value computed inside
    /// a branch survives the merge, because SPIR-V has no mutable locals in the logical addressing
    /// model this crate emits.
    ///
    /// # Errors
    ///
    /// [`BuildError`] if the instruction cannot be emitted.
    pub fn phi(&mut self, result_type: Id, sources: &[(Id, Id)]) -> Result<Id, BuildError> {
        // No capacity hint: it is not observable, so no test can pin it, and an unkillable mutant
        // is worse than the allocation it saves.
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
    // A test may panic — that is how it reports.
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
        // Which is what a forward branch needs: the target has to be nameable before it exists.
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
}
