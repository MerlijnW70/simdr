//! Values that are the same in every lane, and the branches they are allowed to drive.
//!
//! A [`Predicate`] is one boolean per lane. Branching on one splits the subgroup, and a subgroup
//! instruction inside the split answers for whichever lanes happen to be running — see
//! `decisions/DR-0003`. So a branch here does not take a `Predicate`. It takes a [`Uniform`],
//! which only the votes produce, and a vote is uniform by construction: every lane of the
//! subgroup receives the same answer.
//!
//! Per-lane conditionals have a different spelling, [`Lanes::select`], which computes both sides
//! and picks. No divergence, and a reduction inside one is simply not expressible.

use super::{LaneError, Lanes, Predicate};
use crate::module::Id;
use crate::spec::SelectionControl;

/// A boolean every lane of the subgroup agrees on.
///
/// The only way to make one is a vote — [`Lanes::any`], [`Lanes::all`] — which is the point: the
/// type is the argument that a branch on it is safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Uniform {
    id: Id,
}

impl Uniform {
    /// Wrap a value the caller knows is uniform.
    ///
    /// Reserved for the votes, which is why it is not public: the whole guarantee rests on nobody
    /// else being able to make one.
    pub(super) const fn new(id: Id) -> Self {
        Self { id }
    }

    /// The boolean's id.
    #[must_use]
    pub const fn id(self) -> Id {
        self.id
    }
}

impl Lanes<'_> {
    /// True in every lane when the predicate holds in *any* element, as a value a branch accepts.
    ///
    /// # Errors
    ///
    /// As [`Lanes::any`].
    pub fn any_uniform<const LANES: u32>(
        &mut self,
        predicate: Predicate<LANES>,
    ) -> Result<Uniform, LaneError> {
        self.any(predicate).map(Uniform::new)
    }

    /// True in every lane when the predicate holds in *every* element, as a branch condition.
    ///
    /// # Errors
    ///
    /// As [`Lanes::all`].
    pub fn all_uniform<const LANES: u32>(
        &mut self,
        predicate: Predicate<LANES>,
    ) -> Result<Uniform, LaneError> {
        self.all(predicate).map(Uniform::new)
    }

    /// Run `body` only when `condition` holds, and rejoin afterwards.
    ///
    /// The whole subgroup takes the branch together or none of it does, so a reduction inside the
    /// body reduces over every lane — which is the only reading that has a `Simd` meaning.
    ///
    /// Values computed inside the body do not survive the merge: SPIR-V has no mutable locals in
    /// the logical addressing model, and carrying one out needs an `OpPhi` the caller assembles
    /// from [`crate::module::Module::phi`]. What the body is for is a store, or a computation
    /// whose result the body itself writes.
    ///
    /// # Errors
    ///
    /// [`LaneError::Build`] if any instruction cannot be emitted, or whatever `body` returns.
    pub fn if_uniform<F>(&mut self, condition: Uniform, body: F) -> Result<(), LaneError>
    where
        F: FnOnce(&mut Self) -> Result<(), LaneError>,
    {
        let then_block = self.module().alloc_id()?;
        let merge_block = self.module().alloc_id()?;

        // The merge has to be declared before the branch: a consumer needs to know where the arms
        // rejoin before it has walked either of them.
        self.module()
            .selection_merge(merge_block, SelectionControl::None)?;
        self.module()
            .branch_conditional(condition.id(), then_block, merge_block)?;

        self.module().label_at(then_block)?;
        body(self)?;
        self.module().branch(merge_block)?;

        self.module().label_at(merge_block)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::F32;
    use crate::module::{Module, Version, op};

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    fn opcodes(words: &[u32]) -> Vec<u16> {
        decode::body(words)
            .map(|instruction| instruction.opcode())
            .collect()
    }

    /// A uniform condition, built the only way there is.
    fn condition(lanes: &mut Lanes<'_>) -> Uniform {
        let zero = lanes
            .splat_bits::<F32, 32>(0.0_f32.to_bits())
            .expect("zero");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");
        let over = lanes.greater_than(one, zero).expect("compared");
        lanes.any_uniform(over).expect("voted")
    }

    #[test]
    fn a_selection_declares_its_merge_before_it_branches() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);

        lanes.if_uniform(when, |_| Ok(())).expect("emitted");

        let words = module.finish();
        let seen = opcodes(&words);
        let merge = seen
            .iter()
            .position(|opcode| *opcode == op::SELECTION_MERGE)
            .expect("a merge was declared");
        let branch = seen
            .iter()
            .position(|opcode| *opcode == op::BRANCH_CONDITIONAL)
            .expect("a branch was emitted");

        assert!(
            merge < branch,
            "SPIR-V requires the merge instruction immediately before the branch"
        );
    }

    #[test]
    fn a_selection_opens_two_blocks_and_closes_the_first() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);

        lanes.if_uniform(when, |_| Ok(())).expect("emitted");

        let words = module.finish();
        assert_eq!(count(&words, op::LABEL), 2, "the body and the merge");
        assert_eq!(count(&words, op::BRANCH), 1, "the body falls through");
        assert_eq!(count(&words, op::BRANCH_CONDITIONAL), 1);
    }

    #[test]
    fn the_body_lands_between_the_two_labels() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);

        lanes
            .if_uniform(when, |lanes| {
                let value = lanes.splat_bits::<F32, 32>(2.0_f32.to_bits())?;
                lanes.add(value, value)?;
                Ok(())
            })
            .expect("emitted");

        let words = module.finish();
        let seen = opcodes(&words);
        let first_label = seen
            .iter()
            .position(|op| *op == op::LABEL)
            .expect("a label");
        let add = seen
            .iter()
            .position(|opcode| *opcode == op::F_ADD)
            .expect("the body ran");
        let branch = seen
            .iter()
            .position(|opcode| *opcode == op::BRANCH)
            .expect("the body closed");

        assert!(first_label < add && add < branch);
    }

    #[test]
    fn a_body_that_fails_carries_its_error_out() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);

        let refused = lanes.if_uniform(when, |lanes| {
            // 12 lanes has no mapping onto a 32-wide subgroup.
            lanes.splat_bits::<F32, 12>(0)?;
            Ok(())
        });

        assert!(matches!(refused, Err(LaneError::NoMapping { .. })));
    }

    #[test]
    fn selections_nest() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);

        lanes
            .if_uniform(when, |lanes| lanes.if_uniform(when, |_| Ok(())))
            .expect("emitted");

        let words = module.finish();
        assert_eq!(count(&words, op::SELECTION_MERGE), 2);
        assert_eq!(count(&words, op::LABEL), 4);
    }

    #[test]
    fn a_uniform_can_only_come_from_a_vote() {
        // Not a runtime assertion — the point is that `Uniform::new` is private, so this is the
        // only route. The test states the invariant so a future `pub` on it is a visible change.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let when = condition(&mut lanes);

        let words = module.finish();
        let votes = count(&words, op::GROUP_NON_UNIFORM_ANY);

        assert_eq!(votes, 1);
        assert_ne!(when.id().word(), 0);
    }
}
