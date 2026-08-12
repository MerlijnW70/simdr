//! Votes over a predicate — what `Mask::any`, `Mask::all` and a ballot become.
//!
//! A [`Predicate`] is one boolean per element. Asking a question *about* the whole predicate
//! crosses lanes, so unlike the comparison that produced it, these cost an instruction.
//!
//! # What a strip-mined vote means
//!
//! With one element per lane, `any` is one `OpGroupNonUniformAny` and that is the whole answer.
//! With several, each strip has its own vote and they have to be combined — this folds them with
//! a logical or (for `any`) or and (for `all`), which is the only reading that matches what
//! `Mask::any` means for a `Simd` of that width.
//!
//! # Scope
//!
//! Every vote here is over the **subgroup**, and for a clustered mapping that is wider than the
//! vector. `Simd<f32, 8>` on a 32-lane subgroup has four vectors sharing the hardware, and
//! `OpGroupNonUniformAny` has no clustered form — so a vote would answer for all four at once.
//! [`Lanes::any`] refuses that rather than returning a plausible wrong answer.

use super::{LaneError, Lanes, Mapping, Predicate};
use crate::module::Id;
use crate::spec::Capability;

impl Lanes<'_> {
    /// True in every lane when the predicate holds in *any* element — `Mask::any`.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchForm`] for a clustered mapping, where the vote would span vectors that
    /// are not this one. Otherwise [`LaneError::Build`].
    pub fn any<const LANES: u32>(&mut self, predicate: Predicate<LANES>) -> Result<Id, LaneError> {
        self.vote::<LANES>("any", predicate, Combine::Or)
    }

    /// True in every lane when the predicate holds in *every* element — `Mask::all`.
    ///
    /// # Errors
    ///
    /// As [`Lanes::any`].
    pub fn all<const LANES: u32>(&mut self, predicate: Predicate<LANES>) -> Result<Id, LaneError> {
        self.vote::<LANES>("all", predicate, Combine::And)
    }

    /// The predicate's first strip gathered into a bitmask, one bit per lane.
    ///
    /// A four-component vector of `u32`, which is 128 bits — enough for the widest subgroup any
    /// implementation reports. This is the raw `Mask` a caller inspects when the boolean answers
    /// above are not enough.
    ///
    /// Only the first strip: a ballot is per *lane*, and a strip-mined vector has more elements
    /// than lanes, so there is no single mask that describes it.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoSuchForm`] when the vector is strip-mined, otherwise [`LaneError::Build`].
    pub fn ballot<const LANES: u32>(
        &mut self,
        predicate: Predicate<LANES>,
    ) -> Result<Id, LaneError> {
        if matches!(self.mapping::<LANES>()?, Mapping::Strips { .. }) {
            return Err(LaneError::NoSuchForm {
                operation: "ballot",
                because: "a ballot has one bit per lane and a strip-mined vector has more \
                          elements than lanes",
            });
        }

        let uint = self.module().type_int(32, false)?;
        let mask = self.module().type_vector(uint, 4)?;
        let scope = self.scope();
        self.module()
            .require_capability(Capability::GroupNonUniform)?;
        self.module()
            .require_capability(Capability::GroupNonUniformBallot)?;

        let first = predicate
            .strips()
            .first()
            .copied()
            .ok_or(LaneError::no_strips())?;
        Ok(self.module().subgroup_ballot(mask, scope, first)?)
    }

    /// One vote per strip, folded together.
    fn vote<const LANES: u32>(
        &mut self,
        name: &'static str,
        predicate: Predicate<LANES>,
        combine: Combine,
    ) -> Result<Id, LaneError> {
        if let Mapping::Clusters { .. } = self.mapping::<LANES>()? {
            return Err(LaneError::NoSuchForm {
                operation: name,
                because: "SPIR-V's votes have no clustered form, so the answer would cover \
                          every vector sharing the subgroup rather than this one",
            });
        }

        let boolean = self.module().type_bool()?;
        let scope = self.scope();
        self.module()
            .require_capability(Capability::GroupNonUniform)?;
        self.module()
            .require_capability(Capability::GroupNonUniformVote)?;

        let mut answer = None;
        for &strip in predicate.strips() {
            let voted = match combine {
                Combine::Or => self.module().subgroup_any(boolean, scope, strip)?,
                Combine::And => self.module().subgroup_all(boolean, scope, strip)?,
            };
            answer = Some(match answer {
                None => voted,
                Some(previous) => match combine {
                    Combine::Or => self.module().logical_or(boolean, previous, voted)?,
                    Combine::And => self.module().logical_and(boolean, previous, voted)?,
                },
            });
        }

        answer.ok_or(LaneError::no_strips())
    }
}

/// How the per-strip votes are folded into one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Combine {
    Or,
    And,
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

    /// A predicate over `LANES`, built from a comparison so the strips are real.
    fn predicate<const LANES: u32>(lanes: &mut Lanes<'_>) -> Predicate<LANES> {
        let zero = lanes
            .splat_bits::<F32, LANES>(0.0_f32.to_bits())
            .expect("zero");
        let one = lanes
            .splat_bits::<F32, LANES>(1.0_f32.to_bits())
            .expect("one");
        lanes.greater_than(one, zero).expect("compared")
    }

    #[test]
    fn a_full_width_any_is_one_vote() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let mask = predicate::<32>(&mut lanes);

        lanes.any(mask).expect("voted");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_ANY), 1);
        assert_eq!(count(&words, op::LOGICAL_OR), 0, "nothing to fold");
    }

    #[test]
    fn a_strip_mined_any_votes_per_strip_and_ors_them() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let mask = predicate::<128>(&mut lanes);

        lanes.any(mask).expect("voted");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_ANY), 4);
        assert_eq!(count(&words, op::LOGICAL_OR), 3, "four votes, three folds");
    }

    #[test]
    fn all_uses_the_other_instruction_and_the_other_fold() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let mask = predicate::<64>(&mut lanes);

        lanes.all(mask).expect("voted");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_ALL), 2);
        assert_eq!(count(&words, op::LOGICAL_AND), 1);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_ANY), 0);
    }

    #[test]
    fn a_clustered_vote_is_refused_because_it_would_answer_for_the_neighbours() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let mask = predicate::<8>(&mut lanes);

        assert!(matches!(
            lanes.any(mask).err(),
            Some(LaneError::NoSuchForm { .. })
        ));
        assert!(matches!(
            lanes.all(mask).err(),
            Some(LaneError::NoSuchForm { .. })
        ));
    }

    #[test]
    fn a_ballot_yields_a_four_component_mask() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let mask = predicate::<32>(&mut lanes);

        lanes.ballot(mask).expect("ballot");

        let words = module.finish();

        // Follow the ballot's own result type rather than the first declaration of each kind:
        // `Lanes::new` already declares an unsigned integer for the scope constant, so looking
        // at "the first `OpTypeInt`" would find that one and say nothing about this instruction.
        let ballot = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_BALLOT)
            .expect("a ballot was emitted")
            .operands()
            .to_vec();

        let vector = declaration(&words, op::TYPE_VECTOR, ballot[0]);
        assert_eq!(vector[2], 4, "four components");

        // And they are unsigned: `OpGroupNonUniformBallot`'s result is a vector of four 32-bit
        // *unsigned* integers, and a signed component type is a different type and the wrong one.
        let component = declaration(&words, op::TYPE_INT, vector[1]);
        assert_eq!(component[1], 32);
        assert_eq!(component[2], 0, "unsigned");
    }

    /// The operands of the `opcode` instruction whose result id is `id`.
    fn declaration(words: &[u32], opcode: u16, id: u32) -> Vec<u32> {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .find(|instruction| instruction.operands().first() == Some(&id))
            .expect("the type was declared")
            .operands()
            .to_vec()
    }

    #[test]
    fn a_strip_mined_ballot_is_refused() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let mask = predicate::<64>(&mut lanes);

        assert!(matches!(
            lanes.ballot(mask).err(),
            Some(LaneError::NoSuchForm { .. })
        ));
    }

    #[test]
    fn a_vote_declares_the_vote_capability_and_not_the_ballot_one() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let mask = predicate::<32>(&mut lanes);
        lanes.any(mask).expect("voted");

        let words = module.finish();
        let declared: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CAPABILITY)
            .filter_map(|instruction| instruction.operands().first().copied())
            .collect();

        assert!(declared.contains(&Capability::GroupNonUniformVote.word()));
        assert!(!declared.contains(&Capability::GroupNonUniformBallot.word()));
    }
}
