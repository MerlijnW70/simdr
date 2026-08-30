use super::{Element, LaneError, Lanes, Mapping, Predicate, Vector};
use crate::module::Id;
use crate::spec::Capability;

impl Lanes<'_> {
    pub fn any<const LANES: u32>(&mut self, predicate: Predicate<LANES>) -> Result<Id, LaneError> {
        self.vote::<LANES>("any", predicate, Combine::Or)
    }

    pub fn all<const LANES: u32>(&mut self, predicate: Predicate<LANES>) -> Result<Id, LaneError> {
        self.vote::<LANES>("all", predicate, Combine::And)
    }

    pub fn all_equal<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        if let Mapping::Clusters { .. } = self.mapping::<LANES>()? {
            return Err(LaneError::NoSuchForm {
                operation: "all_equal",
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

        let strips = vector.strips().to_vec();
        let first = strips.first().copied().ok_or(LaneError::no_strips())?;
        let lanes_agree = self.module().subgroup_all_equal(boolean, scope, first)?;

        let mut same_here: Option<Id> = None;
        for &strip in strips.iter().skip(1) {
            let matched = self.module().binary(T::EQUAL, boolean, strip, first)?;
            same_here = Some(match same_here {
                None => matched,
                Some(previous) => self.module().logical_and(boolean, previous, matched)?,
            });
        }

        let Some(same_here) = same_here else {
            return Ok(lanes_agree);
        };
        let strips_agree = self.module().subgroup_all(boolean, scope, same_here)?;
        Ok(self
            .module()
            .logical_and(boolean, lanes_agree, strips_agree)?)
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Combine {
    Or,
    And,
}

#[cfg(test)]
mod tests {
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
    fn a_vote_on_a_value_is_one_instruction_and_asks_no_predicate() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.all_equal(value).expect("voted");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_ALL_EQUAL), 1);
        assert_eq!(
            count(&words, op::F_ORD_GREATER_THAN),
            0,
            "no comparison was needed to ask it"
        );
        assert_eq!(
            count(&words, op::GROUP_NON_UNIFORM_ALL),
            0,
            "and it is not `all`"
        );
    }

    #[test]
    fn a_vote_on_a_value_declares_the_vote_capability_and_not_the_arithmetic_one() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");
        lanes.all_equal(value).expect("voted");

        let declared: Vec<u32> = decode::body(&module.finish())
            .filter(|instruction| instruction.opcode() == op::CAPABILITY)
            .filter_map(|instruction| instruction.operands().first().copied())
            .collect();

        assert!(declared.contains(&Capability::GroupNonUniform.word()));
        assert!(declared.contains(&Capability::GroupNonUniformVote.word()));
        assert!(!declared.contains(&Capability::GroupNonUniformArithmetic.word()));
    }

    #[test]
    fn a_strip_mined_vote_on_a_value_asks_both_halves_of_the_question() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let wide = lanes
            .splat_bits::<F32, 128>(1.0_f32.to_bits())
            .expect("splat");

        lanes.all_equal(wide).expect("voted");

        let words = module.finish();
        assert_eq!(
            count(&words, op::GROUP_NON_UNIFORM_ALL_EQUAL),
            1,
            "the lanes are asked about one strip, not about each"
        );
        assert_eq!(
            count(&words, op::F_ORD_EQUAL),
            3,
            "every strip against the first"
        );
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_ALL), 1);
        assert_eq!(
            count(&words, op::LOGICAL_AND),
            3,
            "two folding the comparisons, one joining the two halves"
        );
    }

    #[test]
    fn a_whole_subgroup_vote_on_a_value_stays_one_instruction() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.all_equal(value).expect("voted");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_ALL_EQUAL), 1);
        assert_eq!(count(&words, op::F_ORD_EQUAL), 0);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_ALL), 0);
        assert_eq!(count(&words, op::LOGICAL_AND), 0);
    }

    #[test]
    fn a_vote_on_a_value_refuses_the_one_mapping_it_would_answer_wrongly_for() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let narrow = lanes.splat_bits::<F32, 8>(0).expect("splat");
        let wide = lanes.splat_bits::<F32, 128>(0).expect("splat");

        assert!(matches!(
            lanes.all_equal(narrow).err(),
            Some(LaneError::NoSuchForm {
                operation: "all_equal",
                ..
            })
        ));

        assert!(lanes.all_equal(wide).is_ok());
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

        let ballot = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_BALLOT)
            .expect("a ballot was emitted")
            .operands()
            .to_vec();

        let vector = declaration(&words, op::TYPE_VECTOR, ballot[0]);
        assert_eq!(vector[2], 4, "four components");

        let component = declaration(&words, op::TYPE_INT, vector[1]);
        assert_eq!(component[1], 32);
        assert_eq!(component[2], 0, "unsigned");
    }

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
