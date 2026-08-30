use super::{BadLength, Fold, folds};
use crate::Error;
use crate::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;

pub(crate) struct Stage {
    pub(crate) words: Vec<u32>,
    pub(crate) workgroups: u32,
}

pub(crate) fn stages(
    width: u32,
    elements: usize,
    map: Option<&[u32]>,
) -> Result<Vec<Stage>, Error> {
    let minimum = 2 * WORKGROUP_SIZE as usize;

    if !elements.is_power_of_two() {
        return Err(Error::BadLength(BadLength::NotAPowerOfTwo(elements)));
    }
    if elements < minimum {
        return Err(Error::BadLength(BadLength::TooSmall {
            length: elements,
            minimum,
        }));
    }

    let plan: Vec<Fold> = folds(elements);
    let mut stages = Vec::with_capacity(plan.len() + 2);

    if let Some(words) = map {
        stages.push(Stage {
            words: words.to_vec(),
            workgroups: (elements / WORKGROUP_SIZE as usize) as u32,
        });
    }

    for step in &plan {
        stages.push(Stage {
            words: kernels::fold_by(width, step.factor, step.stride).map_err(Error::Emit)?,
            workgroups: step.workgroups,
        });
    }

    stages.push(Stage {
        words: kernels::workgroup_sum::<F32>(width).map_err(Error::Emit)?,
        workgroups: 1,
    });

    Ok(stages)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::reduction::dispatches_for;

    const WIDTH: u32 = 32;

    #[test]
    fn a_plain_reduction_is_one_fold_per_halving_and_one_finisher() {
        for power in 7..=20 {
            let elements = 1_usize << power;
            let stages = stages(WIDTH, elements, None).expect("planned");

            assert_eq!(
                stages.len(),
                dispatches_for(elements),
                "{elements} elements"
            );
        }
    }

    #[test]
    fn a_mapped_reduction_is_exactly_one_pass_longer() {
        let elements = 8_192;
        let plain = stages(WIDTH, elements, None).expect("planned");
        let mapped = stages(WIDTH, elements, Some(&[1, 2, 3])).expect("planned");

        assert_eq!(mapped.len(), plain.len() + 1);
    }

    #[test]
    fn the_map_is_the_first_pass_and_keeps_the_words_it_was_given() {
        let words = vec![0xdead_beef, 0x1234_5678];
        let stages = stages(WIDTH, 8_192, Some(&words)).expect("planned");

        assert_eq!(stages[0].words, words, "the map is not the first pass");
    }

    #[test]
    fn the_map_covers_every_element_one_per_invocation() {
        for power in 7..=20 {
            let elements = 1_usize << power;
            let stages = stages(WIDTH, elements, Some(&[1])).expect("planned");

            assert_eq!(
                stages[0].workgroups as usize * WORKGROUP_SIZE as usize,
                elements,
                "{elements} elements: the map would miss some of them"
            );
        }
    }

    #[test]
    fn the_finisher_is_last_and_runs_one_workgroup() {
        let stages = stages(WIDTH, 8_192, None).expect("planned");
        let last = stages.last().expect("a finisher");

        assert_eq!(last.workgroups, 1);
    }

    #[test]
    fn the_stages_dispatch_exactly_what_the_fold_plan_says() {
        for power in 7..=20 {
            let elements = 1_usize << power;
            let stages = stages(WIDTH, elements, None).expect("planned");
            let plan = crate::reduction::folds(elements);

            let dispatched: Vec<u32> = stages
                .iter()
                .take(stages.len() - 1)
                .map(|stage| stage.workgroups)
                .collect();
            let planned: Vec<u32> = plan.iter().map(|fold| fold.workgroups).collect();

            assert_eq!(dispatched, planned, "at {elements} elements");
            assert_eq!(
                stages.last().map(|stage| stage.workgroups),
                Some(1),
                "the finisher is one workgroup"
            );
        }
    }

    #[test]
    fn a_length_that_is_not_a_power_of_two_is_refused_before_anything_is_built() {
        assert!(matches!(
            stages(WIDTH, 8_000, None),
            Err(Error::BadLength(BadLength::NotAPowerOfTwo(8_000)))
        ));
    }

    #[test]
    fn a_length_below_two_workgroups_is_refused() {
        assert!(matches!(
            stages(WIDTH, WORKGROUP_SIZE as usize, None),
            Err(Error::BadLength(BadLength::TooSmall { .. }))
        ));
        assert!(stages(WIDTH, 2 * WORKGROUP_SIZE as usize, None).is_ok());
    }

    #[test]
    fn a_width_no_kernel_can_be_built_for_is_refused_by_the_emitter() {
        assert!(matches!(stages(24, 8_192, None), Err(Error::Emit(_))));
    }
}
