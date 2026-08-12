//! What a held reduction will run, decided before any of it is allocated.
//!
//! Split from [`super::held`], which owns Vulkan objects and is therefore excused from the
//! mutation gate as FFI — a mutant that passes a wrong handle kills the process rather than
//! failing a test. None of the arithmetic below is that. A map dispatched over the wrong number of
//! workgroups, or a fold list missing its finisher, is a **wrong number**, and it belongs inside
//! the gate.
//!
//! This is the third time that seam has been worth cutting: `dispatch.rs` gave up 200 lines of
//! conversion that had been sitting behind a blanket FFI exemption, `chain.rs` gave up the copy
//! planning, and this is the reduction's. The rule that keeps producing it: **a file is excused
//! for containing `unsafe`, not for being near it.**

use super::{BadLength, Fold, folds};
use crate::Error;
use crate::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;

/// One pass of a reduction: the module to run and how many workgroups of it.
pub(crate) struct Stage {
    /// The SPIR-V.
    pub(crate) words: Vec<u32>,
    /// Workgroups to dispatch.
    pub(crate) workgroups: u32,
}

/// Every pass a reduction over `elements` runs, in order.
///
/// A `map`, if there is one, then a fold per halving, then the workgroup reduction that finishes.
/// Built entirely before anything is allocated, so a module that will not build fails before a
/// buffer exists rather than half way through.
///
/// # Errors
///
/// [`Error::BadLength`] if `elements` is not a shape this can fold, [`Error::Emit`] if a module
/// cannot be built.
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

    // The map covers every element, one per invocation. `elements` is a power of two of at least
    // two workgroups and `WORKGROUP_SIZE` is a power of two, so the division is exact — and it is
    // computed here rather than taken as an argument precisely so it cannot disagree with the
    // length the folds below were built for.
    if let Some(words) = map {
        stages.push(Stage {
            words: words.to_vec(),
            workgroups: (elements / WORKGROUP_SIZE as usize) as u32,
        });
    }

    for step in &plan {
        stages.push(Stage {
            words: kernels::fold_halves(width, step.half).map_err(Error::Emit)?,
            workgroups: step.workgroups,
        });
    }

    // The last pass crosses between the subgroups of the final workgroup, through shared memory
    // and a barrier, so every one of its invocations holds the whole answer.
    stages.push(Stage {
        words: kernels::workgroup_sum::<F32>(width).map_err(Error::Emit)?,
        workgroups: 1,
    });

    Ok(stages)
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::reduction::dispatches_for;

    /// A width every kernel here can be built for.
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
        // Ordering is the whole of it: a map appended rather than prepended would run *after* the
        // reduction, over one number, and still produce a plausible total.
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
    fn the_folds_halve_and_the_first_one_covers_half_the_input() {
        // Between the map and the finisher, each fold dispatches half the workgroups of the one
        // before it. A list that stopped halving would still reduce, and would reduce the wrong
        // elements.
        let elements = 1_usize << 16;
        let stages = stages(WIDTH, elements, None).expect("planned");

        let folds: Vec<u32> = stages
            .iter()
            .take(stages.len() - 1)
            .map(|stage| stage.workgroups)
            .collect();

        assert_eq!(
            folds[0] as usize * WORKGROUP_SIZE as usize,
            elements / 2,
            "the first fold does not cover half the input"
        );
        for pair in folds.windows(2) {
            assert_eq!(pair[1], pair[0] / 2, "the folds stopped halving: {folds:?}");
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
        // And exactly two workgroups is accepted, so the boundary is the one it says it is.
        assert!(stages(WIDTH, 2 * WORKGROUP_SIZE as usize, None).is_ok());
    }

    #[test]
    fn a_width_no_kernel_can_be_built_for_is_refused_by_the_emitter() {
        assert!(matches!(stages(24, 8_192, None), Err(Error::Emit(_))));
    }
}
