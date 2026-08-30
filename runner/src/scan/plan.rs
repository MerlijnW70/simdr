//! ```text
//!   2^20 elements   →  16384 block totals  →  256  →  4  →  one workgroup scans it
//! ```

use crate::Error;
use crate::kernels::WORKGROUP_SIZE;
use crate::reduction::BadLength;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Level {
    pub(crate) elements: usize,
    pub(crate) capacity: usize,
    pub(crate) workgroups: u32,
}

pub(crate) fn levels(elements: usize) -> Result<Vec<Level>, Error> {
    let block = WORKGROUP_SIZE as usize;

    if elements < block || !elements.is_multiple_of(block) {
        return Err(Error::BadLength(BadLength::TooSmall {
            length: elements,
            minimum: block,
        }));
    }

    let mut levels = Vec::new();
    let mut count = elements / block;

    loop {
        let capacity = count.next_multiple_of(block);
        let workgroups = (capacity / block) as u32;
        levels.push(Level {
            elements: count,
            capacity,
            workgroups,
        });

        if workgroups <= 1 {
            return Ok(levels);
        }
        count = workgroups as usize;
    }
}

pub(crate) fn dispatches(levels: usize, mapped: bool) -> usize {
    let map = usize::from(mapped);
    let up = 1 + levels.saturating_sub(1);
    let top = 1;
    let down = levels;
    map + up + top + down
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::{dispatches, levels};
    use crate::Error;
    use crate::kernels::WORKGROUP_SIZE;

    const BLOCK: usize = WORKGROUP_SIZE as usize;

    #[test]
    fn one_workgroups_worth_of_blocks_needs_one_level() {
        let plan = levels(BLOCK * BLOCK).expect("planned");

        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].elements, BLOCK);
        assert_eq!(plan[0].workgroups, 1);
        assert_eq!(dispatches(plan.len(), false), 3);
        assert_eq!(dispatches(plan.len(), true), 4, "and one more with a map");
    }

    #[test]
    fn a_million_elements_need_three_levels_and_seven_dispatches() {
        let plan = levels(1 << 20).expect("planned");

        let counts: Vec<usize> = plan.iter().map(|level| level.elements).collect();
        assert_eq!(counts, vec![16384, 256, 4]);
        assert_eq!(dispatches(plan.len(), false), 7);
    }

    #[test]
    fn every_level_holds_one_total_per_workgroup_of_the_one_below() {
        for elements in [BLOCK, BLOCK * 2, BLOCK * BLOCK, 1 << 20, 1 << 24] {
            let plan = levels(elements).expect("planned");

            assert_eq!(
                plan[0].elements,
                elements / BLOCK,
                "the first level is one total per input block, at {elements}"
            );
            for pair in plan.windows(2) {
                assert_eq!(
                    pair[1].elements, pair[0].workgroups as usize,
                    "at {elements}: {pair:?}"
                );
            }
        }
    }

    #[test]
    fn the_top_level_is_always_exactly_one_workgroup() {
        for elements in [BLOCK, BLOCK * 3, BLOCK * BLOCK, 1 << 20, 1 << 24] {
            let plan = levels(elements).expect("planned");
            assert_eq!(plan.last().expect("a level").workgroups, 1, "at {elements}");
        }
    }

    #[test]
    fn every_buffer_is_a_whole_number_of_workgroups_and_holds_its_level() {
        for elements in [BLOCK, BLOCK * 3, BLOCK * 100, 1 << 20] {
            for level in levels(elements).expect("planned") {
                assert!(
                    level.capacity >= level.elements,
                    "a level must fit in its own buffer: {level:?}"
                );
                assert!(
                    level.capacity.is_multiple_of(BLOCK),
                    "a partial workgroup would read past the buffer: {level:?}"
                );
                assert_eq!(level.capacity / BLOCK, level.workgroups as usize);
            }
        }
    }

    #[test]
    fn a_length_that_is_not_a_whole_number_of_workgroups_is_refused() {
        for elements in [0_usize, 1, BLOCK - 1, BLOCK + 1, BLOCK * 3 + 7] {
            assert!(
                matches!(levels(elements), Err(Error::BadLength(_))),
                "{elements} was accepted"
            );
        }
    }

    #[test]
    fn the_levels_shrink_until_they_stop() {
        let plan = levels(1 << 24).expect("planned");

        for pair in plan.windows(2) {
            assert!(pair[1].elements < pair[0].elements, "{pair:?}");
        }
        assert!(plan.len() >= 3, "2^24 is deep enough to be worth the check");
    }

    #[test]
    fn the_dispatch_count_is_the_sum_of_the_three_phases() {
        for count in 1..8_usize {
            assert_eq!(dispatches(count, false), 2 * count + 1, "{count} levels");
            assert_eq!(
                dispatches(count, true),
                2 * count + 2,
                "{count} levels and a map"
            );
        }
    }
}
