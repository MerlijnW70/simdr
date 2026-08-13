//! How many levels a scan of a given length needs, and how big each one is.
//!
//! Split from [`super::held`] for the reason that keeps producing this seam: **a file is excused
//! from the mutation gate for containing `unsafe`, not for being near it.** Everything below is
//! arithmetic, and a level sized wrongly or a dispatch count off by one is a *wrong number* rather
//! than a crash — which is exactly what the gate is for. This is the fourth time that cut has been
//! worth making.
//!
//! # The shape of a long scan
//!
//! One workgroup scans [`WORKGROUP_SIZE`] elements. Past that, the input is cut into blocks, each
//! block is scanned on its own, and each block is then told the total of every block before it.
//! Those block totals are themselves a shorter array needing a scan — so the same three steps run
//! again one level up, until a level is short enough for one workgroup to finish.
//!
//! ```text
//!   2^20 elements   →  16384 block totals  →  256  →  4  →  one workgroup scans it
//! ```
//!
//! Three levels, and `2 × 3 + 1 = 7` dispatches. The count is decided here and nowhere else.
//!
//! # Why the buffers are wider than the level
//!
//! A level of 4 totals is still scanned by a workgroup of 64 invocations, because that is what a
//! workgroup is. The buffer is therefore rounded up to a whole workgroup and the tail is written
//! once, at build time, with zeros.
//!
//! **Not because the arithmetic needs it.** Garbage in the padding would propagate only into
//! positions that are themselves padding — a block's offset is the sum of the totals *before* it,
//! so rubbish at high indices cannot reach a low one. It is zeroed because the alternative is a
//! kernel reading memory nobody wrote, and this project has already recorded three tests that
//! assumed such memory is zero on the two devices where it happened to be, and was not on
//! lavapipe.

use crate::Error;
use crate::kernels::WORKGROUP_SIZE;
use crate::reduction::BadLength;

/// One level of block totals, above the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Level {
    /// How many totals this level actually holds — one per workgroup of the level below.
    pub(crate) elements: usize,
    /// How many its buffer holds, rounded up to a whole workgroup.
    pub(crate) capacity: usize,
    /// How many workgroups scan it, and therefore how many totals the level above holds.
    pub(crate) workgroups: u32,
}

/// The levels a scan over `elements` needs, innermost first.
///
/// The last one always takes a single workgroup, which is what ends the recursion: there is
/// nothing above a level one workgroup can scan by itself.
///
/// # Errors
///
/// [`Error::BadLength`] unless `elements` is a whole number of workgroups and at least one. A scan
/// of a partial block is not refused because it is hard — it is refused because padding it here
/// would silently return a longer answer than the caller asked for, and padding it *there* is a
/// decision the caller can make with their own zeros.
pub(crate) fn levels(elements: usize) -> Result<Vec<Level>, Error> {
    let block = WORKGROUP_SIZE as usize;

    if elements < block || !elements.is_multiple_of(block) {
        return Err(Error::BadLength(BadLength::TooSmall {
            length: elements,
            minimum: block,
        }));
    }

    let mut levels = Vec::new();
    // The first level up holds one total per block of the input.
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
        // Each workgroup of this level produces one total for the level above it.
        count = workgroups as usize;
    }
}

/// How many dispatches a scan over these levels runs.
///
/// Up one side and down the other: a block scan at the input, a block scan per level below the
/// top, the single workgroup at the top, then an offset addition per level on the way back down.
/// `2 × levels + 1`, and it is written as the sum of its parts rather than that formula so that a
/// reader can check it against the loop in [`super::held`].
pub(crate) fn dispatches(levels: usize) -> usize {
    let up = 1 + levels.saturating_sub(1);
    let top = 1;
    let down = levels;
    up + top + down
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::{dispatches, levels};
    use crate::Error;
    use crate::kernels::WORKGROUP_SIZE;

    const BLOCK: usize = WORKGROUP_SIZE as usize;

    #[test]
    fn one_workgroups_worth_of_blocks_needs_one_level() {
        // The case `runner/tests/scan.rs` composes by hand: 64 blocks, one workgroup scans their
        // totals, three dispatches.
        let plan = levels(BLOCK * BLOCK).expect("planned");

        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].elements, BLOCK);
        assert_eq!(plan[0].workgroups, 1);
        assert_eq!(dispatches(plan.len()), 3);
    }

    #[test]
    fn a_million_elements_need_three_levels_and_seven_dispatches() {
        let plan = levels(1 << 20).expect("planned");

        let counts: Vec<usize> = plan.iter().map(|level| level.elements).collect();
        assert_eq!(counts, vec![16384, 256, 4]);
        assert_eq!(dispatches(plan.len()), 7);
    }

    #[test]
    fn every_level_holds_one_total_per_workgroup_of_the_one_below() {
        // The invariant the recursion rests on. If it ever fails, some block's total has nowhere
        // to go, or a level scans totals that were never written.
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
        // What ends the recursion. A plan whose last level needed two workgroups would leave
        // half its totals unscanned and return a plausible wrong answer for everything after the
        // first 64 blocks.
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
        // Padding it here would return a longer answer than the caller asked for. Refused with the
        // number they passed, so the message says what to change.
        for elements in [0_usize, 1, BLOCK - 1, BLOCK + 1, BLOCK * 3 + 7] {
            assert!(
                matches!(levels(elements), Err(Error::BadLength(_))),
                "{elements} was accepted"
            );
        }
    }

    #[test]
    fn the_levels_shrink_until_they_stop() {
        // Termination, asserted rather than assumed: each level is strictly shorter than the one
        // below it, so the loop cannot run for ever on any length this accepts.
        let plan = levels(1 << 24).expect("planned");

        for pair in plan.windows(2) {
            assert!(pair[1].elements < pair[0].elements, "{pair:?}");
        }
        assert!(plan.len() >= 3, "2^24 is deep enough to be worth the check");
    }

    #[test]
    fn the_dispatch_count_is_the_sum_of_the_three_phases() {
        // Derived independently of the implementation, so the two have to agree rather than being
        // the same expression twice.
        for count in 1..8_usize {
            assert_eq!(dispatches(count), 2 * count + 1, "{count} levels");
        }
    }
}
