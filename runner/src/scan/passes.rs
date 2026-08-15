//! Which module each pass of a held scan runs, over which buffers, at how many workgroups.
//!
//! **Split from [`super::held`] for the reason that keeps producing this seam: a file is excused
//! from the mutation gate for containing `unsafe`, not for being near it.** `held.rs` allocates
//! buffers and builds pipelines, both of which are FFI; deciding that the *third* pass reads the
//! second level's totals and writes its offsets is neither. It is index arithmetic, and index
//! arithmetic that is wrong gives a plausible number several levels away from its cause.
//!
//! It is the fifth time that cut has been worth making — `dispatch/step.rs` out of `chain.rs`,
//! `reduction/plan.rs` and `scan/plan.rs` out of their `held.rs`, `step::upload_bytes` out of
//! `dispatch/upload.rs` — and it is the one with the most to say: this wiring is the most intricate
//! addressing in the crate and, until it moved here, **nothing tested it without a device**. The
//! tests at the bottom of this file are what that bought.
//!
//! # The order
//!
//! ```text
//!   map     elementwise over the input, where there is one                    1 dispatch
//!   up      scan each block of the input, keeping every block's total         1 dispatch
//!           scan each block of those totals, exclusively                      per level below the top
//!   top     one workgroup scans what is left, exclusively                     1 dispatch
//!   down    add each level's offsets to the level below                       per level
//! ```
//!
//! Every pass reads what an earlier one wrote. That is the property the tests assert, and it is the
//! one a wrong index breaks: a pass reading a buffer nothing has written yet reads the zeros the
//! builder put there, which is a smaller answer rather than an error.

use super::plan::Level;
use crate::Error;
use crate::kernels::WORKGROUP_SIZE;

/// The modules a scan runs, so a pass list takes one argument rather than five.
pub(super) struct Modules<'a> {
    pub(super) blocks: &'a [u32],
    pub(super) blocks_exclusive: &'a [u32],
    pub(super) top: &'a [u32],
    pub(super) add: &'a [u32],
    /// One elementwise pass over the input first, and `None` when there is none.
    pub(super) map: Option<&'a [u32]>,
}

/// Where each of a level's buffers sits in the scanner's buffer list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Slots {
    /// The block totals this level holds.
    pub(super) totals: usize,
    /// This level scanned within its own blocks, and `None` at the top, where the scan of the
    /// level *is* the offsets and no second buffer is needed.
    pub(super) scanned: Option<usize>,
    /// What each block of the level below owes the blocks before it.
    pub(super) offsets: usize,
}

/// The buffers at the ends of the chain — the input's own, rather than a level's.
///
/// Grouped for the same reason [`Modules`] is: the list below was growing an argument per buffer,
/// and a caller passing `scanned` where `output` belongs would build a scanner that scans its own
/// answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Ends {
    /// What the host writes.
    pub(super) input: usize,
    /// What the map writes and the first scan reads, when there is a map.
    pub(super) mapped: Option<usize>,
    /// The input's blocks, scanned from their own starts.
    pub(super) scanned: usize,
    /// Where the answer lands.
    pub(super) output: usize,
}

/// One dispatch of a held scan, before anything has been allocated for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Pass<'a> {
    /// The module to run.
    pub(super) module: &'a [u32],
    /// The buffers it binds, as `(slot, bytes)` **in binding order** — 0 first.
    pub(super) bound: Vec<(usize, u64)>,
    /// How many workgroups of it.
    pub(super) workgroups: u32,
}

/// Every pass a scan over these levels runs, in order.
///
/// `elements` is how many words the input holds; the byte sizes below follow from it and from each
/// level's capacity, so nothing here takes a size it could disagree with.
///
/// # Errors
///
/// [`Error::NoPipeline`] when the levels and the slots do not describe the same scan — a map with
/// nowhere to write, a level with no slots, a non-top level with no scanned buffer. Every one is a
/// construction mistake rather than a caller's, and each is refused rather than skipped: a scan
/// missing a pass returns a number.
pub(super) fn passes<'a>(
    levels: &[Level],
    slots: &[Slots],
    ends: Ends,
    modules: &Modules<'a>,
    elements: usize,
) -> Result<Vec<Pass<'a>>, Error> {
    let Ends {
        input,
        mapped,
        scanned,
        output,
    } = ends;

    let words = size_of::<f32>() as u64;
    let bytes = elements as u64 * words;
    let workgroups = (elements / WORKGROUP_SIZE as usize) as u32;
    let level_bytes = |level: &Level| (level.capacity as u64) * words;

    let mut list = Vec::with_capacity(super::plan::dispatches(levels.len(), modules.map.is_some()));

    // The map, when there is one: elementwise over the whole input, writing where the first block
    // scan will read. Its output never crosses the bus, which is the whole point.
    let first_read = match (modules.map, mapped) {
        (Some(map), Some(mapped)) => {
            list.push(Pass {
                module: map,
                bound: vec![(input, bytes), (mapped, bytes)],
                workgroups,
            });
            mapped
        }
        // A map with nowhere to write, or a buffer with no map, is a construction bug rather than
        // a caller's mistake — the two are decided together in `build_scanner`.
        (Some(_), None) | (None, Some(_)) => return Err(Error::NoPipeline),
        (None, None) => input,
    };

    let (Some(first), Some(first_slots)) = (levels.first(), slots.first()) else {
        return Err(Error::NoPipeline);
    };

    // Up, from the input: every block scanned inclusively, every block's total kept.
    list.push(Pass {
        module: modules.blocks,
        bound: vec![
            (first_read, bytes),
            (scanned, bytes),
            (first_slots.totals, level_bytes(first)),
        ],
        workgroups,
    });

    // Up, through the levels below the top: the same, but **exclusively**, because what a block
    // owes is the total of the blocks before it and not including it.
    for depth in 0..levels.len().saturating_sub(1) {
        let (Some(level), Some(here), Some(above)) =
            (levels.get(depth), slots.get(depth), slots.get(depth + 1))
        else {
            return Err(Error::NoPipeline);
        };
        let (Some(upper), Some(scanned_here)) = (levels.get(depth + 1), here.scanned) else {
            return Err(Error::NoPipeline);
        };

        list.push(Pass {
            module: modules.blocks_exclusive,
            bound: vec![
                (here.totals, level_bytes(level)),
                (scanned_here, level_bytes(level)),
                (above.totals, level_bytes(upper)),
            ],
            workgroups: level.workgroups,
        });
    }

    // The top: one workgroup, scanning what is left straight into the offsets it produces.
    let (Some(last), Some(last_slots)) = (levels.last(), slots.last()) else {
        return Err(Error::NoPipeline);
    };
    list.push(Pass {
        module: modules.top,
        bound: vec![
            (last_slots.totals, level_bytes(last)),
            (last_slots.offsets, level_bytes(last)),
        ],
        workgroups: 1,
    });

    // Down: each level takes the offsets from the level above and pays its own blocks.
    for depth in (0..levels.len().saturating_sub(1)).rev() {
        let (Some(level), Some(here), Some(above)) =
            (levels.get(depth), slots.get(depth), slots.get(depth + 1))
        else {
            return Err(Error::NoPipeline);
        };
        let (Some(upper), Some(scanned_here)) = (levels.get(depth + 1), here.scanned) else {
            return Err(Error::NoPipeline);
        };

        list.push(Pass {
            module: modules.add,
            bound: vec![
                (scanned_here, level_bytes(level)),
                (above.offsets, level_bytes(upper)),
                (here.offsets, level_bytes(level)),
            ],
            workgroups: level.workgroups,
        });
    }

    // And the input's own blocks, which is the answer.
    list.push(Pass {
        module: modules.add,
        bound: vec![
            (scanned, bytes),
            (first_slots.offsets, level_bytes(first)),
            (output, bytes),
        ],
        workgroups,
    });

    Ok(list)
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used)]

    use super::{Ends, Modules, Pass, Slots, passes};
    use crate::kernels::WORKGROUP_SIZE;
    use crate::scan::plan::{self, Level};

    /// Four distinguishable modules, so a pass naming the wrong one is visible.
    ///
    /// Word streams rather than real SPIR-V: nothing here builds a pipeline, and a module that is
    /// its own name is what makes the assertions readable.
    const BLOCKS: &[u32] = &[1];
    const BLOCKS_EXCLUSIVE: &[u32] = &[2];
    const TOP: &[u32] = &[3];
    const ADD: &[u32] = &[4];
    const MAP: &[u32] = &[5];

    /// Which bindings each module reads, as indices into a pass's `bound` list.
    ///
    /// **Not "all but the last".** `Gpu::run_bound`'s convention is that the last binding is
    /// written, and these kernels are the exception that makes the rule worth stating: a block scan
    /// writes *two* — its scanned blocks and their totals — and `add_offsets` reads two, the blocks
    /// and the offsets they owe. The table is here rather than inferred so that a test about the
    /// wiring is not resting on a convention the wiring does not follow.
    fn reads(module: &[u32]) -> &'static [usize] {
        match module {
            MAP | BLOCKS | BLOCKS_EXCLUSIVE | TOP => &[0],
            ADD => &[0, 1],
            _ => &[],
        }
    }

    /// Which bindings each module writes.
    fn writes(module: &[u32]) -> &'static [usize] {
        match module {
            MAP | TOP => &[1],
            BLOCKS | BLOCKS_EXCLUSIVE => &[1, 2],
            ADD => &[2],
            _ => &[],
        }
    }

    fn modules(map: bool) -> Modules<'static> {
        Modules {
            blocks: BLOCKS,
            blocks_exclusive: BLOCKS_EXCLUSIVE,
            top: TOP,
            add: ADD,
            map: map.then_some(MAP),
        }
    }

    /// The slots `build_scanner` allocates, in the order it allocates them.
    ///
    /// Reproduced here rather than imported, because what is under test is whether `passes` reads
    /// them the way they were laid out — and a helper shared with the allocator would move in step
    /// with it and assert nothing.
    fn layout(levels: &[Level], first_free: usize) -> Vec<Slots> {
        let mut next = first_free;
        let mut slots = Vec::with_capacity(levels.len());

        for (depth, _) in levels.iter().enumerate() {
            let at_top = depth + 1 == levels.len();
            let totals = next;
            let scanned = if at_top {
                None
            } else {
                next += 1;
                Some(next)
            };
            next += 1;
            let offsets = next;
            next += 1;

            slots.push(Slots {
                totals,
                scanned,
                offsets,
            });
        }
        slots
    }

    /// The whole plan for a scan of `elements`, with or without a map.
    fn plan_for(elements: usize, map: bool) -> (Vec<Level>, Vec<Slots>, Ends, Vec<Pass<'static>>) {
        let levels = plan::levels(elements).expect("levels");

        // Ends first, in `build_scanner`'s own order: input, then the map's buffer if there is
        // one, then the input's scanned blocks, then the output.
        let input = 0;
        let mapped = map.then_some(1);
        let scanned = if map { 2 } else { 1 };
        let output = scanned + 1;
        let ends = Ends {
            input,
            mapped,
            scanned,
            output,
        };

        let slots = layout(&levels, output + 1);
        let list = passes(&levels, &slots, ends, &modules(map), elements).expect("passes");
        (levels, slots, ends, list)
    }

    #[test]
    fn the_pass_count_is_the_one_the_plan_predicts() {
        // **Two derivations of the same number, made to agree.** `plan::dispatches` says how many
        // dispatches a scan of this depth takes; this list emits them one at a time. `held.rs`
        // compares the two after the fact, which catches a disagreement only on a machine with a
        // device — this catches it anywhere.
        for elements in [64_usize, 4096, 1 << 16, 1 << 20] {
            for map in [false, true] {
                let (levels, _, _, list) = plan_for(elements, map);
                assert_eq!(
                    list.len(),
                    plan::dispatches(levels.len(), map),
                    "{elements} elements, map={map}"
                );
            }
        }
    }

    #[test]
    fn every_pass_reads_only_what_an_earlier_pass_wrote() {
        // **The property a wrong index breaks, and the one no device test can state.** A pass that
        // reads a buffer nothing has written yet reads the zeros the builder put there — a smaller
        // answer rather than an error, arriving levels away from its cause.
        //
        // Which bindings a module reads and writes is per module — see `reads` and `writes`. The
        // input and the level buffers are filled before the chain starts, so they are what a first
        // pass may legitimately read.
        for elements in [64_usize, 4096, 1 << 20] {
            for map in [false, true] {
                let (levels, slots, ends, list) = plan_for(elements, map);

                // Everything the builder writes before the chain runs: the host's input, and every
                // level's buffers, which `Held::zeroed` fills.
                let mut written: Vec<usize> = vec![ends.input];
                for level in &slots {
                    written.push(level.totals);
                    written.extend(level.scanned);
                    written.push(level.offsets);
                }

                for (index, pass) in list.iter().enumerate() {
                    for &binding in reads(pass.module) {
                        let slot = pass.bound.get(binding).map(|&(slot, _)| slot);
                        assert!(
                            slot.is_some_and(|slot| written.contains(&slot)),
                            "pass {index} of a {elements}-element scan ({} levels, map={map}) \
                             reads binding {binding} at slot {slot:?}, which nothing has written",
                            levels.len()
                        );
                    }
                    for &binding in writes(pass.module) {
                        let slot = pass.bound.get(binding).map(|&(slot, _)| slot);
                        assert!(slot.is_some(), "pass {index} binds nothing at {binding}");
                        written.extend(slot);
                    }
                }
            }
        }
    }

    #[test]
    fn the_last_pass_writes_the_answer_and_nothing_else_does() {
        // The scanner reads its answer out of one slot. If any earlier pass wrote there it would
        // be overwritten later — right — but if the *last* pass wrote somewhere else the scanner
        // would read a buffer full of zeros and report it as a scan.
        for elements in [64_usize, 4096, 1 << 20] {
            let (_, _, ends, list) = plan_for(elements, false);

            let written: Vec<usize> = list
                .iter()
                .flat_map(|pass| {
                    writes(pass.module)
                        .iter()
                        .filter_map(|&binding| pass.bound.get(binding).map(|&(slot, _)| slot))
                })
                .collect();

            assert_eq!(
                written.last().copied(),
                Some(ends.output),
                "the last pass of a {elements}-element scan does not write the answer"
            );
            assert_eq!(
                written.iter().filter(|&&slot| slot == ends.output).count(),
                1,
                "the answer is written more than once"
            );
        }
    }

    #[test]
    fn the_map_is_the_first_pass_and_the_scan_reads_what_it_wrote() {
        // The whole point of `scanner_of`: the map's output never leaves the device, because the
        // pass after it reads the buffer the map wrote rather than the input.
        let (_, _, ends, list) = plan_for(4096, true);

        let first = list.first().expect("a map pass");
        assert_eq!(first.module, MAP);
        assert_eq!(first.bound.first().map(|&(slot, _)| slot), Some(ends.input));
        assert_eq!(first.bound.last().map(|&(slot, _)| slot), ends.mapped);

        let second = list.get(1).expect("a block scan");
        assert_eq!(second.module, BLOCKS);
        assert_eq!(
            second.bound.first().map(|&(slot, _)| slot),
            ends.mapped,
            "the block scan reads the input rather than what the map wrote"
        );
    }

    #[test]
    fn without_a_map_the_first_scan_reads_the_input_itself() {
        let (_, _, ends, list) = plan_for(4096, false);

        let first = list.first().expect("a block scan");
        assert_eq!(first.module, BLOCKS);
        assert_eq!(first.bound.first().map(|&(slot, _)| slot), Some(ends.input));
    }

    #[test]
    fn the_top_is_one_workgroup_and_the_only_one() {
        // The recursion ends where a single workgroup can scan what is left. A top dispatched over
        // more than one would scan each block of the top level separately and never combine them —
        // which is the same wrong answer as having no top at all.
        for elements in [4096_usize, 1 << 16, 1 << 20] {
            let (_, _, _, list) = plan_for(elements, false);

            let tops: Vec<&Pass<'_>> = list.iter().filter(|pass| pass.module == TOP).collect();
            assert_eq!(tops.len(), 1, "a {elements}-element scan has one top");
            assert_eq!(tops.first().map(|pass| pass.workgroups), Some(1));
        }
    }

    #[test]
    fn each_pass_dispatches_a_workgroup_per_block_of_what_it_reads() {
        // A pass over a level dispatches that level's own workgroup count, not the input's. Using
        // the input's everywhere would run the top level's kernel over sixteen thousand blocks of
        // a four-element array — which `dispatch::extent` now refuses, and which used to be a
        // dispatch that read past the end of every level buffer.
        let (levels, _, _, list) = plan_for(1 << 20, false);
        let input_groups = (1 << 20) / WORKGROUP_SIZE as usize;

        assert_eq!(
            list.first().map(|pass| pass.workgroups),
            Some(input_groups as u32),
            "the first block scan covers the whole input"
        );

        for pass in &list {
            let expected = pass.workgroups == 1
                || pass.workgroups == input_groups as u32
                || levels
                    .iter()
                    .any(|level| level.workgroups == pass.workgroups);
            assert!(
                expected,
                "a pass dispatches {} workgroups, which is neither the input's nor a level's",
                pass.workgroups
            );
        }
    }

    #[test]
    fn a_map_with_nowhere_to_write_is_refused_rather_than_dropped() {
        // The two are decided together in `build_scanner`, so this is a construction mistake — and
        // a scan that quietly dropped the map would return the running total of the *input*, which
        // is a plausible number and the wrong question.
        let levels = plan::levels(4096).expect("levels");
        let slots = layout(&levels, 3);
        let ends = Ends {
            input: 0,
            mapped: None,
            scanned: 1,
            output: 2,
        };

        assert!(passes(&levels, &slots, ends, &modules(true), 4096).is_err());

        // And the other way round: a buffer for a map that does not exist.
        let ends = Ends {
            input: 0,
            mapped: Some(1),
            scanned: 2,
            output: 3,
        };
        assert!(passes(&levels, &slots, ends, &modules(false), 4096).is_err());
    }

    #[test]
    fn levels_without_slots_are_refused() {
        // Every `Option` in this file is a construction invariant, and each is refused by name
        // rather than skipped. A pass list one pass short still runs, and still returns a number.
        let levels = plan::levels(1 << 20).expect("levels");
        let ends = Ends {
            input: 0,
            mapped: None,
            scanned: 1,
            output: 2,
        };

        assert!(passes(&levels, &[], ends, &modules(false), 1 << 20).is_err());
        assert!(
            passes(
                &levels,
                &layout(&levels, 3)[..1],
                ends,
                &modules(false),
                1 << 20
            )
            .is_err()
        );
    }
}
