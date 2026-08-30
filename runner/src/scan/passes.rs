//! ```text
//!   map     elementwise over the input, where there is one                    1 dispatch
//!   up      scan each block of the input, keeping every block's total         1 dispatch
//!           scan each block of those totals, exclusively                      per level below the top
//!   top     one workgroup scans what is left, exclusively                     1 dispatch
//!   down    add each level's offsets to the level below                       per level
//! ```

use super::plan::Level;
use crate::Error;
use crate::kernels::WORKGROUP_SIZE;

pub(super) struct Modules<'a> {
    pub(super) blocks: &'a [u32],
    pub(super) blocks_exclusive: &'a [u32],
    pub(super) top: &'a [u32],
    pub(super) add: &'a [u32],
    pub(super) map: Option<&'a [u32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Slots {
    pub(super) totals: usize,
    pub(super) scanned: Option<usize>,
    pub(super) offsets: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Ends {
    pub(super) input: usize,
    pub(super) mapped: Option<usize>,
    pub(super) scanned: usize,
    pub(super) output: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Pass<'a> {
    pub(super) module: &'a [u32],
    pub(super) bound: Vec<(usize, u64)>,
    pub(super) workgroups: u32,
}

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

    let first_read = match (modules.map, mapped) {
        (Some(map), Some(mapped)) => {
            list.push(Pass {
                module: map,
                bound: vec![(input, bytes), (mapped, bytes)],
                workgroups,
            });
            mapped
        }
        (Some(_), None) | (None, Some(_)) => return Err(Error::NoPipeline),
        (None, None) => input,
    };

    let (Some(first), Some(first_slots)) = (levels.first(), slots.first()) else {
        return Err(Error::NoPipeline);
    };

    list.push(Pass {
        module: modules.blocks,
        bound: vec![
            (first_read, bytes),
            (scanned, bytes),
            (first_slots.totals, level_bytes(first)),
        ],
        workgroups,
    });

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
    #![allow(clippy::expect_used)]

    use super::{Ends, Modules, Pass, Slots, passes};
    use crate::kernels::WORKGROUP_SIZE;
    use crate::scan::plan::{self, Level};

    const BLOCKS: &[u32] = &[1];
    const BLOCKS_EXCLUSIVE: &[u32] = &[2];
    const TOP: &[u32] = &[3];
    const ADD: &[u32] = &[4];
    const MAP: &[u32] = &[5];

    fn reads(module: &[u32]) -> &'static [usize] {
        match module {
            MAP | BLOCKS | BLOCKS_EXCLUSIVE | TOP => &[0],
            ADD => &[0, 1],
            _ => &[],
        }
    }

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

    fn plan_for(elements: usize, map: bool) -> (Vec<Level>, Vec<Slots>, Ends, Vec<Pass<'static>>) {
        let levels = plan::levels(elements).expect("levels");

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
        for elements in [64_usize, 4096, 1 << 20] {
            for map in [false, true] {
                let (levels, slots, ends, list) = plan_for(elements, map);

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
        for elements in [4096_usize, 1 << 16, 1 << 20] {
            let (_, _, _, list) = plan_for(elements, false);

            let tops: Vec<&Pass<'_>> = list.iter().filter(|pass| pass.module == TOP).collect();
            assert_eq!(tops.len(), 1, "a {elements}-element scan has one top");
            assert_eq!(tops.first().map(|pass| pass.workgroups), Some(1));
        }
    }

    #[test]
    fn each_pass_dispatches_a_workgroup_per_block_of_what_it_reads() {
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
        let levels = plan::levels(4096).expect("levels");
        let slots = layout(&levels, 3);
        let ends = Ends {
            input: 0,
            mapped: None,
            scanned: 1,
            output: 2,
        };

        assert!(passes(&levels, &slots, ends, &modules(true), 4096).is_err());

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
