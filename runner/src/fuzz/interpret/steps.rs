//! What each step of a generated program does to the values, on the CPU.
//!
//! Split from the reduction in [`super`] because they answer different questions: this one is
//! elementwise-or-shuffle bookkeeping over the whole invocation grid, and that one is about which
//! lanes a reduction covers. Both have to be obviously right — a reference with a clever bug in it
//! turns every fuzzing round into a false alarm — and they read more easily apart.

use crate::fuzz::program::{Op, Program};

/// One step, over every invocation at once.
pub(super) fn apply(program: &Program, held: &[Vec<u32>], step: Op) -> Vec<Vec<u32>> {
    let domain = program.domain;
    let width = program.subgroup as usize;

    match step {
        Op::AddConstant(operand) => {
            let operand = domain.encode(operand);
            elementwise(held, |value| domain.add(value, operand))
        }
        Op::MulConstant(operand) => {
            let operand = domain.encode(operand);
            elementwise(held, |value| domain.mul(value, operand))
        }
        Op::ClampBelow(floor) => {
            let floor = domain.encode(floor);
            elementwise(held, |value| domain.max(value, floor))
        }
        // The extended-instruction trio. Written through `Domain::min` and `max`, which are
        // defined from `greater` — the same ordering the emitted `*Min` and `*Max` use, and the
        // same one `ClampBelow`'s compare-and-select above resolves to. Three spellings of one
        // ordering, and the fuzzer's job is to find the round where they stop agreeing.
        Op::MinConstant(operand) => {
            let operand = domain.encode(operand);
            elementwise(held, |value| domain.min(value, operand))
        }
        Op::MaxConstant(operand) => {
            let operand = domain.encode(operand);
            elementwise(held, |value| domain.max(value, operand))
        }
        Op::ClampBoth { low, high } => {
            let low = domain.encode(low);
            let high = domain.encode(high);
            // `min(max(x, low), high)`, which is what GLSL.std.450 defines `*Clamp` to be — and
            // the order matters when the bounds cross. They cannot cross here, because the
            // generator draws `high` from `low`, so this is the definition rather than a guess.
            elementwise(held, |value| domain.min(domain.max(value, low), high))
        }
        // A shift by zero is the identity, and the operation carries no other distance — the
        // operand it used to take was always zero and this arm ignored it either way. See
        // `Op::ShiftUp`: SPIR-V leaves the out-of-range lanes undefined, so there is no reference
        // for a real one to be compared against.
        Op::ShiftUp | Op::ShiftDown => held.to_vec(),
        // The *bit* shift, which crosses no lane — so unlike the two above it is elementwise, and
        // unlike everything else here it reads its operand as bits rather than as a number.
        // `Domain::bit_shift` is where that reading lives, and where the asymmetry that makes the
        // two right shifts worth having is written down.
        Op::BitShift { kind, by } => elementwise(held, |value| domain.bit_shift(kind, value, by)),
        // Elementwise like the shift, and the one step here whose answer at a single input is not
        // compared at all: `predictable` in the parent refuses a round whose magnitude meets a
        // two's-complement minimum, because that value negates to itself and GLSL.std.450 does not
        // promise it. Everything else about the arm is ordinary.
        Op::Absolute => elementwise(held, |value| domain.abs(value)),
        // A multiply and an add, written as a multiply and an add. **Not folded into one
        // expression on the host** for the reason `RepeatAdd` is not folded into a multiply: the
        // device fuses these into a single rounding, and the two agree here only because every
        // value is a small integer that neither rounding touches. Writing it the long way is what
        // leaves that agreement checkable rather than assumed.
        Op::FusedMulAdd { by, plus } => {
            let factor = domain.encode(by);
            let addend = domain.encode(plus);
            elementwise(held, |value| domain.add(domain.mul(value, factor), addend))
        }
        Op::AddIfAllAbove {
            when_all_above,
            add,
        } => {
            let threshold = domain.encode(when_all_above);
            let add = domain.encode(add);

            // `all` where the neighbour has `any`, and per subgroup for the same reason: the vote
            // is uniform, so the whole subgroup takes the step or none of it does. Strips included,
            // because a strip-mined vector's every element is in the subgroup being asked.
            held.chunks(width)
                .flat_map(|subgroup| {
                    let takes = subgroup
                        .iter()
                        .flatten()
                        .all(|value| domain.greater(*value, threshold));

                    subgroup.iter().map(move |elements| {
                        elements
                            .iter()
                            .map(|value| {
                                if takes {
                                    domain.add(*value, add)
                                } else {
                                    *value
                                }
                            })
                            .collect()
                    })
                })
                .collect()
        }
        Op::BroadcastLane(source) => {
            // Inside the *vector*, exactly as the rotate below: `min(lanes, width)` invocations, so
            // a clustered vector reads position `source` of its own cluster and a subgroup-wide one
            // reads lane `source` of the subgroup. Reading `source` as a subgroup lane instead
            // would agree with this for the first cluster of every subgroup and differ for the
            // rest, which is the wrong answer this models against.
            let size = (program.lanes.min(program.subgroup) as usize).max(1);
            let source = source as usize % size;
            held.iter()
                .enumerate()
                .map(|(invocation, elements)| {
                    let from = invocation / size * size + source;
                    elements
                        .iter()
                        .enumerate()
                        // Per strip: a strip-mined vector broadcasts within each strip rather than
                        // collapsing them, which is what the emitted shuffle does per load.
                        .map(|(strip, value)| {
                            held.get(from)
                                .and_then(|held| held.get(strip))
                                .copied()
                                .unwrap_or(*value)
                        })
                        .collect()
                })
                .collect()
        }
        // Both loops add the same constant `times` times over. Written as a loop rather than as
        // one multiplication on purpose: in the wrapping domains the two are equal, and in the
        // float domain they are equal only because the values are small integers. Folding it to a
        // multiply here would quietly assume the thing the fuzzer is checking.
        Op::RepeatAdd { times, add } | Op::RolledAdd { times, add } => {
            let step = domain.encode(add);
            elementwise(held, |value| {
                (0..times).fold(value, |carried, _| domain.add(carried, step))
            })
        }
        Op::RolledCounterAdd { times } => elementwise(held, |value| {
            // 0 + 1 + … + times-1, accumulated the same way the body does.
            (0..times).fold(value, |carried, iteration| {
                domain.add(carried, domain.encode(iteration))
            })
        }),
        Op::SelectEqual { to, then } => {
            let target = domain.encode(to);
            let then = domain.encode(then);
            elementwise(held, |value| {
                if domain.equals(value, target) {
                    then
                } else {
                    value
                }
            })
        }
        Op::AddIfAllEqual { add } => {
            let add = domain.encode(add);

            // Per *subgroup*, like the other vote — and over every element the subgroup holds,
            // strips included, because that is what `all_equal` asks: a strip-mined vector agrees
            // only when its lanes agree *and* its strips do.
            held.chunks(width)
                .flat_map(|subgroup| {
                    let mut elements = subgroup.iter().flatten();
                    let first = elements.next().copied();
                    let agreed = first.is_some_and(|first| {
                        subgroup
                            .iter()
                            .flatten()
                            .all(|value| domain.equals(*value, first))
                    });

                    subgroup.iter().map(move |elements| {
                        elements
                            .iter()
                            .map(|value| {
                                if agreed {
                                    domain.add(*value, add)
                                } else {
                                    *value
                                }
                            })
                            .collect()
                    })
                })
                .collect()
        }
        Op::AddIfAnyAbove {
            when_any_above,
            add,
        } => {
            let threshold = domain.encode(when_any_above);
            let add = domain.encode(add);

            // The vote is per *subgroup*, so it is worked out once per subgroup and applied to
            // every lane in it. Doing it per lane would be the bug DR-0003 exists to make
            // unwriteable, and a reference that made it would agree with a wrong kernel.
            //
            // `chunks` rather than a lookup table indexed by `invocation / width`. The table
            // version needed a default for an index that could not occur, and an unreachable
            // default is a branch no test can ever take — a mutation run flipped it and nothing
            // noticed, because nothing could. Chunking makes the grouping structural.
            held.chunks(width)
                .flat_map(|subgroup| {
                    let takes = subgroup
                        .iter()
                        .flatten()
                        .any(|value| domain.greater(*value, threshold));

                    subgroup.iter().map(move |elements| {
                        elements
                            .iter()
                            .map(|value| {
                                if takes {
                                    domain.add(*value, add)
                                } else {
                                    *value
                                }
                            })
                            .collect()
                    })
                })
                .collect()
        }
        Op::RotateUp(delta) => {
            // Inside the *vector*, which is `min(lanes, width)` invocations — a clustered vector
            // rotates within its own cluster and a subgroup-wide one within the subgroup. The
            // wrap is what a shift does not have, and it is the whole of what this checks.
            let size = (program.lanes.min(program.subgroup) as usize).max(1);
            let delta = delta as usize % size;
            (0..held.len())
                .map(|invocation| {
                    let base = invocation / size * size;
                    let within = (invocation + size - delta) % size;
                    held.get(base + within).cloned().unwrap_or_default()
                })
                .collect()
        }
        Op::ButterflyAdd(mask) => held
            .iter()
            .enumerate()
            .map(|(invocation, elements)| {
                // The exchange is *within a subgroup*: the partner is found by flipping bits of
                // the lane index, not of the global one.
                let lane = invocation % width;
                let partner = invocation / width * width + (lane ^ mask as usize);

                elements
                    .iter()
                    .enumerate()
                    .map(|(strip, value)| {
                        let other = held
                            .get(partner)
                            .and_then(|held| held.get(strip))
                            .copied()
                            .unwrap_or_else(|| domain.zero());
                        domain.add(*value, other)
                    })
                    .collect()
            })
            .collect(),
    }
}

/// Apply `op` to every element of every invocation.
fn elementwise(held: &[Vec<u32>], op: impl Fn(u32) -> u32 + Copy) -> Vec<Vec<u32>> {
    held.iter()
        .map(|elements| elements.iter().copied().map(op).collect())
        .collect()
}
