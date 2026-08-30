use crate::fuzz::program::{Op, Program};

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
            elementwise(held, |value| domain.min(domain.max(value, low), high))
        }
        Op::ShiftUp | Op::ShiftDown => held.to_vec(),
        Op::BitShift { kind, by } => elementwise(held, |value| domain.bit_shift(kind, value, by)),
        Op::Absolute => elementwise(held, |value| domain.abs(value)),
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
            let size = (program.lanes.min(program.subgroup) as usize).max(1);
            let source = source as usize % size;
            held.iter()
                .enumerate()
                .map(|(invocation, elements)| {
                    let from = invocation / size * size + source;
                    elements
                        .iter()
                        .enumerate()
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
        Op::RepeatAdd { times, add } | Op::RolledAdd { times, add } => {
            let step = domain.encode(add);
            elementwise(held, |value| {
                (0..times).fold(value, |carried, _| domain.add(carried, step))
            })
        }
        Op::RolledCounterAdd { times } => elementwise(held, |value| {
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

fn elementwise(held: &[Vec<u32>], op: impl Fn(u32) -> u32 + Copy) -> Vec<Vec<u32>> {
    held.iter()
        .map(|elements| elements.iter().copied().map(op).collect())
        .collect()
}
