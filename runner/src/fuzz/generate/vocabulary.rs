use super::Rng;
use crate::fuzz::domain::{BitShift, Domain};
use crate::fuzz::program::Op;

#[cfg(test)]
use crate::fuzz::domain::ALL_DOMAINS;
#[cfg(test)]
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Kind {
    AddConstant,
    MulConstant,
    ClampBelow,
    MinConstant,
    MaxConstant,
    ClampBoth,
    RepeatAdd,
    RolledAdd,
    RolledCounterAdd,
    ButterflyAdd,
    AddIfAnyAbove,
    AddIfAllEqual,
    SelectEqual,
    RotateUp,
    ShiftUp,
    ShiftDown,
    BroadcastLane,
    ShiftLeft,
    ShiftRightLogical,
    ShiftRightArithmetic,
    Absolute,
    FusedMulAdd,
    AddIfAllAbove,
}

pub(super) const CLUSTERED: &[Kind] = &[
    Kind::AddConstant,
    Kind::MulConstant,
    Kind::ClampBelow,
    Kind::MinConstant,
    Kind::MaxConstant,
    Kind::ClampBoth,
    Kind::RepeatAdd,
    Kind::RolledAdd,
    Kind::RolledCounterAdd,
    Kind::SelectEqual,
    Kind::RotateUp,
    Kind::BroadcastLane,
];

pub(super) const WHOLE: &[Kind] = &[
    Kind::AddConstant,
    Kind::MulConstant,
    Kind::ClampBelow,
    Kind::MinConstant,
    Kind::MaxConstant,
    Kind::ClampBoth,
    Kind::RepeatAdd,
    Kind::RolledAdd,
    Kind::RolledCounterAdd,
    Kind::SelectEqual,
    Kind::RotateUp,
    Kind::ButterflyAdd,
    Kind::AddIfAnyAbove,
    Kind::AddIfAllAbove,
    Kind::AddIfAllEqual,
    Kind::ShiftUp,
    Kind::ShiftDown,
    Kind::BroadcastLane,
];

pub(super) const STRIPPED: &[Kind] = &[
    Kind::AddConstant,
    Kind::MulConstant,
    Kind::ClampBelow,
    Kind::MinConstant,
    Kind::MaxConstant,
    Kind::ClampBoth,
    Kind::RepeatAdd,
    Kind::RolledAdd,
    Kind::RolledCounterAdd,
    Kind::SelectEqual,
    Kind::ButterflyAdd,
    Kind::AddIfAnyAbove,
    Kind::AddIfAllAbove,
    Kind::AddIfAllEqual,
    Kind::ShiftUp,
    Kind::ShiftDown,
    Kind::BroadcastLane,
];

pub(super) const fn by_element(domain: Domain) -> &'static [Kind] {
    const SHIFTS: &[Kind] = &[
        Kind::ShiftLeft,
        Kind::ShiftRightLogical,
        Kind::ShiftRightArithmetic,
    ];
    const SHIFTS_AND_MAGNITUDE: &[Kind] = &[
        Kind::ShiftLeft,
        Kind::ShiftRightLogical,
        Kind::ShiftRightArithmetic,
        Kind::Absolute,
    ];
    const SINGLE: &[Kind] = &[Kind::Absolute, Kind::FusedMulAdd];
    const HALF: &[Kind] = &[Kind::Absolute];

    match domain {
        Domain::Unsigned | Domain::UnsignedByte | Domain::UnsignedShort => SHIFTS,
        Domain::Signed | Domain::Byte | Domain::Short => SHIFTS_AND_MAGNITUDE,
        Domain::Float => SINGLE,
        Domain::Half => HALF,
    }
}

fn shift_by(rng: &mut Rng, domain: Domain) -> u32 {
    rng.below(u64::from(domain.bits())) as u32
}

pub(super) fn fill(rng: &mut Rng, domain: Domain, subgroup: u32, lanes: u32, kind: Kind) -> Op {
    match kind {
        Kind::AddConstant => Op::AddConstant(rng.below(16) as u32),
        Kind::MulConstant => Op::MulConstant(1 + rng.below(3) as u32),
        Kind::ClampBelow => Op::ClampBelow(rng.below(8) as u32),
        Kind::MinConstant => Op::MinConstant(rng.below(u64::from(domain.ceiling())) as u32),
        Kind::MaxConstant => Op::MaxConstant(rng.below(u64::from(domain.ceiling())) as u32),
        Kind::ClampBoth => {
            let low = rng.below(u64::from(domain.ceiling())) as u32;
            Op::ClampBoth {
                low,
                high: low.saturating_add(1 + rng.below(u64::from(domain.ceiling())) as u32),
            }
        }
        Kind::SelectEqual => Op::SelectEqual {
            to: rng.below(u64::from(domain.ceiling())) as u32,
            then: rng.below(u64::from(domain.ceiling())) as u32,
        },
        Kind::AddIfAllEqual => Op::AddIfAllEqual {
            add: 1 + rng.below(8) as u32,
        },
        Kind::RotateUp => Op::RotateUp(rng.below(u64::from(subgroup.max(1))) as u32),
        Kind::RepeatAdd => Op::RepeatAdd {
            times: rng.below(5) as u32,
            add: 1 + rng.below(8) as u32,
        },
        Kind::RolledAdd => Op::RolledAdd {
            times: rng.below(5) as u32,
            add: 1 + rng.below(8) as u32,
        },
        Kind::RolledCounterAdd => Op::RolledCounterAdd {
            times: rng.below(6) as u32,
        },
        Kind::ButterflyAdd => Op::ButterflyAdd(1 << rng.below(distances(subgroup))),
        Kind::AddIfAnyAbove => Op::AddIfAnyAbove {
            when_any_above: rng.below(u64::from(domain.ceiling())) as u32,
            add: 1 + rng.below(8) as u32,
        },
        Kind::ShiftUp => Op::ShiftUp,
        Kind::ShiftDown => Op::ShiftDown,
        Kind::BroadcastLane => {
            Op::BroadcastLane(rng.below(u64::from(lanes.min(subgroup).max(1))) as u32)
        }
        Kind::ShiftLeft => Op::BitShift {
            kind: BitShift::Left,
            by: shift_by(rng, domain),
        },
        Kind::ShiftRightLogical => Op::BitShift {
            kind: BitShift::RightLogical,
            by: shift_by(rng, domain),
        },
        Kind::ShiftRightArithmetic => Op::BitShift {
            kind: BitShift::RightArithmetic,
            by: shift_by(rng, domain),
        },
        Kind::Absolute => Op::Absolute,
        Kind::FusedMulAdd => Op::FusedMulAdd {
            by: 1 + rng.below(3) as u32,
            plus: rng.below(16) as u32,
        },
        Kind::AddIfAllAbove => Op::AddIfAllAbove {
            when_all_above: rng.below(3) as u32,
            add: 1 + rng.below(8) as u32,
        },
    }
}

fn distances(subgroup: u32) -> u64 {
    u64::from(subgroup.trailing_zeros()).clamp(1, 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVERY_KIND: [Kind; 23] = [
        Kind::ShiftLeft,
        Kind::ShiftRightLogical,
        Kind::ShiftRightArithmetic,
        Kind::Absolute,
        Kind::FusedMulAdd,
        Kind::AddIfAllAbove,
        Kind::AddConstant,
        Kind::MulConstant,
        Kind::ClampBelow,
        Kind::MinConstant,
        Kind::MaxConstant,
        Kind::ClampBoth,
        Kind::RepeatAdd,
        Kind::RolledAdd,
        Kind::RolledCounterAdd,
        Kind::SelectEqual,
        Kind::RotateUp,
        Kind::ButterflyAdd,
        Kind::AddIfAnyAbove,
        Kind::AddIfAllEqual,
        Kind::ShiftUp,
        Kind::ShiftDown,
        Kind::BroadcastLane,
    ];

    #[test]
    fn the_three_pools_are_a_mapping_apiece_and_agree_about_what_is_in_them() {
        let by_type: BTreeSet<Kind> = ALL_DOMAINS
            .iter()
            .flat_map(|domain| by_element(*domain))
            .copied()
            .collect();

        for kind in EVERY_KIND {
            assert!(
                WHOLE.contains(&kind) || by_type.contains(&kind),
                "{kind:?} is in no pool, so the generator can never draw it"
            );
        }
        assert_eq!(WHOLE.len() + by_type.len(), EVERY_KIND.len());

        for pool in [WHOLE, CLUSTERED, STRIPPED] {
            for kind in &by_type {
                assert!(
                    !pool.contains(kind),
                    "{kind:?} is gated by the element type and sits in a pool gated by the mapping"
                );
            }
        }

        for kind in CLUSTERED {
            assert!(WHOLE.contains(kind));
            assert!(
                !matches!(
                    kind,
                    Kind::ButterflyAdd
                        | Kind::ShiftUp
                        | Kind::ShiftDown
                        | Kind::AddIfAnyAbove
                        | Kind::AddIfAllEqual
                ),
                "{kind:?} answers for every vector sharing the subgroup"
            );
        }
        assert!(CLUSTERED.contains(&Kind::RotateUp));
        assert!(CLUSTERED.contains(&Kind::BroadcastLane));

        for kind in STRIPPED {
            assert!(WHOLE.contains(kind));
        }
        assert!(!STRIPPED.contains(&Kind::RotateUp));
        assert!(STRIPPED.contains(&Kind::ButterflyAdd));
        assert_eq!(
            STRIPPED.len(),
            WHOLE.len() - 1,
            "the rotate, and nothing else"
        );
    }
    use super::distances;

    #[test]
    fn the_element_pool_agrees_with_what_builds() {
        use crate::fuzz::program::{Finish, Program, ProgramError};

        const GATED: [Kind; 5] = [
            Kind::ShiftLeft,
            Kind::ShiftRightLogical,
            Kind::ShiftRightArithmetic,
            Kind::Absolute,
            Kind::FusedMulAdd,
        ];

        for domain in ALL_DOMAINS {
            let offered = by_element(domain);
            for kind in GATED {
                let mut rng = Rng::new(7);
                let program = Program {
                    domain,
                    subgroup: 32,
                    workgroup: 64,
                    groups: 1,
                    lanes: 32,
                    steps: vec![fill(&mut rng, domain, 32, 32, kind)],
                    finish: Finish::Sum,
                };

                let built = program.build();
                assert_eq!(
                    offered.contains(&kind),
                    built.is_ok(),
                    "{domain:?} and {kind:?} disagree: the table offers it {}, and `build` {}",
                    offered.contains(&kind),
                    match &built {
                        Ok(_) => "accepted it".to_owned(),
                        Err(why) => format!("said `{why}`"),
                    }
                );
                if let Err(why) = built {
                    assert!(
                        matches!(why, ProgramError::NotInThisDomain { .. }),
                        "{domain:?} refused {kind:?} as `{why}`, which is a width problem rather                          than a missing instruction — the two are apart for exactly this reason"
                    );
                }
            }
        }
    }

    #[test]
    fn a_subgroup_of_one_still_leaves_something_to_draw_from() {
        assert_eq!(distances(1), 1);
        assert_ne!(distances(1), 0, "a modulus of zero is a panic");
    }

    #[test]
    fn the_distances_stay_inside_the_subgroup() {
        for width in [1_u32, 2, 4, 8, 16, 32, 64] {
            let count = distances(width);
            assert!(count > 0, "a subgroup of {width} left nothing to draw");

            let largest = 1_u32 << (count - 1);
            if width > 1 {
                assert!(
                    largest < width,
                    "a subgroup of {width} would draw a distance of {largest}"
                );
            }
        }
    }

    #[test]
    fn a_wide_subgroup_is_capped_rather_than_growing_with_it() {
        assert_eq!(distances(32), 4);
        assert_eq!(distances(64), 4);
        assert_eq!(distances(16), 4);
        assert_eq!(distances(8), 3, "1, 2 and 4");
        assert_eq!(distances(4), 2, "1 and 2");
        assert_eq!(distances(2), 1, "1 alone");
    }
}
