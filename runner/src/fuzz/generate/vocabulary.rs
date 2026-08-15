//! Which operations the generator may draw from, and how their operands are filled in.
//!
//! Split from [`super`] so the generator itself is only the drawing: pick a pool, pick from it,
//! fill it in. What is *in* the pool is a policy question — which operations are legal under which
//! mapping — and it reads better as a table than as arms of the match that draws.

use super::Rng;
use crate::fuzz::domain::{BitShift, Domain};
use crate::fuzz::program::Op;

#[cfg(test)]
use crate::fuzz::domain::ALL_DOMAINS;
#[cfg(test)]
use std::collections::BTreeSet;

/// Which operation to generate, before its operands are drawn.
///
/// A named list rather than integers and `match` guards. The guards were the obvious spelling and
/// they carried an arithmetic relationship between a bound and a set of arm labels — mutate one
/// comparison and two operations swap places, which every test still passed because both were
/// still reachable. Killing that would have meant pinning an arbitrary index-to-operation map, and
/// a test of an arbitrary internal mapping proves nothing and makes refactoring painful.
///
/// A table has no arithmetic in it to get wrong.
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

/// What a vector **narrower** than the subgroup may do.
///
/// Elementwise work, which crosses no lane at all, plus the rotate — which crosses lanes and stays
/// inside the cluster while doing it, so a `Simd<f32, 8>` rotates its own eight elements and reads
/// none of the twenty-four beside them.
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

/// What a vector **exactly the subgroup's width** may do: all of it.
///
/// A shuffle or a vote on a vector that shares its lanes with three others is refused by the lane
/// API, and the generator respects that rather than leaning on `build` to say no — a run made
/// mostly of refusals tests very little.
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

/// What a **strip-mined** vector may do: everything but the rotate.
///
/// A rotate over a vector wider than the subgroup moves elements *between* strips — a shuffle per
/// strip plus a rotation of the strips themselves — and `Lanes::rotate_up` refuses it by name. The
/// other three lane-crossing operations are fine: a shuffle applies per strip, and the votes fold
/// their strips together.
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

/// What a domain may do that its *element type* decides, rather than its mapping.
///
/// A second axis beside the three pools above, and it needs its own lists because it is a different
/// question. Those three ask which lanes a vector may read; a bit shift, a magnitude and a fused
/// multiply-add read no lane but their own, so all three mappings would hold identical copies. What
/// gates these is the bound `Lanes` puts on the element: `Integer` for the shifts — six domains —
/// `Signed` for the magnitude — five — and `F32` concretely for the fused multiply-add, which is
/// **one**.
///
/// Three memberships that do not nest, so this is four slices and a match rather than a set
/// operation. It is meant to be read as a table, which is the argument [`Kind`] itself is built on:
/// a table has no arithmetic in it to get wrong.
///
/// The table is not the authority — `Emit` is, and a domain listed here that cannot emit what it is
/// offered would produce refusals instead of rounds. `the_element_pool_agrees_with_what_builds`
/// holds the two together by building one program per pairing.
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

/// How far to shift, in a domain of `bits`.
///
/// **Up to one below the element's width, and reaching that far is the point.** SPIR-V leaves a
/// shift by at least the width undefined, so the top of the range is fixed by the specification
/// rather than chosen — and the whole range is drawn rather than a small constant, because
/// `OpShiftRightLogical` and `OpShiftRightArithmetic` agree on every value whose top bit is clear.
/// Every value this generator draws has a clear top bit. A left shift of 31 is what puts one there,
/// and the right shift after it is then a question with two different answers.
fn shift_by(rng: &mut Rng, domain: Domain) -> u32 {
    rng.below(u64::from(domain.bits())) as u32
}

/// Draw the operands for `kind`.
///
/// Loop trip counts and constants stay small: a rolled loop of four is the same shape as one of
/// four hundred — four blocks and two phis — and the short one leaves every sum well inside the
/// float domain's exactly-representable range, which is what lets the comparison be exact at all.
///
/// **`lanes` as well as `subgroup`, because they are different bounds.** A butterfly's mask has to
/// stay inside the *subgroup* — a lane XOR'd past its last one is undefined. A broadcast's position
/// has to stay inside the *vector*, which for a clustered one is narrower, and `Lanes::broadcast`
/// refuses a position outside it by name. Drawing both from the subgroup would make a third of the
/// clustered rounds refusals, and "a run made mostly of refusals tests very little" is the argument
/// the pools below are built on.
pub(super) fn fill(rng: &mut Rng, domain: Domain, subgroup: u32, lanes: u32, kind: Kind) -> Op {
    match kind {
        Kind::AddConstant => Op::AddConstant(rng.below(16) as u32),
        Kind::MulConstant => Op::MulConstant(1 + rng.below(3) as u32),
        Kind::ClampBelow => Op::ClampBelow(rng.below(8) as u32),
        // Bounds drawn across the whole input range rather than from a small constant, so that
        // some elements are inside them and some are not. A bound nothing ever crosses makes the
        // step an identity, which agrees with every reference including a wrong one.
        Kind::MinConstant => Op::MinConstant(rng.below(u64::from(domain.ceiling())) as u32),
        Kind::MaxConstant => Op::MaxConstant(rng.below(u64::from(domain.ceiling())) as u32),
        Kind::ClampBoth => {
            // `high` is drawn *from* `low` and not independently: `*Clamp` with the bounds crossed
            // is undefined, and a reference cannot predict undefined. This is the one operand here
            // that has a relationship to keep rather than a range to stay inside.
            let low = rng.below(u64::from(domain.ceiling())) as u32;
            Op::ClampBoth {
                low,
                high: low.saturating_add(1 + rng.below(u64::from(domain.ceiling())) as u32),
            }
        }
        // Drawn from inside the corpus's own range, so that some elements match and some do not.
        // A target nothing equals makes the step an identity, and an identity agrees with every
        // reference including a wrong one — the same trap `MinConstant` documents above.
        Kind::SelectEqual => Op::SelectEqual {
            to: rng.below(u64::from(domain.ceiling())) as u32,
            then: rng.below(u64::from(domain.ceiling())) as u32,
        },
        // The vote about a value. On a corpus of distinct elements it almost never passes, which
        // is the point: a reference that got the *condition* backwards would add everywhere.
        Kind::AddIfAllEqual => Op::AddIfAllEqual {
            add: 1 + rng.below(8) as u32,
        },
        // Any distance at all: a rotate wraps, so unlike `ShiftUp` below there is no undefined
        // lane to steer around. The draw is deliberately wider than the smallest vector the
        // generator makes, because the reduction modulo the width is part of what is under test.
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
        // **The distance must stay inside the subgroup.** `OpGroupNonUniformShuffleXor` with a
        // mask that takes a lane past the subgroup's last one is undefined, and the CPU reference
        // cannot predict undefined — it computed `lane ^ mask` and cheerfully read the *next
        // subgroup's* invocation, which is a well-defined answer to a different question.
        //
        // This was `1 << rng.below(4)` — distances 1, 2, 4 and 8 — and every one of those is
        // inside a 32- or 64-wide subgroup, so it was right on both pieces of real hardware here
        // and wrong on an 8-wide one. Found by lavapipe on seed 3, as a disagreement the fuzzer
        // reported against itself.
        Kind::ButterflyAdd => Op::ButterflyAdd(1 << rng.below(distances(subgroup))),
        Kind::AddIfAnyAbove => Op::AddIfAnyAbove {
            // Thresholds straddling the input's range, so some rounds take the branch and some do
            // not. A threshold nothing ever meets would test one arm forever.
            when_any_above: rng.below(u64::from(domain.ceiling())) as u32,
            add: 1 + rng.below(8) as u32,
        },
        // No operand to draw. A non-zero shift reads lanes that do not exist for some invocations
        // and SPIR-V leaves those undefined, so the operation carries no distance at all — it is
        // the identity, and it exists to prove the instruction is emitted and harmless.
        Kind::ShiftUp => Op::ShiftUp,
        Kind::ShiftDown => Op::ShiftDown,
        // Inside the vector rather than the subgroup: a clustered vector is narrower than the
        // subgroup and `Lanes::broadcast` refuses a position past its own width by name. Every
        // position is drawn, zero included — a broadcast of position zero is not the identity and
        // a reference that returned the values unchanged would be caught by it.
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
        // No operand at all, and the only step here with none. Its edge is a *value* rather than an
        // operand — a two's-complement minimum has no magnitude at its own width — so there is
        // nothing to draw around; `interpret` refuses the round instead.
        Kind::Absolute => Op::Absolute,
        // Small, and multiplicative rather than additive: a product of two draws from the whole
        // range would leave the float domain's exact limit at once, and a round the reference
        // refuses proves nothing about the instruction that caused it.
        Kind::FusedMulAdd => Op::FusedMulAdd {
            by: 1 + rng.below(3) as u32,
            plus: rng.below(16) as u32,
        },
        // **Drawn low where `AddIfAnyAbove` straddles**, and the asymmetry is the whole operand.
        // A threshold *some* element exceeds is easy to draw; one *every* element exceeds is not,
        // because the corpus runs from a magnitude of 1 upwards — so `0` is the only threshold the
        // whole subgroup clears outright, and anything above it is one that some element fails.
        // Three values, and both arms are reached across a sweep.
        //
        // In the signed domains every fourth element is negative and no threshold clears them, so
        // the passing arm arrives *through another step*: an `Op::Absolute` earlier in the program
        // makes the vote reachable. Two operations that only work together are worth having for
        // the same reason `RepeatAdd` and `RolledAdd` are worth having apart.
        Kind::AddIfAllAbove => Op::AddIfAllAbove {
            when_all_above: rng.below(3) as u32,
            add: 1 + rng.below(8) as u32,
        },
    }
}

/// How many butterfly distances fit inside a subgroup of `subgroup` lanes.
///
/// The distances are `1, 2, 4, …`, and the largest usable one is half the width — a lane XOR'd
/// with the width itself lands in the next subgroup. So a 32-wide subgroup has 1, 2, 4, 8 (capped
/// at four for the sake of short programs) and an 8-wide one has 1, 2, 4.
///
/// Never zero: a subgroup of one has no partner to exchange with, and `below(0)` would divide by
/// zero. No device reports a subgroup of one, and a generator that divides by zero on a device
/// nobody has is still a generator that divides by zero.
fn distances(subgroup: u32) -> u64 {
    // A clamp rather than three branches. The `if usable < 4 { usable } else { 4 }` this replaces
    // returns 4 at `usable == 4` either way, so flipping the comparison changed nothing and the
    // mutation gate reported it as a survivor no test could kill. Deleting the branch was the fix,
    // as it has been every other time an equivalent mutant turned up here.
    u64::from(subgroup.trailing_zeros()).clamp(1, 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every operation this file knows, so the pools can be checked against a whole.
    ///
    /// Spelled out rather than derived: `Kind` has no iterator and adding one would be a second
    /// list to keep true. What keeps *this* one honest is the test below — a `Kind` missing from it
    /// is a `Kind` in no pool, and a `Kind` in no pool is an operation the generator can never
    /// draw.
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
        // Three lists with the same operations in most of them is a drift risk, and the drift would
        // be silent: a `Kind` dropped from one pool means a program shape the fuzzer stops
        // generating, which looks exactly like a fuzzer that keeps agreeing.
        //
        // So the relationships are asserted rather than maintained by hand. `WHOLE` is the whole
        // vocabulary — a vector the subgroup's own width can do everything — and the other two are
        // it minus what their mapping refuses.
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

        // **The two axes do not overlap, and that is the claim worth pinning.** A shift in `WHOLE`
        // would be drawn in the float domains, where `spirv-val` rejects the module it builds — and
        // the generator's own sweeps would report it as a refusal rather than as a mistake, because
        // a refusal by name is a legitimate answer here.
        for pool in [WHOLE, CLUSTERED, STRIPPED] {
            for kind in &by_type {
                assert!(
                    !pool.contains(kind),
                    "{kind:?} is gated by the element type and sits in a pool gated by the mapping"
                );
            }
        }

        // A clustered vector shares its lanes with three others: no shuffle across the subgroup and
        // no vote. The rotate stays, because it wraps inside the cluster.
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
        // **The broadcast is the second operation a cluster may cross lanes for**, and the reason
        // is the rotate's: it reads position `source` of its *own* cluster, so every lane it touches
        // belongs to the vector asking. The shifts stay out for the reason they always have — the
        // lanes below a cluster's first are another vector's, and the hardware hands them over
        // without a word.
        assert!(CLUSTERED.contains(&Kind::BroadcastLane));

        // A strip-mined vector may shuffle and vote, and may not rotate: that would move elements
        // between strips.
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

    /// The table above and the `Emit` impls are one claim, and here they meet.
    ///
    /// `by_element` is a hand-written list and `Emit` is the authority — a domain offered an
    /// operation its element cannot emit produces `Outcome::Refused` instead of a round, which the
    /// sweeps *count and print* rather than fail on. So the drift would be silent in the direction
    /// that matters: a fuzzer generating refusals looks exactly like a fuzzer that keeps agreeing.
    ///
    /// So every pairing is built. Offered means it builds; not offered means it is refused **as a
    /// missing instruction** rather than as a width — the two errors exist apart for this reason.
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
        // The lower half of the clamp, and why it is not zero: `distances` feeds `Rng::below`,
        // which is a modulus — so returning zero is a division by zero rather than a smaller
        // choice. A subgroup of one has no partner to exchange with and no device reports one,
        // but "no device reports it" is not a reason for a generator to divide by zero if one
        // ever does.
        assert_eq!(distances(1), 1);
        assert_ne!(distances(1), 0, "a modulus of zero is a panic");
    }

    #[test]
    fn the_distances_stay_inside_the_subgroup() {
        // The largest distance drawn is `1 << (distances(w) - 1)`, and it has to be below `w`:
        // `lane ^ mask` with a mask at or past the width names a lane in the next subgroup, which
        // SPIR-V leaves undefined. This is the property the whole function exists for.
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
        // Four distances is a policy about program length, not about the hardware: a 64-wide
        // subgroup could take 1 through 32 and the generator stops at 8 so the programs stay
        // short. Worth pinning, because raising it is a deliberate change rather than a fix.
        assert_eq!(distances(32), 4);
        assert_eq!(distances(64), 4);
        assert_eq!(distances(16), 4);
        assert_eq!(distances(8), 3, "1, 2 and 4");
        assert_eq!(distances(4), 2, "1 and 2");
        assert_eq!(distances(2), 1, "1 alone");
    }
}
