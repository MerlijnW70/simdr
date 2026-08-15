//! Which operations the generator may draw from, and how their operands are filled in.
//!
//! Split from [`super`] so the generator itself is only the drawing: pick a pool, pick from it,
//! fill it in. What is *in* the pool is a policy question — which operations are legal under which
//! mapping — and it reads better as a table than as arms of the match that draws.

use super::Rng;
use crate::fuzz::domain::Domain;
use crate::fuzz::program::Op;

/// Which operation to generate, before its operands are drawn.
///
/// A named list rather than integers and `match` guards. The guards were the obvious spelling and
/// they carried an arithmetic relationship between a bound and a set of arm labels — mutate one
/// comparison and two operations swap places, which every test still passed because both were
/// still reachable. Killing that would have meant pinning an arbitrary index-to-operation map, and
/// a test of an arbitrary internal mapping proves nothing and makes refactoring painful.
///
/// A table has no arithmetic in it to get wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    Kind::AddIfAllEqual,
    Kind::ShiftUp,
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
    Kind::AddIfAllEqual,
    Kind::ShiftUp,
];

/// Draw the operands for `kind`.
///
/// Loop trip counts and constants stay small: a rolled loop of four is the same shape as one of
/// four hundred — four blocks and two phis — and the short one leaves every sum well inside the
/// float domain's exactly-representable range, which is what lets the comparison be exact at all.
pub(super) fn fill(rng: &mut Rng, domain: Domain, subgroup: u32, kind: Kind) -> Op {
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
    const EVERY_KIND: [Kind; 15] = [
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
        for kind in EVERY_KIND {
            assert!(
                WHOLE.contains(&kind),
                "{kind:?} is in no pool, so the generator can never draw it"
            );
        }
        assert_eq!(WHOLE.len(), EVERY_KIND.len());

        // A clustered vector shares its lanes with three others: no shuffle across the subgroup and
        // no vote. The rotate stays, because it wraps inside the cluster.
        for kind in CLUSTERED {
            assert!(WHOLE.contains(kind));
            assert!(
                !matches!(
                    kind,
                    Kind::ButterflyAdd | Kind::ShiftUp | Kind::AddIfAnyAbove | Kind::AddIfAllEqual
                ),
                "{kind:?} answers for every vector sharing the subgroup"
            );
        }
        assert!(CLUSTERED.contains(&Kind::RotateUp));

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
