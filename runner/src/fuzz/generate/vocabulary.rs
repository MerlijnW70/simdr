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
    ShiftUp,
}

/// Legal under every mapping: nothing here reads another lane.
pub(super) const ELEMENTWISE: &[Kind] = &[
    Kind::AddConstant,
    Kind::MulConstant,
    Kind::ClampBelow,
    Kind::MinConstant,
    Kind::MaxConstant,
    Kind::ClampBoth,
    Kind::RepeatAdd,
    Kind::RolledAdd,
    Kind::RolledCounterAdd,
];

/// The above, plus the three that need a vector at least as wide as the subgroup.
///
/// A shuffle or a vote on a vector that shares its lanes with three others is refused by the lane
/// API, and the generator respects that rather than leaning on `build` to say no — a run made
/// mostly of refusals tests very little.
pub(super) const EVERYTHING: &[Kind] = &[
    Kind::AddConstant,
    Kind::MulConstant,
    Kind::ClampBelow,
    Kind::MinConstant,
    Kind::MaxConstant,
    Kind::ClampBoth,
    Kind::RepeatAdd,
    Kind::RolledAdd,
    Kind::RolledCounterAdd,
    Kind::ButterflyAdd,
    Kind::AddIfAnyAbove,
    Kind::ShiftUp,
];

/// Draw the operands for `kind`.
///
/// Loop trip counts and constants stay small: a rolled loop of four is the same shape as one of
/// four hundred — four blocks and two phis — and the short one leaves every sum well inside the
/// float domain's exactly-representable range, which is what lets the comparison be exact at all.
pub(super) fn fill(rng: &mut Rng, domain: Domain, kind: Kind) -> Op {
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
        Kind::ButterflyAdd => Op::ButterflyAdd(1 << rng.below(4)),
        Kind::AddIfAnyAbove => Op::AddIfAnyAbove {
            // Thresholds straddling the input's range, so some rounds take the branch and some do
            // not. A threshold nothing ever meets would test one arm forever.
            when_any_above: rng.below(u64::from(domain.ceiling())) as u32,
            add: 1 + rng.below(8) as u32,
        },
        // Zero, always: a non-zero shift reads lanes that do not exist for some invocations, and
        // SPIR-V leaves those undefined. A reference cannot predict undefined, so this stays the
        // identity and exists to prove the instruction is emitted and harmless.
        Kind::ShiftUp => Op::ShiftUp(0),
    }
}
