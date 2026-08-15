---
id: DR-0003
title: A branch is uniform across the subgroup, or it is refused
status: prose-only
---

## The problem

A subgroup instruction is defined in terms of the lanes that are *active*. Put one inside a branch
that some lanes take and others do not, and the answer changes according to which lanes happen to
be executing — which is a different program from the one that was written, and one whose result
the specification declines to pin down.

```
if (lane_value > threshold) {      // some lanes in, some out
    total = subgroupAdd(x);        // over which lanes?
}
```

Real shading languages let you write that and leave the consequences to the reader. Every GPU
programmer has been bitten by it.

## The decision

**[`Lanes`] branches on a condition that is uniform across the subgroup, and refuses one that is
not.** The condition for a branch is not a [`Predicate`] — a per-lane boolean — but the *result of
a vote*, which by construction holds the same value in every lane of the subgroup.

```rust
let over = lanes.greater_than(value, limit)?;   // per lane: a Predicate
let any  = lanes.any(over)?;                    // uniform: every lane agrees
kernel.if_uniform(any, |kernel| { … })?;        // and so this is safe
```

There is no `if_per_lane`. Per-lane conditionals are spelled [`Lanes::select`], which computes
both sides and picks — no divergence, no reconvergence, and a subgroup operation inside it is
simply not expressible.

## What it turned out to also buy — 2026-08-12

A second consequence, unplanned, found when workgroup shared memory arrived.

`OpControlBarrier` must be reached by **every** invocation of the workgroup. A barrier inside a
branch that only some of them take is undefined behaviour rather than a slow path, and on real
hardware it usually appears to work at small workgroup sizes — which is the worst way for a rule to
be broken.

Nothing in the emitter can check that; it is a property of where the caller puts the barrier. But a
caller who cannot write a divergent branch in the first place cannot easily write a barrier some
lanes miss. The rule that was adopted to keep subgroup reductions meaningful also makes the one
piece of workgroup synchronisation hard to misuse.

That is not an argument for the decision — it was right on its own terms — but it is worth
recording that a restriction paid for itself twice.

## And what a second device confirmed — 2026-08-12

The refusal has a sibling that had never fired. A vote answers for the whole subgroup, and SPIR-V
has no *clustered* vote — so a `Simd<T, 32>` on a **64-wide** subgroup shares its subgroup with
another vector, and asking it to vote would return an answer covering both. `Lanes` refuses that by
name, with `LaneError::NoSuchForm`.

On a 32-wide device a `Simd<T, 32>` is the whole subgroup and the refusal is unreachable. Running
the suite on a 64-wide device turned four kernels that had always built into four kernels that
would not, for exactly the right reason. The error message was already written and had never been
read by anything.

A rule that cannot fire on the hardware to hand is not tested. It is asserted.

## What that costs

Real divergent control flow is genuinely useful and this does not have it. A kernel that wants to
skip expensive work for the lanes that do not need it cannot say so; it computes both sides.

The trade is deliberate. This crate's whole claim is that `Simd<T, N>` semantics survive the trip
to a GPU, and `Simd` has no divergence either — `select` is exactly how portable SIMD spells a
conditional, for exactly the same reason. Offering a divergent branch would let a caller write
something with no `Simd` meaning at all, and the first thing they would put inside it is a
reduction.

## Consequences

- `if_uniform` takes an [`Id`] that came from a vote or another uniform value, and the type system
  cannot check that — a caller who hands it a raw per-lane boolean gets undefined behaviour rather
  than an error. The signature takes `Uniform`, a newtype the votes return, so the mistake needs
  effort.
- Loops have the same rule and the same reason: the trip count is uniform or the loop is refused.
- If a future slice does want divergence, `OpGroupNonUniform*` inside it needs a separate
  argument, not an extension of this one.

## What enforces this

**The type system, completely.** `Lanes::if_uniform` takes a `Uniform`, and the only things that
produce one are `any_uniform` and `all_uniform` — the votes, which answer for the whole subgroup by
construction. A per-lane `Predicate` is a different type and cannot be handed to a branch.

There is no runtime refusal to test here because there is no runtime path to one.
