---
id: DR-0003
title: A branch is uniform across the subgroup, or it is refused
status: prose-only
---

## The Measurement

Building the same `Simd<f32, 32>` at both widths in this machine on 2026-08-26, `Lanes::any` and
`Lanes::all_equal` return `LaneError::NoSuchForm` at subgroup **64** and build without complaint at
subgroup **32**; `Lanes::ballot` builds at both, because its refusal is written for a strip-mined
vector and a 32-lane vector on a 64-wide subgroup is clustered. `src/lanes/vote.rs` produces
`NoSuchForm` at three sites and `src/lanes/shuffle.rs` at two. The type that carries a vote's answer
is `Uniform`, and the only functions returning one are `Lanes::any_uniform` and
`Lanes::all_uniform`; `Lanes::if_uniform` takes it, and a per-lane `Predicate` is a different type
the call will not accept.

## The Decision

A branch takes a `Uniform`, which only a vote produces, and a per-lane conditional is spelled
`Lanes::select`, which computes both arms and picks between them. There is no `if_per_lane`, and a
subgroup instruction inside a per-lane conditional is therefore not expressible. Loops carry the
same rule: the trip count is uniform or the loop is refused.

## The Rejected Route

Offering a divergent branch was rejected because a subgroup instruction inside one answers for
whichever lanes are still executing, which the specification declines to pin down — and **NOT
MEASURED**: no figure was taken for what a divergent branch would save over `select` on either
device, because the API offers no divergent form to time against.

## The Limit

The refusal above is unreachable on a 32-wide device for a vector of 32 lanes, so a machine holding
only the RTX 4080 exercises none of it — the error existed and had never been returned until a
64-wide device was run. `Lanes::if_uniform` cannot check that a caller's `Id` came from a vote,
only that its type did, and `OpControlBarrier` reaching every invocation is a property of where the
caller puts it that nothing in the emitter inspects.
