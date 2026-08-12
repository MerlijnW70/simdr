---
id: DR-0002
title: A module is built for a known subgroup width
status: prose-only
---

## The tension

A `Simd<T, N>` fixes `N` when the code is written. A subgroup's width is fixed by the hardware and
is only knowable at runtime — 32, 64 and 8 have all been measured here, and `VkPhysicalDeviceSubgroupProperties`
is what says which. The lowering has to decide between `Reduce` and `ClusteredReduce` and, if it
clusters, name a `ClusterSize`.

`ClusterSize` is an operand of the instruction and must be a constant in the module. So the choice
cannot be deferred to the device: **by the time the module exists, the decision is already made.**

> **Corrected 2026-08-12.** The paragraph above overstated the case, and `decisions/DR-0005`
> records the experiment that showed it. A specialization constant *is* a constant instruction, so
> `ClusterSize` **can** be supplied at pipeline creation — the validator accepts it and an RTX 4080
> runs it at 4, 8 and 16 from one module. What cannot be deferred is the thing this record is
> actually about: which of three *instruction sequences* to emit. `Reduce`, `ClusteredReduce` and
> the strip-mined fold differ in their instructions, not in an operand, and no value arriving at
> pipeline time can add instructions that were never emitted. The decision below is unchanged; only
> the reason given for it in this paragraph was too strong.

## What we tried not to do

Three ways out were considered and rejected:

- **Read `SubgroupSize` at runtime and branch.** The built-in exists, but a branch would have to
  contain every mapping's instructions and pick between them at runtime — which is a larger module
  doing strictly more work than one built for the width it will run on. (This bullet also used to
  say a branch "cannot supply a `ClusterSize`, which is a compile-time operand"; see the correction
  above. A specialization constant can. A *branch* still cannot, because a value read from a
  built-in is not a constant instruction.)
- **Require `VK_EXT_subgroup_size_control` and pin the width.** Real, and it narrows the devices we
  run on for a problem that is not actually about control.
- **Assume 32.** It is right on NVIDIA and wrong on half of AMD, and wrong silently.

## The decision

**[`Lanes`] is constructed with the subgroup width it is building for, and the caller gets that
width from the device.** The runner already reports it: `gpu.limits().subgroup_size`. A module is
therefore specialised to a device family rather than universal, and that is stated rather than
implied — `Lanes::new` takes the number, so no caller can forget it exists.

This is not a workaround. It is what the hardware is: a `Simd<f32, 8>` means something different on
a 32-lane machine than on a 64-lane one, and pretending otherwise is how a portable-looking
abstraction becomes an unportable one.

## Consequences

- Building one module per target width is the caller's job. That is cheap — a module is a few
  hundred words — and it is honest about what varies.
- `reduce_*` picks its shape from `N` against the width, and `lanes::Mapping` is where that choice
  is made once:
  - equal → `Mapping::WholeSubgroup`, a plain `Reduce`.
  - a divisor → `Mapping::Clusters`, a `ClusteredReduce` whose size is `N` itself, so the lanes
    that would otherwise idle are running other copies of the same vector.
  - a multiple → `Mapping::Strips`, several elements per lane. **This is built.** The strips fold
    within each lane first — `strips - 1` scalar operations — and then one subgroup instruction
    runs over the partials. Because the strips never left their lane, that last step is a *plain*
    `Reduce` and not a clustered one, which is the part that is easy to get backwards.
  - anything else — 12 lanes on a 32-wide subgroup — has no mapping at all and is refused as
    `LaneError::NoMapping`, which names both numbers.
- Strip mining is bounded: `lanes::MAX_STRIPS` elements per lane, and more is
  `LaneError::TooManyStrips` rather than a silent truncation. A reduction over fewer lanes than
  were asked for is a wrong answer, not a smaller one.
- A test that asserts a reduction's result must compute its reference from the *device's* width,
  never from a literal 32. `runner/tests/lanes.rs` does, and `runner/tests/execution.rs` does for
  the shapes that predate the lane API.

## What this record got wrong

Until 2026-08-11 the list above said strip mining "is not built", and named an error —
*LaneError::TooWide*, written here without backticks because it does not exist and never did — as
the thing that would say so. Strip mining had been built for some time. `noha gate` printed a tick
beside this file throughout, because its decision check reads a record's front matter and not its
claims.

`tests/integrity.rs` is the answer: it extracts every `Thing::member` written **in backticks** in
this directory and fails when `src/` no longer defines one. It would have caught that name on the
day it changed, in about a millisecond.

The backticks are the whole convention, and it cuts both ways. Code spelled as code is a claim this
crate defines it, and is checked. A dead name being discussed — as just above — is prose, and is
not. So a retraction can name what it retracts without the check mistaking the obituary for a
promise.
