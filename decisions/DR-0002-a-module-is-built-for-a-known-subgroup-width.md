---
id: DR-0002
title: A module is built for a known subgroup width
status: prose-only
---

## The Measurement

`simdr probe` reads the two devices in this machine at subgroup **32** on an RTX 4080 and subgroup
**64** on integrated AMD Radeon Graphics, both with a workgroup ceiling of 1024 invocations — 32
subgroups in one against 16 — and timestamp periods of 1.0 ns and 10.0 ns per tick. The same runner
suite over the same tree passes 409 tests with **0 skips** at 32 and 398 with **17** at 64, where 16
of the seventeen give the reason `written for a 32-wide subgroup` and the seventeenth is a driver
fault; `runner/tests/fuzzing.rs` terminates with a segmentation fault at 64 and does not at 32.
`src/lanes/mapping.rs` names three mappings, `src/lanes/vector.rs` bounds strip mining at
`MAX_STRIPS` = 8, and `src/kernel/binding.rs:50` refuses a shape whose subgroup is zero or not a
power of two with `LaneError::BadWidth`.

## The Decision

`Lanes::new` takes the subgroup width, and `Shape` carries it beside the workgroup and the buffer
count, so a module is specialised to a device family rather than universal and no caller can forget
the number exists. `lanes::Mapping` decides once: `N` equal to the width is `WholeSubgroup`, a
divisor is `Clusters`, a multiple is `Strips`, and anything else is refused as `LaneError::NoMapping`
naming both numbers.

## The Rejected Route

Assuming 32 was rejected at 64, measured on the second device in this machine on 2026-08-26.
Reading `SubgroupSize` at runtime and branching was rejected because the branch would have to hold
every mapping's instructions, and the three mappings differ in instruction count rather than in an
operand — a strip-mined fold emits `strips - 1` scalar operations that no runtime value can add.
Requiring `VK_EXT_subgroup_size_control` was rejected on device coverage and **NOT MEASURED**: no
figure was taken for how many devices it excludes.

## The Limit

The 17 skips at width 64 are the suite declining to run, not the emitter refusing, so they measure
what the tests were written for and not what the mapping does. Only two widths exist on this
machine; 4, 8 and 16 are reached in CI on a software rasteriser and were not run here. The
segmentation fault at 64 was observed once, in one binary, and no reduced case for it was produced.
