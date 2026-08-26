---
id: DR-0007
title: The lane API declares what it needs, when it needs it
status: prose-only
---

## The Measurement

Removing the one line in `Module::builtin_input` that adds a variable to the entry point's
interface, and running `tests/kernels.rs` on 2026-08-26, leaves **20 of its 21 tests failing** under
`spirv-val`; the line restored, all 21 pass. That file hands 23 modules to the validator. Every
device in this machine returns right answers either way, so the failure mode is an invalid module
that runs.

`Lanes::lane_index` reads `SubgroupLocalInvocationId`, which is what the clustered scan's mask
needs. On the three implementations reachable from here, `local & (width - 1)` gives the same number,
and it does so because subgroups are cut from consecutive local invocations — which Vulkan promises
for a pipeline that asked for full subgroups and not otherwise.

## The Decision

An operation in `crate::lanes` that needs a built-in declares it itself, at the point of use, and
nothing declares one up front. `Module` holds the entry point and its interface as data and
re-renders `OpEntryPoint` whenever either grows, so a built-in discovered while the body is being
built still reaches a list written out earlier. `Module::builtin_input` declares, decorates and
interfaces in one call and does it once per built-in.

## The Rejected Route

Declaring `SubgroupLocalInvocationId` in `kernel::binding` was rejected because it costs every
kernel the `GroupNonUniform` capability, and a module declaring a capability it does not use is
refused by devices that would have run it — a kernel that only scales declares none, and a test
asserts it does not. Threading a lane index through `Lanes::new` was rejected because the number it
would carry is the workgroup index masked, which agrees with the lane's own id by a Vulkan
guarantee this project deliberately does not require.

## The Limit

The measurement above is one deletion on one machine, and what it establishes is that `spirv-val`
sees the omission — not that any driver does. All three implementations here went on returning
right answers from the invalid modules, so no device in reach can distinguish the two states, and
whether some other driver would reject them is **NOT MEASURED**. Nothing checks that a future
built-in is declared on demand rather than up front; the rule is followed by one call site.
