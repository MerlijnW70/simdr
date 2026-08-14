---
id: DR-0007
title: The lane API declares what it needs, when it needs it
status: prose-only
---

## The decision

An operation in [`crate::lanes`] that needs a built-in variable **declares it itself**, at the point
it is used, and nothing declares one up front for the operations a kernel might reach for. The
entry point's interface list is therefore held as data and rendered whenever it grows, rather than
written out once when the entry point is emitted.

Applies to `SubgroupLocalInvocationId` today, which is what the clustered scan's mask needs. The
same shape holds for whatever the next one turns out to be.

## What forced the question

`Lanes::prefix_sum` refused a vector narrower than the subgroup for a fortnight. SPIR-V has a
`ClusteredReduce` and no clustered scan, so that mapping costs a `log2(N)`-step ladder — and the
ladder needs the invocation's position inside its cluster, which `Lanes` had no way to reach: it is
handed a module and a width, not an invocation.

The ladder was written in `runner::kernels` instead, where a `Kernel` could pass it the workgroup
index. That worked and it was the wrong place: the README's mapping table describes the *lane API*,
and one of its three rows was a kernel in the crate above.

`notes/NEXT.md` wrote down two ways in and said neither was obviously right:

1. **Thread a lane index into `Lanes::new`**, which `Kernel::lanes` could supply and a direct caller
   could not.
2. **Declare `SubgroupLocalInvocationId` in `kernel::binding`**, which costs *every* kernel an
   `Input` variable and — the real objection — the `GroupNonUniform` capability, which a kernel that
   only scales does not declare and a test asserts it does not.

## Why neither, and what the third way costs

Both were answers to "where does the number come from". The third is to notice that the objection to
(2) is entirely about *when*: a capability declared for every kernel is a capability declared for
kernels that do not use it, and a surplus capability makes a module refuse to run on a device that
would have run it. Declared **on demand**, it costs nothing to anyone who does not scan a clustered
vector.

That also settles which number it should be, and the two options above differ there in a way that is
easy to miss. A kernel already knows its index within the *workgroup*, and on all three
implementations here `local & (width - 1)` gives the same answer as the lane's own id — because
subgroups happen to be cut from consecutive local invocations, which Vulkan promises for a pipeline
that asked for full subgroups and not otherwise. `SubgroupLocalInvocationId` is *defined* to be the
lane's position. Option (1) would have carried the coincidence into the layer whose whole value is
that it is defined; this project has an entry in `notes/FINDINGS.md` for each of the three times a
number that agreed on the hardware to hand turned out not to be the number that was meant.

The price is one mechanism underneath. `OpEntryPoint` lists every `Input` and `Output` variable the
entry point reaches, and below SPIR-V 1.4 only those — so a built-in discovered while the *body* is
being built has to reach a list that was written out long before. `Module` keeps the entry point and
its interface as data and re-renders the instruction whenever either changes, which also means a
caller may declare a variable before its entry point exists. `Lanes` is handed a `&mut Module` and
may be building a fragment that has none.

## What says it is right

The failure mode is not a wrong number, it is an invalid module — and every driver here runs it
anyway. Removing the one line that adds the variable to the interface leaves 19 of `tests/kernels.rs`'s
20 modules rejected by `spirv-val` and every device still returning the right answers. That check was
run, and is what `a_clustered_scan_is_valid_spirv` exists for.

## Consequences

- `Module::entry_point` and `Module::require_interface` replace emitting `OpEntryPoint` by hand.
  Both re-render the instruction; the words are built before they are installed, so a refused
  instruction leaves the section as it was.
- `Module::builtin_input` declares, decorates and interfaces a built-in in one call, and does it
  once per built-in. Two variables decorated with one built-in is not a duplicate declaration, it is
  an invalid module.
- `Lanes::lane_index` is private. A public method with no caller is not unused, it is unverified —
  `notes/NEXT.md` item 17 has the `OpUDot` that was invalid for a week to prove it.
- The clustered ladder declares `GroupNonUniform` and `GroupNonUniformShuffleRelative`, and neither
  arithmetic capability: every instruction in it is a scalar one or a shuffle. A kernel that scales
  still declares nothing.
