---
id: DR-0010
title: A rolled loop is offered where the buffers are, not only where the lanes are
status: prose-only
---

## The decision

The four-block loop shape is generic over what emits it — a crate-internal `Emits` trait with one
method — and both `Lanes` and `Kernel<T>` offer `repeat_rolled` over it. What differs is only what a
body is handed: `Lanes` hands its body a `Lanes`, and `Kernel` hands its body the kernel.

## What forced the question

`Lanes::repeat_rolled` had no callers, and it turned out it could not have had a useful one.

A `Lanes` holds a module and a subgroup width. It has no bindings — those live on `Kernel`, which is
what `load_at` and `store_at` hang off. So a rolled body could compute and could not *fetch*, which
rules out the one shape that most wants a rolled loop: a reduction over a run whose length is not
known when the module is built.

Every kernel in the downstream engine therefore unrolls its strips. That is right for a run of sixty
four and wrong for one of eight thousand: a softmax over a key-value cache block unrolls one body a
strip three times over, and the caller had to cap the block at 256 subgroups of positions and say in
its own documentation that the limit was this emitter's rather than its arithmetic's.

## Why generic rather than a second copy

Because the shape is exacting and the module note above it already says so: the phis must be first
in the header, the merge must be second-to-last in its block, and the branch must follow it
immediately. Each of those has been got wrong here once. A second hand-written copy on `Kernel` is
a second chance to get them wrong, and the two would drift.

The trait is `pub(crate)` and has one method, so it is not an extension point — it is the smallest
thing that lets one routine serve two hosts.

## What it costs

One trait and one indirection, both invisible outside the crate. Nothing about the emitted module
changes: `Lanes::repeat_rolled` produces the words it produced before, which the existing tests
still assert instruction for instruction.

## What is unchanged, deliberately

The trip count is still fixed when the module is built. A count that varied per lane would diverge,
and a subgroup instruction inside a diverged loop answers for whoever is still there — `DR-0003`.
A *uniformly* varying runtime count would be safe and is still not offered, because nothing has
needed one: a caller that dispatches one workgroup a strip already carries the runtime part in its
workgroup count.

## What enforces this

**A unit test, and only for the shape.** `src/kernel/mod.rs` builds a rolled loop whose body loads
from a buffer and asserts the decoded module has one merge, two phis and *one* body — which is what
says the loop rolled rather than unrolled, and that the body reached a binding at all. The existing
tests in `src/lanes/loops.rs` assert the lane version's words instruction for instruction, so a
change that made the generic routine emit something different would fail there.

Nothing enforces that the two hosts stay in step, because there is nothing to enforce: they are one
routine, which is the point of the record.

## What is not verified here

That the emitted loop is valid SPIR-V, on this machine. `spirv-val` is not installed on it, so
`tests/control_flow.rs` skips — the two tests added there assert the module builds, and the unit test
in `src/kernel/mod.rs` decodes it and asserts one body, one merge and two phis. The validator and a
real dispatch are what would settle it, and the downstream engine's device tests are where that
happens first.
