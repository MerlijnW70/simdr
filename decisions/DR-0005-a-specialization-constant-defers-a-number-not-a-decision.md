---
id: DR-0005
title: A specialization constant defers a number, not a decision
status: prose-only
---

## The decision

A specialization constant may carry any *value* a kernel needs at pipeline creation — an addend, a
scale, a fold size, a cluster size. It may not carry anything the **emitter** has to reason about
while it is building the module.

`Lanes::new` still takes the subgroup width, and `decisions/DR-0002` still holds.

## The experiment that could have overturned it

`notes/NEXT.md` asked whether `ClusterSize` could be a specialization constant. The specification
says it "must come from a constant instruction", and `OpSpecConstant` **is** a constant
instruction, so the answer looked like it might be yes and the case for DR-0002 rested on it being
no.

It is yes. Both authorities agree:

- `spirv-val --target-env vulkan1.1` accepts a clustered `OpGroupNonUniformIAdd` whose
  `ClusterSize` operand is an `OpSpecConstant` — `tests/kernels.rs`.
- An RTX 4080 runs it and produces the right per-cluster sums at 4, 8 and 16, from **one module**,
  with the default of 32 still giving a whole-subgroup reduction —
  `runner/tests/specialized.rs`.

So the sentence in DR-0002 that reads "`ClusterSize` is a compile-time operand, so the choice
cannot be deferred to the device" is **wrong as stated**, and this record exists partly to say so.

## Why DR-0002 survives anyway

Because the choice was never the cluster size. It is *which instruction to emit*.

A `Simd<T, N>` on a subgroup of width `W` reaches one of three different instruction sequences:
`OpGroupNonUniformIAdd` with `Reduce` when `N == W`; the same opcode with `ClusteredReduce` and a
size operand when `N < W`; and a fold of `N / W` scalar operations followed by a reduction when
`N > W`. Those are not one instruction with a parameter. They are three shapes, with different
operand counts, different capability requirements, and — in the strip-mined case — a different
number of instructions in the function body.

A specialization constant arrives long after that is decided. It can change the *operand* of the
clustered form; it cannot turn the clustered form into the strip-mined one, because the strip-mined
one has instructions in it that were never emitted.

The corrected sentence, then: **the cluster size can be deferred, and the mapping cannot.** The
lane API's front door takes a width because it picks a shape, not because the number is needed
early.

## What is deferred today, and why the obvious caller did not happen

Nothing, in the kernels this repository ships. The mechanism exists, is tested end to end, and the
one place it was expected to pay off turned out not to.

`notes/NEXT.md` argued that `Gpu::sum` building fourteen modules for fourteen fold sizes is
expensive in *pipeline creation*. `runner/examples/specialize.rs` measured it — RTX 4080, a
reduction over 2²⁰ elements, buffers allocated once so that what is timed is pipelines:

| | all fourteen folds | per fold |
| --- | --- | --- |
| emitting the modules | 74.2 µs | 5.3 µs |
| a pipeline each, from fourteen modules | 809.6 µs | 57.8 µs |
| a pipeline each, from **one** specialized module | 793.0 µs | 56.6 µs |

Setting the two *strategies* against each other — fourteen modules and fourteen pipelines, against
one module and fourteen pipelines — specializing removes **9.7%** of the setup, 85.6 µs of 883.9.
Emission is 9.2% of what building the pipelines costs and that is what a specialization constant
can remove, because it is fixed *at* pipeline creation and fourteen values still need fourteen
pipelines.

The premise was wrong in a way worth stating plainly: one module per parameter value is **cheap**.
One *pipeline* per parameter value is not, and no specialization constant makes it less so.

> **Re-measured 2026-08-25, on two devices, and the 9.7% is now a loss on both.** The RTX 4080 the
> table above was taken on is no longer in the machine. Re-run on what is:
>
> | device | fourteen modules → fourteen pipelines | one module → fourteen pipelines | specializing gives |
> | --- | --- | --- | --- |
> | RTX 4080 (above, kept for the record) | 809.6 µs | 793.0 µs | **+9.7%** |
> | RTX 5060 Ti (width 32) | 1313.4 µs | 2974.3 µs | **−126.5%** |
> | integrated Radeon (width 64) | 1271.4 µs | 1872.1 µs | **−47.2%** |
>
> Two independent drivers, the same sign, and one of them more than doubles the setup it was
> supposed to shave. Specializing a module is not free to *compile*: the driver sees a new constant
> and re-specializes the shader per pipeline, and on these two it re-specializes for more than the
> module emission it saves — 20.1 µs and 17.6 µs a fold, against hundreds.
>
> **The decision does not move, and it is worth being clear that it never rested on this number.**
> The argument is that a specialization constant is fixed *at* pipeline creation, so fourteen
> values need fourteen pipelines however few modules they came from. That is structural and holds
> at any sign. What changed is that the measurement underneath it stopped being a small win and
> became a large loss — so the sentence "specializing removes 9.7% of the setup" is now true of one
> device that is gone and false of both that are here.
>
> The 9.7% row is kept rather than replaced, because a decision record that quietly restates its
> evidence teaches nobody that evidence moves under it. `runner/examples/specialize.rs` prints
> today's number on whatever runs it; it needs no edit, and it reported the loss as
> `removes -126.5%` without being asked to.

> **Corrected 2026-08-12, later the same day.** This table first read 485.5 µs per pipeline and
> "saves 1.0%". Both numbers were wrong. The probe took one module and allocated two buffers *per
> call*, so what it reported was pipeline creation plus two allocations — and allocation is the
> larger half. The error surfaced when `runner/examples/reducer.rs` timed a whole fourteen-fold
> reduction at 3.1 ms, which is less than fourteen pipelines at 485 µs would have cost on their
> own. [`Gpu::probe_pipelines`] takes a batch now and allocates once. The conclusion did not
> change; the size of it did, from 1.0% to 9.7%.

What the measurement does say is that setup is ~884 µs against a dispatch's 0.8 µs, and that the
reduction chain's real saving is to **hold its pipelines** across calls rather than defer its
constants. That is `Reducer`, and it is built: `runner/examples/reducer.rs` measures **5.0×** on a
reduction over 8 192 elements and **2.2×** over 2²⁰, where the setup is a smaller share of a larger
call.

> That last clause said "where the arithmetic starts to dominate", and the arithmetic does not.
> The same example was later made to break the remaining time down, and it is the host round trip
> and the chained dispatches — not the sums. The ratio also moved from 1.6× once the reduction
> stopped copying its whole output buffer home to read one number out of it.

`kernels::fold_halves_open` and `Kernel::load_offset_by` are kept rather than deleted: they are
what made the comparison possible, they cost one `OpIAdd` per strip against the baked-in form, and
`runner/tests/specialized.rs` checks that an offset arriving at pipeline time reads the same
elements as one baked in.

## Consequences

- `Module::spec_constant`, `spec_constant_bool` and `spec_constant_op` emit the constants;
  `Specialization` in `runner` supplies the values at `vkCreateComputePipeline`.
- Specialization constants are **not deduplicated**, unlike every other constant this crate emits.
  Two with the same default are two different values as soon as a pipeline sets one of them.
- A `VkSpecializationInfo` entry naming an id no constant carries is ignored rather than refused.
  That is Vulkan's rule and it is pinned by a test, because code that sets a superset of what a
  module declares is easy to write and the alternative would make it a crash.
- `OpSpecConstantOp` carries its opcode as a **literal**, in the word where an operand would
  normally go. An instruction one word short of that decodes cleanly and computes something else.

## What enforces this

**Weakly — this is the loosest of the eight.** `Module::spec_constant` returns an `Id`, a value like
any other, so a specialization constant can be added, multiplied and stored and cannot change which
instructions a module contains: the emitter has finished by the time anyone picks its value.

That is structural rather than checked. An `Id` is an `Id`, and nothing here would stop an emitter
branching on the *default* and shipping a module that only appears to defer the decision.
`runner/examples/specialize.rs` measures what deferring is worth; nothing tests what it must not be.
