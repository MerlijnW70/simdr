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

> **Re-measured 2026-08-25 on two devices, and the 9.7% stands.** The RTX 4080 this table was
> taken on is no longer in the machine. On the RTX 5060 Ti that replaced it, `specialize` reports
> **+12.2%, +13.7% and +10.4%** across the runs whose repeats agreed — the same figure as above,
> within the noise of a different card and driver.
>
> ### The retraction this paragraph replaces, which is the more useful half
>
> Earlier the same day this record carried a correction saying the 9.7% had **inverted**: a 126.5%
> *loss* on the 5060 Ti and a 47.2% loss on the integrated Radeon, "two independent drivers, the
> same sign". It was wrong, and it was wrong in the way this project keeps finding things: both
> figures were **single unrepeated samples**, read off one run of an example that timed each
> measurement exactly once and printed it bare.
>
> Repeated five times over, the 5060 Ti gives +12.2%, +4.2%!, −10.7%!, +13.7%, +10.4% — where `!`
> marks a run whose own repeats disagreed by more than a fifth. The three that hold cluster at
> **+12%**. The Radeon never settles at all: +0.7%!, +5.3%!, +12.6%!, every one of them marked, so
> there is no quotable figure for that device — and no support for a 47% loss either. What the
> single sample caught was one batch of pipeline builds landing slow, attributed wholly to whichever
> strategy happened to run second.
>
> The retracted paragraph is described rather than deleted for the same reason the 485 µs row above
> is: a decision record that silently swallows its own wrong turns teaches nobody what a wrong turn
> looked like. This one looked *completely convincing*. It had two devices agreeing, a plausible
> mechanism written out — "the driver re-specializes the shader per pipeline" — and a number the
> example had printed itself. None of that is evidence, and the only thing missing was a second run.
>
> `runner/examples/specialize.rs` now repeats every measurement five times, reports medians, and
> marks any figure whose repeats disagreed. The `!` is what stands between this record and the
> next confident inversion.

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
