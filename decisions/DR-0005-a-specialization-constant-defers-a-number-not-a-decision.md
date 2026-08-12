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

## What is deferred today

Nothing, in the kernels this repository ships. The mechanism exists, is tested end to end, and has
no caller in `runner/src/kernels` beyond the ones written to test it.

That is deliberate rather than unfinished: `Gpu::sum` building ten modules for ten fold sizes is a
real cost, and the fix is a change to `reduction.rs` that is worth making on its own terms rather
than as a demonstration. The measurement that would justify it — pipeline creation against module
emission — is in `runner/examples/overhead.rs` and has not been re-run since.

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
