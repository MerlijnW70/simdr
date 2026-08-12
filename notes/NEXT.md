# What is worth doing next

Rewritten 2026-08-12, after eight items were worked through in two sittings. Every item here has a
number behind it or a named thing it blocks. Ordered by value per line of work, not by size.

The point of the file is that the ordering should be arguable. If a later reader disagrees, the
measurements are here to disagree *with* — and two of the eight were **refuted by their own
measurement**, which is what the ordering is for.

## What is done, and what each one actually turned out to be

Listed with their outcomes rather than crossed off, because four of the eight came out differently
from the argument that put them on the list — two of them differently enough that the work was not
done at all.

### 1. Narrow element types — `i8`, `u8`, `i16`, `u16`, `f16` — **built**

The prediction was 4× less memory traffic on a bandwidth-bound kernel, and therefore something
close to 4× the speed. **The prediction was wrong in both directions.**

`runner/examples/narrow.rs`, RTX 4080, a clamp over 16 777 216 elements:

| kernel | per pass | GB/s | against `i32` |
| --- | --- | --- | --- |
| `Simd<i8, 32>` — one element per lane | 127 µs | 264 | 1.67× |
| `Simd<i8, 128>` — four strips | 33 µs | 1016 | **6.45×** |
| `Simd<i16, 32>` | 127 µs | 527 | 1.67× |
| `Simd<i16, 64>` — two strips | 65 µs | 1038 | 3.29× |
| `Simd<i32, 32>` | 213 µs | 630 | 1.00× |

One element per lane gives **1.67×**, not 4: an invocation that loads one byte costs the same as
one that loads a word, so a byte-per-lane kernel runs at a quarter of the achievable rate. Strip
mining recovers it and then some — 6.45× — and part of that is cache residency rather than bytes,
because at this size the `i32` buffers are 64 MB each and land in the unsteady regime below.

At 1 048 576 elements every unstripped row takes the same 9 µs whatever its width, because at that
size the dispatch is not bandwidth-bound at all and the narrow types buy nothing.

`decisions/DR-0004` records the mapping decision the measurement vindicated: one element per lane,
and reach for `Simd<i8, 128>` rather than for a packed mapping.

### 2. A subgroup width other than 32 — **done, and there was hardware here all along**

`notes/NEXT.md` said this needed AMD hardware or a software implementation. `vulkaninfo` says the
machine has **two** GPUs: the RTX 4080 at width 32 and an integrated `AMD Radeon(TM) Graphics` at
width **64**. The runner preferred the discrete one and nothing had ever asked for the other.

`SIMDR_DEVICE` now picks a device by name, and `simdr list` names them. The whole execution suite
and 30 000 fuzz rounds are green on both.

**It found ten test defects and no emitter defects.** Four tests could not even build at 64: they
asked for a vote on a `Simd<_, 32>`, which is a *cluster* of a 64-wide subgroup, and the lane API
refuses votes on clusters by name — correctly. The other six had references that grouped by the
device's width while the kernel reduced 32 lanes. Every one of them had conflated "32 lanes" with
"the subgroup", and on one device those are the same number.

The emitter was right at both widths. The harness was written by someone who had one device.

### 3. Specialization constants — **built, and they overturned part of DR-0002**

`OpSpecConstant`, `OpSpecConstantOp`, the `SpecId` decoration, and `VkSpecializationInfo` at
pipeline creation. One module, several pipelines, different answers.

The question worth asking was whether `ClusterSize` could be one, since DR-0002's stated reason for
taking the width up front was that it could not. **It can.** `spirv-val` accepts it and the 4080
runs the same module at cluster sizes 4, 8 and 16. `decisions/DR-0005` writes it up and DR-0002
carries the correction; the decision survives for a better reason than the one it was given.

Nothing in `runner/src/kernels` defers a value outside the tests written for it, so `Gpu::sum` still
builds ten modules for ten fold sizes. That is the next item below.

### 4. GLSL.std.450 — **built, and it bought no speed, as predicted**

`min`, `max`, `clamp`, `abs`, `sqrt`, `inverse_sqrt`, `exp`, `log`, `fma`. A clamp is one
instruction where it was two compares and two selects.

The strip fold in `reduce_min`/`reduce_max` **deliberately did not change**. Compare-and-select is
*defined* for NaN and `FMax` is explicitly undefined; this machine returns the non-NaN operand
either way, and agreeing on one device is not the same claim as being defined. Trading a defined
behaviour for an undefined one to save an instruction that was measured not to matter is the wrong
side of that trade.

### 5. Atomics — **built**

`OpAtomicIAdd`, `OpAtomicIIncrement`, `OpAtomicExchange`, `OpAtomicLoad`, `OpAtomicStore`, and
`Kernel::atomic_add_at` for a slot the *data* chooses. Histograms and an allocator both run.

The strongest test is the allocator: `OpAtomicIAdd` returns the previous value, so the slots handed
out are `0..n` with no repeats — and a lost atomic shows up as a duplicate rather than as a total
that is one short.

### 6. Make something use a specialization constant — **measured, and not worth doing**

The previous version of this file put this first, on the grounds that fourteen modules for fourteen
fold sizes is expensive in pipeline creation. `runner/examples/specialize.rs` measured it and the
argument does not survive:

| | all fourteen folds | per fold |
| --- | --- | --- |
| emitting the modules | 71.5 µs | 5.1 µs |
| a pipeline each, from fourteen modules | 6796.8 µs | 485.5 µs |
| a pipeline each, from one specialized module | 6726.0 µs | 480.4 µs |

**1.0%.** A specialization constant is fixed *at* pipeline creation, so fourteen values still need
fourteen pipelines; all it removes is the emission, and emission is 1.1% of the total. One module
per parameter value is cheap. One *pipeline* per parameter value is not.

What the measurement points at instead is item 1 below.

### 7. A width that is neither 32 nor 64 — **done, at 8**

Lavapipe (Mesa's software Vulkan, `llvmpipe`) reports **subgroup width 8** and runs on the CPU.
Installed at `H:\tools\mesa\msvc`, selected with `VK_ICD_FILENAMES`, and the whole execution suite
plus 12 000 fuzz rounds pass on it. That also means the loop runs on a machine with no GPU.

It found one defect in the product's own checking machinery and eight in the tests:

- **The fuzzer generated butterfly distances that leave the subgroup.** `1 << below(4)` gives 8,
  which is inside a 32- or 64-wide subgroup and is the *width* of an 8-wide one — and
  `OpGroupNonUniformShuffleXor` past the last lane is undefined. The CPU reference computed
  `lane ^ mask` and read the next subgroup's invocation, which is a defined answer to a different
  question. Seed 3 reported it as a disagreement the fuzzer had with itself.
- **`whole_subgroup!` listed two widths**, so every kernel using it refused to build with
  `BadWidth` on a device that could have run them. It lists 4, 8, 16, 32 and 64 now.
- **Three tests assumed uninitialised device memory is zero.** The histogram kernels *accumulate*,
  and `Gpu::run` does not initialise the output buffer; two drivers hand back zeros and lavapipe
  does not. They zero it through a `Session` now, which is the only path that can write a binding
  the kernel also reads.
- The rest were the same width-blindness the 64-wide device found: a butterfly distance of 8 in a
  test, a reference summing "everything after the first subgroup" when there are eight of them,
  and lane counts of 32 where the test meant "the subgroup".

**And one device difference worth knowing.** Lavapipe's `Fma` rounds *twice* — it agrees with
`x * x + x` rather than with a fused multiply-add, where both hardware devices agree with the
fused form. `runner/tests/extended.rs` observes which of the two it gets rather than asserting one.

Still unrun: 4 and 16. `whole_subgroup!` can build for them and no implementation here reports
them.

### 8. Narrow types in the fuzzer — **done for the four integers**

`Domain` has `Byte`, `UnsignedByte`, `Short` and `UnsignedShort` now, and `fuzz::check` packs the
buffer at the domain's stride — the one place in the harness where "element values, one per `u32`"
meets "four elements share a word".

`Domain` got *smaller* doing it. Seven domains times eight operations would have been fifty-six
match arms; it is written in terms of `bits()` instead, so `add` is one wrapping add and a mask.

The sweeps agree at every width: 3 000 rounds per domain on the 4080 and the Radeon, 1 500 on
lavapipe, no disagreements.

**`f16` is deliberately not fuzzed.** A half represents integers exactly only to 2048, and a sum
over sixty-four lanes leaves that range immediately — so the exactness argument the float domain
rests on does not hold, and a tolerance would be checking something other than the emitter.
`runner/tests/narrow.rs` covers `f16` against expectations reasoned from the format.

---

## 1. Hold the reduction chain's pipelines

**What it costs today.** A pipeline costs **485 µs** and a dispatch costs **0.8 µs**, measured in
`runner/examples/specialize.rs`. `Gpu::sum` builds one per fold on every call — fourteen for a
buffer of 2²⁰ elements, 6.8 ms — and throws them all away. That is the same waste `Session` was
built to remove for a single pipeline, and it is three orders of magnitude more than the
specialization constant this replaces would have saved.

**What it needs.** `Pass` borrows words and a workgroup count; it would have to carry a built
pipeline instead, and something has to own those across calls the way `Session` does. The awkward
part is that the fold count depends on the input length, so the cache is keyed by length rather
than being one object.

**What to watch.** Pipelines hold descriptor sets pointing at particular buffers, so a cache that
outlives a buffer is a use-after-free waiting to be written in safe-looking code. `Session` gets
this right by owning both together; a chain cache has more moving parts.

## 2. Integer dot product — `OpSDot` and friends

**What it blocks.** The packed `i8` mapping `decisions/DR-0004` declines to build.
`VK_KHR_shader_integer_dot_product` gives a four-element dot product in one instruction, which is
the thing that would make packing worth the fourth mapping.

**Why it is not urgent.** DR-0004's measurement says strip mining already recovers the bandwidth,
so this is about *arithmetic* throughput on a kernel that is not arithmetic-bound. It would need a
kernel that is.

## 3. Multi-dimensional dispatch

`cmd_dispatch(x, 1, 1)` and a one-dimensional address. Everything with a natural 2-D shape — an
image, a matrix tile — has to linearise itself before it reaches a kernel.

---

## Deliberately not doing

**Chasing the large-working-set cliff.** Past ~50 MB the timings stop being steady, and three
explanations have now been tested and refuted: L2 capacity, eviction of a single allocation, and
placement under three simultaneous allocations — all three buffers land device-local up to 3 GB
resident, 18% of the heap. The honest position is recorded and holds.

**A larger CLI.** `simdr probe` answers the question the design forces and `simdr list` answers the
one a second device raised. `simdr validate` would be three lines around `spirv-val`; `simdr emit`
would need a kernel description language, which is a second and worse API beside one whose entire
value is that kernels are Rust with types.

**Mutating the FFI half of `runner`.** A mutant that passes a wrong handle or frees twice kills the
process rather than failing a test. `tests/integrity.rs` holds that exclusion with a reason per
file and checks each one still contains the `unsafe` that excused it.

**Switching the extremes' strip fold to `FMax`.** See item 4 above. One instruction saved, a
defined behaviour lost.

---

## Kept in view

- **`Gpu::run` still assumes input length equals output length.** `run_bound` and `Session` do not.
- **The chain copies the whole buffer between passes.** For a shrinking reduction that is mostly
  copying elements nobody will read.
- **`whole_subgroup!` is a macro in a codebase with no other macros.** It exists because the list
  of widths appeared in twelve places and a list in twelve places drifts. If a third width is ever
  added, that is the one line to change — which is the argument for it.

## Point the gate at the checking machinery, not only at the code

Twelve mutants reproduced in one night of running the mutation gate over the whole workspace: eight
real gaps and four equivalent mutants. **Nine of the twelve were in the fuzzer or its CPU
reference.** None was in the emitter.

The second device made the same point in a different way. Ten tests failed at width 64 and the
emitter was right in every one of them — the harness had a width baked into it while reading as
though it adapted, because it took the width as a parameter and then ignored it.

The concrete habit: when a test takes a parameter, check that it *uses* it. `let width =
limits.subgroup_size` at the top of a test whose expectations say `32` further down is the shape to
look for, and it is invisible on a machine where those are the same number.

## A habit worth keeping

**Re-read `SAFETY` comments when a second caller appears.** `Buffer::write` argued from the set of
callers rather than from a check — "this crate always allocates from the same element count it
writes" — and `Session` falsified it six hours later without touching the file. Nothing automated
caught it: not clippy, not the mutation tester, not 353 tests, not the fuzzer.

There is no tooling suggestion here, which is the uncomfortable part. What there is: when you add a
caller to an `unsafe fn`, read what the old one promised on your behalf.
