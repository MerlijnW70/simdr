# What is worth doing next

Rewritten 2026-08-12, after the five items the previous version listed were built. Every item here
has a number behind it or a named thing it blocks. Ordered by value per line of work, not by size.

The point of the file is that the ordering should be arguable. If a later reader disagrees, the
measurements are here to disagree *with*.

## The five that are done, and what each one actually turned out to be

Listed with their outcomes rather than crossed off, because three of the five came out differently
from the argument that put them on the list.

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

---

## 1. Make something actually use a specialization constant

**What it costs today.** `Gpu::sum` builds ten modules for ten fold sizes; `clipped_dot` rebuilds
for every offset. The mechanism to stop doing that is built and tested and has no caller.

**Why it is first.** It is the only item here whose groundwork is already done, and the measurement
that would justify it — pipeline creation against module emission — is in
`runner/examples/overhead.rs` and has not been re-run since specialization existed. Doing this is
mostly deleting the loop that rebuilds.

**What to watch.** A specialization constant is a constant at pipeline time, so a fold size that
becomes one still can't change the *mapping*. `reduction.rs` picks its lane count per pass; that
part stays as it is.

## 2. Run at a width that is neither 32 nor 64

Both real hardware widths are covered now. What is not covered is 4, 8 or 16 — which is what a
software implementation of Vulkan reports, and what an Intel part can be asked for through
`VK_EXT_subgroup_size_control`.

**Why it is worth anything.** `whole_subgroup!` lists exactly two widths, and the mapping code has
never seen a subgroup narrower than the workgroup by more than a factor of two. A width of 8 makes
`WORKGROUP_SIZE / width` eight subgroups per workgroup, which no test has ever produced.

**What it needs.** Lavapipe (Mesa's software Vulkan) reports 4 or 8, runs on CPU, and would put the
whole loop in CI on a machine with no GPU. That is the cheap half and it is still not done.

## 3. Narrow types in the fuzzer

The differential fuzzer covers `u32`, `i32` and `f32`. The five narrow types have direct device
tests and **no fuzzing at all**, which makes them the least-checked surface in the tree by the
project's own standard.

**What it needs.** `Domain` gains variants that wrap at 8 and 16 bits, and the harness has to
upload and read back at the right stride — `run_bytes` and `run_halves` exist, but `fuzz::mod`
assumes one element per word throughout: `input_len`, the reference, and the comparison.

**Why it is not first.** It is the largest change to the checking machinery, and
`notes/FINDINGS.md` is emphatic that the checking machinery is where the bugs were.

## 4. Integer dot product — `OpSDot` and friends

**What it blocks.** The packed `i8` mapping `decisions/DR-0004` declines to build. `VK_KHR_shader_integer_dot_product` gives a four-element dot product in one
instruction, which is the thing that would make packing worth the fourth mapping.

**Why it is not urgent.** DR-0004's measurement says strip mining already recovers the bandwidth,
so this is about *arithmetic* throughput on a kernel that is not arithmetic-bound. It would need a
kernel that is.

## 5. Multi-dimensional dispatch

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
