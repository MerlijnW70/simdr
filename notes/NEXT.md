# What is worth doing next

Rewritten 2026-08-12, after eleven items were worked through in three sittings. Every item here has
a number behind it or a named thing it blocks. Ordered by value per line of work, not by size.

The point of the file is that the ordering should be arguable. If a later reader disagrees, the
measurements are here to disagree *with* — and two of the eleven were **refuted by their own
measurement** and left undone, which is what the ordering is for.

## What is done, and what each one actually turned out to be

Listed with their outcomes rather than crossed off, because six of the eleven came out differently
from the argument that put them on the list — two of them differently enough that the work was not
done at all, and one of which produced a better finding than the item it was part of.

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

Nothing defers a value outside the tests written for it — item 6 is why.

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
| emitting the modules | 74.2 µs | 5.3 µs |
| a pipeline each, from fourteen modules | 809.6 µs | 57.8 µs |
| a pipeline each, from one specialized module | 793.0 µs | 56.6 µs |

**9.7%** of the setup, comparing the two strategies whole: 85.6 µs of 883.9. A specialization
constant is fixed *at* pipeline creation, so fourteen values still need fourteen pipelines; all it
removes is thirteen module emissions. One module per parameter value is cheap. One *pipeline* per
parameter value is not.

> This table first said 485 µs per pipeline and 1.0%, and both were wrong — the probe allocated two
> buffers per call and was reporting allocation as pipeline creation. See `notes/FINDINGS.md`.

What the measurement points at instead is item 1 below, which is built and measured at **5.0×**.

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

### 9. Hold the reduction chain's pipelines — **built, and measured at 5.0×**

`Gpu::reducer(elements)` builds every pipeline a reduction needs and keeps them, with the buffers
they are bound to, in one object. `runner/examples/reducer.rs`, RTX 4080:

| elements | folds | `Gpu::sum` | `Reducer::sum` | faster |
| --- | --- | --- | --- | --- |
| 8 192 | 8 | 967.0 µs | 191.7 µs | **5.0×** |
| 1 048 576 | 15 | 3069.6 µs | 1941.2 µs | 1.6× |

The saving is per *call*, not per element — about 800 µs to 1.1 ms either way — so it is most of a
small reduction and a smaller share of a large one, which is what a setup cost looks like.

It is built for a length, because how many folds a reduction needs depends on how many elements it
is reducing. A different length needs a different `Reducer`, and that is in the type rather than
behind a resize that would rebuild what the object exists to keep.

**The thing to be careful about** was real: a pipeline holds a descriptor set, and a descriptor set
points at particular buffers. Caching pipelines apart from their buffers would be a use-after-free
in safe-looking code. One type owns both and drops the pipelines first.

### 10. Integer dot product — **built, and it depends entirely on the device**

`OpSDot`, `OpUDot`, `OpSUDot` and `OpSDotAccSat`, over four 8-bit components packed into a 32-bit
integer. Both devices here support it and both report the packed signed form as accelerated.

One instruction against the eleven it replaces — four shifts up, four bitcasts, four shifts down,
four multiplies and three adds. `runner/examples/dot.rs`:

| kernel | RTX 4080 | integrated Radeon |
| --- | --- | --- |
| one dot product per element, 262 144 invocations | 1.00× | 1.52× |
| thirty-two per element, 262 144 invocations | 1.18× | **9.08×** |

The first row is memory-bound and the second is not, which is why both are there. The discrete part
has enough integer throughput that eleven instructions cost nearly what one does; the integrated
part does not, and the difference is nine times.

**It does not overturn `decisions/DR-0004`.** The packing is in the instruction's operands, not in
the vector: a `Simd<u32, N>` is still one `u32` per lane and `OpSDot` reads each of them as four
bytes. DR-0004 carries the table and says so.

Along the way the lane API gained the shifts — `shift_left`, `shift_right_logical`,
`shift_right_arithmetic` — because the written-out twin needs them, and the two right shifts are
another pair that agree on every value with the top bit clear.

### 11. Multi-dimensional dispatch — **built, and it costs nothing**

`Shape::grid`, `Kernel::load_row` / `store_row` / `load_row_at`, and a `runner::Grid` that
dispatches along y. A kernel addresses `row × pitch + column`, where the column is the same
expression a one-axis kernel uses — the same code, not a second copy that agrees.

The prediction was that the extra multiply and add would be invisible on a memory-bound kernel, and
`runner/examples/plane.rs` says so on both hardware devices: 3.38 µs against 3.38 on the 4080,
42.99 against 42.92 on the Radeon.

**The first version of that measurement reported 2× and was measuring something else.** A grid
`rows` deep has `subgroup × rows` invocations per workgroup, so comparing it against a one-axis
kernel of `subgroup` invocations moved the occupancy at the same time. The example is a two-by-two
now. `notes/FINDINGS.md` has both halves and `decisions/DR-0006` records why there is no z.

### 12. The workgroup size — **swept, and the constant does not move**

The confound above was a finding in its own right, and `runner/examples/occupancy.rs` chased it
down: three kernel shapes across every workgroup size, on all three devices, with the invocation
count and the total work held fixed.

| 262 144 invocations | 1 subgroup | best | at | spread |
| --- | --- | --- | --- | --- |
| RTX 4080, memory-bound | 5.35 µs | 2.13 µs | 16 | **2.51×** |
| RTX 4080, arithmetic-bound | 14.94 µs | 13.32 µs | 16 | 1.12× |
| integrated Radeon, memory-bound | 40.61 µs | 40.61 µs | **1** | 1.07× |
| integrated Radeon, arithmetic-bound | 1422 µs | 1422 µs | **1** | 1.29× |

**64 is optimal on the Radeon and 1.54× off on the 4080**, and the two want opposite things — so
there is no better constant. `Gpu::limits()` reports `maxComputeWorkGroupInvocations` now, the
`occupancy` kernels take the size as an argument, and `notes/FINDINGS.md` says which sizes are
worth trying. Wiring a heuristic to three data points would be inventing a device model.

It is also **not an occupancy effect in general**: the arithmetic-bound kernel moves 1.12× where
the memory-bound one moves 2.51×, so whatever the larger workgroup buys, it buys in the load path.

**And the arithmetic row was folded away twice before it was arithmetic.** `x × f + s` composes
into a closed form and the driver found it; `min(_, u32::MAX)` is the identity and was deleted, so
the fold came back. Both were caught by running the loop at 64 iterations and at 512 and seeing the
number not move. `notes/FINDINGS.md` has it under the heading that names it.

## 1. The reduction chain copies the whole buffer between passes

`Gpu::replay` copies `bytes` from destination back to source after every fold, and `bytes` is the
*original* buffer's size for all of them. A 15-fold reduction over 2²⁰ elements copies 4 MB fifteen
times, and after the first fold all but 2¹⁹ of those elements are ones nobody will read again.

The hypothesis is that this is most of what `Reducer::sum` still costs at large sizes — it is 5.0×
faster than `Gpu::sum` over 8 192 elements and only 1.6× over 2²⁰, and a per-call setup saving does
not explain a gap that *shrinks* with size. Copying that grows with size does.

The copy length is known at every step: the chain already knows how many elements each fold leaves.
So this is arithmetic rather than architecture, and the measurement to beat is in
`runner/examples/reducer.rs`.

**What would refute it:** the copy being a negligible share of a 1.9 ms reduction, in which case
the 1.6× is the arithmetic and there is nothing here. Time the copies before shortening them.

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

**A third dispatch axis.** `decisions/DR-0006` has the argument: the term is easy and the *layout*
is not, and a z count above 1 today would run every workgroup again over the same elements. `Grid`
has no z field, so that dispatch cannot be written by accident.

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
