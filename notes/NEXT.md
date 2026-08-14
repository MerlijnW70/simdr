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

> Both run now — see item 16. "No implementation here reports them" was true of the *default*
> lavapipe build and of nothing else: its subgroup width is llvmpipe's vector width over 32, and
> that is an environment variable. The claim was about a setting, and read as a claim about the
> hardware.

### 8. Narrow types in the fuzzer — **done for the four integers**

`Domain` has `Byte`, `UnsignedByte`, `Short` and `UnsignedShort` now, and `fuzz::check` packs the
buffer at the domain's stride — the one place in the harness where "element values, one per `u32`"
meets "four elements share a word".

`Domain` got *smaller* doing it. Seven domains times eight operations would have been fifty-six
match arms; it is written in terms of `bits()` instead, so `add` is one wrapping add and a mask.

The sweeps agree at every width: 3 000 rounds per domain on the 4080 and the Radeon, 1 500 on
lavapipe, no disagreements.

> **Superseded by item 19.** `f16` is fuzzed now, 253 of 256 seeds compared exactly. The paragraph
> below is right about the arithmetic and wrong about what follows from it: the answer was to refuse
> the rounds that leave the range, not the domain.

**`f16` was deliberately not fuzzed.** A half represents integers exactly only to 2048, and a sum
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

### 13. The chain's between-pass copy — **shortened, and the premise was wrong**

`Gpu::replay` copied the whole buffer back after every fold. `Pass::writing` says how many words a
pass produced and the copy is that long, which for a 15-fold reduction over 2²⁰ takes the traffic
from 56 MB to 4 MB.

The item predicted this was most of what `Reducer::sum` still costs. It is not:

| where a held reduction over 4 MB goes | per call | share |
| --- | --- | --- |
| fourteen full-buffer copies — before | 385.6 µs | 22% |
| the same fourteen, shortened — after | 274.3 µs | 16% |
| host upload | 338.5 µs | 19% |
| host download | 662.4 µs | 38% |

**And a fifth was not a fifth to be had.** End to end the change bought **85 µs**, because a
whole-buffer step is 27.5 µs of which 19.0 µs is the two pipeline barriers around the copy and only
8.6 µs is the data. The barriers stay whatever the copy carries. `notes/FINDINGS.md` has the third
chain that separated them.

Worth doing at 85 µs? Marginally, and it was worth *measuring* regardless — it is the only reason
the two items below are known to be the real ones.

### 14. The ping-pong — **built, and it is not the speed-up it was scheduled as**

Two buffers alternating instead of a copy between passes. The prediction was ~250 µs of a 1900 µs
reduction, on the grounds that a chained step was 27.5 µs of which 19.0 was its *pair* of barriers.

One barrier costs nearly what two did — a step went 19.0 → 16.7 µs — so it saved about 32. The
refutation clause this item was written with is exactly what happened.

Paired against the old build on the same machine, alternating runs:

| device | with copies | ping-pong | |
| --- | --- | --- | --- |
| RTX 4080 | 1929 µs | 1914 µs | nothing |
| integrated Radeon | 3792 µs | 3631 µs | **5.5%**, every round |
| lavapipe | 4064 µs | 4038 µs | nothing |

**Kept for being shorter, not faster.** `chain.rs` lost the copy and both barrier constants, `Pass`
lost `outputs` and `writing`, `Step` stopped existing — and with them a copy-length bug that
returned the previous call's data. What arrived instead is one sharp question, which is that the
answer now moves between the two buffers by parity.

### 15. The download — **half of it was four bytes wearing four megabytes**

`Reducer::sum` copied the whole answer buffer home and called `.first()` on it. A reduction
produces one number; the rest is the last fold's leftovers. It reads one word now, and `Gpu::sum`
does the same through `Gpu::run_chain_head`.

| paired, 2²⁰ elements | `Reducer::sum` | `Gpu::sum` |
| --- | --- | --- |
| RTX 4080 | 1866 → 1250 µs, **33%** | 3442 → 2728 µs, 20% |
| integrated Radeon | 3663 → 2550 µs, **30%** | 5270 → 4375 µs, 17% |
| lavapipe | 4844 → 3619 µs, **25%** | 87.0 → 79.2 ms, 9% |

Every round on every device. The first change in four that helped everywhere, because it removes
traffic without adding instructions — and the three before it all split by device for exactly that
reason.

**Worth carrying:** the breakdown that found this was built two items earlier, to test a guess about
the copies. It answered, and then three more measurements were taken before anyone acted on its
largest row. *Act on a breakdown's biggest row before its most interesting one.*

### 16. Widths 4 and 16 — **run, and they found undefined behaviour that was already running**

`notes/NEXT.md` and `README.md` both listed these as unreachable. They are not: llvmpipe's subgroup
width is its vector width over 32, so `LP_NATIVE_VECTOR_WIDTH` of 128 and 512 give **4** and **16**,
with `min` equal to `max` at each so the width is pinned.

Five defects, none in the emitter — the same score as 64 and as 8. The one that matters:
`kernels::scale`, the control kernel, said `load::<32>`, which is one element per invocation at 32
and 64 lanes and **eight strips** at four. It had been reading and writing past its buffer at width
8 for a day, returning zeros, in the green column — until four lanes turned it into an access
violation. Three more kernels carried the same literal, one of them the twin of a kernel that had
already been fixed.

The rest were the family the other widths keep finding: a NaN placed at a literal index that is in
the *second* subgroup at four lanes; a "has no mapping" assertion about 12 lanes, which is a
multiple of 4; a full-width case that skipped every width but 32 and 64. And the patch for those
introduced the same bug twice more, because at four lanes a four-lane cluster *is* the subgroup —
so the lane-count test now works from a list of distinct sizes and asserts it has at least two.

**Still open there:** at 128 and 512 bits lavapipe is unstable under `cargo test`'s parallelism —
~40% of runs disagree at some seed, never twice the same, all reproducing green single-threaded and
in one process. Ruled out on our side: no shared state, a device per test, and the 256-bit build
green 8 of 8 under the same parallelism. Documented as a flag rather than chased.

### 17. A self-audit — **and `OpUDot` had been invalid since the day it shipped**

Every claim the project makes about itself, checked. Zero dependencies, no lint escape outside a
test, no missing `# Safety` section, `decode` unable to panic or loop, `tests/integrity.rs` doing
what it says — all held.

What did not: the audit asked *which public operations appear in no test that runs `spirv-val`* and
found **fifteen**. Writing those tests took twenty minutes and the first run rejected
`Lanes::dot_unsigned`, which had been emitting `OpUDot` with a **signed** result type. It had no
caller, no unit test, and no validator coverage — three layers, and it fell between all of them.

**A public method with no caller is not unused, it is unverified.** Every layer reports green about
it by saying nothing at all.

Also fixed: three doc drifts where the *reasoning* had been refuted by this project's own later
measurements, and `sign_extend`'s unstated precondition — the same "every caller passes a safe
value" argument that `Buffer::write` had expire on it once already.

**Reported and not changed:** `clippy::undocumented_unsafe_blocks` fires 78 times in `runner`, but
almost all are one `ash` call inside an `unsafe fn` whose `# Safety` section already covers it. The
missing thing is a way to tell those from a block that needs a *new* argument — not 78
restatements.

### 18. The host round trip — **the map moved into the chain, and it is worth two crossings**

This item said the upload needed *a shape rather than an optimisation*, and warned that nothing
here had data on the device to begin with, so the API would be for a caller that had not arrived.

The caller was one level up. Σ f(x) is a map and a reduce, and doing it costs three host crossings
of which two are the whole buffer. `Gpu::reducer_of(elements, map)` makes the map the first pass of
the same chain, so the intermediate never leaves the device.

| Σ x², both routes held | three crossings | one crossing | |
| --- | --- | --- | --- |
| RTX 4080, 2²⁰ | 2326 µs | 1331 µs | **1.7×** |
| integrated Radeon, 2²⁰ | 5411 µs | 2759 µs | **2.0×** |

The 993 µs saved is the download (718) plus the upload (294) this file measured separately — 1012
predicted against 993 observed, which is the only reason to believe either number.

**The first version said 2.9×**, because the route being replaced was written as `gpu.run`, which
allocates and builds a pipeline every call. Fourth time in two days. *Give the thing you are
replacing every advantage you would give the replacement.*

### 19. `f16` in the fuzzer — **the exclusion was one step too far**

Item 8 left `f16` out on the grounds that a half counts integers only to 2048, so a sum over a few
hundred lanes leaves the range and a tolerance would check the rounding rather than the emitter.

That argues for *noticing* when a round leaves the range, not for skipping the domain.
`Domain::exact_limit` says where the range ends, the reference reports whether it stayed inside, and
a round that did not is **refused** rather than loosened. `Half` runs **253 of 256 seeds compared
exactly**, with 3 refused — coverage, not a domain that looks supported and never runs. A test now
insists on that distinction, because a domain refused every round looks exactly like one that always
agreed.

The same three lines revealed that `Domain::Float` rested on the identical argument one exponent
wider and had never been checked either. It is checked now. It has never fired, which is what the
assumption predicted and the first time anything has confirmed it.

**Eight mutation survivors along the way**, one of which took three attempts: `_ => vec![false; …]`
was a genuine equivalent mutant whose comment said the alternatives were worse and listed the two
that had been tried. The third — compute the vote *inside the arm that reads it* — deletes the
branch and shortens the file.

### 20. The breakdown that came to half a call — **and the 52% row hiding in the gap**

A full mutation run over all 83 targets: **419/419 killed, 0 survivors**, and the three entries in
the ratchet floor confirmed dead — one of them naming a path that had not existed since a file was
split. The floor is empty.

Then the reduction breakdown was re-read and its rows came to about **half** the call they were
breaking down. Two were missing, both skipped by the *measurement* rather than by the call: the
`f32` → `u32` copy, which the upload row hoisted out of its own timed loop, and the fixed cost of
one submission, which the per-step row cancels out by being a difference.

The conversion was **596 µs, 52% of the call** — larger than the fourteen chained dispatches and
the upload together, and it computed nothing. `Buffer::write_floats` copies the caller's slice
straight into the mapping instead.

| `Reducer::sum`, 2²⁰ | via `Vec<u32>` | direct | |
| --- | --- | --- | --- |
| RTX 4080 | ~1342 µs | ~524 µs | **2.6×** |
| integrated Radeon | ~2543 µs | ~1749 µs | **1.5×** |

**Fourth mismeasurement of the week, and the first that hid a win rather than flattering one.** The
breakdown now comes to 109% instead of 52%; over is honest, since the rows overlap.

**Left undone deliberately:** `Gpu::run` converts the same way. It also allocates three buffers and
builds a pipeline per call, so the conversion is a small share of it — and it is the test-shaped
API, where clarity is worth more than the microseconds. `Session` is what a caller in a loop should
reach for, and it takes words already.

### 21. Three submissions to do one thing — **now one**

Splitting the upload row — a full write against a one-word write, which pays the same map, unmap
and submission and almost none of the copying — showed its fixed half was **73 µs**, and the row
beside it priced a bare submit-and-fence at **65**. So the fixed cost of an upload was not the
mapping; it was a whole submission.

`Reducer::sum` made three: one to move the input into place, one for the chain, one to bring the
answer back. `Gpu::replay` takes an optional `before` and `after` copy now and records them inside
the chain's own command buffer — a barrier each instead of a submission each. `Gpu::run_chain` had
the same shape and got the same treatment.

| `Reducer::sum`, 2²⁰ | three | one | |
| --- | --- | --- | --- |
| RTX 4080 | ~548 µs | ~424 µs | **1.29×** |
| integrated Radeon | ~1751 µs | ~1045 µs | **1.68×** |

~124 µs saved against a predicted 2 × 62.

**Where the reduction stands:** 11.2× `Gpu::sum` over 8 192 elements and 5.6× over 2²⁰, against
2.1× at the start of the day — and 2²⁰ went ~1930 → ~424 µs across three changes, none of which
was an algorithm. Each was a cost the *measurement* had been hiding, and the breakdown found all
three only once it was made to add up.

### 22. Folding by sixteen — **built, and worth a quarter of what was predicted**

The breakdown's largest row was the chain: fourteen steps, 56% of the call. `kernels::fold_by` adds
`factor` elements per invocation instead of two, and `folds()` picks the widest factor that still
leaves a whole workgroup. **Five dispatches instead of fifteen** at 2²⁰, three instead of eight at
8 192.

| 2²⁰ | halving | by sixteen | |
| --- | --- | --- | --- |
| `Reducer::sum` | ~442 µs | ~407 µs | 8% |
| `Gpu::sum` | ~2357 µs | ~2203 µs | 6% |

Nothing at all at 8 192, where the chain was short already.

**Both arguments for it were wrong, optimistically.** "It halves the memory traffic" is true as a
ratio and worth ~6 µs, because the first pass reads N either way and the levels that differ are the
tail. "Ten dispatches at ~15 µs" was ~35 µs total, because that per-step figure comes from a chain
of *empty* kernels where a barrier has nothing to overlap with.

Kept for being shorter — five pipelines to build and hold instead of fifteen — rather than for the
8%. The per-step row in `runner/examples/reducer.rs` now says it is an upper bound.

**Fifth measurement lesson of the week and a new kind:** the first four mismeasured a *change*, this
one mismeasured a *component*. A cost measured in isolation is not the same cost measured in company.

## 1. A buffer the caller already owns

`reducer_of` covers Σ f(x). What it does not cover is a caller whose data was produced by some
*other* dispatch it owns: the reducer's bindings are private, and the map has to be a pass of its
chain, so there is no way to hand it a buffer that already holds the right numbers.

That needs `Reducer` to expose a binding, or to take a `Session`'s buffer — and it has the same
problem this item had before `reducer_of`: **no caller in this repository wants it yet.** Every
path here starts from a `Vec`. Left unbuilt on purpose, and the note is here so the next reader
knows it was considered rather than missed.

---

## The list, rewritten 2026-08-13, worst first

Twenty-two items above were finished in a fortnight, so this is what a survey of the tree turned up
once the obvious work ran out. It is ordered by how badly it is wrong rather than by how good it
would be, because the first four are all *things that are already false* and the rest are things
that are merely absent.

**Nine of the thirteen are done**, in one sitting on 2026-08-13/14: 1, 3, 4, 5, 6, 7, 9, 13 and the
file split under 3. Two of those ended differently from how they were written — item 3's fix needed
nothing declared, and item 9 turned out not to be worth doing. What is left is **8** (the strip-mined
and clustered scan, which is the interesting one), **10** (the breakdown that reads 123%), **11**
(deferred on purpose) and **12** (a third vendor, which needs hardware that is not here).

The three most useful things that happened were not on the list. Item 3 found **eleven tests reading
past their input**; item 4 found the **MSRV was wrong by nine releases**; and item 7 stopped being a
nicety when it turned out a float scan cannot recover its exclusive form by subtraction.

One item was prompted from outside. VectorWare published a piece describing the same premise this
project runs on — a warp is a vector unit, so `Simd<T, N>` lowers onto lane instructions — from a
compiler backend consuming `core::simd`. Their post is honest about the same hard part this file
has circled for a month: what to do when `N` does not equal the width. They idle lanes for a small
`N` and do not detail the large case. That is item 8 below, and it is the one place where finishing
the work would make this project's central claim true for both of its algorithms rather than one.

### Tier 1 — things that are actually wrong

**1. A fresh clone cannot run the test suite — done.** Four of `tests/integrity.rs`'s five tests
panicked on any clone, because they `expect`ed a `noha.yaml` that a global ignore keeps out of every
repository on this machine. They skip loudly now and `cargo test --workspace` from a clone passes.

The plan was a committed manifest of the source list. That turned out to be the wrong shape: the
committed list would be a second copy of a thing derivable from the tree, and the *interesting*
invariant needed no list at all. The excuse for not mutating a file is "it is FFI, so it contains
`unsafe`", and only one direction of that was checked. The other — **every file containing `unsafe`
is excused** — runs on a clone, needs no config, and guards something the first direction does not:
an expired excuse costs coverage, while unsafe code left inside the gate costs the mutation run,
since a mutant that passes a wrong handle kills the process instead of failing a test.

It is also the rule this project had applied by hand three times without anything enforcing it.

**2. The runner's whole kernel library is never validated.** `spirv-val` runs over kernels built in
`simdr`'s own tests. Everything in `runner/src/kernels/` — scan, reduce, dot, narrow, plane,
network, scatter, occupancy — goes straight to a driver, and the dependency arrow means `simdr`'s
tests *cannot* reach it even in principle. Drivers are lenient about things the validator is not:
`OpUDot` with a signed result type ran correctly on two devices for weeks and was invalid the whole
time. That is the gap this leaves open, and it is the highest value per line on the list.

**3. `dispatch::extent` cannot see strip-mining — done, and nothing had to be declared.** The plan
was to carry the lane count out of `Kernel::finish` or decorate the module with it. Neither was
needed: every access starts from `Kernel::run_start`, which emits `group × (workgroup × strips)`,
and the workgroup size is already read from `LocalSize` — so dividing the constant by it gives the
strip count back. A second copy of a number is a second thing to keep true, and there is no second
copy.

`kernels::lane_affine::<32>` at width 4 is eight elements per invocation and is now refused rather
than run, which is the exact shape of the `kernels::scale` bug. What remains outside the check is a
constant offset past the run — `load_offset` reading `in[i + half]` — and that direction
under-counts, so it refuses less than it might and never more.

**It found eleven tests reading past their input on the first run**, across five files, every one of
them green at widths 4, 8 and 16 since those widths were added. `notes/FINDINGS.md` has the table
and what it says about the width sweep: an out-of-bounds *read* is only an access violation when it
crosses a page, so the sweep catches this class when it is unlucky and the bounds check catches it
always.

**4. No CI and no pinned toolchain — done, and the MSRV was wrong by nine releases.**
`.github/workflows/ci.yml` runs formatting, clippy at `-D warnings`, the emitter's suite against
`spirv-val`, the integrity checks, and the whole runner suite on lavapipe at widths 4, 8 and 16.
Widths 32 and 64 stay manual because they need the two GPUs, and the workflow says so rather than
leaving it to be assumed.

**And its first run on a device found a portability bug.** A test asserted that the sum of
sixty-four negative zeros keeps its sign, which IEEE 754 says and Vulkan does not require — it is
the optional `shaderSignedZeroInfNanPreserveFloat32`, binding only a module that declares the
matching execution mode, which this emitter does not. Two GPUs and a locally built lavapipe all
preserved it; the Mesa in Ubuntu 24.04 folds it to `+0.0`. A shared runner turns out to be a
*fourth implementation* rather than only automation, which is a better argument for CI than the one
this item was written for.

`rust-version` said **1.97** under a comment reading *"Measured, not assumed"*. It was neither: 1.97
is the version that happened to be installed, and nothing had ever built the workspace with anything
else. The true floor is **1.88**, where `if let` chains stabilised —
`runner/src/fuzz/generate/coverage.rs` uses one and 1.87 fails on exactly that. Nine releases of
users were excluded for no reason, and all 706 tests pass on 1.88. CI holds it there now, which is
what makes the first word of that comment true.

### Tier 2 — the scan, and what it needs

**5. `WorkgroupId` wired into `Kernel` — done, and it was smaller than it looked.** The built-in
had been loaded since the beginning and used internally to work out where a workgroup's run starts;
nothing exposed it. `Kernel::workgroup_index` returns it and `Kernel::store_at` writes at a slot the
caller names, which together are the "one value per workgroup" that was missing.

`kernels::scan::scan_blocks` is the first user: the same scan with each block's total written to a
third binding. It is what item 6 needs and is worth having on its own.

The gate found the interesting part. Skipping the offset work on the final subgroup cannot change
the answer — the boundary would be `workgroup - 1` and no lane's index exceeds it — so no
behavioural test can see the difference between doing it and not. It is still one comparison and
one select the module should not contain, so the test counts them: every subgroup's slot is read
for the total, and one fewer select is emitted than there are subgroups.

**6. A scan across more than one workgroup — done.** `Gpu::scanner` holds a buffer and a pipeline
per level and runs the whole thing in one submission: `2 × levels + 1` dispatches, up one side and
down the other. 2²⁰ elements is three levels and seven dispatches, and it matches the CPU element
for element on all five widths.

`scan/plan.rs` decides the levels and is inside the gate; `scan/held.rs` owns the Vulkan objects and
is not — the fourth time that seam has been worth cutting. The two derivations of the dispatch
count are made to agree at build time rather than trusted: the plan says `2 × levels + 1` and the
recording loop emits them one at a time, and a disagreement would otherwise show up as a wrong
answer at some depth instead of a failure where the mistake is.

Item 7 turned out to be a prerequisite rather than a nicety: see below.

**7. The exclusive scan — done, and it was load-bearing.** The doc used to say a caller who wants
it can subtract their own element. That is true of integers and **false of floats**: subtracting a
large running total back off itself loses precisely the low bits the scan just accumulated. A long
scan needs the exclusive form for its block offsets, so this stopped being a nicety.

`GroupOperation::ExclusiveScan` had been sitting in `spec::group` since the beginning with nothing
ever emitting one. `Lanes::prefix_sum_exclusive` asks for it, and the two scans now share a builder
and differ in one literal.

**8. A strip-mined and clustered scan — the strips are built; the clusters are SPIR-V's gap.**
`Lanes::prefix_sum` said *"a strip-mined scan must carry a running total between strips, which is
not built"*. It is now: one scan per strip and one `Reduce` for every strip but the last, carried
forward. The vector order makes it work — lane `l` holds `l`, `l + width`, `l + 2·width`, so a
strip is a consecutive run of the vector and every element of strip `s - 1` comes before every
element of strip `s`. Verified at 2, 4 and 8 strips on three devices.

The carry is a `Reduce` and not the last lane of the scan, which matters only in the exclusive form
— an exclusive scan hands no lane the strip's whole total, so reading the carry off it would be
short by exactly one lane's element.

**And the clustered scan is built too, as a kernel.** The sentence that used to be here said it
needed a shuffle from a lane that differs per lane. That was wrong, and finding out how wrong is
the useful part:

* It needs **no dynamic shuffle**. A Hillis-Steele ladder over subgroup lanes does it —
  `log2(cluster)` steps of shift, compare and select — and the mask is what keeps each cluster's
  scan inside itself.
* It needs **no subtraction**, which is the cheap alternative and the wrong one: taking a large
  running total back off itself loses precisely the low bits the scan just accumulated.
* It does need `OpBitwiseAnd`, which was not in `module::op`. Read out of Khronos' assembler rather
  than a table — `spirv-as` was given a module containing one and the emitted word carried 199.
  DR-0001 says the number comes from the authority; the authority answers questions as well as
  publishing them, and the grammar JSON is not installed on this machine while the tool that
  consumes it is.

`kernels::scan::scan_clusters` runs it, correct at clusters of 2, 4 and 8 on every width wide
enough to hold them.

**What is left is where it lives, not whether it works.** `Lanes::prefix_sum` still refuses a
clustered vector, because the mask needs the invocation's position inside its cluster and `Lanes`
is handed a module and a width — not an invocation. Moving it there means one of:

* threading a lane index into `Lanes::new`, which `Kernel::lanes` could supply and a direct caller
  could not;
* or declaring `SubgroupLocalInvocationId` in `kernel::binding`, which costs every kernel an
  `Input` variable and — the real objection — the `GroupNonUniform` capability, which a kernel that
  only scales currently does not declare and a test asserts it does not.

Neither is hard and neither is obviously right, so it is written down rather than guessed at.

### Tier 3 — plumbing, and one measurement

**9. `run_bound` pays a submission per input — measured, and not worth fixing.** The measurement
was the first thing to do and it settled the item. A second binding costs about **330 µs** on an
RTX 4080, and that figure is *flat* across an eight-fold change in data size — so it is fixed setup
rather than transfer, and a submission at 50–80 µs is a fifth of it. The rest is one more buffer
allocated and one more descriptor in the set.

Recording the uploads in one command buffer would recover the fifth and leave the rest. A caller who
minds has a better answer already: `Session` allocates once, and since `Buffer::shared` its writes
land straight in the binding, so a held session pays no upload submission at all. Making `run_bound`
allocate shared buffers is the other half of that and is already refused — per-call BAR allocation
cost `Gpu::sum` 62%.

`runner/examples/bindings.rs` prints the table; `notes/FINDINGS.md` has the argument. Third item on
this list refuted by its own measurement.

**10. The breakdown reads 123% of the call — done, and the guilty row was out by five times.**
`Reducer::sum_timed` writes a timestamp into the chain's own command buffer after every dispatch,
so each pass is measured beside the passes it runs beside. The probe said the chained steps cost
~70 µs of a 296 µs call; in place they are **~12 µs**. A chain of empty kernels gives a barrier
nothing to overlap with, which is the direction that probe was always going to err in.

It also found something neither probe nor arithmetic could: **the profile's shape belongs to the
device.** The integrated Radeon is bandwidth-bound and falls away 92% → 5% → 1% across the passes;
the 4080 is flat at ~2 µs a pass, because at its bandwidth the tail is too small to cost anything
but the dispatch. Same chain, opposite answers. `notes/FINDINGS.md` has both columns.

**11. A buffer the caller already owns.** Unchanged and still deferred; the argument is above under
its own heading.

### Tier 4 — reach

**12. A third vendor.** Two vendors and a CPU implementation is enough to have caught real bugs —
ten tests at width 64, undefined behaviour at 4 and 8. A third driver is where "portable" stops
being a claim resting on two data points. Intel integrated is the cheap one.

**13. Name the neighbours in the README — done.** "What this is not" now points at VectorWare
beside rust-gpu, with what actually differs: they compile the `Simd` you already wrote, so one
source targets three architectures; this is a builder that needs no compiler and works on stable.
And with what they say about a smaller `N`, which is the case `ClusteredReduce` exists for.

---

## The list, rewritten 2026-08-14, after the first one ran out

Eleven of the thirteen above are done and the other two are deferred on purpose or need hardware
that is not here. This is what a second survey turned up, and its shape is different: the first list
was mostly *things that were false*, and this one is mostly **holes the last week's work opened**.
Building the scan three times over left the layers around it out of step.

### Tier 1 — the code contradicts itself

**1. Two doc comments say things the code beside them disproves — done.** Both corrected, and the
correction says what was wrong rather than quietly replacing it: the exclusive scan is built, and
the subtraction the old comment recommended is the thing it exists to avoid.

A third was found while fixing them. `fuzz::mod`'s header said **"`f16` is not fuzzed"** and had
said so since before `Domain::Half` was added to the sweep — the fuzzer runs it 256 times a domain
and refuses the two rounds that leave a half's exact range. That claim was older and more wrong
than either of the two on the list.

**~~1.~~ The original wording, for the record:** `kernels::scan::scan_workgroup`
still reads *"the exclusive form is this shifted by one and is not built — a caller who wants it can
subtract its own element"*, and `scan_workgroup_at` still reads *"a strip-mined scan would have to
carry a running total between strips, which is not built"*. Both were built this week. Worse, the
subtraction that first comment recommends is the exact thing `prefix_sum_exclusive` exists to avoid:
over floats it takes a large running total back off itself and loses the low bits the scan just
accumulated.

The cheapest item here and the one that misleads a reader fastest, because it sits directly above
the function that refutes it.

**2. The fuzzer has never generated a scan — done.** `Finish::Scan` and `Finish::ScanExclusive`
are generated, built, and modelled by the reference. 25 of every 256 rounds end in a scan in each
of the eight domains, and 72 strip-mined scans agree across the forced-wide sweep.

The reference had to grow something the reductions never needed: it models the **lane order**.
Element `j` of a prefix depends on exactly which elements the hardware puts before `j`, so
`interpret::scanned` reproduces the addressing — lane `l` holding `l`, `l + width`, `l + 2·width`
— rather than only the arithmetic.

**It has teeth, and that was checked rather than assumed.** Two deliberate breakages were tried:
writing the inclusive answer where the exclusive one belongs, and reading the vector blocked rather
than strided. Both were caught at seed 1. `the_fuzzer_notices_when_a_scan_is_wrong` keeps a
scan-specific version of that permanently, because the existing teeth test stops at the first
sensitive program and may never reach a scan.

The gate found three survivors, one of them a second copy of an unfalsifiable branch that
`interpret::strips_of` had already been fixed for — `if lanes > subgroup` returns the same answer as
the division at equal widths, so nothing could tell the arms apart.

**~~2.~~ The original wording, for the record:** `fuzz::program::Finish` offers `Sum`, `Max`, `Min` and
`SumOrMax`. Nothing prefixes.

That matters more than a missing case usually would. The scan is now the most intricate thing in
the tree — three mappings, two directions, a carry between strips, a mask between clusters — and
**every test of it is hand-written**. That is precisely the state the reduction was in when the
differential fuzzer found `reduce_min` folding its strips with a *maximum*: right for every mapping
but the strip-mined one, so no hand-written test had ever looked. A scan has three mappings to get
wrong instead of one, and two directions in each.

The CPU reference has to grow a prefix, which is where the work is: a scan's expected answer depends
on the lane order, so the reference has to model the mapping rather than the arithmetic.

### Tier 2 — asymmetries the last week opened

**3. `Scanner` is missing what `Reducer` has — half done.** `Gpu::scanner_of` fuses an elementwise
map into the chain's first pass, and it is worth **2.0–3.0×** depending on the length and the
device: two crossings of the buffer removed, against a scan that grows faster than they do.
`runner/examples/scanner.rs` prints the table and `notes/FINDINGS.md` records it.

`scan_timed` is built too, and it did say something the reduction's could not: **the depth is
nearly free and the two ends are the whole cost.** The five middle passes of a 2²⁰ scan come to
about 10 µs against 21 for the first and last on an RTX 4080, because everything between them works
on block totals rather than the buffer. A longer input costs two more dispatches and almost no more
device time.

And the seven dispatches together are 3% of the call. `notes/FINDINGS.md` has both tables.

**4. There is no one-shot `Gpu::scan` — done.** It builds a `Scanner`, uses it once and drops it,
which is ceremony traded for economy and the doc says so: a dozen buffers and `2 × levels + 1`
pipelines rebuilt per call, all of which `Gpu::scanner` keeps. It exists for symmetry with
`Gpu::sum` and for the cases where building an object to use it once is the wrong shape.

**5. `Lanes::prefix_sum` still refuses a clustered vector.** The ladder works and is proven on three
devices, and it lives in `runner::kernels` — so the *lane API*, which is what the README's mapping
table describes, cannot reach it. Two ways in are written down above and neither is obviously right;
that is the decision to make rather than the code to write.

### Tier 3 — hygiene

**6. `kernels::scan::mod` is 672 lines again.** It was split at 639 four days ago and the clustered
ladder put it straight back. There are three concerns in it now, not two: the workgroup scan and the
arithmetic everything shares, the block composition (already next door in `blocks.rs`), and the
clustered ladder — which shares nothing with either and is the one that grew.

### Tier 4 — carried over, unchanged

**7. A buffer the caller already owns.** Still no caller in this repository wants it.

**8. A third vendor.** Still needs hardware that is not in this machine. Intel integrated is the
cheap one, and CI on a shared runner has already shown what a fourth implementation is worth — it
found a signed zero two GPUs and a local lavapipe all agreed on.

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

The scan's block limit and `dispatch::extent`'s blind spot were both here and are now items 6 and 3
above, with the work they need spelled out. What is left is the one thing that is neither a gap nor
a plan:

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
