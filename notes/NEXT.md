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

One instruction against the nineteen it replaces — four shifts up, four bitcasts, four shifts down,
four multiplies and three adds. `runner/examples/dot.rs`:

| kernel | RTX 4080 | integrated Radeon |
| --- | --- | --- |
| one dot product per element, 262 144 invocations | 1.00× | 1.52× |
| thirty-two per element, 262 144 invocations | 1.18× | **9.08×** |

The first row is memory-bound and the second is not, which is why both are there. The discrete part
has enough integer throughput that nineteen instructions cost nearly what one does; the integrated
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

> **All six are done as of the same day**, and the last of them — item 5, the clustered scan —
> was written here as a decision rather than as work. It was: the two options offered were both
> wrong, and finding the third took less time than either would have. What is left below is items
> 7 and 8, which are the two that were carried over deliberately.
>
> The thing worth carrying forward is not any of the six. It is that finishing item 5 let the
> *fuzzer* reach the clustered scan for the first time, and the first sweep found a driver that
> faults compiling a valid module. Two of the four sittings this file has recorded ended the same
> way: the coverage that mattered came from removing a refusal, not from adding a test.

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

**5. `Lanes::prefix_sum` refused a clustered vector — done, by a third way neither option saw.**
The two written down were "thread a lane index into `Lanes::new`" and "declare
`SubgroupLocalInvocationId` in `kernel::binding`". The objection to the second was entirely about
*when*: it costs every kernel an `Input` variable and the `GroupNonUniform` capability. Declared
**on demand**, by `Lanes` itself, it costs nothing to a kernel that does not scan a clustered
vector — and a test still asserts that a kernel which only scales declares no subgroup capability.

That also settles which number it is. A kernel knows its index within the *workgroup*, and on all
three implementations here `local & (width - 1)` is the same number — because subgroups happen to
be cut from consecutive local invocations, which Vulkan promises for a pipeline that asked for full
subgroups and not otherwise. The first option would have carried that coincidence into the layer
whose value is that it is defined. `decisions/DR-0007` writes it up.

The price is one mechanism underneath: `OpEntryPoint` lists the `Input` variables the entry point
reaches, and a built-in the *body* asks for arrives long after that instruction was emitted, so
`Module` holds the entry point and its interface as data and renders the instruction whenever
either grows.

**The failure mode is an invalid module that every driver runs anyway**, so the check that carries
this is `spirv-val` and nothing else: deleting the line that adds the variable to the interface
leaves 19 of `tests/kernels.rs`'s 20 modules rejected and all three devices still returning right
answers. That was run rather than assumed.

The kernel that held the second copy of the ladder is now a load, a scan and a store — and the
fuzzer generates clustered scans, which it never could while they were refused. Two deliberate
breakages were caught at seeds 0 and 1. **It also found a driver that faults compiling a valid
module**; `notes/FINDINGS.md` has the bisection and what the suite does about it.

### Tier 3 — hygiene

**~~6.~~ `kernels::scan::mod` is 672 lines again — done, and then done again.** Split when the
clustered ladder was carved out into `clusters.rs`; that file then lost the ladder itself to
`Lanes::prefix_sum` under item 5, and what is left in it is the kernel that runs the mapping.

**6. The original wording, for the record:** It was split at 639 four days ago and the clustered
ladder put it straight back. There are three concerns in it now, not two: the workgroup scan and the
arithmetic everything shares, the block composition (already next door in `blocks.rs`), and the
clustered ladder — which shares nothing with either and is the one that grew.

### Tier 4 — carried over, unchanged

**7. A buffer the caller already owns.** Still no caller in this repository wants it.

**8. A third vendor.** Still needs hardware that is not in this machine. Intel integrated is the
cheap one, and CI on a shared runner has already shown what a fourth implementation is worth — it
found a signed zero two GPUs and a local lavapipe all agreed on.

---

## The list, rewritten 2026-08-14, after the second one ran out

The second list finished the same day it was written, so this is a third survey of the tree. Its
shape is different again. The first list was mostly *things that were false*; the second was
**holes the week's work opened**; this one is mostly **refusals that are stronger than the hardware
they are protecting** — and the reason it has that shape is item 5 of the second list. Removing one
refusal turned out to be worth more than any test written that week, so the survey went looking for
the others.

### Tier 1 — things that were already false, and are fixed

**1. `kernels::scan::mod` said the multi-block scan "is not built" — done.** It sat in a section
headed *What it does not do*, four lines below the sentence naming `blocks.rs` as the file that
does it, and two days after `Gpu::scanner` started running the whole thing in one submission. The
heading is *What a kernel here does not do* now, and the limit it states is the kernel's rather
than the crate's.

**2. The README's test counts were 333 and 706 — done.** They are 348 and 740. Also in that table:
`spirv-val`'s row now records what it is the only layer that can see, which is an entry point whose
interface omits a built-in the body loads.

### Tier 2 — refusals stronger than the hardware

**3. `Lanes::butterfly` refused a clustered vector, and for a small mask it need not — done.** A
butterfly pairs lane `l` with `l ^ mask`. Clusters are aligned runs of `LANES` lanes and `LANES` is
a power of two, so a `mask < LANES` flips only bits below the cluster's own width and **cannot
leave it**. Nothing to mask off, no lane index needed, no undefined lane anywhere: the change is a
bound, and a `mask` at or above `LANES` is refused by name rather than clamped.

The refusal's stated reason — "a shuffle reads a lane of the subgroup, and a narrower vector shares
those lanes with other vectors" — was true of the *shifts* and false of the butterfly, and it made
the one shuffle a clustered vector can have unreachable.

`kernels::butterfly_cluster_sum` is what it buys, and it is the kernel this could not have: four
independent trees of `log2(cluster)` steps, checked on a device against the single `ClusteredReduce`
that computes the same thing. Two implementations, neither of them the reference — which is what
`kernels::reduce`'s own header says it is for.

**And the bound caught the mistake before the comparison did.** The teeth check for that test was to
fold `log2(width)` times instead of `log2(cluster)` — the plausible wrong kernel, which returns the
subgroup's total in every lane. It does not return anything: the first mask that reaches outside the
cluster is refused at build time, by name.

**4. `Lanes::broadcast` refused one too — done.** Broadcasting position `source` of a clustered
vector means reading subgroup lane `(l & !(LANES - 1)) + source`, which differs per lane — and
`OpGroupNonUniformShuffle` takes a **dynamic** id, so it is one `OpBitwiseAnd`, one `OpIAdd` and the
instruction that was already being emitted. It needs the invocation's own lane, which `Lanes` has
had since `decisions/DR-0007` and did not when this was refused.

`OpGroupNonUniformBroadcast` is the one that would not do: its id must be *dynamically uniform*,
which is exactly what this is not. That the lane API had always emitted a shuffle instead turned out
to be what made this three lines.

`kernels::broadcast_in_cluster` runs it: four vectors, each reading its own position 3 — subgroup
lanes 3, 11, 19 and 27. The wrong implementation reads `source` as a subgroup lane, and agrees with
this one for the first cluster of every subgroup.

**5. What a cluster's edge means — answered, and the answer was a third operation.**
`shift_up`/`shift_down` really do read a neighbouring vector's lanes. The two obvious answers were
both bad: call the edge *undefined*, which promises less than the hardware does and leaves a caller
holding a value it cannot use; or mask it to something, which invents a semantics SPIR-V does not
have and pays for it in every call. The project has refused that second trade twice already — see
the `FMax` note under item 4 of the first list.

The third answer is that the operation a caller wants at an edge is the one that has none.
**`Lanes::rotate_up` wraps inside the vector**: every lane reads a lane of its own vector, so there
is nothing undefined and nothing to mask, and it is allowed for a clustered vector and a
subgroup-wide one from the same four instructions — `(l & !(size - 1)) | ((l + size - delta) & (size
- 1))`, where a subgroup-wide vector's first half is zero and it collapses to the wrap alone.

`OpBitwiseOr` is **197**, below the and: the bitwise instructions run *downwards* in the grammar, so
the number beside `BITWISE_AND` is not the one it needs. Out of `spirv-as`, per DR-0001 — the second
time today that rule earned its keep.

The shifts stay refused for a clustered vector, and that is now a decision with a reason rather than
a gap. A rotate by a multiple of the width emits nothing, and a strip-mined rotate is refused by
name: it moves elements *between* strips, which is a shuffle per strip plus a rotation of the strips
themselves — a different algorithm, not a different operand.

### Tier 3 — verification debt

**6. The mutation gate had not run over the clustered scan — it has now.** 1 138 changed lines
scoped, **33 of 35 viable mutants killed, 94.3%**, and the two survivors were the same mutation in
two places:

```
src/lanes/mod.rs:242     [false→true]  let uint = self.module().type_int(32, false)?;
src/lanes/reduce.rs:214  [false→true]  let uint = self.module().type_int(32, false)?;
```

**The identical survivor this project has already recorded once.** `Kernel::index_type` carries the
note: flipping that `false` changes the module and nothing observable, because SPIR-V's signedness
is not what decides how `OpIAdd` — or here `OpBitwiseAnd` and `OpUGreaterThan` — behave. The fix
then was "not a test for it but not writing it down twice", and it is the same fix now: both sites
say `self.type_of::<U32>()`, which is the lane API's own `u32`, declared once in `element.rs` where
the signedness *is* load-bearing and mutants there do die.

A third copy in `broadcast_within_cluster` went the same way, before the gate could be pointed at
it.

**And the review pass beside it found four things the gate could not.** A mutation survives or dies;
it cannot say that a module contains instructions computing an answer that was already known:

- A **one-lane cluster** — `Simd<T, 1>`, which the mapping accepts — went through the whole ladder:
  a built-in, a load, a mask, a shuffle and a select, to compute the element itself. Both scans and
  the clustered broadcast answer it before emitting anything now, which is the rule the workgroup
  scan already followed for its final subgroup's offset.
- The clustered broadcast's **device test ran at one width**, because it sat behind a helper that
  refuses anything but 32 lanes. Its expectation has no width in it, and the two devices that found
  this project's last ten bugs were skipping it.
- Two **hand-emitted `OpEntryPoint`s** were left in `examples/emit_minimal.rs` and
  `tests/validated.rs` — the pattern that is now a trap, since a module built that way plus any lane
  operation that declares a built-in is invalid with no diagnostic. Both go through
  `Module::entry_point`; the one that remains is the unit test asserting an exact word stream, which
  is hand-built on purpose.
- `broadcast_in_cluster` covered clusters 2 to 16, leaving out the one-lane case and the 32-lane one
  that only a 64-wide device has.

**The day after, over 968 lines of new work: 27 of 32, five survivors, then 32 of 32.** None of the
five was wrong code — two were claims nothing checked, and the third was an operation that could not
pass on the fuzzer's own corpus. `notes/FINDINGS.md` has that one; it is the sharpest thing the gate
has said here.

**And the gate was run again over the fixes themselves** — 86 changed lines, **8 of 8 killed, no
survivors**. That is the part which makes the item closed rather than moved: the first run said the
clustered scan was 94.3% covered and named what was missing, and the second says the code written to
answer it is covered too. A fix that arrives with no mutant of its own is a fix nobody has checked.

### Tier 4 — carried over, unchanged

**7. A buffer the caller already owns.** Still no caller in this repository wants it.

**8. A third vendor.** Still needs hardware that is not in this machine — and the integrated
Radeon's compiler fault is a second argument for it, because with three implementations a fault on
one is a defect and with two it is an argument.

---

## The public surface, audited — 2026-08-14

Item 17 asked which public operations no `spirv-val` test reached, and the first thing it found was
invalid SPIR-V. The surface has grown by a dozen items since, so the question was asked again over
all **201** public functions, in its strongest form: *which have no consumer at all* — no caller, no
unit test, no validator.

**Four**, and `notes/FINDINGS.md` has the table. One was deleted as a second spelling of an
instruction the typed path already emits; three were finished, each with a `Kernel` or `Lanes`
entry point, a kernel, validator coverage at every width, and a device test. All four turned out to
be *valid*, which is worth stating: the check's value is the difference between "nothing has looked"
and "something looked and it was right", and only one of those is a claim.

The audit also found `Limits` reporting `subgroup_ballot` and no `subgroup_vote` while three kernels
used votes — right on every device here because no implementation offers one without the other.

**Pulling that thread found a second missing bit and a wrong tool.** Laying all seven capabilities
this emitter can declare beside the feature bits the runner reports: `BASIC` was missing, which
every lane kernel declares, and `SHUFFLE_RELATIVE` was missing, which the whole scan rests on. The
gates named the arbitrary shuffle instead. And `simdr probe` — the command that exists so nobody has
to guess — listed `any, all` under *ballot* and `shift_up, shift_down` under *shuffle*.

The fix is a mapping rather than four more gates: `Limits::supports(Capability)` writes the
correspondence down once and `Limits::unsupported_in(&spirv)` reads the requirement out of the
module's own `OpCapability` instructions, so a kernel that needs something new brings its own gate.
`notes/FINDINGS.md` has the table.

### What it leaves open

**9. `Lanes` had no elementwise equality — done, and the strip-mined vote came with it.**
`greater_than` was the only comparison, and `simd_eq` is the one a `Simd` API is asked for first.

`Element::EQUAL` is `OpFOrdEqual` for the floats and `OpIEqual` for **every** integer, signed or
not — the one place in that trait where the two integer families share an instruction, because two
bit patterns are equal or they are not and no reading of the sign bit changes it. A test says so
against `greater_than`, which is three instructions across the same three types.

`OpFOrdEqual` is **180**, and the comparisons are not consecutive in the grammar: `OpIEqual` is 170
and `OpFOrdGreaterThan` is 186. The number came out of `spirv-as`, the way DR-0001 says to, and a
number remembered from the neighbourhood would have assembled into something else.

**And the strip-mined `all_equal` is built on it.** It is two questions rather than one folded vote:
every lane holds the same strip 0 (one `AllEqual`), *and* in every lane the other strips equal strip
0 (`strips - 1` comparisons folded with `and`, then one `All`). Neither says it alone.

The device test is the one the old refusal existed for: two strips, each internally uniform, holding
different values. A folded vote answers 1 and the answer is 0. Checked by breaking it — dropping the
second question makes exactly that case fail, and nothing else.

**10. Twelve more functions are used only inside the emitter**, their own tests included. That is a
weaker finding than the four above and not nothing: a unit test written beside the function it
tests agrees with the author, which is how `reduce_min` folded its strips with a maximum for weeks.
The list is in the audit script's second section; `Module::f_add` and `Module::f_mul` are the two
whose only callers are tests of the *block* machinery rather than of themselves.

---

## Deliberately not doing

**Putting a latency-bound caller on this device.** `decisions/DR-0008` is the boundary and
`runner/examples/latency.rs` is the check that re-tests it on any machine. One answer from a held
`Session` costs ~100 µs on an RTX 4080 and ~779 µs on the integrated Radeon, of which the device
itself is **2.9%** and **0.3%** — the rest is submission, fence and copy-back, and nothing in this
repository can move it. A caller needs `R / (c - d)` *independent* answers pending before the device
is worth asking, and **never**, at any batch size, when its CPU cost per answer is at or below this
device's.

The case that settled it is written up in full there: a chess engine next door, whose NNUE layer
`kernels::network::clipped_dot` was modelled on. Break-even ~9 700 evaluations at once against the
one that alpha–beta pruning supplies — and two ceilings above that arithmetic which the engine had
already measured for itself. The answer is *stay on the CPU*, and it is a decision rather than a
note because the pressure to re-open it is predictable ("the kernel is only a few microseconds" — it
is 2.9% of what the caller waits).

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

**Working around the AMD driver's clustered-scan fault.** It dies inside
`vkCreateComputePipelines` on a module `spirv-val` accepts and two other implementations run
correctly, so there is nothing here to fix — and shaping the emitter around one vendor's crash
would mean carrying a workaround with no test that could ever say it was still needed. The suite
probes for it, replaces those rounds with a reduction, and prints the count.

**A clustered scan by subtraction, as the cheap path for that device.** It would dodge the fault
and it is the trade this project has refused twice: over floats it takes a large running total back
off itself and loses precisely the low bits the scan just accumulated. An exact answer on two
devices beats an approximate one on three.

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

---

## The list, rewritten 2026-08-15, after asking a different question

The three lists above all asked *what is missing*. This one asked, of every check already in the
tree, **where is it called from** — and four of them turned out to guard less than they are about.
That is a different failure from a gap, and it is invisible in exactly the way a gap is not: a check
that covers one caller passes its own tests, reports green, and reads in a table like a check that
covers all of them.

`notes/FINDINGS.md` has the four in full. What follows is what they leave.

### Tier 1 — done, and each was a check that had stopped covering its subject

**1. `dispatch::extent` guarded one of six dispatch entry points — done.** `Gpu::run` was checked;
`run_bound`, `Session::dispatch`, `run_chain`, `Gpu::reducer` and `Gpu::scanner` were not. It is
per **binding** now rather than per module, which is what the differently-sized buffers of
`run_bound` and `Session` need, and `runner/tests/bounds.rs` asks each of the six doors the same
question.

**2. A shuffle's operand was bounded for the clustered mapping and for neither of the others —
done.** `butterfly(value, 4096)` on a 32-wide subgroup built a module `spirv-val` accepts in which
every lane reads a lane that does not exist. One bound for every mapping now, and 32 000 fuzzing
rounds refuse none of the programs the generator makes.

**3. `Kernel::new` checked three of `Shape`'s four numbers — done.** A subgroup width of zero, or
24, built a kernel and finished a valid module. The width is what `decisions/DR-0002` makes the
whole module specific to.

**4. The address arithmetic saturated — done.** `strip × workgroup + offset` turned an index nobody
can express into one that exists. Refused by name, in `u64`, with the number that did not fit.

**5. The documentation build had never passed — done.** `README.md` listed it among the checks;
nothing ran it; twelve intra-doc links were dead, one of them a public function whose return type
could not be named. It is a CI step now.

**6. The scan's pass wiring was excused from the mutation gate for being near `unsafe` — done.**
`runner/src/scan/passes.rs`, and four properties of the wiring that no device test could state.

### Tier 2 — what the same question leaves open

**7. Nothing asked the "where is it called from" question automatically — it does now.** This item
said to wait for a third instance, because two is a coincidence and the check would be shaped around
them. The third arrived the same day: `Module::memory_barrier`, emitting an `OpMemoryBarrier` whose
semantics Vulkan forbids, with no caller and no validator behind it.

`tests/integrity.rs` asks it on every run: every `pub fn` in `src/` must be named by something
outside the file that declares it. Five are excused with a reason each — `require_extension`, which
`require_capability` reaches in the same file, and the four `subgroup_*` wrappers that are readable
spellings of what `subgroup_reduce` emits — and each excuse expires by itself, because a second test
fails if an excused operation gains a caller.

It is a **floor rather than a proof**, in the direction that costs coverage rather than truth: two
files may declare the same method name — `word` is a `pub const fn` on eight `spec` enums — and a
reference to either counts for both. Sharpening that means resolving names to modules, which is a
parser rather than a grep, and nothing yet needs it.

Both directions were checked by breaking them. The first version of the check *missed* a throwaway
`pub fn` appended to `module/mod.rs`, because it stopped reading at the first `#[cfg(test)]` and the
probe was after it — which is the check having exactly the blind spot it exists to find, and the
reason it now reads whole files.

**The other half of the original item is answered too, and the family was already in the tree.**
This asks whether an operation is reached at all, not whether it reaches *the validation its
siblings reach* — `Gpu`'s dispatch family had six members and one bound check, and nothing here
would have said so because all six were consumed.

Every dispatch in this crate goes through `Pipeline::new`, and that call is handed both halves of
the bound at once: the module, and how much of each buffer the shader may see. So the family is
"files that build a pipeline", and what its members owe is a mention of `overrun`. Seven build one;
six bound it; the seventh is `probe_pipelines`, which times pipeline creation and submits nothing,
excused by name with an excuse that expires by itself.

It is a floor rather than a proof — it asks whether the file names the check, not whether every path
through it reaches one, and `runner/tests/bounds.rs` asks the sharper question of each door. What it
adds is the thing that had no mechanism: noticing a **seventh door**.

**8. `Kernel::load_offset`'s offset was outside the dispatch bound — done, and something did need
it.** The item said the under-count cost nothing because the fold kernels size their dispatch
narrower than their buffer. `kernels::network::clipped_dot` does not: activations occupy the first
`offset` elements of binding 0 and the weights the rest, so it reads *exactly twice its run*. Handed
a buffer of only the run it dispatched and returned 512 words of zeros on an RTX 4080 — read from
past the end of a storage buffer, through `Gpu::run`, from safe code.

It was one more step of the same walk, as written, and it needed nothing declared for the same
reason the strip count did not: `Kernel::address` folds `strip × workgroup + offset` into one
constant, so `shift - (strips - 1) × workgroup` is the caller's offset and the strip term is a
number `addressing.rs` already recovers. Checked by breaking it — with the offset multiplied by
zero the device test fails by *succeeding*, which is the whole finding in one line of output.

**9. A grid kernel's `row × pitch` was outside it too — done, and it was the larger half.** Rows are
`pitch` elements apart whether or not the dispatch covers a row, so a kernel reading a narrow slab of
a wide matrix reaches its last row `(rows - 1) × pitch` elements in while the invocation product
counts only the columns dispatched. Not off by a constant — they diverge with the pitch: 800
elements measured as 128 at four rows of 256, and **258 080 measured as 2 048** at the 64-of-4096
shape `plane.rs`'s own header describes and calls supported.

Every grid test in this crate dispatches `pitch / width` workgroups across, which covers a whole row
— and there `(rows - 1) × pitch + columns` *is* `rows × columns`, exactly. The two readings agree on
every test that exists and diverge without bound off them.

Two more in the same walk while it was open. `LocalSize` was read as a product where the addressing
wants x alone, so a grid two rows deep with two strips would have recovered a strip count of
`2 / 2 = 1`; nothing has both today. And the row was matched by its *shape*, which
`(group.y × pitch) + run` shares — the first version found that instead, reported no pitch, and
looked exactly like working.

~~What is left outside is `Kernel::load_offset_by`, whose offset is a specialization constant chosen
after the module was built with no literal in it to read.~~ **Closed 2026-08-16, and it was the one
fail-open site in the file.** Every other thing this walk cannot read makes the check *weaker*; that
one made it **permissive**, because a term counted as zero is a reach the bound allows. The number
was not unknowable, only written somewhere else: a pipeline is created *with* its specialization, so
`Bounds::of` takes one. `notes/FINDINGS.md` has the account.

**And the gate was pointed at all of it, five times.** 63% → 85.2% → 86.7% → 90% → **100%, 30 of
30**. The first run's survivors clustered in the four conditions that identify a grid's row, because
a workgroup one row deep never builds the sum they match — so that branch had no test at all, in
either half of the walk written that day.

Three survivors were then argued to be *equivalent mutants*, each with a sound argument about every
module this emitter produces. All three were wrong, and for the same reason: this file decodes
SPIR-V rather than only the SPIR-V it wrote. Four edits to a real kernel — a sum duplicated, a sum
over the row's base that is not a row, a constant spliced in off the lane, and the workgroup size
removed — killed every one of them. `runner/src/dispatch/extent/addressing.rs` has the table.

### Tier 3 — carried over, unchanged

**10. A buffer the caller already owns.** Still no caller in this repository wants it.

**11. A third vendor.** Still needs hardware that is not in this machine.

## What a check is for, restated

Three of the four in Tier 1 were **refusals**, and the fourth was a saturation standing in for one.
The project already had the rule — *refused by name rather than clamped* — and applied it every time
somebody wrote a new operation. What it had not had was a way to notice when an old refusal stopped
reaching a new caller.

The habit that found them, which costs nothing and is not automatable yet:

**Ask where a check is called from, not whether it works.**

---

## Where the work goes now — 2026-08-15, after DR-0008

The four lists above asked *what is missing*, *what holes did the week open*, *which refusals are
stronger than the hardware*, and *where is each check called from*. This one asks a question none of
them did: **which workloads is this repository actually the right answer for**, now that the cost of
being asked has been measured properly.

`decisions/DR-0008` answers it with one number. A round trip is fixed at ~100 µs on the discrete
device and ~779 µs on the integrated one, and the device's own share of that is **2.9%** and
**0.3%**. So:

> **Batch size beats kernel quality by two orders of magnitude.** Raising the answers per round trip
> from 2 to 2 048 is 800×. No kernel change recorded anywhere in this file has been worth more
> than 3×.

That reorders everything. What follows is the tree read against it.

### Tier 1 — already the right shape, and worth deepening

**1. The differential fuzzer is the most valuable thing here, and it is not a speed workload.**
30 000 generated programs, each answered by a device and by a CPU reference. It found `reduce_min`
folding its strips with a maximum, an `OpUDot` invalid since the day it shipped, and a driver that
faults compiling a valid module. Every one of those is a correctness result that no benchmark would
have produced, and the workload is throughput-shaped by construction — thousands of independent
questions, no caller waiting on any one of them.

It is also where the mutation gate keeps landing: *"nine of the twelve were in the fuzzer or its CPU
reference; none was in the emitter."* Widening the vocabulary is the single highest-value direction
in the tree.

**Three passes at it, and each found what the one before could not see.** `notes/FINDINGS.md` has
them in full:

* **What it could not generate.** `Lanes::broadcast` was reached by no generated program at all —
  the cross-lane operation whose answer is fully determined and whose mapping does the most work —
  and `shift_down` had no `Op` while `shift_up` did. A wrong broadcast mapping is caught at seed 5.
* **The gate over the whole module**, scoped by path rather than by diff for the first time: 170 of
  171, and the one survivor was a comparison written twice. Merging the two spellings made *existing*
  coverage reach it — the duplication was the survivability. 172 of 172 after.
* **The generator's modules were never validated.** Thousands a run, straight to a driver, in the
  component that produces the shapes nobody designed. 232 now go through `spirv-val` at every width,
  no device required.

What is left here is the rest of the surface: the bit shifts, `abs`, the conversions and the integer
dot products are all exactly checkable in the integer domains and none of them is generated.

**And the reason is sharper than "the generator has no domain-aware pools".** Preparing that pass
found that `Lanes::shift_left` took `T: Element`, `F32` is an `Element`, and a shift of a vector of
floats built a module `spirv-val` rejects — `OpUDot`'s shape exactly, in three public operations.
It is `T: Integer` now, so the call cannot be written, with a `compile_fail` doctest as the only
artefact that can assert what a program *cannot be*.

Which was also the blocker: `Program::build` emits through one function generic over `Element`, and
a shift needs `Integer`. Generating one means emitting **per domain** rather than filtering a pool.

**Built on 2026-08-15, and the obvious reading of that sentence was the wrong one.** "Per domain"
looks like a second copy of the const-generic width ladder, one for the six integer types and one
for the two floats — a relationship decided twice, which is the shape `notes/FINDINGS.md` catalogues
more often than any other. The ladder stays single and the *element type* carries what it can do:
`Emit` is one method, implemented by macro for the integers and by a refusal for `F32` and `F16`.

`Op::BitShift` reaches all three shifts. The distance is drawn across the element's whole width,
because the two right shifts agree on every value whose top bit is clear and every value this
generator draws has one — so a small distance would have generated two instructions and proved one.
`ProgramError` names the second kind of no that arrived with it.

What is left of this item: `abs` — gated by `Signed` rather than by `Integer`, so it is a second
group on the same mechanism and now cheap — the conversions, which change the element type mid-
program and so do not fit the single-accumulator shape without thought, and the integer dot
products, whose input is four bytes in a word and which the sandbox covered at all three mappings
before it was deleted — see `notes/FINDINGS.md`, which is where that coverage now lives as a record
rather than as a runnable check.

**2. The reduction and scan chains are the shape DR-0008 rules in** — one question over a whole
buffer, one submission. `Reducer::sum` at 11.4× over 8 192 elements and 9.2× over 2²⁰, a map fused into a scan at
2.1–3.1×, and a held `Session` at 17× per dispatch against rebuilding on the RTX 4080 and 6× on the
integrated part — all re-measured 2026-08-26. The 52× this line used to give was a `Session` figure
attributed to `Reducer`, and it was three times what the same test reports today. Nothing here needs
re-arguing; it needs keeping, and re-measuring before it is quoted.

**3. The emitter is the part with no round trip at all.** It takes no dependencies, forbids
`unsafe`, and produces SPIR-V on stable Rust — and the boundary above does not touch it, because
nothing is dispatched. Every hour spent on `spirv-val` coverage, on the lane API's refusals, and on
the mutation gate pays regardless of what any device costs.

**And the whole tree has now been through the gate at full scope: 93 targets, 639 mutants, 100%.**
*Re-run whole on 2026-08-16: 651 mutants, still every shard at 100% with no survivors — twelve more
than the first time, which is the vocabulary and the bound growing since. The tree holds 94 targets
now rather than 93: `extent.rs`'s test module became a file of its own that day, which is one more
file over the same code.*
It had never been — a single run exceeds the mutant cap and outlives the client, so every run before
this was diff-scoped, which answers "did this change arrive covered" and never "is this file
covered". `NOHA_ONLY` shards it by path; five runs did it. `notes/FINDINGS.md` has the table and the
three findings, one of which was a public operation building invalid SPIR-V.

### Tier 2 — the shape a future caller would need

**4. There is no API for "here are N independent problems, answer them all".** Every entry point
takes one module and one grid; a caller with a batch has to lay it out itself, and the one thing
DR-0008 says matters is exactly that layout. `Gpu::run_bound` and `Session` are the pieces, and what
is missing is the shape that makes batching the default rather than the caller's idea.

Written down rather than built, and deliberately: it needs a caller. The last three times this file
guessed at an API before one existed — the third dispatch axis, the caller-owned buffer, the larger
CLI — the guess was refused by its own argument. A batch API invented without a batch would be the
fourth.

### Tier 3 — carried over, unchanged

**5. A buffer the caller already owns.** Still no caller in this repository wants it.

**6. A third vendor.** Still needs hardware that is not in this machine.

### What this list retires

Anything that begins *"if the kernel were faster"*. The kernel is 2.9% of what a caller waits for on
the best device in this machine and 0.3% on the other. A proposal has to move the round trip or the
batch size, and only one of those is reachable from here.

## The baseline, frozen — 2026-08-15

`baseline-2026-08-15` is an annotated tag on the tree with nothing owed on it: every branch reached
by a test, every opcode emitted by something, every public operation consumed outside its own file,
every pipeline builder bounding its dispatch, every decision record saying what enforces it, and
every number the documents state about this repository resolved against the tree.

It is a floor rather than a milestone. What it is for is the sentence that comes after it: **work
after this may add work, and may not add debt.** A `git diff baseline-2026-08-15` that adds a public
operation with no consumer, an opcode with no emitter, a dispatch with no bound or a number with no
counter is a change that undid something, and the tag is what makes that one command rather than an
argument.

The bit shifts above are the first extension built on it, and they were held to it: six tests, 124
mutants at 100%, `spirv-val` before any device, and five widths across three vendors after.

## Where the work goes now — 2026-08-15, with the vocabulary at 20

The bit shifts are in, and they were the item this file named three lists running. So the question
is the next one, asked the same way: **which operations does the emitter offer that no generated
program can reach, and of those, which have an exact reference?** The second half is the filter that
does the work — a fuzzer round is only a test if something can disagree with it, and disagreement
needs an answer that is *right* rather than close.

`src/lanes/` declares <!--count:lane-operations-->68 public functions, of which perhaps forty are
operations rather than plumbing — an inexact split, so it is stated inexactly rather than given a
number nothing checks. What follows is the named list instead.

### Tier 1 — ~~ripe~~ built, 2026-08-16

All three, in the order they were listed. What each turned out to be is in `notes/FINDINGS.md`; the
short version is that `Emit` stopped being an arrangement for one operation. Six domains, five and
one, in three methods with a refusing default — three memberships that do not nest, which is what
makes it a mechanism rather than a special case.

`abs` arrived needing exactly the treatment predicted for it, and from the predicted cause: a
magnitude has no answer at a two's-complement minimum, and the bit shifts had just made that value
reachable. The round is refused rather than compared. `fma` needed its own doc comment corrected —
it argues against fuzzing a fused operation, which is right in general and wrong for a corpus of
small integers. And `all_uniform` needed no gate and a carefully argued threshold.

The list below is what it was, kept because the reasoning is what the next one should be asked with.

### The original entries — ripe, in order of what they cost

**1. `all_uniform` — the third vote, and two of three are generated.** `Op::AddIfAnyAbove` reaches
`any_uniform` and `Op::AddIfAllEqual` reaches `all_equal_uniform`. The vote that asks whether a
predicate held *everywhere* has unit tests and no generated program — `runner/src/kernels/unrun.rs`
names it in the same breath as `shift_down` and `broadcast`, which is the list this file has emptied
twice by fuzzing what was on it.

**No new axis at all.** It is a `Kind` in `WHOLE` and `STRIPPED` beside the vote already there, one
arm in `emit`, one in `interpret`, and the reference is the one beside it with `any` changed to
`all`. Cheapest thing on this list and it closes an asymmetry of exactly the shape `ShiftUp` and
`ShiftDown` had — *"the shift up was here from the beginning and its twin was not"*.

The one thing to get right: a vote nothing ever passes is an identity, and an identity agrees with
every reference including a wrong one. `any_uniform`'s threshold is drawn straddling the corpus for
that reason; an `all` vote passes far less often, so its threshold has to be drawn from the *bottom*
of the range or the step is a no-op in almost every round.

**2. `abs` — and it is a third gating axis, which is the interesting part.** `Lanes::abs` takes
`T: Signed`, which is `F32`, `I32`, `I8`, `I16` and `F16` — **five domains**, not the six `Integer`
gates and not the two a float-only operation would. So it is a second method on `Emit` with a
different membership, and building it is what turns that trait from a special case into a mechanism.

**And it has a real edge, which the bit shifts just made reachable.** GLSL.std.450's `SAbs` of the
most negative representable value has no positive answer to give. Before this week a signed domain's
values were small and that edge was unreachable; a left shift of 31 now puts a value there, so
`abs` arrives already needing the `ShiftTooFar` treatment — a refusal by name, or a draw that cannot
produce it. Design, not a table entry, which is the same sentence this file used about the shifts.

**3. `fma`, and the doc comment beside it argues against this.** `Lanes::fma` says: *"never
bit-identical [to a multiply and an add], so a kernel that must agree with a CPU reference exactly
has to make the same choice on both sides. That is why the fuzzer's vocabulary has `min`, `max` and
`clamp` in it and not this."*

That is right in general and **not right for this corpus**, which is the kind of inherited reasoning
this file keeps catching. Every float value the generator draws is a small integer below the exact
limit, where a product and a sum are both exact — so a fused operation and an unfused one give the
same bits, and the fuzzer can hold them to it. It is also the mirror of the bit shifts' axis:
float-only where those are integer-only, which is the second half of the proof that `Emit` is a
mechanism.

Worth doing *after* `abs`, and worth rewriting that doc comment when it is done rather than leaving
a sentence that was true of the general case standing over a case where it is not.

### Tier 2 — the family with no fuzz coverage and a history

**4. The four packed dot products, `reinterpret` and `to_f32`.** `OpUDot` is the instruction that
shipped **invalid** — correct on two devices for weeks, caught the first time `spirv-val` was
pointed at it — and none of the four is in the generator's vocabulary.

The sandbox covered them at all three mappings with an exact reference and has been deleted, so the
blind spot is **open again as a check and closed as a question**: all twelve combinations were shown
to agree on three devices, and nothing runs that comparison today. What was missing even then is the
*generated combination*, a dot product with a rotate and a rolled loop around it. The cost is real: the input is four bytes in a word, which is a different
buffer shape from every other step here, and the accumulator's type changes mid-program. That is the
first step that would not fit the single-accumulator straight line this generator is built on.

### Tier 3 — deliberately not, and the reason is the filter

**5. `sqrt`, `inverse_sqrt`, `exp`, `log`.** No exact reference exists. GLSL.std.450 specifies these
in ULPs of tolerance rather than exactly, so a comparison would be checking a device's rounding
against the host's — two approximations agreeing, which says nothing about the mapping. This is the
same argument `Domain::exact_limit` makes about a half leaving its range, and the answer there was
to refuse the round rather than loosen the comparison. There is no round to keep here.

**6. No new domain.** All eight are generated and all eight run at five widths. This axis is closed.

### What the next big feature is, and why its blocker just moved

`decisions/DR-0008` measured the thing that decides everything above it: a round trip costs ~100 µs
on the discrete device and the device's own share of it is **2.9%**. Raising the answers per round
trip from 2 to 2 048 is 800×; no kernel change recorded anywhere in this file has been worth more
than 3×.

So the feature is **an API for "here are N independent problems, answer them all"** — and this file
has refused to build it three times, each time for the same good reason: *"it needs a caller. The
last three times this file guessed at an API before one existed the guess was refused by its own
argument."*

**That objection got an answer, and then the answer was thrown away — which was the plan.** The
sandbox was a caller, and the design pressure it supplied was a *mistake* rather than a requirement:
two of its three tools ran one dispatch per seed where the seeds varied only the data and shared the
module. A batching API was built there, used, measured, and deleted with the directory.

What survives is the shape, in `notes/FINDINGS.md`. A batch is **N problems laid out so that the
invocation's own index selects the problem**, which is a constraint on the *kernel* rather than on
the buffer — so an API here owns the arithmetic a kernel has to be built against, not a container
for the words. What made the one real workload un-batchable was a constant element offset sized for
one workgroup, which is right at a batch of one and wrong at every larger size.

That is a better starting point than the file had before, and it is still not a caller. This entry
stays open.

## Where the work goes now — 2026-08-16, after the sandbox

The generator draws from <!--count:fuzz-operations-->23 operations, against the
<!--count:lane-operations-->68 public functions `src/lanes/` declares. That ratio is the one number
this list is written against, and it is now checked at both ends rather than typed.

### Tier 1 — the hole the deletion opened, closed

**1. The packed dot products at all three mappings — done, and in the suite this time.** Deleting
the sandbox took the only check that ran `OpSDot`, `OpUDot`, `OpSUDot` and `OpSDotAccSat` anywhere
but whole-subgroup, because every kernel in `runner::kernels::dot` is built through
`whole_subgroup!`. `runner/tests/mappings.rs` is that check, in the tree that runs on every push,
and it builds its own modules precisely because the macro is what it is testing around.

Checked by breaking it: reading `OpSUDot`'s second operand as signed disagrees at **all three**
mappings on the first seed. And it batches — one workgroup per seed, the offset the whole batch's —
which is the shape the sandbox left behind rather than the code it left behind.

**2. The fuzzer keeps watching without being asked.** Every push already sweeps it at 256 rounds a
domain on lavapipe at 4, 8 and 16. `.github/workflows/nightly.yml` runs 8 000 a domain at 03:00 UTC and
on demand, and fails when fewer than eight domains report — a domain refused every round has no
coverage and looks exactly like one that always agreed.

Two levels of the same argument: the short run says the harness works, the long run is the search.
30 000 programs are what found `reduce_min` folding its strips with a maximum.

### Tier 2 — unchanged, and worth restating against what is now checked

**3. The vocabulary's remaining reach.** The dot products are the family with the most history and
the least *generated* coverage — the twelve combinations agree, and no generated program has ever
put a rotate and a rolled loop around one. It stays hard for the reason it always was: the input is
four bytes in a word and the accumulator's type changes mid-program, which is the first step that
would not fit the single-accumulator straight line the generator is built on.

**4. The batch API still has no caller.** The sandbox was one and was thrown away on purpose; what
it left is the shape, in `notes/FINDINGS.md`. `runner/tests/mappings.rs` uses that shape and does
not need an API for it — one test is not a second caller.

**5. A buffer the caller already owns**, and **6. a third vendor.** Carried over, unchanged.

### And the thing that is not on any tier

The emitter has no external callers, and every promise it makes is checked by something in this
repository rather than by anything outside it. `USING.md` is the first attempt at writing down what
somebody outside would be relying on and what holds each of those up — not a feature, and the one
piece of work here whose absence nothing in the tree can detect.

## The horizon, after the night of 2026-08-16

The whole tree through the mutation gate in six shards — 93 targets, every shard **100%**, no
survivors — and then the three things that survey turned up.

**What is closed.** The dispatch bound's last blind term, above. `notes/CLAIMS.md` item 2, which
asked for a sweep of `src/module/` and turns out to have nothing in it to sweep: that layer has no
generic parameter anywhere, so a bound cannot be too wide there, and what guards it is `spirv-val`
alone. And the file-length question, which was worth measuring rather than assuming — one file
needed splitting and the other eight named as candidates did not.

**What is open, in the order it is worth doing.**

1. **The examples still run nowhere but a workstation.** `notes/CLAIMS.md` item 3, and the one
   remaining check with no instrument: seventeen programs that compile in CI and are never run, of
   which sixteen need a device. It is one step in the lavapipe job, and what stops it being one line
   is that nobody has measured what those sixteen cost on a software driver at four lanes.
2. **The mapping relationship is still only half mechanised.** `Mapping::of` is the single rule and
   the fuzzer asserts it takes its mapping from there. What no check asserts is the *absence* of a
   fourth copy — and this file has paid for that three times.
3. **The dot products have no generated coverage.** `runner/tests/mappings.rs` holds the twelve
   combinations; no generated program has ever put a rotate and a rolled loop around one. The
   blocker is unchanged and real: the input is four bytes in a word and the accumulator's type
   changes mid-program, which is the first step that does not fit the generator's single-accumulator
   line.
4. **The batch API still has no caller**, and the sandbox that was one was deleted on purpose.
## Where the work goes now — 2026-08-25, after an audit rather than a feature

Nine days without a list, and four commits of feature work in them: the five arithmetic
instructions on 2026-08-18, `Kernel::repeat_rolled` and its many-carry form on 2026-08-19, and
`decisions/DR-0010` recording the second. None of that reached this file, which is the smallest of
the five things the audit found and the one that hid the other four — a list that stops being
written is a list that stops being read against.

**Four of the five were an instrument that had quietly stopped reaching its subject.** That is the
same shape `notes/CLAIMS.md` is about, four times over, and it is worth stating as the finding
rather than as four repairs.

### Tier 1 — found and closed, 2026-08-25

**1. `README.md` had been mojibake for nine days, and nothing here could see it.** One edit on
2026-08-16 round-tripped the file through Windows-1252: 174 sequences over 134 of its 840 lines, so
every em dash arrived as three characters and every micro sign as two, on the front page. The
damage is described rather than quoted here for the reason `notes/FINDINGS.md` gives at length:
this file is one the new check reads, and an example would be an instance.

The commit that did it shows **136 insertions and 136 deletions** in a diff whose message is about
the fuzzer's gating, which is what a whole-file re-encode looks like when nobody is looking for one.
It survived eight commits.

The repair was mechanical and is worth writing down as a method rather than an outcome, because the
file was **mixed** — nineteen em dashes had survived, added by later edits — so decoding the whole
thing again fails on the first character that was never damaged. What works is undoing the round
trip *per run*: take each character back to the byte Windows-1252 would have decoded it from, and
accept the run only when those bytes form a character. 840 lines before and after, and the ASCII
skeleton of every changed line identical — which is the proof that only the encoding moved.

**Why nothing caught it is the part worth keeping.** `tests/documented.rs` checks the numbers a
document states and the backticked names it uses. Both are ASCII. A document can therefore rot in
every character those two do not read and pass the suite written to keep it honest.
`every_document_is_the_text_it_was_written_as` is the fourth claim in that file now, and it reaches
`.rs` as well as `.md`: the editor that did this has no opinion about extensions, and a doc comment
carries the same punctuation as a paragraph.

**2. `spirv-val` was not installed on the machine with the GPUs, and the fallback pointed at a
drive that is not mounted.** One runner sweep printed **631 skips**, every one of them
`spirv-val not found`. So the workstation — the only place widths 32 and 64 run at all — had never
validated a module, and `decisions/DR-0010` says so about itself in its *What is not verified here*
section.

`tests/common/spirv_val.rs` read `H:\tools\spirv-tools\install\bin\spirv-val.exe` when `SPIRV_VAL`
was unset. There is no `H:` here. So "look in the usual place" had meant "find nothing" for as long
as the letter had been wrong, two lines under a doc comment about exactly that failure — the file
already knew a set-and-wrong path is worse than an unset one, and had a hard-coded one of its own.
The fallback searches `PATH` now, which is the usual place and cannot go stale when a drive is
remounted or a checkout moves.

With the validator present the two suites come to **395 and 482 with no skips at all**, and every
one of those 631 modules is legal — which is the answer rather than the point. The point is that
nothing had asked, on the machine where it mattered most.

**3. The five arithmetic instructions and the two rolled loops had never run.** `f_sub`, `f_div`,
`f_negate`, `i_sub` and `u_div` had **one consumer between them** — `tests/instructions.rs`, which
builds one module and hands it to `spirv-val`. `Kernel::repeat_rolled` and `repeat_rolled_many` had
their own unit tests and `tests/control_flow.rs`, and no dispatch.

Both passed `every_public_operation_has_a_consumer_outside_its_own_file`, because a test is a
consumer. That is the check doing exactly what it says, and the gap is one it was never asked to
close: an operation whose only consumer is the test written to satisfy it. `decisions/DR-0009`'s
territory, and the record `notes/CLAIMS.md` lists as backed by nothing.

The validator is a weaker witness for these than for anything else in the tree, and `DR-0001` says
why: an opcode number read wrong assembles into a **different well-formed instruction**. `OpFNegate`
at the wrong number is still a valid module. Four kernels in `runner/src/kernels/unrun.rs` run them
now — a centre-and-scale whose every step is exact so the comparison is on bits, a remainder written
as the divide, multiply and subtract it actually is, a rolled loop reading a different block of a
buffer each trip, and two running totals out of one pass whose difference is stored so that both
phis are load-bearing. Each was **checked by breaking it**: an `f_sub` written as an `f_add`, an
`i_sub` as an `i_add`, a loop whose every trip reads block zero, a second phi wired to the first.
Four mutations, four red tests, each for its own reason.

They declare no subgroup capability and are built through `whole_subgroup!`, so they run at every
width and the per-width skip counts in `ci.yml` are unchanged. That was a design constraint rather
than a happy result: a test that skips at 4, 8 and 16 would have moved three numbers in a matrix and
taught nobody anything.

**4. The mutation gate could not run from this checkout, and the four commits above had not been
through it.** `noha.yaml` is excluded by this machine's global gitignore — policy, deliberate, and
the reason `tests/integrity.rs` skips four checks rather than panicking. What the policy does not
say is that the file should be *absent*, only that it is never committed, and it was absent.

Rebuilt from the tree: the source list is generated as everything under `src/`, `runner/src/` and
`cli/src/` minus `NOT_MUTATED`, so the two agree by construction rather than by hand — 94 targets
over 15 excused FFI files. Generating one list from the other is the only honest way to rebuild it,
because a hand-typed list that nothing compares against reality is the drift those tests exist to
catch. All 17 integrity checks run here now, with no skips.

`test:` is the **whole workspace** and not the root, which is the one thing easy to get wrong: a
mutant in `runner/src` is only killed by the device suite, and a root `cargo test` runs the emitter
alone — so every runner mutant would survive and bury the emitter's real score underneath them.

**5. This file was nine days stale**, which is the entry you are reading.

### Tier 2 — carried over, unchanged and still right

None of the four the audit inherited was closed by it, and none was made easier.

**6. The examples still run nowhere but a workstation.** `notes/CLAIMS.md` item 3: seventeen
programs that compile in CI and are never run there, sixteen of which need a device. The nightly
liveness job runs them; a push does not. What stops it being one step is unchanged — nobody has
measured what those sixteen cost on a software rasteriser at four lanes.

**7. The mapping relationship is still only half mechanised.** `Mapping::of` is the single runtime
rule and the fuzzer asserts it takes its mapping from there. What no check asserts is the *absence*
of a fourth copy, and this file has paid for that three times.

**8. The dot products have no generated coverage.** `runner/tests/mappings.rs` holds the twelve
combinations; no generated program has ever put a rotate and a rolled loop around one. The blocker
is unchanged and real: the input is four bytes in a word and the accumulator's type changes
mid-program, which is the first step that does not fit the generator's single-accumulator line.

**9. The batch API still has no caller**, and the sandbox that was one was deleted on purpose.

### What this list is asking the next one

Every one of the four instruments that had failed was **reporting green while reaching nothing**: a
skip nobody counted, a path on a drive that had gone, a check whose alphabet excluded the damage, a
config whose absence was a shrug. This repository is unusually good at noticing that a *claim* has
no instrument. What it had not done is ask, of the instruments it already has, **whether each one
still touches its subject** — which is a different question and, on this evidence, the one with more
in it.

A candidate for the next sitting, in that spirit: nothing here checks that a test which is *supposed*
to run actually ran. The lavapipe job asserts a skip count per width and is the only place in the
tree that does. The workstation asserts nothing of the kind, which is how 631 of them went unnoticed.

## Where the work goes now — 2026-08-26, and the question the last list asked

The section above closed by asking, of the instruments this repository already has, **whether each
one still touches its subject**. It was taken up the next day. Both answers were no, and neither
was found by a gate — one by asking which public functions `cargo test` never runs, the other by
reading documents against the tree.

### Tier 1 — found and closed, 2026-08-26

**1. The whole measurement surface had never been run by the suite.** `Gpu::time_grid`,
`time_repeated`, `memory_types`, `probe_memory`, `probe_resident`, `Reducer::sum_timed` and
`Scanner::scan_timed` were reachable from `runner/examples/` and from nowhere else. Every one
passed `every_public_operation_has_a_consumer_outside_its_own_file`, because an example is a
consumer — the check doing exactly what it says, on a question nobody had asked it.

Asked instead: which `pub fn` is named by *no test at all*. Nineteen of four hundred. Seven were
the above; twelve were `Module` emitters run only through their callers, which is the shape
`dot_unsigned` had. `src/module/arithmetic.rs` had 167 lines and **no test module at all**.

`runner/tests/instruments.rs` holds the first seven. It makes no wall-clock claim — `ci.yml`
already argues that at length — so `time_grid`'s second axis, the only thing it offers that
`Gpu::time` does not, is held through the **bounds check** instead: eight workgroups by eight and a
linear sixty-four must be refused for the same extent, which is false the moment `grid.y` is
dropped. Checked by dropping it and watching the test go red. The twelve are held to their opcode
*and their operand order*, because an `OpFAdd` where an `OpFSub` belongs still returns a number.

**2. `Gpu::time_repeated` documented an error its own code could not produce.** The `# Errors`
section promised `Error::NoPipeline` "if `repeats` is zero and there is nothing to summarise", over
a loop reading `repeats.max(1)`. Zero has always taken one sample.

The floor stayed and the sentence went. `NoPipeline` reads *the driver returned no compute
pipeline*, which is not what a caller who passed a zero did; `Gpu::probe_resident` floors its own
count at the same malformed question and promises nothing; and the result does not lie, because
`Timing::repeats` reports the one sample taken rather than the zero asked for. That is a different
case from `Timing::of(&[])`, which would have to invent a duration and still refuses.

**3. Three counts written as words, and the instrument that now reads them.** `notes/CLAIMS.md`
understated the size of the counter registry, in words, inside the very paragraph arguing there is
"no argument for leaving these to prose" — while stating it correctly two hundred lines further
down. The wrong number is described rather than quoted, for the reason the mojibake note gives:
this file is one the new check reads, and a quotation would be an instance. `README.md` and
`notes/CLAIMS.md` disagreed with each other about how many decision records there are, and both
were short of the <!--count:decisions-->10 in `decisions/`. Two more undercounted the examples by
one, where there are <!--count:examples-->17.

None was reachable, and the reason is structural rather than an oversight: **a marker resolves to
digits, so it cannot stand in front of a spelled numeral.** Every count written as an English word
was outside the instrument by construction.

`every_spelled_count_of_a_standing_set_is_the_number_that_is_there` reads them now, and its
vocabulary was *measured* rather than chosen. `tests`, `operations`, `opcodes` and `jobs` sit beside
a spelled numeral forty-odd times between them and are almost all accounts of something that
happened — "Eleven tests reading past their input", "seven opcodes deleted in one commit". Those
are true permanently, and a gate that failed them would fail a document telling the truth, which is
the red-for-the-wrong-reason `ci.yml` is about. A noun qualifies when it names a set that **exists
now**. Three do.

`COUNTERS` also counts itself now, which is the smallest instrument in the file and the one whose
absence started this: how many claims a document may make about the tree is a claim about the tree.

### What this list is asking the next one

**This file went stale again, and the commits above are what made it stale.** The section before
this one was written *because* nine days of work had not reached it; two days later two more
commits had not either. Nothing checks that a note is current, and the audit that found everything
above did not close that — it is the one finding with no instrument behind it.

Three things were looked at and not finished, stated so they are not mistaken for covered:

- `notes/FINDINGS.md` — 4 117 lines, and by `notes/CLAIMS.md`'s own count 222 measured numbers —
  was not audited at all.
- Of the `# Errors` blocks naming a concrete variant, two are now known to promise a *condition*
  the body contradicts. **The second was found by a method rather than by reading everything, and
  the method is the part worth keeping**: enumerate the sites that *produce* a variant, list the
  docs that *claim* it, and compare. Matching a variant name against the body finds nothing,
  because almost everything delegates and the refusal is two layers down — but a variant with few
  producers can be read against its claimants by hand in minutes.

  `LaneError::NoSuchForm` has six producing sites and seven docs claiming it. Six matched.
  `kernels::broadcast_in_cluster` was the odd one: it promised that variant for a `source`
  outside the vector and emits `LaneOutOfRange`, because `Lanes::broadcast` refuses through
  `Lanes::within_group`, which names the operand and the width it passed. Corrected, and pinned
  by a test that goes red when `within_group` is made to name the other variant.

  It is still not an instrument. Doing it for every variant means a call graph, and the six
  variants with many producers would need one. What it is, is a pass somebody can repeat.
- Branch reachability was sampled, not exhausted: the `ok_or` and `unwrap_or` sites were read, the
  thirty-odd `saturating_*` calls and the match arms were not.
