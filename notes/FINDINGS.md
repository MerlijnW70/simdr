# FINDINGS

What each slice cost beyond its description, which assumption turned out wrong, and which dead
ends were walked. Read the section for the area you are about to touch — most of what is here
exists because something was already derived once, wrongly.

## Validation

**`spirv-val` without `--target-env` validates almost nothing you care about — 2026-08-11.**

The default is the *universal* SPIR-V environment, which is far laxer than any real consumer. The
first four validation tests were written without the flag, all passed, and were reported as
"Khronos accepts the module". Then the teeth test — a `GLCompute` entry point with no `LocalSize`,
written to prove the gate could go red — **also passed**, because `LocalSize` is a Vulkan
requirement rather than a SPIR-V one.

So for one commit this project had a validation gate that could not fail. Every call in
`tests/validated.rs` now names its environment: `vulkan1.0` for SPIR-V 1.0 modules, `vulkan1.1` for
1.3 ones.

Two things worth carrying forward:

- **Write the test that proves the gate can go red, and write it early.** It cost one test and
  found that the other four were weaker than claimed. Without it the discovery would have come
  much later, with much more built on top.
- **A tool exiting zero is not the same as a tool agreeing with you.** Check what it actually
  validated, not just that it was happy.

## Performance, measured against device-local memory

**2026-08-11, RTX 4080, `cargo run -p runner --release --example bench`.** Buffers are device-local
now and the host's copies are separate submissions, so what is timed is the kernel reading VRAM.
Everything got roughly fifteen times faster, which is the measure of how badly the previous
numbers were dominated by the bus.

| kernel | 4 096 wg (4 MB) | 65 536 wg (67 MB) |
| --- | --- | --- |
| empty (dispatch only) | 3 us | 34 us |
| scale (no reduce) | 4 us / 73 G | 45 us / 93 G |
| `Simd<f32,8>` clustered | 4 us / 72 G | 34 us / 123 G |
| `Simd<f32,32>` whole | 4 us / 71 G | 34 us / 122 G |
| `Simd<f32,64>` 2 strips | 4 us / 142 G | 39 us / **215 G** |
| `Simd<f32,128>` 4 strips | 4 us / 246 G | 279 us / **60 G** |

**The reduction is still free**, and now that is a real measurement rather than an artefact of
everything being bus-bound: a clustered reduce, a full-subgroup reduce and no reduction at all all
land within a few percent of the dispatch floor.

**Strip mining pays until the working set stops fitting.** Two strips reaches 215 G elements per
second — 860 GB/s of traffic, which is *above* this card's ~717 GB/s of VRAM bandwidth and
therefore cache. Four strips at the same dispatch size collapses to 60 G.

**The cause is the working set, not the layout.** The same four-strip kernel runs at 246 G on the
small dispatch and 60 G on the large one, and what changes between them is 4 MB of data against
67 MB.

**~~And the 4080's L2 is 64 MB, so the cliff sits exactly where the data stops fitting in it.~~**
**Retracted the same day — see below.** That sentence was written from one number and a plausible
coincidence, and a sweep refuted it within the hour. It is left struck through rather than deleted
because the pattern is the point: a mechanism that explains one data point is not a finding.

That retracts the earlier reading of this as a striding problem twice over: blocking by workgroup
did not move the numbers, and the numbers that mattered were hidden behind PCIe anyway. **A
throughput figure needs its memory path checked before it is believed**, and this one took two
attempts to get right.

## The sweep, instrumented — and almost none of it survives

**2026-08-11, after adding spread reporting and a memory-placement check.** Both were added
because the previous section could not tell a result from noise. They answered immediately.

**Placement is fine, so eviction is not the story.** `probe_memory` reports a 16.8 GB device-local
heap and confirms a 96 MB request lands device-local. The harness holds three buffers, so 288 MB
at the largest point — nowhere near pressure. That hypothesis is dead too.

**And with the spread visible, seventeen of nineteen rows are not evidence.** `Timing` runs five
repeats and flags any point whose slowest is more than a fifth above its fastest. Almost
everything is flagged, some rows by 20× within a single sweep:

| kernel | working set | best | median | spread |
| --- | --- | --- | --- | --- |
| 4 strips | 32 MB | 20 us | 20 us | **1.0x** |
| 4 strips | 72 MB | 315 us | 2776 us | 10.4x |
| 4 strips | 96 MB | 192 us | 197 us | 22.8x |
| 2 strips | 16 MB | 24 us | 24 us | **1.0x** |
| 2 strips | 80 MB | 203 us | 243 us | 20.9x |

Two steady points out of nineteen, and neither locates a cliff.

**So every cliff claim in this file above is withdrawn**, including the ones that survived the
first retraction. What looked like a reproducible boundary at 48–56 MB was two runs of a
measurement that, given five, disagrees with itself by an order of magnitude. The shape kept
recurring, which is what made it convincing; recurrence is not stability.

**What is left standing** is the small-dispatch part of the benchmark, where the numbers *were*
consistent run to run: the reduction costs nothing measurable against the dispatch floor. That one
holds because it was never near the noisy regime.

**What it would take to answer the original question.** More repeats with outlier rejection, a
machine not also driving a display, and probably device timestamp queries rather than fence
timing — the host-side clock includes scheduling this harness has no view of. None of that is
built, and until it is, this project has no performance claim about large working sets.

## Superseded: the cache-capacity story, which was also wrong

**`cargo run -p runner --release --example sweep`, twice, 2026-08-11.** The experiment: run a
four-strip and a two-strip kernel over the *same* working sets. A two-strip kernel reaches any
given number of megabytes at twice the workgroup count, so if the cliff is cache capacity it lands
at the same megabytes for both. It does not.

| working set | 4 strips, run 1 / run 2 | 2 strips, run 1 / run 2 |
| --- | --- | --- |
| 32 MB | 21 / 20 us | 64 / 97 us |
| 48 MB | 29 / 30 us | 236 / 289 us |
| 56 MB | **112 / 153 us** | 3142 / 299 us |
| 64 MB | 130 / 129 us | 2682 / **3641 us** |
| 96 MB | 211 / 199 us | 4131 / 4989 us |

**What is established.** The four-strip cliff is real and reproducible: fast through 48 MB, four
to five times slower from 56 MB, in both runs. And the L2 hypothesis is dead — the two kernels do
not break at the same working set, so a simple capacity threshold is not what is happening.

**What is not established, and must not be quoted.** Everything about the two-strip column. Its
pre-cliff times vary by 50% between runs, and the point where it collapses *moved* from 56 MB to
64 MB. A measurement whose cliff wanders is measuring something outside the kernel.

The ~20 GB/s figures are PCIe-shaped, which makes eviction of the device-local allocation a
suspect — this card is also driving a display, and the harness allocates three buffers of the full
input size. But that is a hypothesis of exactly the kind that just cost a retraction, so it stays
a hypothesis.

**What the harness needs before it can answer this.** It reports one number per point; it should
report the spread across repeats, so an unstable measurement announces itself instead of reading
like a result. And it should check that the allocation it got is genuinely device-local rather
than assuming the request was honoured. Neither is built.

## Superseded: the host-visible measurements

**The buffers are host-visible, and on a discrete GPU that means system memory over PCIe.**
Measured 2026-08-11 with the blocked layout in place:

| kernel | 4 096 wg | 65 536 wg |
| --- | --- | --- |
| empty (dispatch only) | 4 us | 38 us |
| scale (no reduce) | 41 us | 677 us |
| `Simd<f32,32>` whole | 41 us | 646 us |
| `Simd<f32,64>` 2 strips | 41 us | 672 us |
| `Simd<f32,128>` 4 strips | 41 us | 2769 us |

The `empty` row is the tell: a dispatch of four million invocations costs 38 µs, so launch
overhead is nothing and everything else is memory. And the throughputs sit between 24 and
50 GB/s, which is PCIe territory rather than the 4080's ~700 GB/s of VRAM.

**So changing the strip layout did not change the numbers, and could not have.** Blocking by
workgroup is the right layout — `runner/tests/lanes.rs` proves a second workgroup reads its own
run — but a measurement pinned at the host-memory bandwidth cannot see a cache effect. The
earlier reading of the four-strip collapse as a locality problem was a guess dressed as a finding;
what it actually shows is a kernel reading 67 MB across the bus.

**What would make this measurable:** device-local buffers with a staging copy, so the kernel reads
VRAM. That is perhaps eighty lines in `runner/src/buffer.rs` and it is the prerequisite for any
performance claim this project makes. Until then the honest statement is that the reduction is
*not* the bottleneck, and nothing more.

## Earlier performance notes, superseded by the above

**Measured 2026-08-11, RTX 4080, `cargo run -p runner --release --example bench`.** Elements
reduced per second, so a strip-mined kernel is not flattered for doing more work per dispatch.

| kernel | 4 096 workgroups | 65 536 workgroups |
| --- | --- | --- |
| scale (no reduce) | 6.5 G | 6.5 G |
| `Simd<f32,4>` clustered | 6.5 G | 6.5 G |
| `Simd<f32,8>` clustered | 6.5 G | 6.3 G |
| `Simd<f32,32>` whole subgroup | 6.4 G | 6.5 G |
| `Simd<f32,64>` 2 strips | 12.8 G | 11.8 G |
| `Simd<f32,128>` 4 strips | 25.4 G | **7.5 G** |

Three things fall out.

**The reduction is free.** A clustered reduce, a full-subgroup reduce and no reduction at all run
at the same rate. Whatever the bottleneck is, `OpGroupNonUniform*` is not it — which is the
strongest support the clustering design has: it costs nothing to use.

**Strip mining wins, up to a point.** Two strips nearly doubles throughput.

**Four strips collapses it, and the layout is why.** With a stride covering the whole dispatch, an
invocation's four elements are 16 MB apart; each strip is coalesced across the subgroup but the
strips are nowhere near each other, and past two of them the cache gives up. A per-workgroup
stride would keep them close — that is the obvious next experiment and it is not done.

**The small size measures nothing.** At 4 096 workgroups every row takes 41 µs whatever it does,
so that column is launch overhead. The two sizes are in the benchmark precisely so that is
visible; one size alone would have been reported as a result.

**And the first version of this benchmark was wrong.** `STRIP_STRIDE` was 64, so at 4 096
workgroups every invocation's second element was one another invocation had already read — the
25.4 G was a cache measurement. The stride is a parameter now. A throughput number needs its
access pattern checked before it is believed.

## Loops

**`OpLoopMerge` must be the *second-to-last* instruction in its block — 2026-08-11.** Not merely
somewhere before the branch: immediately before it. The first rolled loop computed its exit
comparison between the merge and the branch, which reads perfectly naturally and is invalid.

The unit test missed it. It asserted `merge < branch` by position, which was true. **`spirv-val`
caught it the moment a rolled loop was added to the validated kernels** — and it had not been,
because until then nothing built one outside a unit test. The unit test now checks adjacency.

Two things worth carrying: a positional assertion is weaker than an adjacency one and looks the
same, and **a shape with no validated kernel is a shape nothing is really checking**.

**Unrolled by default.** `repeat` takes a count known at build time and emits the body that many
times: no phi, no back edge, no counter, and the driver was going to unroll it anyway.
`repeat_rolled` is the one that emits the four-block form, and it exists for counts large enough
that unrolling would bloat the module. Its body is built *once*, so it cannot depend on the
iteration number — which is why the tree reduction uses the unrolled one.

## Float edge cases

**Both `reduce_max` paths drop a NaN on this device — 2026-08-11.** `OpGroupNonUniformFMax` and
the compare-and-select strip fold agree: over `0..31` with a NaN at lane 7, both return 31. That
was worth checking because the two are genuinely different instructions and an ordered comparison
against NaN is *false*, so the select path had an obvious way to differ.

Recorded rather than asserted. SPIR-V does not fully pin `FMax` with a NaN operand, so
`runner/tests/floats.rs` asserts only what is guaranteed — the answer is one of the inputs or NaN,
never some third number — and prints what the device actually did. **Pinning an answer the
specification declines to give would turn a driver's freedom into our regression.**

What *is* asserted, because IEEE 754 fixes it under any reduction order: an infinity propagates, a
NaN propagates through addition, `-0.0 + -0.0` stays negative, and a value past 2²⁴ swallows the
small ones without trace. Each also checks the *other* subgroup is untouched, which is the part a
broken mapping would break.

## Control flow

**A uniform branch works and a divergent one is not offered — 2026-08-11.** `DR-0003` has the
argument; what the tests add is that the safe half runs correctly on hardware, including the case
that matters: two subgroups in one workgroup taking the branch differently.

**The `Uniform` newtype is the whole enforcement.** Only a vote produces one, and `Uniform::new`
is private, so a caller cannot hand `if_uniform` a per-lane boolean without editing this crate.
The type system cannot prove uniformity, but it can make the mistake need effort.

**A value does not survive a merge.** SPIR-V's logical addressing model has no mutable locals, so
anything computed inside a branch needs an `OpPhi` to come out. That surprised the first kernel
written against it, and the fuzzer's branch operation uses a select on the vote instead — the vote
is uniform, so the two readings agree and the select keeps the value in a register.

## Differential fuzzing

**20 000 generated programs, zero disagreements — 2026-08-11.** `runner/tests/fuzzing.rs`
generates straight-line lane programs from a seed, interprets the same program on the CPU, and
compares. Nothing was found, which is a weaker statement than "there are no bugs" and a stronger
one than any hand-written test can make.

Three choices that make the comparison mean something:

- **`u32`, not `f32`.** A subgroup reduction combines lanes in an order the specification does not
  fix, and floating-point addition is not associative — so an exact float comparison would be
  comparing against one arbitrary order. Integer add and multiply are associative modulo 2³², so
  the answer does not depend on the order at all.
- **The reference models the *mapping*, not the instructions.** It works out where each element
  lives and which lanes share a subgroup, independently of the emitter. If the two ever disagree
  about that, the disagreement is the finding and either could be at fault.
- **`ShiftUp` is always by zero.** A non-zero shift reads lanes that do not exist for the
  invocations near the edge, and SPIR-V leaves those undefined. A reference cannot predict
  undefined, so the operation stays the identity — it proves the instruction is emitted and
  harmless, and nothing more. Wanting more would need the wrap the specification does not give.

And a teeth test, as everywhere else: feed the reference a perturbed input and check that the two
part ways, so a green run cannot mean the comparison never fires.

## The lane API

**The abstraction holds, measured — 2026-08-11.** `kernels::lane_sum::<N>` is four lines that name
no reduction shape and no cluster size; `Lanes` derives both from `N` against the device's width.
On the RTX 4080 over a 0..63 ramp: `Simd<f32,4>` → `[6,6,6,6, 22,22,22,22]`, `Simd<f32,8>` → 28,
`Simd<f32,32>` → 496. One source, three widths, three correct answers.

**Elementwise really is free.** A `Vector<8>` add and a `Vector<32>` add emit the *same single*
`OpFAdd` — there is a test asserting the instruction count is 1 for both. The lane count lives in
the type and never reaches an instruction, which is what makes `Vector<8> + Vector<32>` a compile
error rather than a validator complaint.

**Mutation coverage more than doubled on this slice** — 17 viable mutants to 38, because the lane
layer has genuine branching where the emitter below it was straight-line operand assembly. Worth
noting for the earlier observation that a 100% score said less than it looked: it now covers more.

**A clustered *scan* does not exist.** SPIR-V's clustered form is a reduce only, so `prefix_sum` on
a vector narrower than the subgroup would scan across lanes belonging to a *different* vector.
Refused by name rather than approximated.

## Execution

**The kernels run, and the numbers are right — 2026-08-11.** On an RTX 4080 (subgroup 32,
arithmetic/clustered/shuffle all supported): `subgroupAdd` over a 0..63 ramp returns 496 to the
first subgroup and 1520 to the second, `Clustered { size: 8 }` returns 28 to the first cluster, and
a `shuffle_xor 1` butterfly returns `[1, 1, 5, 5, 9, 9, …]`. Every one matches a CPU reference
computed from the *device's* reported subgroup width rather than an assumed 32.

That makes the clustered design claim measured rather than argued: eight adjacent lanes reduce
independently inside a 32-wide subgroup, so a `Simd<f32, 8>` need not idle twenty-four lanes.

**`ash`, not `wgpu`, and the reason matters.** wgpu routes SPIR-V through naga, which re-parses and
re-emits it — that would test naga's reading of our module rather than the driver's. `ash` passes
the words to `vkCreateShaderModule` untouched, so a disagreement is between us and the driver with
no third opinion in between.

**Write the discriminator, not just the reference.** A reduction that spanned the whole workgroup
instead of the subgroup would give every lane the same number, and a carelessly written reference
could agree with it. Both reduction tests therefore also assert that two groups hold *different*
totals. Same discipline as the validator's teeth test.

## Subgroups

**One instruction, two operand encodings — 2026-08-11.** `OpGroupNonUniformFAdd` takes an
execution scope and a group operation side by side, and they are encoded *differently*: the
grammar calls the scope `IdScope`, meaning it arrives as the **id of a 32-bit integer constant**,
while the `GroupOperation` next to it is a plain literal. The trailing `ClusterSize` is an id
again.

Passing the scope's numeric value where its constant's id belongs produces a module that
assembles and means something else. `Module::scope` exists so no call site handles the number, and
the disassembly is the check: it should read `OpGroupNonUniformFAdd %float %uint_3 Reduce %21` —
an id, then a keyword.

Read the grammar's operand list, not just the opcode, before emitting an instruction for the first
time.

## Types

**Aggregates are the exception to the one-declaration rule, and deduplicating them would be a
bug — 2026-08-11.** §2.8 forbids declaring the same *scalar, vector or matrix* type twice, and the
obvious generalisation is to intern everything. It is wrong: the specification explicitly allows
multiple `OpTypeStruct` and `OpTypeRuntimeArray` declarations with identical operands, *because*
that is how two aggregates with the same shape carry different decorations. Intern them and two
buffers wanting different `ArrayStride`s silently become one.

So `type_struct` and `type_runtime_array` allocate a fresh id every call, and everything else
interns. Read the rule before generalising it.

## Encoding

**A literal string whose byte length is a multiple of four still needs a whole extra word.** The
NUL terminator is part of the literal, not padding that may be elided — a consumer that finds four
non-zero bytes in the last word keeps reading into the next operand. `literal_string_words` folds
this into `len / 4 + 1` so the even case needs no branch.

## Mutation coverage

**`noha prober` reports 100% over a much smaller surface than the code.** As of 2026-08-11 it
generates mutants for `encode.rs` and a little of `module/`, and **none at all** for `src/spec/`,
whose bodies are `match self { X => 1 }` with no predicate to flip. Check `.noha/tia.tsv` for the
per-file counts before quoting a score. What guards those numbers is DR-0001's recipe and the
validator, not the prober.


## Control flow

**A phi names the block a value *arrived through*, which is not the block the arm opened —
2026-08-11.** `choose_uniform` builds `%then`, hands the body a builder, and joins at `%merge`. The
obvious `OpPhi %type %value %then …` is right only while the body stays in one block. Let the body
nest a selection or a loop and it finishes in *that* construct's merge block; the phi then names a
predecessor that no longer branches to the join, and `spirv-val` calls it a dominance failure.

The fix is that `Module` tracks which block is open — `label`/`label_at` set it, every terminator
clears it — and the arm reads it back after the body rather than assuming. An arm that left no open
block at all (it returned) is refused as `LaneError::NoOpenBlock` rather than attributed to
something plausible.

`tests/control_flow.rs` validates both the flat and the nested case, because the flat one passes
either way.

**A loop counter is available for free and was being thrown away.** `repeat_rolled` already emits a
counter phi to test the trip count against; handing it to the body costs nothing and is the only
way a body built *once* can index anything. The test that pins it compares the id the body received
against the `u32` phi in the emitted words — asserting merely that "some id" arrived would have
passed for a copy or a fresh zero.

## Numbers

**`OpUDiv` is 134, not 152.** Guessed 152 while writing the multi-pass reduction; assembling a probe
with `spirv-as` said otherwise. DR-0001 exists for exactly this, and the recipe now has a second
form: with no `spirv.core.grammar.json` to hand, `spirv-as` a one-instruction module and read the
word back out. Khronos' own assembler is as authoritative as the JSON.

Same probe confirmed `BuiltIn SubgroupId = 40` — which then went unused, because the reduction
found a shape that needs no subgroup identity at all.

## Multi-pass

**There is no barrier across a dispatch, so a reduction wider than a workgroup is an algorithm
rather than an operation.** `Gpu::sum` folds `out[i] = in[i] + in[i + half]` until 64 elements
remain, then finishes with one subgroup reduction: 11 dispatches for 65 536 elements, all in one
submission, exact against a CPU sum at every power of two from 2^7 to 2^18.

Two things about it are deliberate and worth not forgetting:

- **The dispatch is sized to the work, so no pass needs a bounds test.** `half` invocations means
  `i + half` is in range by construction. A guarded version would be a divergent branch, which
  DR-0003 does not offer, and it would cost an instruction per element to buy nothing.
- ~~**The last two floats come home.** The final workgroup holds two subgroups and therefore two
  totals; adding them on the device would need a third mechanism for one addition.~~
  **Superseded 2026-08-12.** The third mechanism arrived — workgroup shared memory and a barrier —
  and `kernels::workgroup_sum` combines the subgroups on the device. The host reads one number it
  computed no part of. `Reduction::host_combined` says so, and reports `1`.

  The prediction in the struck-through half was right about what it would take, which is the only
  reason it is worth leaving here: "a third mechanism for one addition" sounded like a bad trade
  and the mechanism turned out to be worth having for its own sake.

Between passes the chain copies the whole output buffer back into the input. For a shrinking
reduction that is mostly copying elements nobody will read, and a ping-pong across two descriptor
sets would avoid it. **This chain is built to be correct, not fast, and nothing here is a
performance claim.**

## Pointed at something real

**A chess engine's NNUE layer, 2026-08-11.** `H:\schaak` is a zero-dependency, no-`unsafe` engine
with a `768 -> 256x2 -> 1` quantised network. Its whole per-evaluation arithmetic, once the
accumulator is current, is two 256-element clipped-ReLU dot products — which `Simd<i32,256>` on a
32-wide subgroup expresses as exactly one subgroup's work, eight strips. `kernels::clipped_dot`
matches the engine's own loop exactly, checked on the device.

Three numbers came out of it, and only the first was expected:

**One evaluation, waited on: ~940 us.** Against the engine's recorded 199 ns, that is ~4700x
slower. Most of that is `Gpu::run` allocating buffers and building a pipeline per call, so it
overstates the floor — but the floor is a submit-and-fence, tens of microseconds, and alpha-beta
cannot ask for two evaluations at once. **GPU evaluation inside a search is arithmetic, not
opinion: it does not work.**

**Batched, it wins by 150x and then stops.** 8192 positions run at 766 M evaluations/s against
5.0 M on one CPU thread. At 65 536 positions — 268 MB of operands — it falls back to 154 M with a
2.2x spread, the same unexplained large-working-set cliff this file already records. The peak sits
somewhere the sweep does not resolve, and no claim is made about where.

**The clamp is free, and that is the interesting one.** `v.clamp(0, qa)` costs four instructions
per element here, because there is no elementwise min or max in the lane API — two compares and two
selects. Timed against the same kernel with the clamp removed: 6.50 us versus 6.47 us, a 0.5%
difference where either side wobbles by 3.5%. **Not measurable.** The kernel is waiting on memory,
not on arithmetic, so an elementwise `min`/`max` would buy nothing here — worth knowing before
building one.

The honest framing of all of it: one whole GPU against one CPU thread, on a workload the engine's
own `SPEED.md` says is ~20% of search time. Free evaluation caps the win near 25%, and the engine
already tried explicit CPU SIMD on these same kernels and measured a wash. **Nothing here argues
the engine should change.** It argues that `simdr` can be pointed at a real workload and produce
numbers that survive being looked at.

## The paperwork drifted, and the gate said green

**Three hand-maintained lists had gone stale, 2026-08-11.** Found by asking what to work on next
rather than by any test:

- `noha.yaml` listed 33 of 38 sources. Five files were never mutated — including
  `src/lanes/branch.rs`, the phi and block-tracking code, the most dangerous thing in the tree.
  "100% mutation coverage" was a true statement about a list that excluded it. Corrected: 75/75
  over all 38, still 100%, so the hand-written tests had been holding.
- `decisions/DR-0001` said `spirv-val` "is not yet installed and is the next real oracle this
  project needs". It had been installed and running at fifteen call sites for some time.
- `decisions/DR-0002` said strip mining "is not built" and named an error to prove it. Strip mining
  had been built for weeks and that error never existed under that name.

`noha gate` printed a tick beside all three decision records throughout — its check reads a
record's front matter, not its claims, and reports "prose-only: recorded, not machine-checked"
followed by a green light. A fail-open in the apparatus whose job is catching fail-open.

`tests/integrity.rs` is the fix and it is deliberately in the emitter's own suite rather than in the
tool's configuration: a check that lives inside the thing it guards cannot be skipped by not running
the tool. It compares the source list against the tree in both directions, and extracts every
`Thing::member` written in backticks under `decisions/` and fails when `src/` no longer defines one.

**The convention that makes the second half work: backticks are the claim.** Code spelled as code
asserts this crate defines it, and is checked. A dead name being discussed in prose is not. So a
retraction can name what it retracts without the check mistaking the obituary for a promise.

## The fuzzer was proving the wrong surface

**Its vocabulary predated three passes of emitter work.** Straight-line programs over one vector:
six elementwise and shuffle operations, two finishes, two domains. Loops, `choose_uniform`, the
block tracking, `load_offset`, `reduce_min` and the whole `i32` path had arrived since and none of
it was generated. "30 000 programs, zero disagreements" was true and was about the code of some
weeks earlier.

Extended with `Op::RepeatAdd` and `Op::RolledAdd` — the same arithmetic through an unrolled loop and
a real four-block one, so the pair must agree while only one has a back edge —
`Op::RolledCounterAdd`, `Finish::Min`, `Finish::SumOrMax` (which carries a value out of a branch
through an `OpPhi`), and `Domain::Signed`.

**It found a real bug on the first run.** `reduce_min` folded its strips with a *maximum*:

```rust
let (left, right) = match extreme {
    Extreme::Max => (partial, next),
    Extreme::Min => (next, partial),      // swapped the operands...
};
let takes_left = binary(GREATER_THAN, boolean, left, right);
partial = select(element, takes_left, left, right);   // ...and swapped which is kept
```

The two swaps cancel. Both ends folded to the maximum. It was invisible because the strip fold only
runs when a vector is wider than the subgroup — with one strip the loop does not execute and the
group instruction does the right thing — and because every hand-written test splatted *one* value
across both strips, so the wrong end returned the same number.

The fix is one comparison, always the same way round, with the extreme deciding which arm the select
keeps. The regression test builds two strips holding **different** constants, which is the property
the old tests lacked.

Then: 18 000 programs across `u32`, `i32` and `f32`, zero disagreements.

**What to take from it.** Three tests over `reduce_min` passed for weeks. What they had in common
was a splat — the same value in every position — which makes an entire class of ordering and
selection bug unobservable. A test whose inputs are all equal is testing that something runs.

## Six operations that had never run

**A coverage sweep across the three evidence layers, 2026-08-12.** Counting mentions of each lane
operation in `src/` (unit), `runner/tests` and `runner/src/kernels` (executed), and
`runner/src/fuzz` (differential) produced this:

| op | unit | gpu | fuzz |
| --- | --- | --- | --- |
| `prefix_sum` | 8 | **0** | **0** |
| `ballot` | 19 | **0** | **0** |
| `shift_down` | 9 | **0** | **0** |
| `broadcast` | 3 | **0** | **0** |
| `all_uniform` | 1 | **0** | **0** |
| `reduce_min` | 7 | **0** | 1 |

Nineteen unit tests over `ballot` and not one dispatch. A unit test here decodes the module and
agrees that the emitter emitted what the test expected — a check on one author's understanding
against itself. `reduce_min` passed seven of them while folding its strips with a maximum.

All six now run. None of them was wrong, which is worth recording precisely because it was not
knowable beforehand. The one most likely to have been: `prefix_sum` is an *inclusive* scan, an
exclusive one is the same instruction with a different `GroupOperation`, they differ by exactly one
element, and every opcode-counting test passes for either.

`Limits` gained `subgroup_ballot` on the way. A kernel using the votes declares
`GroupNonUniformBallot`, and a *surplus* capability fails at pipeline creation rather than at
validation — so the skip is real rather than cautious.

## Control flow that nests

**Neither nesting had ever been built.** A branch inside a loop makes the loop's own bookkeeping —
the copy into the phi's promised name, then the branch to the continue target — land in the
*selection's* merge block rather than in the body block the loop opened. A loop inside a branch
makes the selection's `OpPhi` name the *loop's* merge block rather than the arm's.

Both validate and both compute correctly, which is what `Module::current_block` was built for and
had never been asked to prove.

## Converting a `u32`, and why it is not a bitcast

**`OpConvertUToF = 112`, `OpBitcast = 124`**, both read out of `spirv-as` rather than recalled.

The motivation was concrete: `repeat_rolled` hands its body a `u32` counter, so the fuzzer could
only generate `RolledCounterAdd` in the unsigned domain. `Element::FROM_U32` closes it —
`OpConvertUToF` for a float, `OpBitcast` for `i32` where the widths are equal, and `OpCopyObject`
for `u32` itself so there is no special case and no `Option` for every call site to test.

**It has teeth.** Setting `F32::FROM_U32` to `OpBitcast` and re-running: the fuzzer disagrees at
seed 1 index 0, and the unit test fails too. Iteration 3 read as float bits is a denormal near
zero, so the loop would add nothing and look like a numerical problem rather than a wrong opcode.

## More than two buffers

The emitter's `Shape` always took a buffer count; the runner had two hardcoded. That is why the
NNUE layer had to be handed weights and activations concatenated into one buffer with the join
passed as an offset — which works and is not the shape of the problem.

`Gpu::run_bound` binds one buffer per input plus an output, sized individually. The split form of
the layer agrees with the concatenated one exactly, which is a better check than either against a
reference: two routes, one answer.

## The placement hypothesis is dead too

`probe_memory` carried this in its own documentation: *"It answers for one buffer. A run holds
three of that size... That gap is real and not yet closed."* `Gpu::probe_resident` closes it by
holding all of them at once.

```
      each    all three   one resident three resident    of heap
     56 MB       168 MB   device-local   device-local       1.0%
    256 MB       768 MB   device-local   device-local       4.6%
   1024 MB      3072 MB   device-local   device-local      18.3%
```

Device-local everywhere, up to 3 GB resident. **A third explanation for the large-working-set cliff
is refuted.** L2 capacity, eviction of a single allocation, and now placement under simultaneous
allocations. The project still has no performance claim about large working sets and now has three
dead hypotheses rather than two — which is progress of the only kind available here.

## The night of the five fixes — 2026-08-12

A research pass measured five things worth fixing and then fixed them. Two of the five turned out
to be much larger than they looked from the outside.

### The staging buffer was asking for the wrong memory

`Buffer::staging` asked for `HOST_VISIBLE | HOST_COHERENT`, and `memory_type` returns the **first**
type that satisfies a request. On an RTX 4080 that is index 2 — visible, coherent, *not cached* —
while index 3 offers all three.

Host-visible memory without `HOST_CACHED` is write-combined. Sequential writes into it coalesce and
go at full speed; every **read** is an uncached fetch with no prefetching and no line reuse. And
`Buffer::read` memcpys out of exactly such a mapping on the way home from every single dispatch.

```
                before       after
   64 MB      188 ms       21 ms       8.8x
 transfer    357 MB/s   3138 MB/s
```

One flag. `Buffer::preferring` now takes a required set and a preferred set, and the two-pass
lookup falls back so no device is narrowed out.

**Asking for the flags you want and taking the first match is the obvious thing to write, and it
silently picks the wrong memory whenever a better type sits later in the list.**

### Almost all of a small call was setup, and now there is a way not to pay it

`examples/overhead.rs` subtracts twice: the device clock gives the dispatch, the host clock around
`Gpu::run` gives everything, and varying the buffer size separates fixed cost from per-byte cost.

```
  buffer   round trip     dispatch     overhead
   256 B      807 us        2.5 us       805 us
  allocate + free, any size: ~310 us  ->  three per run
  the same dispatch, amortised over a thousand: 0.8 us
```

So better than 99% of a small call is allocation and pipeline creation. `Gpu::session` holds them:
**52x faster per dispatch** than rebuilding everything, measured in `runner/tests/session.rs` with
a deliberately loose assertion — pinning the ratio would make it a test of whichever machine ran
it.

### The mutation gate had never run most of the suite

This is the one that matters most, and it was found by chasing a flake.

Adding `runner/`'s pure half to the mutation sources produced survivors that changed between runs.
Three of them were applied by hand and **all three were killed instantly by the suite**. The cause
was not contention:

`H:\simdr` is a root package with a member, so plain `cargo test` runs **six suites of nineteen**.
`noha.yaml` said `test: cargo test --quiet`. The whole execution and fuzzing layer had never been
in the kill set, and neither had any `src/` mutant that only a dispatch would catch. The score read
100% over the tests that happened to be in scope.

Every command in `noha.yaml` names `--workspace` now.

**A survivor is only a finding if it reproduces.** That is written into the config, because two of
the three chased that night were innocent.

### Two survivors that reproduced were equivalent mutants

`strips_of` was `if lanes > subgroup { lanes / subgroup } else { 1 }`. At `lanes == subgroup` both
arms give one, so flipping `>` to `>=` changes nothing and no test can ever kill it. Same for
`group_size` with `min`. Both are now written without the comparison — `(lanes / subgroup).max(1)`
and `lanes.min(subgroup)` — which is simpler *and* removes a survivor that could only ever waste
somebody's evening.

A survivor sitting on a branch that cannot be got wrong is pointing at a branch that should not
exist.

### One survivor that reproduced was real, and it was about the fuzzer itself

`z ^ (z >> 31)` in the generator's finaliser, mutated to `&`. An AND biases the random stream hard
toward zero, and the whole suite stayed green — because nothing anywhere asserted that the fuzzer
*explores*. A fuzzer that generates the same three programs forever still reports thousands of
agreements.

Three tests now: every operation reached across 512 seeds, every finish reached, and the top three
bits of the stream landing in all eight buckets. The mutant dies on the first.

### The workgroup boundary is gone

`OpTypeArray = 28`, `OpControlBarrier = 224`, `OpMemoryBarrier = 225`, and
`MemorySemantics::AcquireReleaseWorkgroup = 264` — all read out of `spirv-as` and confirmed by
`spirv-val`, per DR-0001.

`Gpu::sum` used to end with two floats coming home because nothing could combine two subgroups on
the device. `kernels::workgroup_sum` does it now:

```text
  total = reduce_sum(value)     each lane holds its own subgroup total
  shared[local] = total         every invocation writes a different slot
  barrier                       reached by all of them
  answer = shared[0] + shared[w] + ...   constant indices, one per subgroup
```

The final reads are at build-time constants, so every invocation runs identical instructions and
none diverges. `Reduction::tail` is gone; `Reduction::host_combined` reports `1`, meaning none.

**An unplanned consequence, recorded in DR-0003.** A barrier must be reached by every invocation,
and a caller who cannot write a divergent branch cannot easily write a barrier some lanes miss. The
rule adopted to keep subgroup reductions meaningful also makes the one piece of workgroup
synchronisation hard to misuse. It paid for itself twice.

### Six operations that had never run, and two nestings never built

A coverage sweep across the three evidence layers found `prefix_sum`, `ballot`, `shift_down`,
`broadcast`, `all_uniform` and `reduce_min` with unit tests and no dispatch — nineteen unit tests
over `ballot` alone. All six run now and none was wrong, which was not knowable beforehand: an
exclusive scan is the same instruction with a different `GroupOperation` and every opcode-counting
test passes for either.

A branch inside a loop and a loop inside a branch had never been built either. Both validate and
both compute correctly, which is what `Module::current_block` was for and had never been asked to
prove.

### And a CLI, of exactly one subcommand

`simdr probe`. DR-0002 makes the subgroup width an argument so nobody can forget it exists, which
left no way to *ask* what it is without writing a program. It also reports the subgroup features —
a surplus capability fails at pipeline creation rather than at validation — and the memory types,
because that is where the 8x was hiding and looking is now one command.

Nothing else. `simdr validate` is three lines around `spirv-val` and belongs in the README;
`simdr emit` would need a kernel description language, which is a second and worse API beside one
whose entire value is that kernels are Rust with types.
### And the one that mattered most: a comment that stopped being true

`Buffer::write` did not check that the caller's slice fit. Its safety comment explained why it did
not need to:

> the caller's slice is no longer than that because this crate always allocates from the same
> element count it writes

That was **true when it was written**. `Gpu::run` allocates `input.len()` and writes `input`; the
invariant held at every call site because there was one.

`Session` broke it, and the change that broke it did not touch this file. A session's staging
buffer is sized to its *largest* binding, and `Session::write(index, words)` takes a slice from
outside. A caller writing more words than the largest binding holds would have memcpyd past the end
of a mapping — **from safe code, in a crate whose entire claim is that it cannot** — and
`Buffer::read` had the same shape in the worse direction, handing back whatever was next in the
address space.

Nothing caught it. Not clippy, not the mutation tester, not 353 tests, not the fuzzer. It was found
by re-reading a `SAFETY` comment while checking something unrelated.

Both are bounded now, in `Buffer` rather than at the call sites, and the error says both numbers.

**The lesson is about the comment, not the bug.** A safety argument that names *the current set of
callers* has an expiry date and does not say when. "No caller passes more than fits" is a claim
about today; "the length is checked here" is a claim about the code. The first kind should be
treated as a bug waiting for a second caller — and this project now has one data point on how long
that takes: about six hours.
And a second one in the same file, found by asking the same question the other way round.
`Session::write` clamped the copy to the binding's size — so writing 500 words into a 64-word
binding wrote 64 and **reported success**. That is not a safety bug; it is worse in the way this
project cares about, because a short write is a wrong answer arriving later rather than a crash
arriving now, and refusing rather than truncating is a rule everywhere else here.

Both bounds are now against the *binding's* own capacity rather than the staging buffer's, which
also let a parallel `sizes: Vec<u64>` field go: each `Buffer` knows how large it is, and a second
list describing the first is a thing that can drift.
### Two more the extended mutation gate found, both of the same shape

Once the gate was actually running the whole suite, it kept earning its keep.

**`1 << rng.below(4)` mutated to `>>`.** The butterfly mask becomes 0 or 1 forever instead of
1, 2, 4, 8 — and a butterfly of distance zero pairs a lane with itself, which is a perfectly valid
program the reference agrees with. Everything stayed green while the fuzzer quietly stopped
exercising the shuffle.

**Reaching every operation is not the same as reaching every operation's operands.** The test added
the night before checked which `Op` variants appear across 512 seeds; it said nothing about what
was *inside* them. There is now one for the distances too.

**`if best <= 0.0` in `Timing::spread` mutated to `<`.** The denominator. A `best` of exactly zero
would then divide, giving `inf` — so `is_steady` would report the least informative measurement
there is as the loudest possible warning. Nothing covered a zero duration; `Timing::of(&[ZERO])` is
one line to construct.

Both killed by the tests that now exist. **Score after: what the gate reports is finally over the
suite it claims to be over**, which is the thing that changed tonight — the number itself moved
much less than what it means.
### And two more, one of which was the whole point

The gate kept going.

**`groups: 1 + rng.below(2)` mutated to `1 -`.** Half of every fuzz run then dispatches **zero
workgroups**. The kernel computes nothing, the reference computes nothing, and two empty answers
agree — so the sweep counts the round as checked and reports success. Half the fuzzing was testing
nothing at all and the number went up.

That is the third gap of one family, and each had to be closed separately:

1. **The fuzzer must explore** — a degraded RNG generated the same few programs forever.
2. **It must reach its operands** — every `Op` appeared, but the butterfly only ever at distance 0.
3. **It must actually run** — a program that dispatches nothing agrees with everything.

Passing the first two says nothing about the third. Whatever a fuzzer's coverage claim rests on has
to be *asserted*, not inferred from the fact that it produced a large number of agreements.

**The `workgroup_sum` guard, replaced by `false`.** It refuses a subgroup the workgroup is not a
whole number of. Every caller passes a real device's width, so the refusal was unreachable and read
as a safeguard while guarding nothing. Three tests reach it now, including the zero that would
divide.
### Three more, and the shape of the whole exercise

**`lanes < subgroup` mutated to `<=`.** A full-width vector then counts as clustered, so the
generator stops offering it shuffles and votes — on the one mapping where those matter most. The
existing test asked that *narrow* programs have none; nothing asked that wide ones have some.

**A rule worth testing in one direction is usually worth testing in both.**

**`half / WORKGROUP_SIZE` mutated to `half *`.** Every fold of `Gpu::sum` then dispatches four
thousand times too many workgroups — and still returns the right answer, because the surplus
invocations write past the buffer and are discarded. Only the wall clock notices, and no test
watches it. `reduction::folds` now returns the plan and three tests pin it.

**A wrong dispatch size is invisible in the answer and enormous in the cost.** Anything with that
shape has to be pinned rather than inferred from the result being right.

**`unwrap_or(false)` mutated to `true`** on a lookup indexed by `invocation / width`, where the
table is built with exactly `div_ceil(width)` entries. The default could not be reached, so no test
could ever kill it. Rewritten with `chunks`, which makes the grouping structural and removes the
branch entirely — the third time tonight that an unkillable survivor was pointing at a detour.

## The tally, for whoever reads this next

Twelve mutants reproduced in one night of running the mutation gate properly — **eight real gaps
and four equivalent ones**. Counted rather than estimated, because the first version of this
paragraph said "ten" from memory and was wrong:

| what | where | kind |
| --- | --- | --- |
| the disagreement index nothing checked | `fuzz/mod.rs` | real |
| that the generator explores at all | `fuzz/generate` | real |
| which butterfly distances it reaches | `fuzz/generate` | real |
| that a program dispatches anything | `fuzz/generate` | real |
| the clustered boundary, in both directions | `fuzz/generate` | real |
| `Timing::spread` with a best of zero | `timing.rs` | real |
| the `workgroup_sum` guard, unreachable | `kernels/reduce.rs` | real |
| workgroups per fold, invisible in the answer | `reduction.rs` | real |
| `strips_of`'s comparison | `fuzz/interpret` | equivalent |
| `group_size`'s comparison | `fuzz/interpret` | equivalent |
| an `unwrap_or` on an impossible index | `fuzz/interpret` | equivalent |
| the `ELEMENTWISE + 1` match guard | `fuzz/generate` | equivalent |

**Nine of the twelve are in the fuzzer or its reference. None is in the emitter.**

That is not luck. The emitter has four other layers watching it and the checking machinery had
none, which is the ordinary way a test suite rots: whatever is furthest from the thing under test
is furthest from anybody looking.
### The RNG needed a golden test, not a distribution test

Four mutations of `SplitMix64`'s inner rounds survived every distribution check written for it —
including the one added earlier the same night. The algorithm is good enough that mangling one
round still spreads three bits over eight thousand draws. **A distribution test catches a
*collapsed* generator. It cannot catch a *different* one.**

And the distribution test was looking at the wrong end of the word. It bucketed `next() >> 61`, the
top three bits, while every consumer goes through `below(n)` — which is `next() % n`, a modulus,
which reads the **low** bits. It was checking bits nobody uses.

Both fixed. The low bits and `below(5)` are checked now, and above them sits a golden test:

```rust
assert_eq!(drawn, vec![0xE220_A839_7B1D_CDAF, 0x6E78_9E6A_A1B9_65F4, ...]);
```

A golden test is usually a smell and is right here for two reasons. **SplitMix64 is a specified
algorithm** with published constants and a published sequence, so pinning it states something true
rather than freezing an arbitrary internal choice — and those four values were computed from the
published formula in a separate implementation, agreeing with the reference, neither read off the
Rust source. **And the seeds mean something**: this file records which seed found which bug, so
changing the generator silently turns every one of them into a number pointing at a different
program. That should be a failing test.

It kills all four, plus the gamma constant's sign.

### A fifth equivalent mutant, left in place with its reasoning

`_ => vec![false; invocations]` in the reference's vote table — never read, because the only arm
that consults it is the one it is not built for. Flipping it changes nothing.

Left as it is, with a comment saying so. The alternatives are an index-and-default (the shape it
replaced, whose default was *also* unreachable) or nesting two different chunk sizes inside each
other in the function every other layer is checked against. Neither is worth trading a comment for,
and pretending a comment is a fix would be worse than saying which it is.

## The second device — 2026-08-12

**There were two GPUs in this machine the whole time.** `notes/NEXT.md` had "run on a 64-wide
subgroup" filed under *needs AMD hardware*, and `vulkaninfo` says the machine holds an RTX 4080 at
width 32 and an integrated `AMD Radeon(TM) Graphics` at width **64**. The runner picked the
discrete one and nothing had ever asked for the other.

The cost of finding out was one `SIMDR_DEVICE` environment variable. The cost of not having found
out was that DR-0002 — the record the whole lane API is shaped around — had never been tested.

**Ten tests failed at width 64. The emitter was right in every one of them.**

| what failed | why |
| --- | --- |
| four control-flow tests, at *build* time | they asked for a vote on a `Simd<_, 32>`, which is a cluster of a 64-wide subgroup, and votes have no clustered form — the lane API refused by name, correctly |
| `dot.rs`, `narrow.rs` | the reference grouped by the device's width while the kernel reduced 32 lanes |
| two `loops.rs`, one `execution.rs` | a discriminator asserting "the two subgroups disagree" in a workgroup that holds one subgroup at width 64 |
| `lanes.rs` | 512 lanes is sixteen strips at width 32 and eight at width 64, so a count asserted to be refused is accepted |
| `session.rs` | a held pipeline is 52× faster than rebuilding one on the 4080 and **5×** on the integrated part; the test asserted ten |
| `fuzzing.rs` teeth test | one fixed seed, whose generated program at width 64 does not depend on the element the test perturbs |

Every one of them is the same mistake wearing a different hat: **a test that takes the width as a
parameter and then writes `32` further down**. It reads as though it adapts. On a machine with one
device, those are the same number.

The habit that would have caught it earlier: when a test takes a parameter, check that changing it
changes the test. `let width = limits.subgroup_size` followed by an expectation containing a
literal `32` is the shape to grep for.

### And what a second device does not prove

The suite is green at 32 and at 64 and has never run at 4, 8 or 16 — which is what a software
implementation reports. `whole_subgroup!` lists two widths because two is what exists here.

## Narrow element types — the prediction was wrong twice — 2026-08-12

`notes/NEXT.md` argued for `i8` and `i16` on the grounds that a quarter of the bytes should be
close to a quarter of the time. Measured, `runner/examples/narrow.rs`, RTX 4080, a clamp over
16 777 216 elements:

- `Simd<i8, 32>` — one element per lane — is **1.67×** an `i32` kernel, not 4×. It runs at 264
  GB/s against the `i32` kernel's 630. An invocation that loads one byte costs what one that loads
  a word costs, so a byte-per-lane kernel leaves three quarters of the rate unused.
- `Simd<i8, 128>` — four strips, so each invocation moves a full word — is **6.45×**, at 1016 GB/s.
- At 1 048 576 elements all three widths take the same 9 µs, because nothing is bandwidth-bound
  there and the narrow types buy exactly nothing.

Two honest qualifications on the 6.45×. Part of it is cache residency rather than bytes: at that
size the `i32` buffers are 64 MB each and land in the unsteady regime past ~50 MB, while the `i8`
ones do not. And a clamp is one instruction per element, which is as close to pure memory as this
crate can arrange — a kernel doing real arithmetic would see less.

The useful conclusion is the one `decisions/DR-0004` rests on: **strip mining is what makes a
narrow type pay**, and strip mining is a mapping that already existed. The packed mapping the
record declines to build would have bought what `Simd<i8, 128>` already buys.

## A capability the module cannot declare — 2026-08-12

`shaderSubgroupExtendedTypes` gates whether the subgroup instructions accept 8- and 16-bit types,
and **there is no SPIR-V capability for it**. A module reducing over `i8` is byte-for-byte what it
would be on a device that supports it everywhere: `spirv-val` accepts it, and a device without the
feature refuses the *pipeline*.

Which means the validator is strictly weaker than usual here, and `runner/tests/narrow.rs` is the
only layer that can tell. Worth remembering the next time a feature looks like it is covered
because the module validates.

## `ClusterSize` can be deferred, and DR-0002 was right for the wrong reason — 2026-08-12

DR-0002 said the mapping cannot be deferred to the device because `ClusterSize` is a compile-time
operand. A specialization constant **is** a constant instruction, so that reason was wrong: the
validator accepts an `OpSpecConstant` in the `ClusterSize` slot and the 4080 runs one module at
cluster sizes 4, 8 and 16.

The decision survives on the argument it should have been given in the first place: the three
mappings are three *instruction sequences*, and no value arriving at pipeline time can add
instructions that were never emitted. `decisions/DR-0005` has the long form; DR-0002 carries the
correction in place.

**Testing a record's stated reason is worth doing even when the record's conclusion is right.**
This one had been quoted in the README, in the crate docs and in two module headers, all of them
repeating a justification that does not hold.

## The extremes' fold stayed as it was — 2026-08-12

Importing GLSL.std.450 made `FMax` available, and the strip fold in `reduce_min`/`reduce_max` still
uses a comparison and a select. One instruction per extra strip is on the table and was left there.

Compare-and-select is **defined** for NaN: an ordered comparison against one is false, so the fold
keeps the other operand by a rule in the specification. `FMax` with a NaN is explicitly undefined —
"which operand is the result is undefined". This machine returns the non-NaN operand either way
round (`runner/tests/extended.rs` observes both orders), so the two agree here.

Agreeing on one device is not the same claim as being defined, and `notes/NEXT.md` had already
measured this fold as buying no time. A defined behaviour traded for an undefined one, for an
instruction that does not show up in a timing, is the wrong side of the trade.


## Superseded: specialization constants save 1% — 2026-08-12

> **Every figure in this section is wrong, twice over, and it is kept for what it cost.** The
> probe behind it allocated two buffers on every call, so the pipeline column below is pipeline
> creation plus two allocations — *A probe that measured the wrong thing*, later the same day,
> has the corrected 57.8 µs and 809.6 µs and the revised conclusion of 9.7%. Five standalone runs
> on 2026-08-26 then put the specialized column an order of magnitude *above* the un-specialized
> one, inverting the sign as well; `decisions/DR-0005` carries those and records that the
> instrument does not reproduce between harnesses.

`notes/NEXT.md` ranked "make `Gpu::sum` use a specialization constant" first, on the grounds that
fourteen modules for fourteen fold sizes is expensive in pipeline creation.
`runner/examples/specialize.rs` measured it. RTX 4080, a reduction over 2²⁰ elements:

| | all fourteen folds | per fold |
| --- | --- | --- |
| emitting the modules | 71.5 µs | 5.1 µs |
| a pipeline each, from fourteen modules | 6796.8 µs | 485.5 µs |
| a pipeline each, from one specialized module | 6726.0 µs | 480.4 µs |

Emission is 1.1% of the total, and emission is all a specialization constant can remove — it is
fixed *at* pipeline creation, so fourteen values still need fourteen pipelines and fourteen shader
compilations.

**One module per parameter value is cheap. One pipeline per parameter value is not.** The refactor
was not done, and the reason is in `decisions/DR-0005` with the table.

What the measurement does point at: 485 µs per pipeline against 0.8 µs per dispatch, and `Gpu::sum`
builds all fourteen on every call and throws them away.

## Lavapipe, at a subgroup width of 8 — 2026-08-12

Mesa's software Vulkan reports **subgroup width 8** and runs on the CPU. Installed at
`H:\tools\mesa\msvc` from the pal1000/mesa-dist-win **msvc** release; the mingw build's
`vulkan_lvp.dll` would not load here (`error 126`, a missing dependency) and the msvc one loads
with no extra files at all.

It found one defect in the checking machinery and eight in the tests.

**The fuzzer generated shuffles that leave the subgroup.** `1 << below(4)` gives distances 1, 2, 4
and 8. All four are inside a 32- or 64-wide subgroup; 8 is the *width* of an 8-wide one, and
`OpGroupNonUniformShuffleXor` past the last lane is undefined. The CPU reference computed
`lane ^ mask` and read the next subgroup's invocation — a defined answer to a different question —
so the fuzzer reported a disagreement it had caused itself. Seed 3, in every domain at once.

The generator now draws from `log2(width)` distances. That the bug needed a *narrow* subgroup to
appear is the point: two devices agreed for two months about something neither could disprove.

**Three tests assumed uninitialised device memory is zero.** The histogram kernels accumulate into
their output buffer, and `Gpu::run` allocates that buffer without initialising it. Vulkan says
nothing about what is in a fresh allocation. Two drivers hand back zeros; lavapipe does not. The
tests zero it through a `Session` now — the only path that can write a binding the kernel also
reads — and the empty-kernel test no longer asserts anything about the untouched buffer.

**And `whole_subgroup!` listed two widths**, so every kernel using it refused to build with
`BadWidth` on a device that could have run them. A list of the widths that exist is a list that
needs revisiting every time a new device appears; it holds 4, 8, 16, 32 and 64 now.

### Lavapipe's `Fma` rounds twice

`OpExtInst Fma` on both hardware devices agrees with the host's `f32::mul_add` to the bit. On
lavapipe it agrees with `x * x + x` instead — two roundings rather than one. SPIR-V says `Fma`
computes `a * b + c` as a single operation.

`runner/tests/extended.rs` observes which of the two it gets and asserts only that it is one of
them. Pinning either would turn one implementation's behaviour into this suite's regression, and
the difference is real and worth knowing about rather than worth failing over.

## The narrow types are fuzzed now — 2026-08-12

`i8`, `u8`, `i16` and `u16` are domains in the differential fuzzer. Their arithmetic wraps at 8 or
16 bits, wrapping is defined, and the reference wraps identically — so the exactness the fuzzer
needs comes for free and what is checked is instruction selection: `OpSConvert` against
`OpUConvert`, `SMax` against `UMax`, and a buffer whose stride is one byte.

3 000 rounds per domain on each GPU and 1 500 on lavapipe, no disagreements.

`Domain` got **smaller**. Seven domains times eight operations is fifty-six match arms if each is
written out; writing them in terms of `bits()` makes `add` one wrapping add and a mask, and the
file went from 249 lines to 230 with four more domains in it.

**`f16` is not fuzzed and that is a decision.** A half is exact for integers only to 2048, and a
sum over sixty-four lanes leaves that range at once — the argument the float domain rests on does
not hold, and a tolerance would be checking something other than the emitter.


## A probe that measured the wrong thing — 2026-08-12

`Gpu::probe_pipeline` was written to time pipeline creation on its own, and it allocated two
device-local buffers on every call because a descriptor set needs something to point at. So what it
reported was **pipeline creation plus two allocations**, and allocation is the larger half:

| | reported | actual |
| --- | --- | --- |
| a pipeline, per fold | 485.5 µs | 57.8 µs |
| fourteen of them | 6796.8 µs | 809.6 µs |

That wrong number reached `decisions/DR-0005`, `notes/NEXT.md`, this file and a commit message,
where it read as "specializing saves 1.0%". The corrected figure is 9.7%.

**What caught it was a second measurement that did not add up.** `runner/examples/reducer.rs` timed
a whole fourteen-fold reduction — three buffer allocations, fourteen pipelines, the host copies,
fifteen dispatches and a readback — at **3.1 ms**. Fourteen pipelines at 485 µs would have been
6.8 ms on their own. A part cannot cost more than the whole that contains it.

Two things worth keeping from that:

- **A probe has to be told what it is excluding.** `probe_resident` isolates allocation and says so
  in its name; `probe_pipeline` claimed to isolate pipelines and quietly included two of what
  `probe_resident` measures. It takes a batch now and allocates once, so the per-pipeline number is
  a per-pipeline number.
- **The arithmetic between two measurements is a test.** Nothing in the suite could have caught
  this — both numbers were produced by code that ran correctly. What caught it was one number being
  larger than another number it is a part of, and noticing that requires writing both down.

The conclusion did not move: a specialization constant removes the emission and not the pipeline,
so it is worth single digits where holding the pipelines is worth 5×. Only the size of the claim
was wrong, and it was wrong by a factor of eight.

## Superseded: holding a reduction's pipelines is worth 5× — 2026-08-12

> **The ratios and the fold counts below both moved, and a code change is why.** The chain folded
> by two when this was taken — eight folds at 8 192 elements and fifteen at 2²⁰ — and folds by
> sixteen now, which is three and five. `runner/examples/reducer.rs` on 2026-08-26 measures the
> same comparison at **11.4×** and **9.2×** against the 5.0× and 1.6× here. The finding this
> section is kept for is the one in its last paragraph, about ownership, which did not change.

`Gpu::sum` builds a pipeline per fold on every call and destroys them all. `Gpu::reducer(elements)`
builds them once and keeps them, along with the three buffers they are bound to.

| elements | folds | `Gpu::sum` | `Reducer::sum` | faster |
| --- | --- | --- | --- | --- |
| 8 192 | 8 | 967.0 µs | 191.7 µs | 5.0× |
| 1 048 576 | 15 | 3069.6 µs | 1941.2 µs | 1.6× |

The absolute saving is roughly constant — 775 µs and 1128 µs — which is what a *per-call* cost
looks like, and it is why the ratio falls as the arithmetic grows. Both columns run identical
dispatches over identical data.

The ownership is the part that needed care rather than the caching. A pipeline holds a descriptor
set and a descriptor set points at particular buffers, so pipelines and buffers have to be owned by
the same object and released in that order. `Session` had already established the shape; this is
the same trade for a chain of pipelines rather than one.


## Splitting a file moved 200 lines into the mutation gate — 2026-08-12

`runner/src/dispatch.rs` did two jobs: the staging machinery — three buffers, three submissions, a
fence — and the surface over it, one call per way a caller might spell its data. The whole file was
excused from mutation as FFI, which was true of the machinery and not of the surface.

Splitting it along that seam made the excuse fail. `tests/integrity.rs` checks that every excused
file still contains `unsafe`, and `dispatch/run.rs` contains none: it converts `f32`, `u32`, bytes
and halves into words and calls `execute`. So it is mutated now rather than excused, and the
packing that `run_bytes` and `run_halves` do — four elements to a word, little-endian, truncated
back to the caller's length — is inside the gate for the first time.

The check that caught it was written months earlier for a different reason: an excuse that names
FFI should expire when the file stops being FFI. It had never fired. **A file split is exactly the
event that makes a blanket exemption wrong**, and nothing but that check would have noticed.


## The integer dot product is worth nine times, or nothing — 2026-08-12

`OpSDot` computes four 8-bit products and their sum in one instruction. Written out that is eleven:
four shifts up, four bitcasts, four shifts down, four multiplies and three adds — and
`runner/tests/dot_product.rs` runs both spellings and checks they agree, against each other and
against a host reference.

`runner/examples/dot.rs`, 262 144 invocations:

| kernel | RTX 4080 | integrated Radeon |
| --- | --- | --- |
| one dot product per element | 1.00× | 1.52× |
| thirty-two per element | 1.18× | **9.08×** |

Both devices report `integerDotProduct4x8BitPackedSignedAccelerated`. **The flag says the hardware
has the instruction; it does not say anyone will notice.** The discrete part has enough integer
throughput that nineteen instructions cost nearly what one does, and at one dot product per element
it is memory-bound anyway.

Two things worth carrying:

- **The first version of the benchmark measured the wrong thing**, and looked fine doing it: one
  dot product per element is a *memory-bound* kernel, so it reported 1.00× on the 4080 and would
  have been written up as "the instruction does nothing". The second shape — thirty-two dot
  products per element, salted so a driver cannot fold them — is what has the arithmetic in it. A
  benchmark of an arithmetic instruction has to be arithmetic-bound, and saying so in the doc
  comment is not the same as being it.
- **Lavapipe supports the instruction and reports it not accelerated**, which is the case the flag
  exists to distinguish and is why it is reported rather than assumed. Three devices, three
  answers.

### It does not make a packed mapping

`decisions/DR-0004` declines to pack four `i8` into a lane, and this does not change that. A
`Simd<u32, N>` is still one `u32` per lane; `OpSDot` reads each of those `u32`s as four bytes. The
packing is a property of the *instruction's operands*, and the two readings of the same buffer —
`Simd<i8, N>` arithmetic and a packed dot product — coexist without either knowing about the other.


## A second axis costs nothing, and the thing next to it in the table costs 2× — 2026-08-12

Kernels can address `(row, column)` now: `simdr::kernel::Shape::grid`, `Kernel::load_row`, and a
`runner::Grid` that dispatches along y as well as x. `decisions/DR-0006` records why there is no
third axis.

The question worth asking was what the extra arithmetic costs. A grid address is
`row * pitch + column` where a linear one is just the column, so every access pays one multiply and
one add more. `runner/examples/plane.rs` runs the same elementwise kernel five ways over the same
elements, and the answer is **nothing**:

| | one axis | two axes |
| --- | --- | --- |
| RTX 4080, 131 072 invocations, workgroup 32 | 3.38 µs | 3.38 µs |
| RTX 4080, 131 072 invocations, workgroup 256 | 1.66 µs | 1.67 µs |
| integrated Radeon, 262 144 invocations, workgroup 64 | 42.99 µs | 42.92 µs |
| integrated Radeon, 262 144 invocations, workgroup 512 | 48.00 µs | 46.71 µs |

Lavapipe runs it too and its timings move by more between repeats than any of these differences, so
it is not in the table. A number that cannot separate the two cases is not evidence about them.

### The first version of the benchmark said 2×, and was measuring occupancy

The first table compared a one-axis kernel of `width` invocations per workgroup against a grid
**eight rows deep**, and the grid came out at 2×. It would have been written up as "the second axis
is faster", which is absurd on its face — an extra multiply does not make a kernel quicker — and
that is the only reason it got a second look.

A grid `rows` deep has `width * rows` invocations per workgroup. So the comparison moved the
address *and* the occupancy at once, and the occupancy was all of it. The example is a two-by-two
now: down a column is the address, across a row is the workgroup size.

**The habit this is the second instance of.** `notes/FINDINGS.md` already carries a measurement
that was wrong by eight because a probe allocated two buffers per call. Both were one variable
moving that nobody had listed. The check that caught both was the same: the number was better than
the mechanism could explain.

### And the thing it was confounded with is a real finding about `WORKGROUP_SIZE`

Eight subgroups per workgroup instead of one, at 131 072 invocations:

| | workgroup of one subgroup | workgroup of eight | |
| --- | --- | --- | --- |
| RTX 4080 | 3.38 µs | 1.66 µs | **2.04× faster** |
| integrated Radeon | 42.99 µs | 48.00 µs | 1.12× *slower* |

**It goes opposite ways on the two devices.** So there is no constant to change: `WORKGROUP_SIZE`
is 64 across `runner`, and 64 is one subgroup on the Radeon, two on the 4080 and eight on lavapipe.
A device-dependent number would have to come from the device, and nothing here measures it at
startup.

What this does say is that the workgroup size is worth more than most of what has been optimised
here, and that it has never been chosen — it has been 64 since the first kernel. One elementwise
kernel on two devices is not enough to pick a number.

> Swept properly the same day, over three kernel shapes and three devices — see **The workgroup
> size is worth 2.5×, or nothing** below. The answer is that the constant does not move, and the
> reason is stronger than "not enough data": the two devices want opposite things, and within one
> device it depends on whether the kernel is memory-bound.

### What the address arithmetic gained on the way past

Hoisting the row out of the strip loop showed that the *column* had never been hoisted either:
`Kernel::address` recomputed `group * workgroup * strips` once per strip, so a four-strip load
emitted four identical multiplies. Every driver folds those back into one, which is exactly why
nothing had caught it — the answer was right and the module said something the arithmetic does not.

It is one per access now, and `src/kernel/access.rs` has the test. Nothing measurable changed.

## The workgroup size is worth 2.5×, or nothing, and which one is not portable — 2026-08-12

`kernels::WORKGROUP_SIZE` has been 64 since the first kernel in this project and was never
measured. The item before this one turned up a 2× hiding behind it, so `runner/examples/occupancy.rs`
sweeps it: every column holds the invocation count, the element type, the lane mapping and the
total work fixed, and varies only how many subgroups share a workgroup.

`Gpu::limits()` reports `maxComputeWorkGroupInvocations` now, because a sweep needs to know where
to stop and asking for more fails at pipeline creation with no useful message.

### 262 144 invocations, best size per row

| device | shape | 1 subgroup | best | at | spread |
| --- | --- | --- | --- | --- | --- |
| RTX 4080 (width 32, ceiling 32 subgroups) | memory-bound | 5.35 µs | 2.13 µs | 16 | **2.51×** |
| | arithmetic-bound | 14.94 µs | 13.32 µs | 16 | 1.12× |
| | subgroup reduction | 5.34 µs | 2.14 µs | 8 | 2.49× |
| integrated Radeon (width 64, ceiling 16) | memory-bound | 40.61 µs | 40.61 µs | **1** | 1.07× |
| | arithmetic-bound | 1422 µs | 1422 µs | **1** | 1.29×, all of it at 16 |
| | subgroup reduction | 45.77 µs | 43.40 µs | 16 | 1.06× |
| lavapipe (width 8, on the CPU) | all three | — | — | — | no trend above the noise |

Four things fall out of it, and only the first was expected:

- **It is worth 2.5× on the 4080 and nothing on the Radeon.** The discrete part wants many
  subgroups per workgroup; the integrated part is already at its best with one, and the current
  constant of 64 *is* one subgroup there. So 64 is optimal on one device and 1.54× off on the other.
- **It is a memory-system effect, not an occupancy effect in general.** The arithmetic-bound kernel
  moves 1.12× across the whole range on the 4080 while the memory-bound one moves 2.51×. Whatever
  the larger workgroup is buying, it is buying it in the load path.
- **The largest size is never the best.** 32 subgroups is worse than 16 on the 4080 (2.13 to 2.48)
  and 16 is worse than 8 on the Radeon (1422 to 1828, 28%). There is a peak, and running to the
  device's ceiling walks past it.
- **Lavapipe has no trend at all.** Its repeats move by more than the differences between its
  columns, which is the honest reading of a CPU implementation: there is no occupancy to tune.

### So the constant does not move

The two devices want opposite things, and within one device the answer depends on whether the
kernel is memory-bound. There is no number to change 64 to.

What there is: `kernels::occupancy` holds kernels that take the size as an argument, `Gpu::limits()`
reports the ceiling, and this table says which sizes are worth trying. Wiring a heuristic to it
would need a device model this project does not have and cannot get from three data points.

### The arithmetic row was folded away twice before it was arithmetic

It first came out **identical to the elementwise row**, to the hundredth of a microsecond, on both
sizes and all six columns. A kernel doing 512 multiply-adds per element does not cost what one
multiply costs, so the loop was not running.

- **Attempt one** was `x * f + s` per iteration. That is affine, and `times` iterations of it
  compose into a single `x * f^times + c` — which the driver folded, correctly.
- **Attempt two** added `min(_, u32::MAX)` to break the affinity. `min(x, u32::MAX)` is the
  identity function, so it was deleted and the fold came back. This one is worse than the first
  mistake, because the fix was written *specifically* to prevent the fold and looked like it had.

What caught both was the same one-line check: run the loop at 64 iterations and at 512 and see
whether the number moves. It did not — 2.14 µs either way — and 2.14 µs was also what the
elementwise kernel cost.

The limit is `0x00FF_FFFF` now, low enough that it clamps within a few iterations, and 512
iterations cost 13.3 µs against 2.1. `runner/tests/occupancy.rs` asserts that the clamp actually
fires, because the property the benchmark depends on is not the answer — it is that the answer took
work to get.

**The general shape:** a benchmark whose result is *too good* is reporting that the work did not
happen. This is the third time in this file — a probe that measured allocation as pipeline
creation, a grid comparison that measured occupancy as addressing, and now a loop that was not
there. Each was caught by the number being better than the mechanism could explain.

## A test that cannot run from a clone, and that is the deal — 2026-08-12

`tests/integrity.rs` reads `noha.yaml`, compares its source list against the tree in both
directions, and fails when either has something the other does not. It has caught every file added
in the last four sittings.

It cannot run from a fresh clone, because `noha.yaml` is not in the repository and is not going to
be. This machine's **global** gitignore excludes `noha.yaml` and `.noha/` from every repository on
it, under the heading *"local verification toolchain — never commit, in any repository on this
machine"*.

Two things are worth carrying from that.

**A global exclusion is invisible from inside a working tree.** The file is present, `git status`
is clean, the test passes, and nothing distinguishes "tracked" from "there". `git ls-files <path>`
is the whole check, and a test that reads a file it does not own has a second failure mode beside
being wrong — being absent for the next reader.

**A `!` negation in a repository's own `.gitignore` outranks `core.excludesFile`, so the policy is
defeatable in one line — which is exactly why it should not be.** That line was written here and
then removed: the exclusion is deliberate, the consequence is a price the policy chooses to pay,
and the right response is to say so where a reader will look. `README.md` does, next to the
description of the gate.

The repository `.gitignore` had also been asserting the opposite for months —

> `.noha/baseline.tsv` and `.noha/tia.tsv` are *not* ignored

— which was never true, because the global rule had them the whole time. That comment is corrected;
the files are still not committed.
## Two survivors the batched gate never reported — 2026-08-12

The mutation gate is normally run scoped: `NOHA_ONLY` naming the files a piece of work touched.
Running it one file at a time turned up two survivors that the batched runs over the same files had
reported as 100%. Both are real, and neither is a wrong answer in the shipped code.

### `24 - byte * 8` → `24 + byte * 8`, in the written-out dot product

`square_of_byte` extracted packed byte `b` by shifting left `24 - 8b` and arithmetic-shifting right
24. The mutant gives shift counts of 24, **32, 40 and 48**.

SPIR-V leaves a shift at or past the operand's width undefined. This device masks the count to five
bits, so 32, 40 and 48 became 0, 8 and 16 — and the kernel read bytes **0, 3, 2, 1**. Every kernel
using the helper sums the *squares* of all four, and a sum is symmetric, so the answer was
identical. Two GPU kernels and a host reference all agreed, exactly, on the wrong bytes.

Nothing here could have caught it: not the twin-kernel comparison, not the host reference, not the
negative-byte discriminator. The fold that made the two spellings comparable is the same fold that
made the positions invisible.

`kernels::dot::byte_component` is the fix — a kernel that writes one position on its own, checked
against `simdr::lanes::signed_bytes` for each of the four. It kills the mutant, and it is the only
thing in the suite that says which byte is which.

**The general shape:** an operation that folds N things symmetrically cannot test how the N were
chosen. That is worth looking for wherever a test's reference is a sum, a maximum, or a product.

### `type_int(32, false)` → `type_int(32, true)`, twice

Two kernels built their own 32-bit integer type for a value they were about to add to an address.
Flipping the sign changes nothing observable: `OpIAdd` is sign-agnostic, and two's-complement
addition is the same either way. An equivalent mutant.

This project's habit with those is to delete the thing rather than write a test that cannot exist,
and the deletion here is `Kernel::index_type()` — the kernel already decided what its addresses are
computed in, and both call sites were reconstructing that decision from scratch. A caller that
reconstructs it differently (a 64-bit integer, say) gets a validation failure rather than a wrong
number, so the sign was never load-bearing — which is exactly why no test could reach it.

Both files use the accessor now. `runner/src/kernels/scatter.rs` went from one mutant to none, and
its existing test — that the module declares exactly one integer type — still guards the property
the accessor now makes true by construction.

### And a limit of the gate worth knowing

`runner/src/kernels/plane.rs` and `runner/src/kernels/occupancy.rs` generate **zero** mutants. They
are straight-line module construction with no comparison, no boolean literal and no branch, and the
gate has nothing to mutate. A 100% score over them is not evidence of anything.

What covers them is `runner/tests/plane.rs` and `runner/tests/occupancy.rs`, on a device, against
host references. Worth saying out loud: a green mutation score is a statement about the mutants
that were generated, and for some files that set is empty.

## The between-pass copy was a fifth of a reduction, and most of that fifth was the barriers — 2026-08-12

`notes/NEXT.md` proposed shortening the chain's between-pass copies, on the grounds that they were
probably most of what `Reducer::sum` still costs at 2²⁰ elements, and said to time them first.
`runner/examples/reducer.rs` now does. Every row below is a difference between two calls that
differ in one thing, rather than a subtraction from an estimate.

### Where a held reduction over 4 MB goes

| | per call | share of 1762 µs |
| --- | --- | --- |
| fourteen full-buffer copies — what the chain did | 385.6 µs | 22% |
| the same fourteen, shortened — what it does now | 274.3 µs | 16% |
| host upload of the input | 338.5 µs | 19% |
| host download of the output | 662.4 µs | 38% |

**The hypothesis is refuted.** The copies were a fifth, not a majority. The majority is the host
round trip, at 57% between the two — and no kernel change touches it. A caller whose data is
already on the device pays neither.

### And a fifth was not a fifth to be had

Shortening the copies removed 52 of 56 MB of device-to-device traffic and the end-to-end time moved
by **85 µs**, not by 385. The reason is in the same example, from a third chain: 61 empty passes
copying **one word** each, which keeps every barrier and carries nothing.

| one whole-buffer step | 27.5 µs |
| --- | --- |
| the two pipeline barriers around the copy | 19.0 µs |
| the 4 MB itself | 8.6 µs |

So a step is 69% barrier. Fourteen barrier pairs stay whatever the copies carry — a pass still has
to wait for the one before it — and 385.6 − 274.3 = 111 µs is all the payload there was to remove.
The observed 85 µs agrees with that within the spread, which is how the numbers are known to be
describing the same thing.

**What that points at.** The remaining 274 µs is 266 µs of barrier and 8 µs of data. A ping-pong
across two descriptor sets removes the copy *and* one of the two barriers, and `chain.rs` has said
so in a comment since it was written. That is now the largest device-side item, and the host
transfers are twice it again.

### The measurement had to be lengthened before it said anything

The first version compared 15 passes against 1. Both cost about 2 ms, the difference was about a
tenth of either, and the repeats were themselves about a tenth apart — so signal and noise were the
same size. Two runs reported **188 µs and 337 µs for the same quantity**.

Sixty-one passes makes the difference roughly half the measurement instead of a tenth. The repeats
are no steadier; they simply no longer swamp what is being measured. That is the whole fix, and it
is worth remembering as a shape: *a difference of two large numbers needs the difference to be
large, not the numbers to be quiet.*

### What guards the shortened copy

A copy shorter than the next pass reads is not a slower answer, it is a wrong one — the tail holds
whatever the source buffer held before, which on a reducer's second call is the first call's data.
`runner/tests/reducer.rs` runs a reducer twice with different inputs at nine lengths, heavy first
so a stale tail makes the answer too large rather than too small.

It was checked by being broken: halving the declared output length fails **five** tests, including
that one. A guard that has never fired is not known to be a guard.

## The ping-pong is simpler code, and only faster where bandwidth is scarce — 2026-08-12

The chain no longer copies. Two device buffers alternate — pass 0 reads A and writes B, pass 1 reads
B and writes A — so only the descriptor set a pipeline was built with changes, and the module never
learns which end it is on. `runner/src/dispatch/step.rs` has the arithmetic.

The prediction, written in `notes/NEXT.md` the same day: the copy step was 27.5 µs of which 19.0 µs
was its *pair* of pipeline barriers, so removing the copy and one barrier should leave about 9.5 µs
and save ~250 µs of a 1900 µs reduction.

**It saved about 32.** One barrier costs nearly what two did: a chained step went from 19.0 µs
(two barriers plus a one-word copy) to 16.7 µs (one barrier). NEXT.md's own refutation clause —
*"the remaining barrier costing what both did"* — is what happened.

### The A/B, because the machine had drifted

Comparing against numbers taken an hour earlier said the change made things *worse*, which is what
sent this back for a second look: `Gpu::sum` had moved too, in the same direction, on code that had
not changed. Something else on the machine was busy.

So both versions were built to separate binaries and run **alternately**, five rounds each,
2²⁰ elements:

| device | with copies | ping-pong | paired |
| --- | --- | --- | --- |
| RTX 4080 | 1929 µs | 1914 µs | no measurable difference, 2 of 5 rounds favour copies |
| integrated Radeon | 3792 µs | 3631 µs | **5.5%**, all 4 rounds favour ping-pong |
| lavapipe | 4064 µs | 4038 µs | no measurable difference |

The 4080 has bandwidth to spare and does not notice 4 MB of copying. The integrated part does not
and does. That is the same split the workgroup-size sweep and the integer dot product both found,
which is starting to be the most reliable prediction in this file: **anything that trades memory
traffic for instructions is worth something on the integrated device and nothing on the discrete
one.**

### Kept anyway, and not for speed

`chain.rs` lost the copy, both barrier-stage constants and the `Stages` type; `Pass` lost `outputs`
and `writing`; `Step` lost `copy_bytes` and then lost `Step`. A whole class of bug went with them —
a copy shorter than the next pass reads returns the *previous call's* data, which was real enough
to need a test at nine lengths.

What replaced it is one arithmetic question, and it is a sharper one: **the answer moves.** An odd
number of passes leaves it in B and an even number in A. Reading the wrong end returns the
second-to-last fold — roughly double the right answer, and green on any length whose parity happens
to match what the code assumed. `runner/tests/reducer.rs` sweeps nine lengths and asserts both
parities were covered; flipping `answer_in_destination` fails **nine** tests across two levels.

### Two measurements that were the same shape

The copy-shortening before it predicted 111 µs and delivered 85. This predicted 250 and delivered
32. Both times the error was the same: a component was timed *with its barriers included* and then
costed as though removing the component removed the barriers too. The barriers were never the thing
being removed.

## Four bytes were being shipped as four megabytes — 2026-08-12

The breakdown built for the last two items said the host download was 37% of a held reduction over
2²⁰ elements. Three items in a row had been spent shaving the *device* side of that call — 85 µs,
then 32 — while the largest single line in the table went unread.

`Reducer::sum` ended:

```rust
self.gpu.copy(answer, staging, bytes)?;      // bytes = 4 MB
staging.read(self.gpu, self.elements)?       // 1 048 576 words
...
output.first()                               // one of them
```

A reduction produces **one number**. Every invocation of the final workgroup holds the whole total,
so slot zero is the answer and the other 1 048 575 words are the last fold's leftovers. Both the
device-to-host copy and the host read were sized to the buffer.

It is `copy(answer, staging, 4)` and `read(1)` now, and `Gpu::sum` gets the same through
`Gpu::run_chain_head`, which brings a prefix home instead of everything.

### Paired against the previous build, alternating runs, 2²⁰ elements

| device | `Reducer::sum` | | `Gpu::sum` | |
| --- | --- | --- | --- | --- |
| RTX 4080 | 1866 → 1250 µs | **33%** | 3442 → 2728 µs | 20% |
| integrated Radeon | 3663 → 2550 µs | **30%** | 5270 → 4375 µs | 17% |
| lavapipe | 4844 → 3619 µs | **25%** | 87 000 → 79 200 µs | 9% |

Every round on every device, by a wide margin. **This is the first change in four that helped
everywhere** — the workgroup size, the integer dot product and the ping-pong all split by device,
because all three traded memory traffic for instructions and only the integrated part is short of
bandwidth. This one removes 4 MB and adds nothing, so there is nothing for a device to be
indifferent about.

### What it says about the three items before it

They were all real and all small: 85 µs, 32 µs, and this one 616. The difference is not cleverness
— the ping-pong is by far the most intricate of the three — it is that the first two were chosen
from a *guess* about where the time went and this one was chosen from the table.

The table existed for two of those items. It was built to test the guess, it answered, and then
three more measurements were taken before anyone read the largest row in it.

**The habit:** when a breakdown is produced, act on its biggest row before its most interesting
one. `notes/NEXT.md` had "the host round trip is 57%" as item 1 and the reason it read as hard was
the *upload* — which is genuinely hard, because the data has to arrive. The download was sitting in
the same row and was four lines of code.

### What is left

The upload, at ~294 µs of a 1275 µs call, and it is real: a caller passing `&[f32]` has to have it
copied to the device. What removes it is not an optimisation but a shape — a reduction that reads a
buffer already on the device — and that is what `notes/NEXT.md` now heads with.

## Widths 4 and 16, and the undefined behaviour that had been running since the start — 2026-08-12

`README.md` said "Nothing has run at 4 or 16" and treated it as a hardware limit. It is not one.
Neither GPU here offers a range — the RTX 4080 reports `minSubgroupSize` 32 and `max` 32, the
Radeon 32 and 64 — but **llvmpipe's subgroup width follows its vector width**, and that is an
environment variable:

| `LP_NATIVE_VECTOR_WIDTH` | subgroup |
| --- | --- |
| 128 | **4** |
| 256 | 8 (the default, and what this project had been using) |
| 512 | **16** |

`minSubgroupSize` equals `maxSubgroupSize` at each setting, so the width is pinned rather than a
default the driver may vary.

The suite now runs at **4, 8, 16, 32 and 64** — every power of two a Vulkan implementation is known
to report. It found five defects and, as at 64 and at 8, **not one of them was in the emitter**.

### The serious one: a kernel that had been reading past its buffer for months

`kernels::scale` — *the control kernel*, the one whose doc comment says "run it first" — said
`load::<32>`. Thirty-two lanes is one element per invocation on a 32- or 64-wide subgroup and
**eight strips** on a four-wide one, so on a narrow device it read and wrote eight times the buffer
every caller hands it.

At four lanes that is an access violation and the test binary dies with `STATUS_ACCESS_VIOLATION`.
At **eight** lanes it is four strips, which is the same undefined behaviour returning zeros — and
lavapipe at width 8 has been in this project's green column for a full day. The first strip is in
range, every assertion only looks at the first strip, and nothing complains.

Three more kernels had the same literal: `lane_affine::<32>` as every caller spelled it,
`fold_halves_open`, and `specialized_cluster`. `fold_halves_open` is the specialization twin of
`fold_halves`, which had *already* been converted to `whole_subgroup!` — a pair that must agree,
where only one half had been fixed.

**The shape:** `load::<32>` reads as "a vector" and means "one element per invocation" only at two
of the five widths. Every one of these is now `whole_subgroup!`, which takes the count from the
device.

### And three more of the family the other widths already found

- **`floats.rs` put its NaN at index 5** and asserted about "the first subgroup". Five is in the
  *second* subgroup of a four-wide device. One test failed; a sibling with the value at index 7
  went on passing while measuring nothing, because it *reports* rather than asserts.
- **`lane_sum::<F32, 12>` was asserted to have no mapping.** Twelve is a multiple of four, so on a
  four-wide subgroup it is a perfectly good three-strip vector. It is six now, which divides no
  power of two and is divided by none of them, so it has no mapping at any width.
- **The full-width case was a `match` on 32 and 64** that skipped every other width with a message
  — so a narrow device ran neither it nor the discriminator that followed it.

The lane-count test was then rewritten around a *list* of the cluster sizes a given width can hold,
because the patch for the above introduced the same bug twice more: at four lanes `lane_sum::<_, 4>`
is the whole subgroup and at eight `lane_sum::<_, 8>` is, so "these two must differ" asserted that a
module differs from itself. Collecting distinct sizes makes that unrepresentable, and the test now
asserts it has at least two to compare — so it cannot quietly end up with nothing.

### A caveat on how to run it

At **128 and 512** bits, lavapipe is unstable under `cargo test`'s default parallelism: independent
devices in separate threads, and roughly 40% of runs report a disagreement at some seed, always at
the first index, never the same seed twice. The same programs re-run identically in a single
process — 3 584 checks in a row without one — and `--test-threads=1` is green every time.

What that is *not*: our code has no shared state between `Gpu`s, each test opens its own instance
and device, and the default 256-bit build is green 8 runs out of 8 under the same parallelism. What
it is has not been chased further than that, because the fix is a flag:

```powershell
$env:LP_NATIVE_VECTOR_WIDTH = "128"     # or 512
cargo test -p runner -- --test-threads=1
```

## A self-audit, and the invalid instruction it found in four minutes — 2026-08-12

Every claim this project makes about itself, checked rather than re-read. What held is as much the
point as what did not, so both are below.

### Held

- **Zero dependencies in the emitter.** `cargo tree -p simdr` is one line.
- **No lint escape outside a test.** Every `#[allow]` and `#[expect]` in `src/` sits inside a
  `#[cfg(test)]` module — checked mechanically, not by reading. So `unwrap_used`, `expect_used`,
  `panic` and `indexing_slicing` are denied for real in everything that ships.
- **No `# Safety` section missing.** `clippy::missing_safety_doc` over `runner` reports nothing.
- **The only non-FFI `unsafe` is documented and bounds-checked.** `Buffer::write` and
  `Buffer::read` are the two `copy_nonoverlapping` calls in the crate; both carry an argument and
  both refuse a length past the mapping.
- **`decode` cannot panic or loop.** `checked_sub` and `get`, and an instruction always consumes at
  least its own opcode word. It is now *tested* against arbitrary word streams rather than only
  argued — see below.
- **`tests/integrity.rs` checks what it says it checks**, in both directions, including that every
  file excused from mutation still contains the `unsafe` that excused it.

### The one that did not: `OpUDot` with a signed result type

The audit asked a question nothing had asked before — *which public operations appear in no test
that runs `spirv-val`?* — and the answer was **fifteen**, including the whole shuffle and vote
surface, both right shifts, half the dot product, and four GLSL.std.450 instructions.

Writing those tests took twenty minutes. The first run failed:

```
error: line 53: Result must be an unsigned int scalar type.
```

`Lanes::dot_unsigned` emitted `OpUDot` with a `Vector<I32, _>` result. `OpUDot`'s result type must
have a signedness of 0, so **the module was invalid** — and it had shipped that way, because:

- it had **no caller**, anywhere in the workspace;
- it had **no unit test** of its own, while `dot_signed` had three;
- and the validator suite had never been pointed at it.

Three layers, and the operation fell between all of them. `dot_signed` and `dot_mixed` were fine:
a signed result is legal for both, so only the one unsigned instruction had a rule to break.

It is `Vector<U32, _>` now, which is also the right type on its own terms — four products of
unsigned bytes cannot be negative.

**The shape worth carrying:** a public method with no caller is not "unused", it is *unverified*.
Nothing in the ordinary run of the suite touches it, so every layer reports green about it by
saying nothing at all. `dot_unsigned` existed for symmetry with `dot_signed`, which is exactly the
reason a thing gets written and never exercised.

### Drift, in three places, all of it in the *reasoning* rather than the numbers

- `reduction/held.rs` and `decisions/DR-0005` both said the reduction's remaining time at 2²⁰ is
  where "the arithmetic starts to dominate". This project *measured* that six items later and it is
  the host round trip and the chained dispatches. Both now say so, and the 1.6× they quoted is
  2.2×.
- `kernels/reduce.rs` said the whole-subgroup wrapper "instantiates 32 and 64", which stopped being
  true when the width list grew to five.
- The README's layer diagram omitted `encode.rs` entirely and drew `half.rs` and `decode.rs` as
  directories.

### And one argument that had not expired yet

`sign_extend(bits, width)` shifts by `32 - width`. A width of 0 shifts by 32 and one above 32
underflows; both panic in a debug build, in a crate whose first claim is that no input makes it
panic. Every caller passes 8 or 16 — which is *precisely* the argument `Buffer::write`'s safety
comment made before `Session` falsified it six hours later. It is clamped and tested now, rather
than left resting on who happens to call it.

### What was reported and not changed

`clippy::undocumented_unsafe_blocks` fires **78** times across `runner`. That number looks much
worse than it is: almost all of them are a single `ash` call inside a function that is already
`unsafe fn` with a `# Safety` section covering exactly that obligation, and writing 78 restatements
of "the device is live and the enclosing contract holds" would be noise in a codebase whose
comments are supposed to carry an argument.

What is genuinely missing is not the comments but a way to tell the two apart — an FFI call under a
stated contract from a block that needs a new argument. Nothing enforces that distinction, so a
novel `unsafe` block would arrive unremarked among the seventy-eight. Recorded rather than papered
over.

## The map belongs in the chain, and that is worth two crossings — 2026-08-13

`notes/NEXT.md` had the upload as item 1 and said it needed *a shape rather than an optimisation*,
with an honest caveat attached: nothing in this repository had data on the device to begin with, so
the API would be for a caller that had not arrived.

The caller was there all along, one level up. **Σ f(x) is a map and a reduce**, and computing it
costs three host crossings — send the input, run the map, bring the result home, send it back,
reduce — of which two are the whole buffer. `Gpu::reducer_of(elements, map)` makes the map the
*first pass of the same chain*, so its output is handed to the first fold on the device and the
intermediate never crosses at all.

`kernels::square` is the map; Σ x² is the squared L2 norm, which is a real primitive rather than a
demonstration.

| Σ x², both routes held | three crossings | one crossing | |
| --- | --- | --- | --- |
| RTX 4080, 8 192 | 317–373 µs | 175 µs | **1.8–2.1×** |
| RTX 4080, 2²⁰ | 2326–2356 µs | 1331–1355 µs | **1.7×** |
| integrated Radeon, 2²⁰ | 5411 µs | 2759 µs | **2.0×** |

**The saving is the crossings and nothing else.** At 2²⁰ it is 993 µs, against a download of 718 µs
and an upload of 294 µs measured separately in the same file — 1012 µs predicted, 993 observed. Two
numbers arrived at independently that agree is the only reason to believe either.

### The first version of this measurement said 2.9×, for the reason it always does

The old route was written as `gpu.run(&square, …)`, which allocates three buffers and builds a
pipeline on every call. This file had *already measured* that at ~900 µs, and charging it to the
route being replaced made the new one look nearly twice as good as it is.

That is the fourth time in two days: a component timed with its setup included, then costed as
though the change removed the setup too. The rule that keeps not being followed is simple enough to
write down — **give the thing you are replacing every advantage you would give the replacement**.
Both columns run through a held `Session` and a held `Reducer` now, and the only difference left is
where the intermediate went.

### And an assertion that earned its place by failing

The example asserts both routes equal a host reference. At 2²⁰ elements over values 0..7 that
assertion fired: Σ x² is 18.3 million and an `f32` counts exactly to 16.7. The measurement would
otherwise have printed a confident speed-up for two numbers that were both wrong.

The values are 0, 1 and 2 now — small enough to stay exact, and **2 has to be among them**, because
x² and x agree on 0 and 1 and a map that had stopped squaring would go unnoticed without it.

### What this does not reach

A caller whose data was produced by some *other* dispatch it owns still cannot hand that buffer to
a `Reducer` — the reducer's bindings are private, and the map has to be a pass of its chain. That is
a third shape, and it still has no caller in this repository, so it is still not built.

## `f16` is fuzzable after all, by refusing the rounds it cannot check — 2026-08-13

`notes/NEXT.md` had excluded `f16` from the differential fuzzer since the narrow types arrived, with
a stated reason: a half counts integers exactly only to **2048**, so a sum over a few hundred lanes
leaves that range at once, and a tolerance would be checking the rounding rather than the emitter.

The reasoning was right. The conclusion — leave the domain out — was one step too far.

What it argues for is not skipping the domain but **noticing** when a round leaves the range.
`Domain::exact_limit` says where that is, the reference reports whether it stayed inside, and
`fuzz::check` returns `Outcome::Unrepresentable` instead of comparing. So every `Half` round that is
compared at all is compared **exactly**, and the ones that cannot be are counted rather than
quietly loosened.

| domain | agreed | refused | unrepresentable | of 256 seeds |
| --- | --- | --- | --- | --- |
| the seven that already ran | 256 | 0 | 0 | each |
| **Half** | **253** | 0 | **3** | |

99% of rounds are checked, so this is coverage rather than a domain that looks supported and never
runs. That distinction is now a test: `Swept::expect_mostly_checked` insists most rounds compared
something, because a domain refused every round is indistinguishable from one that always agreed if
only the failures are reported.

### And the same check found that `Float` had never been checked either

`Domain::Float` rests on exactly the same argument one exponent wider — every value a small integer,
every partial sum under 2²⁴ — and that had only ever been *assumed*. Nothing verified it at run
time. It is verified now, for both, by the same three lines. It has never fired for `Float`, which
is the answer the assumption predicted and the first time anything has said so.

### Three mutants, and one of them took three attempts to delete

The mutation gate found eight survivors in the new code. Six were straightforward gaps —
`as_f32`'s three readings, the reduction identities' signedness — and two were sharper:

- **`exact = exact && within(…)` flipped to `||` survived my first test.** The test raised a value
  past the limit and asserted the round was inexact — but the *final* answers were also past the
  limit, and the separate check on those caught it either way. The case that isolates the loop is
  the one the comment beside it already described: a value that leaves the range and is **clamped
  back**, so everything compared is in range and only the middle was not.
- **`_ => vec![false; invocations]` was a genuine equivalent mutant** — a vote nothing reads for
  three of the four finishes. Its comment said the alternatives were worse, and listed two that
  were. The third was not tried: compute the vote **inside the one arm that reads it**. There is no
  default to be wrong about now because there is no value to default, and the reference got shorter.
  It costs a recomputation per invocation instead of per subgroup, which a file whose first line
  says *obviously right rather than fast* can afford.

The lesson is the older one arriving again: when a comment argues that an unfalsifiable branch has
to stay, it is usually arguing about the two shapes someone already tried.

## The breakdown was half a breakdown, and the missing half was the biggest row — 2026-08-13

A full mutation run over all 83 targets came back **419/419 killed, 0 survivors**, and the three
entries in the ratchet floor — one of them naming `runner/src/reduction.rs`, a path that became
`reduction/mod.rs` when the file was split — were all confirmed dead. The floor is empty now.

Then the reduction breakdown was re-read, and it did not add up. Its rows came to about **half** of
the `Reducer::sum` they were breaking down, and that gap had gone unremarked through three separate
optimisations built on top of it.

Two rows were missing, and both were things the *measurement* skipped rather than the call:

- **The `f32` → `u32` copy.** `Reducer::sum` takes `&[f32]` and the buffer holds words, so it built
  a `Vec<u32>` of the whole input on every call. The upload row hoisted that conversion out of its
  timed loop — measuring a cost the real call does not have, and missing one it does.
- **One submission and its fence.** The per-step row is a *difference* between a 61-pass chain and a
  1-pass chain, so every cost paid once per call cancels out of it exactly.

Measured, the conversion was **596 µs — 52% of the call**. The largest single item, larger than the
fourteen chained dispatches and the upload together, and it computed nothing: `f32::to_bits` is
*defined* as reinterpreting the bits, and the bits were already the right bits.

### Removing it

`Buffer::write_floats` copies the caller's slice straight into the mapping. `f32` and `u32` have the
same size and alignment, and the bytes are copied rather than read as numbers, so the cast is one
safety argument rather than a conversion.

Paired against the previous build, alternating runs, 2²⁰ elements:

| `Reducer::sum` | via `Vec<u32>` | direct | |
| --- | --- | --- | --- |
| RTX 4080 | ~1342 µs | ~524 µs | **2.6×** |
| integrated Radeon | ~2543 µs | ~1749 µs | **1.5×** |

Every round on both devices. Σ x² through `reducer_of` came down with it — 1390 → 735 µs on the
4080 — and `Reducer::sum` against `Gpu::sum` went from 2.1× to **4.3×**.

The breakdown now comes to 109% of the call rather than 52%. Over is the honest direction: the rows
are measured separately and overlap, since the upload row maps and unmaps the staging buffer and so
does the call.

### What this is the fourth instance of

Every performance item this week has been mismeasured in the same direction, and this one is the
sharpest version of it:

1. A probe that allocated two buffers per call and reported it as pipeline creation — wrong by 8×.
2. A grid comparison that moved the occupancy and the address at once — 2× that was neither.
3. A map-reduce comparison whose old route paid `gpu.run`'s setup — 2.9× that was 1.7×.
4. **A breakdown that hoisted a per-call cost out of its own timed loop** — and so reported half a
   call and left the largest row invisible for three optimisations.

The first three flattered a change. This one hid one. The rule that would have caught all four is
the same: **time the thing the caller actually pays, in the shape the caller actually pays it** —
and when a breakdown does not come close to the whole, the gap is the finding.

## A reduction submitted three times to do one thing — 2026-08-13

With the breakdown finally adding up, the largest row it named was the upload at 287 µs. Splitting
*that* — a full write against a one-word write, which pays the same map, unmap and submission and
almost none of the copying — put **73 µs** of it in the fixed half. And the row beside it said a
bare submit-and-fence costs **65 µs**.

Which meant the fixed cost of the upload was not the mapping. It was a whole submission.

`Reducer::sum` was three of them:

```
staging.write_floats(..)        // host memcpy, no submission
gpu.copy(staging, source, ..)   // command buffer, submit, fence
gpu.replay(..)                  // command buffer, submit, fence
gpu.copy(answer, staging, 4)    // command buffer, submit, fence
staging.read(.., 1)             // host read, no submission
```

Two of the three exist only to move bytes between buffers the third submission already touches.
`Gpu::replay` takes an optional `before` and `after` copy now and records them into the chain's own
command buffer, costing a pipeline barrier each instead of a submission each. `Gpu::run_chain` had
the identical shape and got the same treatment.

Paired against the previous build, alternating runs, 2²⁰ elements:

| `Reducer::sum` | three submissions | one | |
| --- | --- | --- | --- |
| RTX 4080 | ~548 µs | ~424 µs | **1.29×** |
| integrated Radeon | ~1751 µs | ~1045 µs | **1.68×** |

The 4080 saved ~124 µs against a predicted 2 × 62. Prediction and observation agree, which is the
only reason to believe either.

### Where the reduction now stands

Over 8 192 elements `Reducer::sum` is **11.2×** `Gpu::sum`, and over 2²⁰ it is **5.6×** — against
2.1× at the start of the day. Three changes got it there, and the order they were found in is the
whole lesson:

| | 2²⁰, `Reducer::sum` |
| --- | --- |
| where the day started | ~1930 µs |
| reading one word home instead of the buffer | ~1140 µs |
| writing the caller's floats straight into the mapping | ~548 µs |
| one submission instead of three | **~424 µs** |

Not one of the three was an algorithm. Every one was a cost the *measurement* had been hiding: a
download sized to the buffer instead of the answer, a conversion hoisted out of its own timed loop,
and two submissions that a per-step row cancelled out by being a difference.

**The breakdown found all three, and only once it was made to add up.** It read 52% of the call for
weeks, and the missing 48% was where every one of them lived.

## Folding by sixteen: five dispatches instead of fifteen, and a quarter of the predicted saving — 2026-08-13

With the breakdown adding up, its largest row was the chain itself: fourteen steps, 237 µs, 56% of
a 425 µs call. The reduction folded in **halves** — `out[i] = in[i] + in[i + h]` — which takes
`log₂(N/64)` passes. `kernels::fold_by` adds `factor` elements per invocation instead of two, and
`folds()` picks the widest factor that still leaves a whole workgroup for the next pass.

Over 2²⁰ elements that is **five dispatches instead of fifteen**, and over 8 192, three instead of
eight.

Paired against the halving build, alternating runs:

| 2²⁰ | halving | by sixteen | |
| --- | --- | --- | --- |
| `Reducer::sum` | ~442 µs | ~407 µs | 8% |
| `Gpu::sum` | ~2357 µs | ~2203 µs | 6% |

At 8 192 the held reduction does not move at all; only `Gpu::sum` does, and for a reason that has
nothing to do with the chain — it builds a pipeline per pass, so five fewer passes is five fewer
pipelines.

### Both arguments for the change were wrong, and both in the optimistic direction

**"It halves the memory traffic."** True as a ratio, irrelevant as a duration. Halving reads about
`2N` across the chain because every level re-reads what the level above wrote; folding by sixteen
reads about `1.07N`. But the difference is one buffer's worth — 4 MB at 2²⁰ — which is roughly
**6 µs** of bandwidth on this device. The *first* pass reads N either way and dominates both; the
levels that differ are the tail, and the tail is small by construction.

**"Ten fewer dispatches at ~15 µs each is ~150 µs."** It was about 35. The ~15 µs per step comes
from this file's own chain-of-empty-kernels measurement, and that chain has nothing for a barrier
to overlap with. In a real reduction the removed passes are the tail — dispatches of one to sixty-
four workgroups whose launch hides behind the pass before them. **The per-step row is an upper
bound**, and it now says so.

### The fifth measurement lesson of the week, and a new kind

The first four all mismeasured a *change*. This one mismeasured a *component*: the breakdown's
step row was honest about what it measured and wrong about what it predicted, because a cost
measured in isolation is not the same cost measured in company.

Kept anyway, and not for the 8%: five pipelines instead of fifteen is less to build, less to hold
and less to get wrong, and the chain is nowhere near a command buffer's limit at any size this
accepts. But the headline number is 8%, not the 35% the arithmetic promised, and the arithmetic is
written down beside it so the next reader can see which half of it was real.
## Writing the input where the device can already read it — and the premise being wrong twice

`Reducer::sum` over 2²⁰ was **70% host upload** once everything else had been taken out of it. The
input went into staging memory and then across into the buffer the first pass reads: the same four
megabytes moved twice. Asking for memory that is device-local *and* host-coherent removes the
second move, because the first write already landed in the right place.

Paired A/B, alternating which binary ran first in each round, medians of three:

| device | staged | shared source | |
| --- | --- | --- | --- |
| RTX 4080, 2²⁰ | ~404 µs | ~280 µs | **31%** |
| integrated Radeon, 2²⁰ | ~924 µs | ~617 µs | **33%** |
| either, 8 192 | — | — | nothing above the noise |

### The premise was wrong, and the first measurement of it was noise

The change was written as "on an integrated part `device_local` *is* host-visible, so staging is a
pointless second copy". Both halves of that were wrong on this machine, and `runner/examples/
memtypes.rs` prints why:

* The **integrated Radeon** offers a device-local type that is *not* host-visible, at index 0.
  `Buffer::device_local` takes the first match, so it has never once fallen back to host-visible —
  the fallback the code documents as "the normal state of an integrated part" has never fired.
* The **RTX 4080** offers a type at index 4 that is device-local, host-visible *and* host-coherent
  — a resizable-BAR window. The discrete card is the one that could do this all along.

So the first version of the change was dead code on every device here, and it still appeared to
save 19 µs consistently. That was **ordering**: the A binary ran first in every round and paid the
warm-up. Running the B binary first reversed the result exactly. Two behaviourally identical
binaries "differed" by 5% and would have been committed as a win, with a wrong explanation
attached, if the memory tables had not contradicted the story first.

The fix is not to guess better. It is that `Buffer::host_writable()` asks the buffer, and no code
in this crate infers it from the kind of device.

### The same change is a 62% regression one call away

`Buffer::shared` was applied to the one-shot `Gpu::run_chain` at the same time. That path allocates
its buffers on every call, and:

| `Gpu::sum`, RTX 4080 | device-local source | shared source | |
| --- | --- | --- | --- |
| 2²⁰ | ~2153 µs | ~3492 µs | **62% slower** |
| 8 192 | ~748 µs | ~919 µs | 22% slower |

The 8 192 row is what identifies the cause: 32 KB of upload, nothing to gain from removing a copy
of it, and it still lost 22%. Allocating out of a BAR window costs more than the copy it saves, so
the cost is in the allocation. Reverted there and kept in `Reducer` and `Session`, which allocate
once and upload many times.

**The same three lines are a third faster or two thirds slower depending only on how often the
buffer is made.** Nothing about the memory changed between those two columns.

### What this does to the breakdown table

`accounted for` in `runner/examples/reducer.rs` now reads about **123%** of the call. It read 52%
*under* the whole before its missing rows were found, 79% once they were, and past 100% as soon as
the call itself got a third shorter while the rows — each timed in isolation, each paying its own
fixed costs — did not. Two of the three are upper bounds by construction and now say so in the
table itself. It is a ranking of what to attack next, not a budget that adds up.

## Eleven tests were reading past their input, and running at four lanes had not found them

`dispatch::extent` was extended to recover the strip count from a module's own address arithmetic —
`Kernel::run_start` emits `group × (workgroup × strips)`, and the workgroup size is already read
from `LocalSize`, so dividing the constant by it gives the count back. Nothing had to be declared
and no second copy of the number exists.

It refused eleven tests the first time it ran, across five files:

| file | tests | the kernel |
| --- | --- | --- |
| `runner/tests/extended.rs` | 8 | `clamped`, `magnitude`, `larger`, `smaller`, `root`, `fused_square` at 32 lanes |
| `runner/tests/narrow.rs` | 3→1 | `narrow_add`, `narrow_clamp` at 32 lanes |
| `runner/tests/specialized.rs` | 6 | `specialized_add`, `specialized_affine`, `specialized_derived` at 32 lanes |
| `runner/tests/lanes.rs` | 1 | `lane_sum::<F32, 32>` as a discriminator |

Every one of them paired a kernel built for **32 lanes** with a buffer of **one workgroup**. That is
one element per invocation on a 32-wide subgroup and eight on a four-wide one, so on lavapipe each
was handed an eighth of what its kernel reads — right in the first sixty-four elements and off the
end for the rest.

**They had all been passing.** Not skipped, not flaky: green at widths 4, 8 and 16 in every run
since those widths were added, because the part of the answer that was checked happened to be the
part that was in bounds.

### What this says about the width sweep

`README.md` has claimed for weeks that running at 4, 8 and 16 lanes is what catches a kernel
conflating "32 lanes" with "the subgroup" — and it did catch `kernels::scale`, loudly, with an
access violation. The eleven below it were the same bug in the same week and the sweep ran straight
past them, because an out-of-bounds *read* is only an access violation when it crosses a page.

So the sweep finds this class when it is unlucky and the bounds check finds it always. The two are
not substitutes: the sweep is what proves the *answers* are right at every width, and nothing about
the bounds check says a number is correct.

### The fix, and one place it was the wrong fix

Ten of them are a buffer sized to `WORKGROUP_SIZE × strips` rather than `WORKGROUP_SIZE`, and the
helper that computes it now lives once in `runner/tests/common/mod.rs` rather than being copied per
file — it was copied into two before that was obvious.

The eleventh is different. `lanes.rs` used `lane_sum::<F32, 32>` as a *discriminator*: a subgroup
reduction whose answer should differ from a workgroup reduction over the same input. Sizing its
buffer up would have fixed the over-read and left the test comparing two different inputs. The whole
-subgroup form is what it meant, and that in turn made the assertion false on a 64-wide device —
where one subgroup fills the workgroup and the two reductions cover the same lanes. The test now
says which relationship it expects at which width, which is what it should have said in the first
place.

## Superseded: a second binding costs about 330 µs, and the submission is a fifth of it

> **The instrument cannot resolve the quantity this section is about.** Four runs of
> `runner/examples/bindings.rs` on 2026-08-26 put the difference at +877.9, −824.8, +277.0 and
> −88.3 µs at 512 operands, and at +663.1, −1069.7, +181.4 and −235.9 at 4 096 — it changes sign,
> against a recorded +367 and +329. The columns it is subtracted from read 9 000–10 850 µs today
> where they read ~815 and ~1181 here, so the effect being measured is under a tenth of the noise
> on the totals.
>
> No cause is offered. `runner/examples/overhead.rs` puts allocating and freeing a 256-byte buffer
> at 282.9 µs today against the ~310 recorded, so allocation is not eleven times slower; what is,
> is **NOT ESTABLISHED**. This is the second instrument here found unable to reproduce its own
> reading — `decisions/DR-0005` has the first — and both are timings around per-call buffer
> allocation, which is an observation and not a conclusion.
>
> The conclusion below — that item 9 is not worth doing — is unaffected: it turns on the cost
> being fixed setup rather than transfer, and a difference this instrument cannot see is not a
> difference worth chasing either.

`Gpu::run_bound` uploads each input through the shared staging buffer in turn, and each of those is
a `Gpu::copy` — a whole command buffer, submission and fence. So `k` inputs cost `k + 2`
submissions where a chain costs one. That is the shape that was worth 116 µs when it was fixed for
the reduction, and `notes/NEXT.md` listed fixing it here as item 9, with measuring it first.

`runner/examples/bindings.rs` measures it. `clipped_dot` reads its activations and weights from one
buffer with the join as an offset; `clipped_dot_split` reads them from two, and
`runner/tests/network.rs` asserts the two give the same answer — so the difference between them is
the second binding and nothing else.

| operands | one buffer | two buffers | difference |
| --- | --- | --- | --- |
| 512 | ~815 µs | ~1181 µs | **+367 µs** |
| 4 096 | ~790 µs | ~1118 µs | **+329 µs** |

**The difference is flat across an eight-fold change in data.** So it is not transfer, and it is not
the copy: it is fixed setup — one more buffer allocated, one more descriptor in the set, one more
submission. A submission on this device is 50–80 µs, which is a **fifth** of it.

### So item 9 is not worth doing, and that is the third time

Recording `run_bound`'s uploads inside one command buffer would recover the submission and leave the
allocation and the descriptor set, which is most of the cost. And a caller who minds any of it has a
better answer already: `Session` allocates once and holds it, and since `Buffer::shared` was
introduced its writes go straight into the binding on a device that offers such memory — so a held
session pays *no* upload submission at all, not a cheaper one.

Making `run_bound` allocate shared buffers instead would be the other half of that, and it is
already measured and refused: allocating out of a BAR window costs more than the copy it saves when
the buffer is made per call, which cost `Gpu::sum` 62% when it was tried.

The one-shot path is for a caller running a kernel once, and it builds a pipeline every time —
5× the held path by this project's own measurement. Shaving 60 µs off a call that spends 800 is
optimising the wrong thing.

Left undone, deliberately, and this is the third item on that list to be refuted by its own
measurement rather than by an argument.

## The first CI run that reached a device found a signed zero that is not preserved

`negative_zero_and_positive_zero_sum_to_positive_zero` summed sixty-four negative zeros and
asserted the answer was `-0.0`. IEEE 754 says so: `-0.0 + -0.0` is `-0.0` whatever the order. It
held on the RTX 4080, on the integrated Radeon, and on the lavapipe built on this machine.

It does not hold on the Mesa that ships in Ubuntu 24.04. LLVM 20.1.2 folds the sum to `+0.0`, and
**it is entitled to**: Vulkan does not require signed-zero preservation. It is the optional
`shaderSignedZeroInfNanPreserveFloat32` property, and even a device that reports it only binds a
module declaring the `SignedZeroInfNanPreserve` execution mode. This emitter declares no such mode,
so nothing here had asked for the behaviour the test was demanding.

The test now asserts what is guaranteed — the value is numerically zero, which is the comparison
that hides the question — and *observes* the sign, the way the NaN test already treats a maximum.

### What this says about the width sweep, again

Three implementations agreed and a fourth did not. The three that agreed are the three that were
easy to reach: two GPUs in one machine and a Mesa built on it. The fourth was a different build of
the same software renderer, and it disagreed on the first run.

That is the second time in two days that adding a *place to run* found something the existing
places could not, after eleven tests reading past their input showed up when the bounds check
started looking. Both were latent for as long as the code has existed.

**A shared runner is a fourth implementation, not just automation.** That is the strongest argument
for CI here, and it is not the one the item was written for — item 4 was about the suite being run
by hand on one machine. It found a portability bug on its first green device.

## The step row was out by five times, and the two devices disagree about why

`runner/examples/reducer.rs` built its breakdown out of probes: a chain of empty kernels for the
per-step cost, a bare submit-and-wait for the submission, a `Session` write for the upload. The
rows came to **118%** of the call they described, and the file said so without being able to say
which row was wrong.

`Reducer::sum_timed` writes a timestamp into the chain's own command buffer after every dispatch,
so each pass is measured beside the passes it actually runs beside. On an RTX 4080 over 2²⁰:

| | probe | in place |
| --- | --- | --- |
| the chained steps | ~70 µs, 24% of the call | **~12 µs, 4%** |

**Out by roughly five times**, and in the direction the probe was always going to err: a chain of
empty kernels gives a barrier nothing to overlap with, so every step pays its full latency where a
real one hides behind the pass before it. The row was labelled an upper bound after the fold-by-
sixteen pass found it overestimating by four; this says by five and says it from inside the call.

### The profile's shape belongs to the device, not the algorithm

Each pass after the first reads a sixteenth of what the one before it wrote, so a bandwidth-bound
chain should fall away to nothing and a latency-bound one should not. Both happen, on the same
chain, on two devices in the same machine:

| pass | RTX 4080 | integrated Radeon |
| --- | --- | --- |
| 1 of 5 | 3.6 µs — 30% | 115.8 µs — 92% |
| 2 of 5 | 2.0 µs — 17% | 6.6 µs — 5% |
| 3 of 5 | 2.0 µs — 17% | 1.5 µs — 1% |
| 4 of 5 | 2.0 µs — 17% | 1.1 µs — 1% |
| 5 of 5 | 2.0 µs — 17% | 1.0 µs — 1% |
| all five | **11.8 µs** | **126.0 µs** |

The Radeon is bandwidth-bound and falls away exactly as the arithmetic predicts. The 4080 is flat,
because at its bandwidth the later passes are too small to cost anything but the dispatch itself —
a floor per dispatch rather than work.

That is the same floor the fold-by-sixteen pass was chasing, and it is why removing ten dispatches
was worth anything on the 4080 at all. It also explains why the same change was worth so little:
the floor is ~2 µs, and ten of them is ~20 µs against a 400 µs call.

**Neither device's answer generalises.** The example prints what the device in front of it says,
rather than repeating either of these.

## Fusing the map into a scan is worth 2–3×, and the scan itself is where the levels show

`Gpu::scanner_of` makes an elementwise map the first pass of the scan's own chain, so its output
never crosses the bus. The same trade `reducer_of` makes, measured the same way: both columns hold
their pipelines and their buffers, so neither pays for allocation or pipeline creation, and both
were asserted to compute the same numbers before either was timed.

| Σ x² as a running total | three crossings | one crossing | |
| --- | --- | --- | --- |
| 4 096 — RTX 4080 | ~170 µs | ~66 µs | **2.6×** |
| 65 536 | ~237 µs | ~101 µs | **2.4×** |
| 2²⁰ | ~2078 µs | ~1008 µs | **2.1×** |
| 4 096 — integrated Radeon | ~1140 µs | ~375 µs | **3.0×** |
| 65 536 | ~1335 µs | ~518 µs | **2.6×** |
| 2²⁰ | ~4263 µs | ~2168 µs | **2.0×** |

**The multiple falls as the input grows, and that is the honest reading.** What is removed is two
crossings of the buffer, which grows with the input — but so does the scan, and the scan grows
faster because it is seven dispatches over a dozen buffers rather than one. At 4 096 the crossings
are most of the work; at 2²⁰ they are half of it.

### What the scan costs on its own

| elements | levels | dispatches | RTX 4080 | integrated Radeon |
| --- | --- | --- | --- | --- |
| 4 096 | 1 | 3 | ~80 µs | ~389 µs |
| 65 536 | 2 | 5 | ~106 µs | ~610 µs |
| 2²⁰ | 3 | 7 | ~966 µs | ~2228 µs |

Sixteen times the elements between the first two rows and about a third more time, because two more
dispatches are most of what changed. Sixteen times again and it is nine times slower — by then the
buffer is 4 MB and the host write is back to being the largest single row, which is what
`runner/examples/reducer.rs` measures directly for the reduction.

## A scan's depth is nearly free; its two ends are the whole cost

`Scanner::scan_timed` writes a timestamp into the chain's own command buffer after every dispatch,
so a scan over 2²⁰ reports where its seven passes went. Three kinds of pass — block scans on the
way up, one workgroup at the top, offset additions on the way down — and the answer is that only
two of the seven matter.

| pass | RTX 4080 | integrated Radeon |
| --- | --- | --- |
| up: the input | 11.0 µs — 35% | 244.1 µs — 59% |
| up: level 1 | 2.2 µs | 5.4 µs |
| up: level 2 | 2.0 µs | 2.8 µs |
| top: one workgroup | 1.9 µs | 1.4 µs |
| down: level 2 | 2.0 µs | 1.4 µs |
| down: level 1 | 2.0 µs | 2.7 µs |
| down: level 0 | 10.2 µs — 33% | 158.0 µs — 38% |
| all seven | **31.5 µs** | **415.6 µs** |

The first pass reads the whole input and the last writes the whole answer. Everything between them
works on block totals — a sixty-fourth of the buffer, then a sixty-fourth of that — so the five
middle passes come to about 10 µs against 21 for the two ends on the 4080, and about 14 against 402
on the Radeon.

**Two consequences, and the second is the useful one.** A longer input costs two more dispatches
and almost no more device time, so the recursion is not what to worry about. And making a scan
faster means making those two full-buffer traversals faster — shortening the middle would recover
single microseconds.

### And the dispatches are 3% of the call

31.5 µs of device time against ~995 µs of wall clock on the 4080. The rest is the host writing its
input and waiting for one submission, neither of which is inside the command buffer — the same
conclusion `runner/examples/reducer.rs` reaches for the reduction, arrived at independently and by
a different route. On this hardware, at this size, **the arithmetic is not the cost of either
algorithm.**

## The clustered scan moved into the lane API — 2026-08-14

`Lanes::prefix_sum` covers all three mappings now. The clustered one is a Hillis-Steele ladder,
built where the other two are, and the kernel that used to hold a second copy of it is a load, a
scan and a store.

What it needed was the invocation's own lane, and `Lanes` is handed a module and a width. It
declares `SubgroupLocalInvocationId` for itself, on demand, so a kernel that only scales still
declares no `Input` variable and no `GroupNonUniform` capability — `decisions/DR-0007` has the
argument and the two options it was chosen over.

**The failure mode is not a wrong number, it is an invalid module that every driver runs anyway.**
`OpEntryPoint` lists the `Input` variables the entry point reaches, and a built-in the *body* asks
for arrives long after that instruction was emitted. `Module` renders it from data now. Deleting
the one line that adds the variable to the interface leaves **19 of `tests/kernels.rs`'s 20 modules
rejected by `spirv-val`** and all three devices still returning the right answers — which is what
says the validator is carrying this and no execution test could.

### The fuzzer generates clustered scans now, and it took two deliberate breakages

`fuzz::generate` used to exclude the scans from a clustered vector, because they were refused. That
left the ladder — three mappings, two directions, a mask — checked by hand-written tests only,
which is precisely the state the reduction was in when the fuzzer found `reduce_min` folding its
strips with a maximum.

The reference needed one line: the run that scans together is `min(lanes, width)` invocations, not
the width. Both breakages were caught immediately — dropping the mask disagreed at seed 0, and
returning the inclusive answer where the exclusive one belongs disagreed at seed 1.

432 clustered scans agreed on an RTX 4080; 146 strip-mined ones on the same run.

## An AMD driver that faults compiling a valid module — 2026-08-14

The new coverage found this on the first sweep at width 64. The **integrated AMD Radeon** cannot
compile the clustered ladder past a certain size: `vkCreateComputePipelines` dies with
`STATUS_ACCESS_VIOLATION`, taking the test process with it. Smaller programs of the same shape
compile and return *wrong answers*.

Bisected, all at subgroup 64 with `Simd<f32, 16>` — a cluster of 16:

| program | this device |
| --- | --- |
| `[MinConstant, MulConstant, MaxConstant]` + clustered scan | compiles, agrees |
| `[MinConstant, MulConstant, MaxConstant, ClampBoth]` + clustered **sum** | compiles, agrees |
| the same four + clustered **maximum** | compiles, agrees |
| the same four + clustered scan, inclusive | **faults in pipeline creation** |
| the same four + clustered scan, exclusive | **faults in pipeline creation** |

So the ladder is necessary and so is enough code around it; the direction is not. A separate
8-bit case — `OpExtInst UMax` on a `uchar` feeding the same tail — compiles and answers wrongly
rather than faulting, which is what the probe in `runner/tests/fuzzing.rs` uses, because nothing can
catch a fault in order to report it.

**Why it is theirs and not ours.** `spirv-val --target-env vulkan1.1` accepts every one of these
modules. An RTX 4080 and lavapipe compile and run all of them and agree with the CPU reference,
across thousands of rounds. The one control that could not be run is the same *module* on another
64-wide device, because there is only one here — forcing llvmpipe to a 2048-bit vector width gets a
64-wide subgroup and then disagrees about clustered **reductions**, which are years old and green
everywhere, so that configuration is broken rather than a witness. And a driver that faults while
compiling a valid module has a defect whatever the module says.

**What the suite does about it.** `runner/tests/fuzzing.rs` probes once per device with the 8-bit
program that answers wrongly. On a device that fails the probe, the generated sweeps replace a
clustered scan with a **sum** rather than dropping the round — a third of the seeds generate one,
and dropping them would take their steps down with the tail — and the dedicated clustered sweep
skips loudly. Every count is printed: `85 clustered scans replaced by a sum` per domain, of 256.

The filter took three widenings, and each one was it being too clever: "8-bit and a max" (the probe),
then every 8-bit program (a `MinConstant(0)` faulted), then every narrow one (a 16-bit program
disagreed), then the ladder itself (an `f32` program faulted). Each attempt was an instruction in
front of the same tail, which is what said the tail was the part that was broken.

## The gate over the clustered scan, and the survivor this project had already met — 2026-08-14

1 138 changed lines scoped against `HEAD~2`: **33 of 35 viable mutants killed, 94.3%**, two
survivors, and they were one mutation in two files.

```
src/lanes/mod.rs:242     [false→true]  let uint = self.module().type_int(32, false)?;
src/lanes/reduce.rs:214  [false→true]  let uint = self.module().type_int(32, false)?;
```

Flipping the signedness of the integer type the lane index is loaded and masked in changes the
module and nothing else. `OpBitwiseAnd` is sign-agnostic; so is `OpUGreaterThan` as far as the
*type* is concerned, because SPIR-V's `OpTypeInt` signedness is not what selects the comparison —
the opcode is. `spirv-val` accepts either and all three devices return the same numbers.

**`Kernel::index_type` carries this exact note already**, from a survivor in
`runner/src/kernels/reduce.rs` on 2026-08-12: *"the sign was untestable because it was never
load-bearing, and the honest fix is not a test for it but not writing it down twice."* The same fix
applies: both sites now ask for `type_of::<U32>()`, the lane API's own `u32`. It is declared once,
in `element.rs`, where the signedness decides which comparison and which extreme an element reaches
— and mutants there die.

That makes three copies removed in two days by the same argument, which is worth more than the
score: **a constant that no test can pin is a constant that should have one home.**

### What the gate could not have found

Run beside it as a review of the same diff, and every one of these is a module saying something the
kernel does not do — a shape no mutation expresses:

- **A one-lane cluster went through the whole ladder.** `Simd<T, 1>` maps to `Clusters { size: 1 }`,
  which is a case rather than an impossibility, and the mask is then `lane & 0` — so the shuffle's
  result is selected away in every lane and the answer was right all along. Five instructions, an
  `Input` variable and a `GroupNonUniform` capability to compute the element itself. The same was
  true of the clustered broadcast, where every lane read its own value through a mask and an add.
- **The clustered broadcast's device test ran at 32 lanes only**, behind a helper that refuses other
  widths because the *other* tests in that file bake 32 into their expectations. This one's
  expectation has no width in it.
- **Two hand-emitted `OpEntryPoint`s remained.** Harmless today and a trap tomorrow: a module built
  that way plus any lane operation that declares a built-in is invalid, and every driver here runs
  it. Both go through `Module::entry_point` now.

### And over the fixes: 8 of 8, no survivors

The gate was pointed at the commit that answered it — 86 changed lines, **8 viable mutants, all
killed**. The `if size == 1` short-circuits are the interesting ones, because both directions of
that branch are guarded by different tests: flipped one way a one-lane cluster emits the ladder
again and the new unit tests see the instructions appear; flipped the other, every cluster returns
early and the step-count tests see them vanish.

Two runs, and the second is the one that closes the item. A fix that arrives with no mutant of its
own is a fix nobody has checked.

## The public surface, audited a second time — 2026-08-14

Item 17 asked *which public operations appear in no test that runs `spirv-val`* and found fifteen;
writing those tests took twenty minutes and the first run rejected `Lanes::dot_unsigned`, which had
been emitting `OpUDot` with a signed result type. The surface has grown by a dozen items since, so
the question was asked again — this time as *which public functions have no consumer at all*, over
all 201 of them.

**Four.** Not one of them was reachable from a caller, a unit test or the validator:

| | what it is | what happened to it |
| --- | --- | --- |
| `Module::f_ord_greater_than` | `OpFOrdGreaterThan` | **deleted** — a second spelling of an instruction `Lanes::greater_than` already emits through `Element::GREATER_THAN` |
| `Module::atomic_exchange` | `OpAtomicExchange` | finished: `Kernel::atomic_exchange_at`, a kernel, a device test |
| `Module::atomic_load` | `OpAtomicLoad` | finished: `Kernel::atomic_load_at`, a gather kernel, a device test |
| `Module::subgroup_all_equal` | `OpGroupNonUniformAllEqual` | finished: `Lanes::all_equal` and `all_equal_uniform`, a kernel that branches on it, a device test |

`notes/NEXT.md` item 5 recorded all five atomics as "built" on 2026-08-12. Three of them were;
two were *emittable*, which is a different claim and reads the same in a list.

**All four were valid, which is the honest outcome and not the expected one.** The last audit found
an invalid instruction on its first run; this one found none, and the value is the same either way
— the difference between "no test has ever looked" and "a test looked and it was right" is the
whole of what the check is for.

### What the three that were kept turned into

- **The exchange is checked by a chain, not by a total.** Every invocation swaps its own index into
  one slot and keeps what it displaced. Whatever order the scheduler picks, the values handed out
  plus the one left in the slot are exactly the marker together with every index, each once — so a
  lost exchange shows up as a duplicate and a torn one as a value that was never in the chain.
  Neither shows up in a sum, which is what the histogram tests can see.
- **The atomic load is checked by where it reads**, not by what it returns: `out[i] = in[in[i]]`,
  through an index the data chose, over an input with no fixed point — asserted, so that a kernel
  which read the invocation's own slot cannot agree by accident.
- **The vote is checked by a branch and by two inputs.** One where every subgroup agrees and one
  where a single lane of the *first* subgroup differs: a vote stuck at true fails the second, one
  stuck at false fails the first, and a vote that answered for the dispatch rather than for each
  subgroup fails the second on every device wider than one subgroup.

### And a limit that was reporting the wrong feature

`Limits` offered `subgroup_arithmetic`, `subgroup_clustered`, `subgroup_shuffle` and
`subgroup_ballot` — and no `subgroup_vote`, while three kernels used votes and their tests gated on
the ballot instead. `VK_SUBGROUP_FEATURE_VOTE_BIT` and `..._BALLOT_BIT` are different bits, and
`GroupNonUniformVote` and `GroupNonUniformBallot` are different capabilities. Every device here
offers both, so the gate was right by luck on all three — the same shape as a test that takes a
width parameter and then ignores it.

## The feature bits, laid beside the capabilities — 2026-08-14

The audit that found four unreachable functions also found a limit reporting the wrong feature:
`Limits` had `subgroup_ballot` and no `subgroup_vote`, while three kernels used votes and their
tests gated on the ballot. Asking the question properly — *every capability this emitter can
declare, against every feature bit the runner reports* — turns up more of the same.

| capability | Vulkan bit | reported before |
| --- | --- | --- |
| `GroupNonUniform` | `BASIC` | **no** — and every lane kernel in the library declares it |
| `GroupNonUniformVote` | `VOTE` | no, added earlier the same day |
| `GroupNonUniformArithmetic` | `ARITHMETIC` | yes |
| `GroupNonUniformBallot` | `BALLOT` | yes |
| `GroupNonUniformShuffle` | `SHUFFLE` | yes |
| `GroupNonUniformShuffleRelative` | `SHUFFLE_RELATIVE` | **no** — and the whole scan rests on it |
| `GroupNonUniformClustered` | `CLUSTERED` | yes |

Two of seven, and the second is not a nicety: the clustered ladder is `log2(cluster)` `ShuffleUp`s
and `Lanes::shift_up`/`shift_down` are nothing else, so every scan kernel in the library declares
`GroupNonUniformShuffleRelative` — while the tests that run them checked the *arbitrary* shuffle.

**Every one of these was right on all three devices**, because no implementation offers one of these
without the others. That is the same shape as a test that takes a width parameter and then ignores
it: correct until the day it is not, and nothing in the codebase can tell the difference.

### The fix is a mapping rather than four more gates

`Limits::supports(Capability)` writes the correspondence down once, and
`Limits::unsupported_in(&spirv)` reads the requirement out of the **module's own** `OpCapability`
instructions — so a kernel that starts needing something new brings its own gate with it instead of
waiting for a test author to remember. That needed one thing from the emitter: `Capability::ALL` and
`Capability::from_word`, the inverse of the encoder, with a round-trip test that is total.

`runner/tests/execution.rs` now checks the kernels it runs against the device that runs them, and
the fuzzer's gate is `Limits::subgroup_surface()` — the five bits a generated program can reach,
named once, where it used to name three of them at six call sites.

### And `simdr probe` was telling a caller the wrong thing

The tool that exists so nobody has to guess listed four features, two of them under the wrong
heading: `any, all` sat under **ballot** and `shift_up, shift_down` under **shuffle**. It lists
seven now, each naming what that bit actually permits — which is the point of the command, and it
had been wrong since the day the shifts were written.

## A corpus built to expose wrong answers made one operation unobservable — 2026-08-15

The gate over the day's work came back **27 of 32, five survivors**, and none of them was wrong
code. Two were claims nothing checked; the third is worth keeping.

`fuzz::corpus` makes every element **distinct**, and says why: a wrong lane mapping shows up
immediately when no two lanes hold the same number. That property is the whole reason the
differential fuzzer catches mapping bugs — and it is exactly the property that makes a vote *about
agreement* never pass.

So `Op::AddIfAllEqual` was generated in hundreds of rounds and did nothing in every one of them. The
mutation gate said so precisely: flipping the reference's `if agreed` to `false` changed no sweep's
answer. A step that cannot pass is a step nobody is checking, and it looks identical in the counts
to a step that always agrees.

**The general shape:** a corpus is a hypothesis about what makes failures visible, and every
hypothesis excludes something. This one excludes uniformity, which is the input class three of this
crate's operations exist for — the vote about a value, `if_uniform`, and the fast path a kernel
takes when its subgroup wants the same thing.

The fix is not a different corpus — the distinctness is load-bearing — but a second one, used where
the first cannot see: a uniform input, plus the same input with a single lane changed, which is what
separates a per-subgroup vote from a per-dispatch one.

`Domain::equals` was the same shape one layer down. All three mutants of its `is_float()` branch
survived, because on small positive integers a numeric comparison and a bitwise one agree
everywhere. Its two real cases — `+0.0` equals `-0.0`; a NaN equals nothing, and the same bits as an
*integer* equal themselves — are now stated where no generated round has to reach them.

**Re-run after the fixes: 32 of 32 killed, no survivors, over 968 lines.**

### And the width sweep found one more, as it usually does

Running the whole suite at width **4** after the vocabulary changed: `the_fuzzer_notices_when_the
_answer_is_wrong` failed with `TooManyStrips { strips: 16, limit: 8 }`.

Not a new bug — a latent one. The generator draws its lane count from a fixed list, and on a
four-wide subgroup a vector of 64 is sixteen strips, which the emitter refuses by name. Every sweep
in that file already treats a refusal as `Outcome::Refused`; this one test called `.expect("built")`
and had only ever run where the seeds happened not to draw one.

Adding two operations changed which programs the seeds produce, and the test met a refusal for the
first time. **The fifth time a width sweep has found something the width did not cause** — it moves
the dice, and what falls out was already there.

---

## Four checks that were right where they were written — 2026-08-15

An audit that asked, of every check in the tree, *where is this called from* rather than *does this
work*. Four answers were "from fewer places than it is about", and none of the seven layers could
have said so: each of these checks passes its own tests, and a check that guards one caller reads
exactly like a check that guards all of them.

### `dispatch::extent` guarded one of the six ways this crate dispatches

`Gpu::run` was checked. `Gpu::run_bound`, `Session::dispatch`, `Gpu::run_chain`, `Gpu::reducer` and
`Gpu::scanner` were not — and every one of them can write past the end of a binding from safe code,
which is undefined behaviour rather than a wrong number. This is the layer that found **eleven tests
reading past their input** the first time it ran, listed in `README.md` as one of seven, covering a
sixth of its subject for four days.

Extending it needed a different question. One length for every buffer is what `Gpu::run` allocates
and it is the wrong shape for the rest: a reduction reads four strips from binding 0 and writes one
scalar per invocation to binding 1, so a single number has to take the larger and would refuse an
output buffer that is exactly right. The strip count is recovered **per binding** now — the access
chain names the variable, the variable carries its `Binding` decoration, and the `OpIAdd`/`OpIMul`
between them say how many elements each invocation touches. `dispatch::extent::addressing` is that
walk; `runner/tests/bounds.rs` asks each of the six doors the same question.

Two deliberate omissions, both in the safe direction. A binding whose address does not depend on the
invocation — `Kernel::store_at` writing one total per *workgroup* — is left out rather than guessed
at, because over-counting one refuses a dispatch that is correct. And a module whose addressing the
walk cannot read is let through: **"this runner cannot tell" must never be reported as "your module
is wrong"**.

### A shuffle's operand was bounded for one mapping and unbounded for the other two

`Lanes::butterfly` refused a mask at or above the vector's width for a **clustered** vector, from
the day clustered shuffles were allowed. For a whole-subgroup or strip-mined one it refused nothing:

```rust
lanes.butterfly(value, 4096)   // on a 32-wide subgroup
```

builds a module `spirv-val` accepts, that every device runs, and in which every lane reads a lane
that does not exist. SPIR-V leaves the result undefined; a device answers with whatever was in the
register. `broadcast`, `shift_up` and `shift_down` were the same.

Nothing below the API can see this. The validator does not know the subgroup width — it is a
property of the device, not of the module — so this is the second thing in this project that only a
bound in the API can catch, after the one `dispatch::extent` exists for.

The bound is one rule now rather than one row's rule: the operand may not reach outside the lanes
the vector occupies, which are the subgroup's or the vector's own where it is narrower. It costs
nothing — every one of these takes a build-time `u32` — and **32 000 fuzzing rounds refused none of
them**, which is the check that it bounds what it should and not what it should not.

### `Shape` carried four numbers and `Kernel::new` checked three

`Shape::new(0, 64, 2)` built a kernel, stored to a buffer and finished a module the validator
accepts. So did a width of 24. The width is the number `decisions/DR-0002` makes the whole module
specific to — and it was checked in `Lanes::new`, which a kernel with no lane operation never
reaches. The buffers, the workgroup and the rows had all been refused from the first day.

The fix moved the same bound to where the shape is. It also removed a guard next door: the workgroup
scan checked `subgroup == 0 || !workgroup.is_multiple_of(subgroup)` and reported it as
`BadShape { workgroup, buffers: subgroup }` — a subgroup width in a field named for a buffer count,
printing *"a kernel of 64 invocations over 24 buffers describes nothing"*. Half of that condition is
unreachable now and the other half says what it means.

### The address arithmetic saturated

`Kernel::address` computed `strip × workgroup + offset` with `saturating_mul` and `saturating_add`,
with a comment explaining that no shape a device accepts comes near the limit. True of the shape —
and `offset` is an ordinary argument of `Kernel::load_offset`, so an offset near `u32::MAX` produced
a *different* element index rather than a refusal. Saturating turns an address nobody can express
into one that exists, and the module says nothing about it.

Both terms are computed in `u64` and refused by name now, which is what this project does everywhere
else it meets this trade.

### The shape all four share

Not "these were unchecked". Each had a check that was **correct where it was written** and had
stopped covering everything it was about — a new entry point, a new mapping, a new field, a new
argument. That is the same failure `Buffer::write`'s safety comment had, and the same one the
mutation gate keeps finding in the fuzzer: whatever is furthest from the thing under test is
furthest from anybody looking.

The habit that found them is worth more than the four fixes: **ask where a check is called from, not
whether it works.**

## The documentation build had never passed — 2026-08-15

`README.md` has listed this among the commands to run since before `.github/workflows/ci.yml`
existed:

```powershell
$env:RUSTDOCFLAGS = "-D warnings"
cargo doc --workspace --no-deps
```

It fails at `HEAD`, and had been failing for as long as anyone can tell: **twelve broken intra-doc
links**, plus a filename collision that had `cargo doc --workspace` writing the library's front page
and the CLI binary's to the same file.

Rustdoc treats `broken_intra_doc_links` and `private_intra_doc_links` as *warnings*, so `cargo doc`
exits zero over all of it. The `-D warnings` in the README is exactly what turns that into an
answer — and nothing ran the command, so the setting had never been applied.

One of the twelve was not cosmetic: `runner::fuzz::reference` is a public function returning
`Reference`, a type that was never re-exported. A caller could call it and could not name what came
back.

**A check that is written down and not run is worse than one that is neither**, because the writing
down is what stops anybody from adding it. It is a CI step now.

## The scan's pass wiring moved inside the gate — 2026-08-15

`runner/src/scan/held.rs` was 651 lines with no tests — excused from the mutation gate as FFI, which
it mostly was. What it also held was `record`: which module each of the seven dispatches runs, over
which buffers, at how many workgroups. That is index arithmetic, it contains no `unsafe`, and it is
the most intricate addressing in the crate.

It is `runner/src/scan/passes.rs` now, and what the move bought is stated as tests that need no
device: every pass reads only what an earlier pass wrote; the last pass writes the answer and
nothing else does; the top is one workgroup and the only one; a map with nowhere to write is refused
rather than dropped. None of those could be asked before.

**The fifth time this seam has been worth cutting** — after `dispatch/step.rs`, `reduction/plan.rs`,
`scan/plan.rs` and `step::upload_bytes`. The rule keeps producing it: *a file is excused for
containing `unsafe`, not for being near it.*

## An operand that was always zero — 2026-08-15

`fuzz::Op::ShiftUp(u32)` carried a distance, drawn as zero and nothing else, because SPIR-V leaves a
shift's out-of-range lanes undefined and a reference cannot predict undefined. The interpreter
ignored the operand; the generator never varied it; a comment on each said so.

`Program` is public. Anything that built a `ShiftUp(2)` would have been compared against the answer
for a different program — silently, since the reference returns values rather than a verdict.

It is `Op::ShiftUp` now, with no operand at all. The invariant was in two comments and is in the
type. The opposite case keeps its `u32`: `Op::RotateUp` *can* be checked at any distance, because a
rotate wraps and every lane reads inside its own vector.

## The sweep run mechanically, and the barrier it rejected — 2026-08-15

The audit that found `OpUDot` asked which public operations reach no `spirv-val` test, and it was
done by reading. Asked again, this time by grep — *which public functions have no reference outside
the file that defines them* — it turned up ten more. Four were the `subgroup_f_add`/`f_max`/`f_min`/
`i_add` wrappers, whose opcodes the typed path already emits through `subgroup_reduce`, and whose
own unit test pins all four apart. The other six had never been validated at all:

| | what it emits |
| --- | --- |
| `Module::subgroup_elect`, `subgroup_broadcast`, `subgroup_broadcast_first` | instructions **nothing else in the crate emits** — the lane API's `broadcast` is a shuffle, deliberately |
| `Module::atomic_store` | the only atomic with no result id, and the last one in the state the exchange and the load were found in |
| `Module::spec_constant_bool` | the one specialization shape whose *default decides the opcode* rather than an operand |
| `Module::memory_barrier`, `Lanes::exp`, `Lanes::if_uniform_value`, `Kernel::store_row_at` | one instruction family each, composed by nothing |

**Eight of the nine were valid.** That is worth stating rather than skipping: the value of the check
is the difference between "nothing has looked" and "something looked and it was right", and only one
of those is a claim.

The ninth was not.

### `OpMemoryBarrier` may not order nothing

```
error: [VUID-StandaloneSpirv-MemorySemantics-10869]
       MemoryBarrier: MemorySemantics must not use Relaxed memory order with MemoryBarrier
```

CI said the same thing in different words, and that mattered. Ubuntu's `spirv-tools` is an older
build and cites a **different VUID** for the same refusal —
`VUID-StandaloneSpirv-OpMemoryBarrier-04732`, *"Vulkan specification requires Memory Semantics to
have one of the following bits set: Acquire, Release, AcquireRelease or SequentiallyConsistent"*.
The first version of the negative test asserted on the string `MemorySemantics`, which the newer
message has and the older one spells with a space. It passed here and failed there.

Which is this suite's own lesson arriving from the other side: **a test that pins a detail it is not
about is a test about the wrong thing.** The claim is that the module is refused *for the barrier*,
so what is asserted is `MemoryBarrier` — in both messages — and the requirement is written out in
the two doc comments rather than cited by a number that depends on which build of the validator
happens to be installed.

`MemorySemantics::None` encodes to `Relaxed`, and this crate's own documentation recommended it —
*"ordering nothing is cheaper than ordering nothing while saying otherwise"*. That sentence is
correct about **atomics**, where `Relaxed` is legal and is what `Kernel::atomic_add_at` uses on
every device here. It is false about a barrier, where the same mask is not a cheaper barrier but an
invalid module.

So the two operations that take identical-looking scope and semantics operands accept different
values, and nothing said so, because nothing had ever built the second one.

The emitter cannot refuse it: both operands arrive as *ids of constants*, and that layer cannot ask
what value a constant holds. What it can do is say so where a caller reads, which is now on
`Module::memory_barrier` and on `MemorySemantics::None` — and assert the boundary rather than
describe it. `tests/instructions.rs` carries both halves: a barrier with `AcquireRelease` the
validator accepts, and one with `Relaxed` it rejects. The second is there for the reason
`a_compute_entry_point_without_a_workgroup_size_is_refused` is: without it, the first could be
passing because the validator never says no.

### What the two sweeps together say

The first was done by reading and found three. The second was done by grep and found six more, one
of them invalid. The difference is not care — it is that a list of things to check by eye is a list
somebody has to keep, and the grep re-derives it from the tree every time.

**An operation with no consumer is not dead code. It is untested code that reads as dead**, and the
two are the same thing right up until somebody calls it.

## Sixty-one gates that named a feature by hand — 2026-08-15

`Limits::unsupported_in` was built on 2026-08-14 for one reason: three kernels using **votes** were
gated on the **ballot**, which is a different capability and a different feature bit. It reads the
requirement out of a module's own `OpCapability` list, so a gate cannot name the wrong feature.

It was used in one assertion test and **zero gates**. The 61 hand-picked gates it was written to
replace were all still there.

Not a stylistic point. Five of them were **under-specified**, in the same shape as the bug that
prompted the tool:

| where | gated on | the kernel also needs |
| --- | --- | --- |
| `control.rs` ×3 | `subgroup_arithmetic` | `GroupNonUniformVote` — `scale_if_any_above`, `branch_only` |
| `loops.rs` ×2 | `subgroup_arithmetic` | `GroupNonUniformVote` — `sum_or_max` |
| `execution.rs` | `subgroup_arithmetic` | the same, and its message already said "vote" |

On a device with arithmetic and no vote, those would have run and failed at pipeline creation
instead of skipping. All three implementations here offer both, which is why it survived — the same
sentence the ballot bug earned.

And one over-specified, which fails the other way. `unrun.rs::ready` gated on
`subgroup_surface() && subgroup_ballot` — the union of everything *any* kernel in that file reaches
— so a device missing one feature skipped **every** test in the file, including the ones that never
touch it. That is lost coverage, and it is silent.

47 sites now call `common::runnable`, which asks each module. Three exceptions remain, each with the
reason written where it is:

- **`Reducer` and `Scanner` build their modules inside themselves**, from a length rather than from
  a caller's SPIR-V, so there is nothing to ask. `reducer.rs`, `reduction.rs`, `scan.rs` and two
  cases in `bounds.rs`.
- **The fuzzer does not know what it is about to generate.** A program's capabilities depend on the
  draw, so `subgroup_surface` — the union — is the honest gate there, and the one place a union is
  the right shape.
- **`shaderSubgroupExtendedTypes` has no capability at all.** A device may accept
  `OpGroupNonUniformIAdd` on a 32-bit integer and refuse it on an 8-bit one with nothing in the
  SPIR-V to say so, so `narrow.rs` and `fuzzing.rs` ask `Narrow` by hand for that one.

### The verification found a hole in the verification

Checking the change meant checking that no test had *started* skipping — and every run in this
session had been counting `SKIPPED` lines out of captured output. **libtest swallows `eprintln!`
from a passing test.** Every "skips: 0" measured nothing at all.

With `--nocapture`, the real numbers: **0 skips at width 32, 17 at 64, 25 at 4** — and every one of
them a *width-shape* reason ("written for a 32-wide subgroup", "no case written for a subgroup of
4") or the known AMD clustered-ladder fault. **Not one capability-reason skip on any of the three**,
which is what a correct conversion looks like on devices that offer the whole surface.

The project's own rule is that a skipped correctness test which looks green is worse than a red one.
It turns out the check for that had the same shape as the thing it was checking: a number that read
as evidence and was produced by a pipe that could not carry it.

## A misspelt device name turned the whole suite off, quietly — 2026-08-15

Measuring the gate conversion above meant running the suite on lavapipe, so:
`SIMDR_DEVICE=llvmpipe`. It came back **157 skips, zero failures, exit code 0**, every one of them
reading `SKIPPED …: no Vulkan device` — printed by a machine with two Vulkan devices in it.

The name was wrong. lavapipe calls itself `llvmpipe (LLVM …)` only when its ICD is on the loader's
path, and it was not; the two devices that *were* there are called something else. Nothing had gone
wrong with any code under test. The run simply never happened.

`Gpu::open_matching` collapsed two states into one:

```rust
Err(Error::NoComputeDevice) if pattern.is_some() => Ok(None),
```

under a doc comment calling a machine "without the part being asked for" a normal state for a test
suite to find. It is not. **Passing a name is asserting that device is here.** The assertion being
wrong is a typo, an ICD path never exported, a driver that did not install — and every one of those
was indistinguishable from having no GPU at all, which is the one state the suite is designed to
skip over without complaint.

Three outcomes now, each meaning exactly one thing:

| outcome | what it means |
| --- | --- |
| `Ok(None)` | no Vulkan loader. The environment is bare, not broken — skip. |
| `Error::NoComputeDevice` | a loader, and nothing behind it that can compute. |
| `Error::NoSuchDevice { wanted, present }` | devices are here, and none of them is the one named. |

`NoSuchDevice` carries what *is* here, so the message says what to have typed:

```
SIMDR_DEVICE names a device that is not here — no device here is called "llvmpipe"
— SIMDR_DEVICE matches a substring of ["NVIDIA GeForce RTX 4080", "AMD Radeon(TM) Graphics"]
```

and `common::device` panics on that one rather than skipping. Exit code 101 instead of 0.

The second row lost a condition on the way past. `NoComputeDevice` was an error without a pattern
and `Ok(None)` with one — for a machine state that a pattern has no bearing on either way. That
guard existed only to reach the row below it.

### Why the harness has to be the one that fails

A skip is invisible: `libtest` swallows `eprintln!` from a passing test, so 157 of them and 0 of
them print the same summary. That is exactly the shape this suite is built to refuse — and the CI
workflow already refuses it once, failing the lavapipe job outright when no ICD is found, on the
stated grounds that "the tests below would skip and pass over nothing". The same sentence was true
of `SIMDR_DEVICE` and nothing checked it.

`SIMDR_DEVICE` is not a convenience. It is the *only* way the second subgroup width ever runs —
DR-0002 is the record the whole lane API is shaped around, and until that variable existed only one
width had ever been tested. A two-device sweep exists to show that both ran. A typo in the variable
that chooses them must not be able to report that they did.

`runner/tests/selection.rs` now covers all three answers. There had been no test of any of them.

### The same shape, one directory away

`SPIRV_VAL` is how CI says where `spirv-val` lives. `validator()` read it and returned
`path.is_file().then_some(path)` — so a path with nothing at it gave `None`, which every caller
reads as *the validator is not installed* and skips over:

```rust
let Some(tool) = validator() else {
    eprintln!("SKIPPED {label}: spirv-val not found (set SPIRV_VAL)");
    return Ok(());          // <- every validation in both test trees, green
};
```

A typo in one environment variable turned off every validation this project has, in both crates, and
left the run green. It is an `assert!` now, for the same reason: naming a path is asserting
something is at it.

Two variables, the same mistake, and both are the ones CI depends on to be pointing somewhere.

### And the count of what did not run is now a number CI checks

`--nocapture` on all three test steps, because without it none of these skips were ever reaching a
log. Then the lavapipe matrix carries the number of tests that legitimately cannot run at its width,
and the step fails when the run disagrees:

| width | skips | why they skip |
| --- | --- | --- |
| 4 | 25 | 11 written for a 32-wide subgroup, 6 narrow types, 3 with no case at 4, 1 that will not strip-mine |
| 8 | 22 | the same 11, 3 narrow, 3 with no case at 8 |
| 16 | 18 | the same 11, 3 with no case at 16 |

Not one of them a capability reason, at any width, which is what the module-derived gate above
should produce on a device offering the whole surface.

The workflow already refused this once — it fails outright when no lavapipe ICD is found, on the
stated grounds that the tests would otherwise "skip and pass over nothing". That guarded one route
to an empty run. This guards the others, and it means a gate that starts over-skipping shows up as
red rather than as a summary line that looks exactly like success.

### The skip count went red on its own first run, twice, and both were real

The check above was written from three local sweeps: 25, 22 and 18. CI disagreed with all three
immediately, and neither reason was noise.

**It counted 4 of 23.** The grep was `^SKIPPED`, and `--nocapture` is precisely what makes that
wrong: `libtest` writes `test <name> ... ` without a newline and finishes the line only after the
body has run, so a skip's own output lands *mid-line*, behind the name of the test it belongs to:

```
test a_strip_mined_dot_product_folds_four_products_per_lane ... SKIPPED dot-product-strips: written for a 32-wide subgroup
```

Locally the same count matched all 25, because a PowerShell `2>&1 | Out-String` capture merges the
two streams differently than a pipe on Linux does. A counting method that agrees with itself on one
platform is not a measurement.

**And every width was one short.** `session.rs` skips its speed *ratio* when `CI` is set — a
shared runner's wall clock is not evidence about setup cost, which that file argues at length — so
CI legitimately skips one test a workstation never does. The numbers are measured on CI now, where
they are asserted, rather than carried there from a machine that runs a different set.

Both are the same mistake as the one that started this: a number produced somewhere other than
where it is used, read as if it travelled. The check earning its place by failing on the environment
it was written for is the outcome to want.

### One more variable of the same shape

`SIMDR_FUZZ_ROUNDS` is how a longer search is asked for, and it was
`.and_then(|value| value.parse().ok()).unwrap_or(ROUNDS)` — so `100_000`, `1e6` or a stray space
sent the run quietly back to 256. Somebody would have watched a 30 000-round sweep they never ran.
It panics with the value now.

Three variables audited, three of the same bug: `SIMDR_DEVICE`, `SPIRV_VAL`, `SIMDR_FUZZ_ROUNDS`.
Every one of them read as *set-and-wrong is the same as unset*, and every one of them is a variable
whose entire purpose is to say where something is or how much of it to do.

## The constant the bounds check could not see — 2026-08-15

`dispatch::extent` has said this about itself since it was written:

> What is outside it is a **constant offset past the run**: `Kernel::load_offset` reads
> `in[i + half]`, and a fold whose dispatch is deliberately narrower than its buffer looks, to this,
> like a dispatch with room to spare. That direction is safe — it under-counts, so it refuses less
> than it might and never more.

Under-counting is the safe direction and it is not the same as no hole. A buffer exactly as long as
the run is one this said a dispatch fit while the kernel read `half` elements past the end of it,
and `notes/NEXT.md` carried it as open on the grounds that "nothing needs it yet".

**Something did.** `kernels::network::clipped_dot` puts activations in the first `offset` elements
of binding 0 and the weights after them, so it reads *exactly twice its run* and always has. Handed
a buffer of only the run, an RTX 4080 dispatched it and returned 512 words of zeros — a full array
of plausible numbers, read from past the end of a storage buffer. That is the undefined behaviour
this file exists to refuse, and it was reachable from safe code through `Gpu::run`.

### It needed nothing declared, again

The same thing was true of the strip count, and for the same reason: the number is already in the
module. `Kernel::address` folds the strip's stride and the caller's offset into **one** constant at
build time —

```text
address = group × (workgroup × strips)  +  local + (strip × workgroup + offset)
          \_____________ base _______/           \__________ shift __________/
```

— so the constant added to the invocation's own lane carries both terms, and the strip term is a
number this walk has always recovered. `shift - (strips - 1) × workgroup` is the caller's offset and
nothing else, because the largest shift on a binding belongs to its last strip. Every other binding
gives zero, which is the arithmetic agreeing with the answer the file gave before there was an
offset to add.

A second copy of the offset — carried out of `Kernel::load_offset`, or decorated onto the module —
would have been a second thing to keep true. There is no second copy.

### Checked by breaking it

Multiplying the recovered offset by zero and re-running: the unit tests fail, and the device test
fails by *succeeding* — `Ok([0, 0, 0, …])`, 512 words of it, off the end of a buffer on a 4080. The
failure mode and the evidence for it are the same output.

What stays outside, now stated as two things rather than one: a grid kernel's `row × pitch`, which
this does not read at all, and `Kernel::load_offset_by`, whose offset is a *specialization* constant
— a number chosen after the module was built, with no literal in it to find. Both under-count.

### The pitch, which was the larger half — same day

`notes/NEXT.md` item 9 sat beside item 8 with the same wording: outside the check, under-counts, safe
direction. Having just found what item 8's "nothing needs it yet" was worth, item 9 was the obvious
next thing to look at, and it is worse.

A grid's rows are `pitch` elements apart **whether or not the dispatch covers a row**. So a kernel
reading a narrow slab of a wide matrix reaches its last row `(rows - 1) × pitch` elements in, and
the invocation product this compared against counts only the columns dispatched. The two are not
off by a constant — they diverge with the pitch:

| shape | what it reaches | what this compared |
| --- | --- | --- |
| 4 rows, pitch 256, one 32-wide workgroup across | **800** | 128 |
| 64 rows, pitch 4096, one 32-wide workgroup across | **258 080** | 2 048 |

`plane.rs`'s own header describes that second shape and calls it supported — "a buffer whose rows
are 4096 long reads a 64-wide slab of it, and `pitch` is 4096". Every grid test in this crate
dispatches `pitch / width` workgroups across instead, which covers a whole row; and on a whole row
`(rows - 1) × pitch + columns` **is** `rows × columns`, exactly. The two readings agree on every
test that exists and diverge without bound off them.

Checked by breaking it: with the pitch suppressed, the device test fails by succeeding — a session
holding 128 words per binding dispatched a kernel that writes 800 and returned `Ok(4.16µs)`.

### Two more things the same walk had wrong

**`LocalSize` was being read as a product where the addressing wants the x axis.**
`Kernel::run_start` emits `group.x × (Shape::workgroup × strips)`, and `Shape::workgroup` is
`LocalSize`'s x alone — while the strip count was recovered by dividing that constant by
`x × y × z`. For a grid two rows deep with two strips that is `2 / 2 = 1`, and the strip count comes
back as one. Nothing has both today, which is the only reason it held.

**The row was matched by its shape, and its shape is not unique.** `row = group.y × rows + local.y`
and `start = (group.y × pitch) + run` are both an `OpIAdd` over an `OpIMul` over `group.y`. The
first version of `row_of` found `start`, reported no pitch at all, and quietly went back to counting
invocations — which looked exactly like working. `local.y` on the right is what tells them apart.

Three of these in one file in one sitting, and all three have the same shape: an expression that
happens to be unambiguous on every module this crate emits today.

## Five runs of the gate over one file, and the word "equivalent" three times — 2026-08-15

The mutation gate over the day's address-walk work, scoped to the same span each time:

| run | killed | score | what it named |
| --- | --- | --- | --- |
| 1 | 17/27 | **63%** | eight of ten survivors in the four conditions identifying a grid's row |
| 2 | 23/27 | **85.2%** | the four `&&`, after the row-two-deep tests |
| 3 | 26/30 | **86.7%** | one killed, and the uniqueness rule arrived as a survivor of its own |
| 4 | 27/30 | **90%** | the two clauses saying what the row's sum is *over* |
| 5 | **30/30** | **100%** | nothing |

The first run's clustering was the finding, not the score. All eight of those mutants sat in
`row_of`, and they sat there because **a workgroup one row deep computes its row as `group.y` alone
and never builds the sum** — so `row_of` returns before the conjunction, and every grid kernel in
this crate, every test above them and every unit test written that day took the short branch.
`row = group.y × rows + local.y` was decoded by nothing at all.

Which is the branch that had already been wrong once that morning: the sum has the same shape as
`start = (group.y × pitch) + run`, the address the row is *used* to compute, and the first version
matched that instead.

### The word "equivalent", three times, wrongly

Three survivors were argued to be equivalent mutants, each with a careful argument about the module
rather than the tests:

* the only `OpIMul` over `group.y` is the row's base, and the only term using that base is the row —
  so each half of the conjunction selects the same one instruction;
* of the sums in an address exactly one has a constant on its right, and it is the one `shift_in`'s
  left-hand clause already names.

Every one of those is **true of every module this emitter produces**, and that turns out to be a
much smaller claim than it sounds. This file decodes SPIR-V; the clauses exist for the SPIR-V it did
not write. The fix was to write it — four edits to a real kernel, none of which changes an address:

| the edit | what it makes | what it pins |
| --- | --- | --- |
| every `OpIAdd` copied under a fresh id | two rows | the match must be *unique* |
| a sum over the row's base adding something that is not `local.y` | not a row | what the sum is over |
| `i_add(index, k)` spliced into an access chain | a constant off the lane | whose sum carries the offset |
| `OpExecutionMode` removed | no workgroup size | the divisor is not zero |

The third has to be **reachable** where the first two do not, and that difference is the design in
miniature: `row_of` scans every term in the module, while the offset and pitch walks follow only
what an access chain reaches. So the copies are appended and ignored; the splice goes in *in order*,
in front of the chain it repoints, and the module stays well formed.

`k` is the largest constant of the index's own type, because the walk takes a maximum — a constant
under the strip stride would be folded away by it and prove nothing. Applying the mutation by hand
confirms it: the offset comes back as **64 where the kernel reads 0**, which is the loose reading
asking for more buffer than the kernel touches. That is the one direction this check must never
take, and nothing before this could have told.

### And one guard that arrived as its own survivor

Requiring the row to be unique killed a mutant and immediately became one: nothing emits two sums of
that shape, so `if sums.next().is_some()` could not fire. **A guard that cannot fire reads exactly
like a guard that works** — the same sentence as the skipped test that looks green, one level down.

## 97% of one answer is not computation — 2026-08-15

A question from next door: `H:\schaak` is a UCI chess engine whose NNUE layer this project's
`kernels::network::clipped_dot` was modelled on. Can this device evaluate its network and buy Elo?

`runner/examples/latency.rs` exists to answer exactly that — its header has said *"a game-tree
search asking for one evaluation, say"* since it was written. **Its table could not support the
answer**, and fixing it is half of this entry.

### The instrument was wrong first

Rows one and two were host wall clock around `Gpu::run`. Row three was a *device timestamp* from
`time_repeated`, printed under the same `per answer` heading, and divided by the answers it produced
where the others were not. So the batched row read about **500× better than any caller ever sees**:
1.9 µs was what the device spent, not what the host waited, and it was per *dispatch* rather than
per answer.

Two clocks under one heading, one of them measuring something the caller never experiences. The same
failure as the skip count that could not be counted, in the example written to settle the very
question it was misreporting.

Rewritten: every row is wall clock around the whole operation — submit, wait, **read the answer
back** — the device figure is printed *beside* it rather than in place of it, and the per-answer
column is derived from the answers a dispatch actually produces. The held `Session` is measured too,
because `Gpu::run` rebuilds its pipeline every call and understating the device's case would make
the conclusion worthless.

Two of my own errors on the way, both caught before the number was used: the device total was
divided by 200 while the timing helper called the closure 601 times — a **3× overstatement of the
device's share** — and a row labelled "one answer" was dividing by two, because a 64-invocation
workgroup holds two 32-wide subgroups.

### What it says

| RTX 4080 | per call | per answer | of which the device |
| --- | --- | --- | --- |
| 2 answers, built per call | 858 µs | 429 µs | — |
| 2 answers, held session | **100 µs** | 50 µs | 2.9 µs — **2.9%** |
| 2 048 answers, held session | 129 µs | **0.063 µs** | 3.6 µs — 2.8% |

| Integrated Radeon | per call | per answer | of which the device |
| --- | --- | --- | --- |
| 1 answer, held session | **779 µs** | 779 µs | 2.5 µs — **0.3%** |
| 1 024 answers, held session | 878 µs | 0.858 µs | 11.4 µs — 1.3% |

**Ninety-seven per cent of a single answer is submission, fence and copy-back**, and 99.7% on the
integrated part. Which is why two answers cost 100 µs and two thousand cost 129: the round trip is
fixed, and the only lever is how many answers divide it.

### The answer to the question, and why it is a decision

`decisions/DR-0008` records it. The break-even is `R / (c - d)` independent answers, and **never**
when the CPU's per-answer cost `c` is at or below the device's `d`.

schaak's evaluation is **78 ns** in-tree, measured by differential timing in its own `SPEED.md`,
against 63 ns per answer here — so the break-even is **~9 700 evaluations pending at once**, and
alpha–beta supplies **one**, because a node's score decides whether its siblings are searched at
all. That is what pruning is.

Two independent ceilings sit above that arithmetic and were already measured by the engine:
evaluation is ~20% of a 376 ns node, so a *free* evaluation buys ~25% nps ≈ **15 Elo**; and the
network is **not adopted**, since distilling the engine's own hand evaluation asymptotes to parity
with it. Its real bottleneck is playing games — `NNUE.md` calls the WDL loop *"a large, multi-day,
self-play-bound compute project"* and `TUNING.md` records a tune that improved validation loss and
lost **−24.7 Elo** over the board. Self-play is alpha–beta search, which is the one workload a GPU
is worst at.

And a structural refusal underneath: schaak is `#![forbid(unsafe_code)]` with an empty dependency
table, in its crate description. `runner` is `ash`, FFI and `unsafe` by necessity. Linking them ends
both claims.

**The useful half is what it rules in.** Batch size beats kernel quality by two orders of magnitude
— 2 answers to 2 048 is 800×, where no kernel change recorded here has been worth more than 3×. So
the shape to keep building is the one the reduction and scan chains already have: one question over
a whole buffer, one submission. And the most valuable workload in this repository is not a speed one
at all — it is the differential fuzzer, whose 30 000 programs are what find emitter bugs.

## Three passes at the fuzzer, which DR-0008 had just put first — 2026-08-15

The boundary decision reordered the tree: batch size beats kernel quality by two orders of
magnitude, so the differential fuzzer is the most valuable workload here and is not a speed one at
all. Three passes followed from that, and each found something the one before could not see.

### What it could not generate

Fifteen operations against a lane API with fifty-odd. Two of the gaps were the shape this project
has already been caught by.

**`Lanes::broadcast` was reached by no generated program.** It is the cross-lane operation whose
answer is *fully determined* — unlike a shift, every lane reads a lane that exists — and the one
whose mapping does the most work: a whole-subgroup vector reads lane `source` of the subgroup, a
clustered one reads position `source` of **its own cluster**, a strip-mined one does it per strip. A
reference reading `source` as a subgroup lane agrees for the first cluster of every subgroup and
differs for the rest, which is exactly how `reduce_min` came to fold its strips with a maximum.
Checked by breaking it: substituting the subgroup for the vector is caught at **seed 5**, by five
tests including the clustered one.

**`Lanes::shift_down` had no `Op` while `shift_up` did.** An asymmetry rather than a decision, and
of the two directions the fuzzer proved one instruction emitted, declared and harmless while saying
nothing at all about the other.

`fill` takes `lanes` as well as `subgroup` now, because they are different bounds: a butterfly's
mask must stay inside the *subgroup*, a broadcast's position inside the *vector*.

Two things that exposed rather than caused. `the_fuzzer_notices_when_a_scan_is_wrong` rewrites
`program.lanes` under operands drawn for another mapping — fine until an operand is bounded by the
vector, and a butterfly's mask has the same shape and had simply never collided at seed 7. And the
sweep counted refusals without ever saying why; the 32 at subgroup 4 turned out to be the strip
limit, which took a diagnostic to establish rather than a reading.

### The comparison written twice

The gate over the **whole** fuzz module — 10 files, 3 417 lines, scoped with `NOHA_ONLY` rather than
by diff, which it had never been. `notes/NEXT.md` records the reason: *"nine of the twelve were in
the fuzzer or its CPU reference. None was in the emitter."*

**170 of 171, 99.4%, one survivor:**

```
runner/src/fuzz/generate.rs:70  [<→<=]  let clustered = lanes < subgroup;
```

Four lines above it the same relationship is already decided three ways, under a comment saying so:
*"the mapping is a three-way choice and it used to be asked as a yes-or-no."* The pool was fixed and
the finish beside it stayed a yes-or-no.

The mutant is not cosmetic. `clustered` decides one thing — whether `Finish::SumOrMax` can be drawn
— and that is the only finish carrying a value out of a branch through an `OpPhi`, which this
module's header calls *"the failure mode no other layer here catches, because a phi naming the wrong
predecessor validates cleanly and then computes the wrong thing"*. Flipping it tells a
whole-subgroup program it is clustered and silently deletes the phi coverage of the commonest
mapping. Everything stays green.

**The duplication was the survivability, and that is the whole finding.** One `shape_of` returning a
named three-way `Shape`, with both the pool and the finish derived from it — and flipping the single
comparison now fails **four** tests, two of which already existed. Coverage that was there all along
could not reach a second spelling of the thing it covered. Re-run: **172 of 172, 100%.**

### The modules nobody read, handed to nobody

`validated.rs` opens by explaining why it exists — `OpUDot`, valid-looking, correct on two devices
for weeks, caught by the first `spirv-val` run. The kernel library got that layer. **The generator
did not**, and it builds thousands of modules a run: a rolled loop feeding a clustered scan, a
broadcast under a vote, four steps whose combination no author chose. Every one went straight to a
driver.

This pass was planned as something else — "every opcode the emitter knows appears in a validated
module", approximated by asking whether each `pub fn` is named in a validating file. **Measured
before writing: 75 are not, and the instrument is wrong rather than the code.** `Lanes::rotate_up`
is validated through a kernel that calls it without spelling its name; a grep cannot answer a
question about what an instruction stream contains. The measurement turned up the sharper gap
instead.

`every_generated_shape_is_valid_spirv` needs **no device**, which is the point: `Program::build` is
the emitter alone, so it sweeps every width rather than whichever is plugged in. 232 modules over
five widths and eight domains, all three mappings, six finishes — and every one valid.

Nothing was found, and that is worth stating plainly: the value of a check is the difference between
"nothing has looked" and "something looked and it was right", and only one of those is a claim.

## The whole tree at full scope, and the three things it said — 2026-08-15

The mutation gate had never run over this tree whole. A single full-scope run needs ~640 mutants,
which exceeds the configured cap and takes longer than the MCP client will wait — so every run
before this was **diff-scoped**, covering only the lines a commit changed. That answers "did this
change arrive covered", never "is this file covered".

`NOHA_ONLY` takes a comma-separated path list, which turns one impossible run into five possible
ones. Driven from the CLI in a background task rather than through the MCP tool, the client timeout
stops applying at all.

| shard | targets | mutants | score |
| --- | --- | --- | --- |
| `runner/src/fuzz/` | 10 | 172 | 100% *(after a fix)* |
| `runner/src/dispatch|reduction|scan/` | 11 | 139 | **100%** |
| `src/lanes/` | 18 | 105 | **100%** |
| `src/module|spec|kernel/` + `decode`, `encode`, `half`, `lib` | 37 | 162 | **100%** |
| `runner/src/kernels/`, `timing`, `cli` | 17 | 61 | 100% *(after a fix)* |
| **whole tree** | **93** | **639** | **100%** |

Two of the five shards were already perfect, and one of those is the claim this file has carried for
weeks on the strength of an old truncated run: *"none was in the emitter"*. It is measured now
rather than assumed — 267 mutants across the whole of `src/`, including `module/` and `spec/`, which
is where `DR-0001` says a number invented from memory produces a module that assembles cleanly and
means something else. Nothing survived there.

### And a fourth thing, found before the gate ran

Preparing the first pass — widening the fuzzer to the bit shifts — meant reading what they needed.
`Lanes::shift_left`, `shift_right_logical` and `shift_right_arithmetic` took `T: Element`, and
**`F32` is an `Element`**. A shift of a vector of floats compiled, built, and produced
`OpShiftLeftLogical` with a float result type, which SPIR-V forbids. Confirmed with a probe, not
reasoned: `spirv-val` rejects the module.

Reachable from safe code, spelled plausibly, illegal, with nothing refusing it and nothing
validating it. `OpUDot`'s shape exactly.

The fix was already in the crate and had not been applied here. `Signed` exists as a second trait
with its argument written out — *"an `Option<Glsl>` would have made that a runtime error, and an
`OpCopyObject` would have made it silently fine. Refusing at the type is neither."* `Integer` is the
third, so the call **cannot be written**. Its check is a `compile_fail` doctest, which is the only
artefact that can assert what a program cannot be.

A sweep of the rest while it was open: every other `Element`-bounded lane operation is one SPIR-V
defines for all of them, and the float-only maths take no `T` at all. The three shifts were the only
instructions SPIR-V restricts that the bound did not.

### The two survivors, and why the same shape keeps coming back

Both are a *second copy of a comparison*, and neither could be seen by the coverage that already
existed for the first copy.

**`generate.rs:70`** — `let clustered = lanes < subgroup`, four lines under the same relationship
taken three ways. `clustered` decides only whether `Finish::SumOrMax` can be drawn, and that is the
one finish carrying a value out of a branch through an `OpPhi` — *"the failure mode no other layer
here catches"*. Flipping it tells a whole-subgroup program it is clustered and deletes the phi
coverage of the commonest mapping, silently. Merging the two spellings into one `shape_of` made
**existing** tests fail on the mutant: the duplication was the survivability.

**`kernels/reduce.rs:315`** — `if LANES > subgroup`, in a file that already carries a note about
meeting this shape once before. Two mutants at once, invisible for two different reasons: one
consumer *reports* a build refusal rather than failing on it, and the other skips the equal case by
name. A cluster exactly the subgroup's width is the shape neither asserts, and it is not a missing
mapping — it is a whole-subgroup vector. The `cond→false` half is sharper still: without the guard
the call is refused anyway, by the *butterfly's* bound, as a mask reaching outside its subgroup —
a true statement about a different thing, printed identically as "not built". What the guard is
worth is not the refusal but which one.

Three instances of one shape now: `interpret::strips_of`, `Program::input_len`, and these two. The
habit that finds it is worth naming — **a relationship decided in two places is decided in one place
and copied**, and the copy is invisible to every test that guards the original.

## Two shapes, named — 2026-08-15

Everything found this week is one of two things. Both are invisible in the same way: the code reads
correctly, the tests pass, and the thing that is wrong is *somewhere else*.

### A relationship decided twice

A rule written in two places is not two spellings of one rule. It is **two rules that agree on the
inputs anybody draws**, and they diverge in the cases nobody does.

Three instances, and the third was found by accident while fixing the second:

| where | how it was written | how it diverged |
| --- | --- | --- |
| `interpret::strips_of` | `if lanes > subgroup` | the arms are the same answer at equal widths, so nothing could tell them apart |
| `fuzz::generate` | `lanes < subgroup` | called a whole-subgroup vector *clustered* under one mutation, silently deleting `SumOrMax` — the only finish that carries a value out of a branch through an `OpPhi` |
| `kernels::reduce` | `LANES > subgroup` | refused a cluster exactly the subgroup's width, which is not a missing mapping but a whole-subgroup vector |

And the rule they were copies of, `Mapping::of`, decides by **divisibility** rather than by
comparison — with the reason written beside it: *"a comparison would be indistinguishable from `<=`
— divisibility says the same thing and says it once."* So a seven-lane vector on an eight-wide
subgroup is *clustered* to all three copies and **refused** by the original. Only the generator
drawing powers of two kept four different rules in agreement.

**The asymmetry that makes this class dangerous.** A duplicated *branch* has a mutant, so the
mutation gate finds it — both of the live ones were found that way, at 99.4% and 96.7%. A duplicated
claim in *prose* has nothing at all. And the coverage guarding the original cannot see the copy:
merging the fuzzer's two spellings into one made **existing** tests fail on the mutant, two of which
had been guarding that relationship all along.

The copies existed for a real reason rather than carelessness. `Lanes::mapping::<LANES>` takes the
width as a const generic — which is what `decisions/DR-0002` is about — and both callers held a
width they learned at run time, so neither could reach it. The fix was not to delete the copies but
to give the rule a runtime face: `Mapping::of(lanes, subgroup)`, with the const-generic method as
its one-line front.

**The habit:** when a relationship is decided in two places, it is decided in one place and copied.
Find the copy before the gate does.

### A bound wider than the thing it bounds

An operation that accepts more than it can do. Nothing refuses it, because the refusal would have to
come from a layer that was never asked.

`Lanes::shift_left` took `T: Element`. `F32` is an `Element`. So a shift of a vector of floats
compiled, built, and produced `OpShiftLeftLogical` with a float result type — which SPIR-V forbids.
Reachable from safe code, spelled plausibly, illegal.

**No instrument here could see it, and each for its own reason.** The mutation gate could not: there
was no branch to flip, so nothing survives. `spirv-val` could not: it only sees modules a test
builds, and no test built that one. The type system could not: `Element` is exactly the bound that
was wrong. `clippy` could not: it is a valid Rust call.

That is what makes this class the most expensive of the four in `notes/CLAIMS.md` — it produces
**invalid modules that run**, and drivers are lenient about what the validator is not.

The fix is a third trait, and the crate had already made the argument for the second one:

> `SAbs` and no `UAbs`. … A `Option<Glsl>` would have made that a runtime error, and an
> `OpCopyObject` would have made it silently fine. **Refusing at the type is neither.**

`Integer` now bounds the three shifts, so the call cannot be written rather than being caught. Its
check is a `compile_fail` doctest — the only artefact that can assert what a program *cannot be*.

### And a bound that is right leaves a gap of its own

Narrowing a bound tells you which types an operation accepts. It says nothing about whether they
were ever tried.

`Integer` admits six types; `tests/instructions.rs` validated the shifts at **two**. The four narrow
integers had never been handed to the validator. Pulling that thread found the general case: every
module `spirv-val` had ever seen at 8 or 16 bits came from `kernels::narrow`, which reaches seven
operations — `add`, `clamp`, `load`, `reduce_sum`, `splat_bits`, `store`, `store_scalar`. The
comparisons, selects, extremes, shuffles, votes and scans all accept a narrow element and were
validated at 32 bits and nowhere else.

Narrow is where SPIR-V is fussiest: `Int8` and `Int16` must be declared, the group opcodes differ
between the signed and unsigned forms of one width, the conversions reach `OpSConvert` or
`OpUConvert` depending on the *target's* signedness, and `shaderSubgroupExtendedTypes` is a
permission with no capability in the module at all.

`the_lane_surface_is_valid_for_every_narrow_integer` fills the **type × operation** grid at one
width. The **type × width** grid is still open, and the test says so rather than implying otherwise:
a narrow butterfly on a four-wide subgroup is a clustered shuffle and is validated nowhere.

### What the two shapes have in common

Neither is a mistake in the code that contains it. A copy is wrong because of a rule somewhere else;
a bound is wrong because of a rule in a specification. Both are found by asking a question *about*
the code rather than running it — which is why the two instruments that found them were a mutation
gate and a validator, and why `notes/CLAIMS.md` exists to ask which other claims have neither.

## Seven numbers nothing emits, and a float edge nothing could reach — 2026-08-15

Two findings from the sandbox, and neither was the thing it was looking for.

### The consumer audit had a kind it could not see

`every_public_operation_has_a_consumer_outside_its_own_file` asks the question of every `pub fn`, and
it was written after `Module::memory_barrier` turned up emitting an `OpMemoryBarrier` whose semantics
Vulkan forbids, with no caller and no validator behind it.

**An opcode is a `pub const`**, so the same shape in the same tree was invisible to it. There are
**seven**:

| opcode | what it is for |
| --- | --- |
| `F_CONVERT` | `OpFConvert`, a float at a different width — no `f16`↔`f32` conversion is offered |
| `LOGICAL_NOT` | `Module` has `logical_and` and `logical_or`, and nothing negates |
| `GROUP_NON_UNIFORM_I_MUL` | `Element` names `GROUP_ADD`, `GROUP_MAX`, `GROUP_MIN` — no product |
| `ATOMIC_S_MIN`, `ATOMIC_U_MIN`, `ATOMIC_S_MAX`, `ATOMIC_U_MAX` | `Module` has add, exchange, increment, load and store — no minimum or maximum |

All seven are **half of an operation nobody has asked for**.

`decisions/DR-0001` is why that is worse than it looks. The rule is that every opcode was read out of
Khronos' grammar rather than remembered, and what keeps the rule honest is that a wrong number
produces a module `spirv-val` rejects. **A number nothing emits is a copy of the grammar with no
check behind it** — it can be wrong for as long as it sits there, and whoever reaches for it first
inherits the mistake along with the convenience.

`every_opcode_is_emitted_by_something` asks it now, with the seven excused by name and a reason
each, so an eighth cannot appear quietly.

### Float-to-integer: not added, and the reason is DR-0006's

`OpConvertFToS` and `OpConvertFToU` are not in `op.rs` at all — a gap in the *surface* rather than in
the testing of it. Asked whether they should be added, the answer is no, on two grounds:

* **No caller wants it.** The quantised kernels compute in integers throughout, and `to_f32` exists
  for the outgoing direction — an integer sum shown as a float. This is `decisions/DR-0006`'s
  argument for `Grid` having no third axis, verbatim: *"a third term would have no caller… and an
  untested term is worse than a missing one."*
* **One instruction is not the operation.** `OpConvertFToS` truncates toward zero, and a caller
  wanting a *rounded* integer needs `RoundEven` first — so the instruction is half of what anybody
  would ask for. And SPIR-V leaves the result **undefined** where the value does not fit, so a safe
  API would have to clamp, which is inventing a semantics the specification does not have and paying
  for it on every call. This project has refused that trade twice already: the `FMax` note, and the
  clustered scan by subtraction.

The same argument settles `F_CONVERT`, which is the same shape one type up.

### The f16 edges were outside the fuzzer by construction

`src/half.rs` opens by saying every one of the 65 536 half bit patterns is round-tripped through
`to_f32` and back — **on the CPU**. Nothing checked a device agreed, and the layer that might have
could not: `Domain::Half` has a **ceiling of 8** and refuses any round whose arithmetic leaves
±2048, because that is exactly what lets its comparison be exact.

So denormals, infinities, NaNs, negative zero and every rounding boundary sat outside the differential
fuzzer *by construction rather than by oversight* — on the one type whose whole difficulty is at the
edges, and whose storage path was therefore exercised at values below 8 and nowhere else.

All 65 536 now go through an identity kernel — a load and a store, nothing between them, because
every instruction in between would be a licence for the device to change the value. They survive on
the RTX 4080, the integrated Radeon and lavapipe at 4 and 16.

**What is asserted and what is only reported is the whole design**, and this project has been caught
on that line before: a test once asserted that a sum of sixty-four negative zeros keeps its sign,
which IEEE 754 says and **Vulkan does not require** — it is `shaderSignedZeroInfNanPreserveFloat32`,
binding only a module declaring the matching execution mode. Two GPUs and a local lavapipe preserved
it; Ubuntu's Mesa folded it to `+0.0`.

So bit-exactness through a load and a store is **asserted** — no arithmetic happens, so no rounding
mode, denormal flush or NaN quieting is licensed to touch it — and NaNs are counted apart and
**printed**, because a device may reshape a payload and Vulkan permits it. Asserting there would be
the signed-zero mistake a second time.

Exhaustive rather than sampled, for the reason the conversion probes are boundaries rather than
samples: the interesting patterns are a few hundred of 65 536, and a sweep finds them by luck and
proves it by luck too.

### And then they were deleted

The seven were excused for about an hour and then removed, which is the reading `decisions/DR-0001`
actually supports. Keeping a number nothing emits keeps a copy of the grammar that `spirv-val` never
sees; deleting it costs a doc comment and a minute of `spirv-as` on the day somebody wants it back —
**which is the day it becomes checkable.** The excuse list is empty now and the check is an absolute:
every one of the <!--count:opcodes-->100 opcodes this emitter declares reaches a module.

The list stays in place at length zero, because an exception should be a line somebody writes rather
than a silence, and because the expiry test beside it fires the moment an excused opcode gains an
emitter. Its scanner test needed a new negative case for the same reason the list emptied: there is
no dead opcode left to point at, so it asks about a name no opcode has — which is the shape a new
dead one would arrive in.

## The first step that is not available in every domain — 2026-08-15

`notes/NEXT.md` has named the fuzzer's vocabulary as the highest-value direction in the tree for
three passes running, and named the same blocker each time: *"`Program::build` emits through one
function generic over `Element`, and a shift needs `Integer`. Generating one means emitting per
domain rather than filtering a pool."*

That is now built, and the shape it took is worth recording, because the obvious reading of "emit
per domain" is the wrong one.

### The obvious fix was a second copy of the width ladder

`build_in` picks the element type from the domain and hands it to `build_at::<T, LANES>`, which is a
ladder of nine const-generic arms — `1 => …::<T, 1>()`, `2 => …::<T, 2>()`, and so on, because
`LANES` is a const generic and the generator's width is a runtime value. Emitting per domain reads
like: one ladder for the six integer types and another for the two floats.

**Two ladders is a relationship decided twice**, which is the shape this file catalogues more often
than any other — `reduce_min` folding its strips with a maximum, `shape_of` deciding the mapping a
second way, `strips_of` and `input_len` sharing a comparison that could not distinguish its arms.

So the ladder stays single and the *element type* carries what it can do. `Emit` is one method:

```text
fn bit_shift<const LANES: u32>(lanes, kind, value, by) -> Result<Vector<Self, LANES>, ProgramError>
```

implemented for the six integers by a macro that calls the lane API, and for `F32` and `F16` by one
that refuses. A blanket `impl<T: Integer> Emit for T` beside `impl Emit for F32` is what this wants
to be, and Rust will not take it — the compiler cannot know `F32` is not an `Integer`, so the impls
conflict. That is the whole reason for the macros, and it is worth saying so where somebody will
otherwise try to simplify them away.

### Two gating axes, and they are not the same axis

The generator already had three pools — `CLUSTERED`, `WHOLE`, `STRIPPED` — chosen by the mapping,
which says **which lanes a vector may read**. A bit shift reads no lane but its own, so it is legal
under all three and adding it to each would have been three identical copies.

What gates it is the **element type**. So it is a fourth list beside the three rather than three
entries inside them: two axes, two lists. Combining them would have been six.

### A second kind of no

`Outcome::Refused` carried a `LaneError` because, until this, every operation existed in every
domain and the only thing that could go wrong was the *width*. A step some domains have and others
do not is a second kind of no, and folding it into the first would have made a float program holding
a shift look exactly like a vector too wide to map.

`ProgramError` names three: the lane API's refusal, `NotInThisDomain`, and `ShiftTooFar`. That is
`decisions/DR-0009` applied to the type it says is missing an arm.

### The corpus could not have told the two right shifts apart

`OpShiftRightLogical` and `OpShiftRightArithmetic` **agree on every value whose top bit is clear**,
and every value this generator draws is a small positive number. So the naive draw — a small shift
distance, like every other operand here — would have generated both instructions, run both on four
devices, agreed both times, and proved one instruction twice.

The distance is drawn across the element's whole width instead. A left shift of 31 puts a bit at the
top and the next right shift is a question with two different answers. `Domain::bit_shift` is
asserted directly on that: for every integer domain, the two right shifts differ once the top bit is
set — a claim about the *reference*, made where a device is not needed to check it.

**And the ceiling is the specification's rather than a choice.** A shift by at least the operand's
width is undefined in SPIR-V, so `ProgramError::ShiftTooFar` refuses one. That is the `ButterflyAdd`
lesson verbatim: its mask was drawn from `1 << below(4)`, every distance of which is inside a 32- or
64-wide subgroup — right on both devices here, and wrong on an eight-wide one, found by lavapipe on
seed 3 as a disagreement the fuzzer reported against itself.

### The other asymmetry, which the reference had to be written from

`OpShiftRightArithmetic` sign-extends from the **element's own top bit whatever the type's
signedness says**. An `OpTypeInt` with signedness 0 still spreads bit 7 of a byte. So the reference
reads it from `Domain::bits` rather than from `Domain::is_signed` — and the two would agree for
every signed domain, which is exactly how a reference written the other way would have passed and
then disagreed with a device only in `u8` and `u16`.

### What it cost and what it bought

Six new tests, 124 mutants at 100% over the six changed files, and the vocabulary is 20 kinds where
it was 17. All eight domains agree over 256 seeds each on the RTX 4080, the integrated Radeon and
lavapipe at 4, 8 and 16 — 232 generated modules through `spirv-val` at every width first.

The three operations had, before this, a `compile_fail` doctest and one hand-written test apiece.

## Three gates in one trait, and the edge two of them arrived with — 2026-08-16

`Emit` was written for the bit shifts and read like an arrangement for one operation. Three more
went through it the next day, and what came out is the argument for the shape.

### Six, five, and one

| operation | what `Lanes` requires | domains |
| --- | --- | --- |
| the three bit shifts | `T: Integer` | **6** |
| `abs` | `T: Signed` | **5** |
| `fma` | `Vector<F32, _>`, concretely | **1** |

Three memberships that **do not nest**. Any of them alone could have been a special case; together
they are a mechanism, and the alternative — a width ladder per bound — would now be three copies of
nine const-generic arms.

The trait's default is a **refusal**, so an element type that gains no override refuses everything.
That is the direction `noha gate`'s fail-closed check exists to keep things pointing, and it made
the eight impls a table: `emit_for!(I32, shifts, magnitude)` on one line per element, with the
capabilities across.

`Lanes::all_uniform` needed no gate — it is the third vote, available everywhere the other two are,
and it had unit tests and no generated program. The same asymmetry `Op::ShiftDown` had.

### The magnitude has one input with no answer, and the shifts made it reachable

A two's-complement minimum has no positive counterpart at its own width: `-128` as an `i8` negates
to `-128`. Every device does that and **GLSL.std.450 does not promise it**.

Until this week the question could not arise — every signed value the generator drew was small. A
left shift of 7 in a byte domain lands on `0x80` exactly, and roughly half the draws of that
distance do. So the two operations interact, and the interaction is a value the reference is not
entitled to predict.

The answer is the one `Domain::exact_limit` already gives for a half that leaves its range: **refuse
the round rather than compare it**. `Outcome::Unrepresentable`, counted and printed. The first
device sweep reported exactly one such round in `Byte` and one in `Short` — the mechanism firing on
the domains where the minimum is closest.

This is the shape a test in this repository once got wrong in the other direction: it asserted that
a sum of sixty-four negative zeros keeps its sign, which IEEE 754 says and Vulkan does not require.
Two GPUs and a local lavapipe agreed with it; Ubuntu's Mesa folded it to `+0.0`. Asserting what the
hardware happens to do is how a specification's silence becomes a promise nobody made.

### The fused multiply-add is fuzzable for a reason its own doc comment denies

`Lanes::fma` says: *"never bit-identical [to a multiply and an add], so a kernel that must agree
with a CPU reference exactly has to make the same choice on both sides. That is why the fuzzer's
vocabulary has `min`, `max` and `clamp` in it and not this."*

True in general, and **not true of this corpus**. Every float here is a small integer below the
exact limit, where a product and a sum are both exact — so the fused and unfused spellings give the
same bits and the pair can be held to agreeing. That is the same trade `RepeatAdd` and `RolledAdd`
are here for: one answer, two instruction streams, and they must match.

The doc comment is a piece of inherited reasoning that outlived its scope, which is the class this
file catalogues under *a reason that outlived its conclusion*. It now says which case it means.

### The `all` vote needed its operand argued, not drawn

`AddIfAnyAbove` straddles the corpus so both arms are reached. An `all` vote at the same threshold
almost never passes — the corpus runs from a magnitude of 1 upwards, so `0` is the only threshold
the whole subgroup clears outright. Three values, and both arms appear across a sweep.

**In the signed domains the passing arm arrives through another step.** Every fourth element is
negative and no threshold clears them, so the vote only passes in a program that took a magnitude
first. Two operations that only work together, which is a coverage argument rather than a coincidence
— and the reason the coverage test asserts the *threshold's* range rather than the vote's outcome.

### And a meta-test that had been passing by luck

`the_fuzzer_notices_when_a_scan_is_wrong` perturbs one input element and asserts the reference
notices. Widening the vocabulary moved the random stream, and the seed it settled on for `Scan` over
4 lanes in `Float` was a program that maps both inputs to the same answer — a clamp does it, and so
does an `all` vote both inputs clear.

That is the *program* being right and the check being wrong: the search picked a program that
builds, and never asked whether it could tell the two inputs apart. The condition belongs in the
search, and does now — with the same insistence that a seed exists, because skipping the combination
quietly is the failure that whole file is about.

### What it cost

Vocabulary 20 → 23. Six more tests, 176 mutants at 100% over the seven changed files, 232 generated
modules through `spirv-val`, and all eight domains agreeing over 256 seeds each on the RTX 4080, the
integrated Radeon and lavapipe at 4, 8 and 16.

## The sandbox, and what it left behind — 2026-08-16

`proeftuin/` was built on 2026-08-15 as a place to put the engine under pressure without any of it
becoming part of what the engine claims about itself, and deleted on 2026-08-16. Its README opened
by promising that deleting it was one `rm` and one line; this is the record of what it found, kept
here because the directory is not.

**Three tools, each carrying an exact oracle**, which was the entry requirement: a workload is only
a test if something can disagree with it, and disagreement needs an answer that is *right* rather
than close. That ruled out more than it sounds — a fluid simulation is float chaos with no cheap
exact reference, a procedural world has none at all beyond looking right.

### What it found about the engine

* **The packed dot products had only ever run whole-subgroup.** `kernels::dot` builds every one
  through `whole_subgroup!`, so `OpSDot`, `OpUDot`, `OpSUDot` and `OpSDotAccSat` — the family in
  which `OpUDot` shipped **invalid** — had never executed clustered or strip-mined, where they are a
  different instruction sequence with a different fold behind them. Twelve combinations, of which
  four had ever run. All twelve agree, at every width, on three devices.
* **`convert_u32`'s first sentence needed a qualifier.** It is documented as *"a `u32` value's
  number, as a value of `T`"*, and for `i32` the opcode is `OpBitcast` — so `0xFFFF_FFFF` converts
  to −1 rather than to 4 294 967 295. The two readings are identical for every value below
  `i32::MAX`, which is every loop counter, which is what the method exists for. A reference written
  from the *sentence* would have agreed with the implementation about everything except the answer;
  one written from the **opcode table** did not. `src/lanes/mod.rs` carries that table now.
* **Every `f16` bit pattern survives a load and a store.** All 65 536, exhaustively, on four
  devices — where the differential fuzzer could reach none of the edges, because `Domain::Half` has
  a ceiling of 8 and refuses any round leaving ±2048, which is exactly what makes its comparison
  exact.

### What it found about itself, which is the more useful half

* **It dispatched an invalid module and two devices ran it.** The layer stored an `i32` into a
  `u32` buffer — `kernels::dot` does the `reinterpret` this had left out. An RTX 4080 and an
  integrated Radeon each returned 192 correct-looking answers; lavapipe refused it with
  `ERROR_UNKNOWN` and said nothing about why. A sandbox that dispatches without validating
  reproduces the exact failure `runner/tests/validated.rs` opens by describing.
* **It had three copies of one outcome type.** One per tool, the same four "did not run" arms under
  different names — the shape `decisions/DR-0009` prevents *inside* a harness and did not prevent
  *between* harnesses. A fifth reason would have had to be added three times, and adding it to two
  of the three reads exactly like adding it to all.
* **It was spending round trips the way `decisions/DR-0008` says nothing else matters.** Two of its
  three tools ran one dispatch per seed where the seeds varied only the *data* and shared the
  module: 72 in the test and 384 in the report, where 12 would do.

### The batch lesson, which is the one worth carrying forward

`notes/NEXT.md` had refused three times to invent a batching API because it had no caller. The
sandbox was one, and the design pressure it supplied was a **mistake rather than a requirement**,
which is the more useful kind.

**What made the layer un-batchable was one number.** `Kernel::load_offset` reaches a second operand
at a constant element offset, and the layer passed the size of *one workgroup's* operand — correct
for a single dispatch and wrong for every workgroup after the first, which would have read its
neighbour's activations. A batch of one problem is the only size at which the per-problem offset and
the whole-batch offset agree, which is exactly why the mistake survived every test.

So a batch is **N problems laid out so that the invocation's own index selects the problem**, and
that is a constraint on the *kernel* rather than on the buffer. Any future API here owns the
arithmetic a kernel has to be built against, not a container for the words.

**And the exception is as informative as the rule.** The conversion sweep stayed at seventy-two
dispatches on purpose: its probe value is a *constant in the module*, which is what makes twelve
boundaries twelve modules, and batching them would mean loading the probe from a buffer — a driver
may fold a constant conversion where it cannot fold a loaded one. Twelve round trips buys a stronger
question. Not everything with many problems is a batch.

### And what the deletion itself proved

The isolation contract held. `cargo test --workspace`, `noha gate`, CI and the mutation gate were
unchanged by the removal, because the exclusion was structural — an empty `[workspace]` in the
sandbox and one `exclude` line at the root — rather than promised.

**What did not vanish on its own was the prose.** Seven sentences across four documents and one
emitter doc comment named the directory, and `tests/documented.rs` would have failed on every one of
them the moment the files went. That is the check doing its job, and it is also the finding: a
deletable directory leaves its *code* cleanly and leaves citations of itself everywhere the rest of
the tree explained why something is true.

## The one term the dispatch bound counted as zero — 2026-08-16

`runner/src/dispatch/extent` reads a module's workgroup size, element stride and address arithmetic
and refuses a dispatch that would touch more of a binding than the binding holds. Both of its own
module headers carried the same sentence, and it had been there since the check was written:

> One thing stays outside: `Kernel::load_offset_by`'s offset is a *specialization* constant, a
> number chosen after the module was built with no literal in it to find. It under-counts, which is
> the direction this check must always take when it cannot see.

**The first clause is true and the second is exactly backwards.** Everywhere else in that file a
term it cannot read makes the check *weaker* — a binding it cannot decode is not judged, a module
with two candidate rows gives no pitch, a shape it does not recognise falls back to counting
invocations. Those are all refusals to answer. This one *answered*, with zero, and a bound that
under-counts a reach is a bound that **lets the overrun through**. It was the one fail-open site in
a file whose whole subject is failing closed.

### The number was never unknowable — it was known somewhere else

A specialization constant has no literal in the module because it is chosen when the **pipeline** is
created. And every caller that bounds a dispatch has that value in scope at the moment it asks:
`Gpu::execute` takes a `&Specialization`, hands it to `Pipeline::new` two lines later, and passed
the module alone to the bound.

So `Bounds::of` takes the specialization now, `addressing::specialized` resolves each constant's id
through its `SpecId` decoration to the value the pipeline will carry — the caller's, or the module's
own declared default where the caller sets nothing, which is what the driver does with the same two
numbers — and `open_shift_in` adds it like any other term.

**An argument rather than an option, because forgetting it is the failure this had.** Two of the
three call sites specialize nothing and now say so by passing `Specialization::none()`, which is the
same value their pipeline is built with; the third passes the real one. A caller cannot omit it.

### And a guard that was written, tested, and removed the same night

`OpSpecConstantOp` derives one constant from another — `offset x 2` — and this walk cannot evaluate
that. The obvious companion to the fix above was to **drop such a binding from the answer** on the
reasoning that absent and zero are different claims and only one is honest about not knowing.

**The test written to prove that worthwhile disproved it.** A binding with no entry does not go
unjudged: `overrun_uniform` falls back to the invocation reading. So dropping the binding replaces
one under-count with another — and a *worse* one, because an address carrying both a folded constant
and a derived one would lose the constant this can read as well as the one it cannot.

So the guard is gone and the limit is a sentence: the derived term is not counted, and the answer
stays a floor, which is what this file gives for everything it cannot read. **Counting what is
legible and being a floor beats counting nothing and being a cruder floor.** The reasoning that
produced the guard was the right reasoning about the wrong mechanism, and the only thing that could
have told them apart is the test that ran.

### What it was worth, measured

The existing device test `an_offset_supplied_at_pipeline_time_reads_the_same_elements_as_a_baked_in_one`
dispatches that kernel with an offset of a whole workgroup over a buffer of exactly two, and still
passes — the bound now needs `run + offset` and the buffer is exactly that. Calibration confirmed by
the one test that was already sized right.

What was not checked before and is now: the same module and the same buffer, **refused or accepted
according to a number that appears nowhere in it**. That test is in `runner/tests/bounds.rs`, and it
is also what holds the three call sites honest — a future one that passes `none()` where a real
specialization exists fails there rather than in a driver.

### And a file that was three-quarters test

`runner/src/dispatch/extent.rs` was 1 100 lines, of which 755 were its test module — by a distance
the largest file in the tree, and the *code* in it is 345 lines. The tests moved to
`extent/tests.rs`, the shape `runner/src/fuzz/domain/tests.rs` already had.

**No other file needed it, and that is worth measuring rather than assuming.** Sorted by length the
next eight are the same story: `interpret.rs` is 331 lines of code and 460 of tests, `shuffle.rs`
410 and 410, `access.rs` 296 and 360. Take the test modules off every file in the tree and the
longest body of code left is **577 lines** — `runner/src/device.rs`, which is FFI and has no test
module at all.

Two entries need a footnote, and both make the same point. `generate/coverage.rs` measures 638 and
is a test module in its own right — it is `#[cfg(test)] mod coverage;` from the file beside it, so
it is already split. And `fuzz/program.rs` measures 522, of which most is the doc comments on a
23-entry vocabulary: separating a vocabulary from its explanations is the opposite of an
improvement, and the argument for splitting a file has to be about what a reader is looking for.

There is no thousand-line module here to break up. There was one file whose thoroughness had
outgrown reading in one screen.

### The consumer check was scoped to half the workspace, and the other half cost three

`every_public_operation_has_a_consumer_outside_its_own_file` asked its question of `src/` only, and
said why in a paragraph that had been true for months: *"`runner` is `publish = false` and exists to
be consumed by tests; widening to it is a later decision, not an oversight."*

**That is a description rather than a reason.** A public function nobody calls is untested surface
whichever crate it sits in — and the crate that dispatches to a device is the one where untested
surface reaches a driver.

Widened, it cost three of a hundred and seventy-four:

* **`Gpu::run_words`** was public and named by nothing outside its own file, with `Gpu::run_u32` the
  public spelling of the same thing. Private now.
* **`Gpu::time_specialized`** had no caller anywhere, examples included. Deleted — an untested
  timing path that nothing has ever run is a measurement waiting to be quoted.
* **`Specialization::set_f32`** had none either, and trying to give one is what settled it. The
  method *is* tested — `a_float_goes_in_as_its_bits`, in the file that declares it — and cannot be
  tested anywhere else: what that test observes is `data()`, which is `pub(crate)`. An integration
  test, the only kind this check counts as a consumer, has nothing to look at but `len`, and `len`
  cannot tell `set_f32(3, 1.5)` from `set(3, 1)`.

  So it is excused, with that as the reason. **A test in the same file is a weaker consumer than one
  in another crate and it is not *no* consumer**, which is the distinction between this and
  `Module::memory_barrier` — that one had no test at all. The excuse list is exactly where a line
  like this belongs: written down, with what would have to change for it to expire.

Three in a hundred and seventy-four is a good ratio and it is not the point. The point is that the
question had never been asked of that half, and the answer to *"is this scoped narrowly for a
reason?"* was no.

**Asked of the other two kinds of public thing while the question was open**: of 50 public struct
fields across all three crates, **zero** are named nowhere else. Of 103 public enum variants, twelve
— which is the next section, and a different answer.

### Twelve enumerants nothing names, and why they are not the seven opcodes

The opcode sweep asked which `pub const` numbers nothing emits and found seven, and they were
deleted. The same question of the next kind up — which **public enum variant** is never named
outside the file that declares it — has an answer too: **12 of 103**, and every one of them is in
`src/spec/`.

    Decoration     BufferBlock, NonReadable, NonWritable
    StorageClass   Private
    BuiltIn        NumSubgroups, SubgroupSize
    Scope          CrossDevice, Invocation
    LoopControl    Flatten, DontFlatten, Unroll, DontUnroll

**They look like the seven and they are not, and the difference is what an enum is for.** `op.rs` is
a *table*: 95 numbers out of Khronos' several hundred, and the only reason any one of them is there
is that something emits it — so a number nothing emits is a copy of the grammar with nothing
checking it, and deleting it costs a minute on the day somebody wants it back.

An enum that models a closed set from the specification is a different object. `LoopControl` has
four members because the specification has four; a `LoopControl` carrying only the two this crate
emits would be a *wrong type* — it would say the other two do not exist. The same for `Scope`, whose
members are the levels a memory operation can name, and for `Decoration`'s access qualifiers. The
next person to need `Scope::Invocation` would add it back with a guess, and the guess is the failure
`decisions/DR-0001` exists to prevent.

What keeps those twelve honest is weaker than what keeps the opcodes honest, and worth naming: a
unit test asserting each `word()` against the specification, which is the *second copy of the
number* DR-0001 warns about — where an emitted opcode is checked by `spirv-val` seeing a real
module. That is the best available for a number nothing emits, and it is the reason to keep the set
complete rather than to prune it: a partial enum has the same weak check and a worse type.

## The round trip, redrawn on a workload that looks nothing like a chess engine — 2026-08-16

A throwaway demonstration generated procedural worlds on the GPU — a two-octave value-noise
landscape, a cave system packed one bit per layer, an escape-time fractal in fixed point — and
measured all three against one CPU thread at a million answers each, on an RTX 4080:

| world | round trip | host | ratio | the dispatch alone |
| --- | --- | --- | --- | --- |
| landscape | 2.76 ms | 2.15 ms | **0.8×** | 73 µs |
| caverns | 2.84 ms | 8.43 ms | **3.0×** | 126 µs |
| fractal | 3.42 ms | 41.88 ms | **12.2×** | 166 µs |

**The device's own work is 29×, 67× and 253× faster than the host, and the landscape still loses.**
Every one of the three returns the same four bytes per answer, so the transfer is the same ~2.8 ms
of moving eight megabytes in all three rows; what changes is the arithmetic on top of it.

That is `decisions/DR-0008` exactly, on a workload chosen for having nothing in common with the one
that produced it. The record there is a chess engine's NNUE layer and the conclusion was *stay on
the CPU*; the same boundary drawn here says **the crossover is work per byte returned**, and names
where it sits: a kernel doing two octaves of noise per output word is under it, one doing forty
iterations of fixed-point arithmetic is well over.

**And half the transfer was waste.** `Gpu::run_grid` sizes its output from its input, so a generator
that reads nothing at all still uploads four megabytes of zeros. `notes/NEXT.md`'s *"a buffer the
caller already owns"* is the entry that would remove it, and it stays open for want of a caller — a
throwaway is not one.

### The engine's rules cost three things and none of them was a limitation

Worth recording because all three are decisions this repository argued for and none had been tested
against a workload written by somebody who wanted the opposite:

* **No per-lane branch** — `decisions/DR-0003`. Procedural generation is branches all the way down
  (*if the density clears the threshold, place stone*) and every one became a comparison and a
  `select`. That is not a workaround: a divergent branch runs both sides and masks anyway.
* **No exclusive-or.** The hash had to mix with multiply, add and shift alone, so `h ^= h >> 16`
  became `h += h >> 16`. It mixes less per round and it mixed enough.
* **No subtraction and no division.** `zx² − zy²` is `add(zx², mul(zy², -1))` and a halving is a
  shift. One extra instruction a driver folds, in the two places it came up.

### And the shape that made generation-from-nothing work

A `Vector<T, LANES>` at `LANES == subgroup` is one element per **invocation**, so a value splatted
from `Kernel::local_index` is a different number in every lane: the lane's own column. No buffer of
coordinates is uploaded, and the whole world comes out of the dispatch's own geometry.

That is worth knowing about this API and is written down nowhere else — a splat of a *uniform*
constant and a splat of a *per-invocation* built-in are the same call, and they are the difference
between a flat field and a world.

## Four instruments that had stopped touching their subject — 2026-08-25

An audit rather than a feature, and what it found was not four bugs. It was four *checks* reporting
green over nothing, which is the failure this project has written more about than any other and had
not turned on its own instruments.

Listed worst first by how long each had been silent.

### The front page had been mojibake for nine days

`README.md` — 840 lines, 55 measured numbers, the first thing anybody reads — went through a
Windows-1252 round trip on 2026-08-16 and nobody noticed until now. **174 damaged sequences over
134 lines**: every em dash arrived as three characters, and every micro sign, every ratio's
multiplication sign and every superscript as two or three of their own.

**The examples are described rather than shown, and that is not squeamishness.** This file is one
the check below reads, so writing the damage down would *be* damage — the same rule
`tests/documented.rs` follows one level up when it builds its test input out of bytes instead of
typing it, and the same rule again that keeps `OPENS` split in half so a file explaining marker
syntax does not contain a marker. Three times now, in three kinds of markup.

The commit that did it is `f28e29b`, whose message is about the fuzzer's three gating axes and whose
README diff is **136 insertions and 136 deletions**. That number is the whole tell: a documentation
edit does not rewrite every line in a file, and a whole-file re-encode rewrites exactly the lines
with a non-ASCII character on them. Nothing in the diff review asked why an unrelated change touched
136 lines.

**The repair had a wrinkle worth recording**, because the obvious method fails. Decoding the whole
file back through Windows-1252 dies on the first character that was never damaged — nineteen em
dashes had survived, added by later edits, so the file was **mixed**. What works is undoing the round
trip one run at a time: take each character back to the byte Windows-1252 would have decoded it from,
and accept the run only when those bytes form a character. Ten distinct mappings covered all 174.

The proof the repair changed nothing but encoding: 840 lines before and after, 135 lines differing,
and on every one of them the ASCII skeleton byte-identical. Non-ASCII characters went 467 → 175,
which is exactly the arithmetic of collapsing each run to the character it was.

**Why no check could see it is the finding.** `tests/documented.rs` reads two things: the digits after
a `count:` marker, and the names inside backticks. Both are ASCII. So a document may rot in every
character those two do not read and pass the suite whose entire purpose is keeping documents honest —
`notes/CLAIMS.md`'s subject arriving inside the file that implements it.

It is a fourth claim in that file now, and it detects the *cause* rather than a list of known-bad
sequences, so damage this repository has not met yet is covered too. It reads `.rs` as readily as
`.md`: the editor that did this has no opinion about extensions.

### The validator was not installed on the machine with the GPUs, and the fallback pointed at a drive that is gone

One `cargo test -p runner` printed **631 skips**, every one `spirv-val not found (set SPIRV_VAL)`.
The workstation — two GPUs, the only place widths 32 and 64 run at all — had never validated a
module. `decisions/DR-0010` says so about itself, in a *What is not verified here* section written
five days earlier and read by nobody since.

`tests/common/spirv_val.rs` fell back to `H:\tools\spirv-tools\install\bin\spirv-val.exe`. There is
no `H:` on this machine. So "look in the usual place" had meant "find nothing" for however long the
letter had been wrong.

**Two lines above that fallback is a doc comment about exactly this failure.** It explains at length
that a `SPIRV_VAL` pointing at nothing used to return `None`, that every caller reads `None` as "not
installed", and that a typo therefore turned off validation in both test trees while leaving a green
run behind — which is why a set-and-wrong path is now a panic. The file had learnt the lesson for the
environment variable and kept a hard-coded absolute path immediately underneath it.

It searches `PATH` now. That is the usual place, it costs the same lookup, and it does not have an
opinion about which drive a toolchain lives on.

With the validator present: **482 emitter tests and 395 runner tests, with no skips at all**, and all
631 modules legal. That they were legal is the answer and not the point. The point is that nothing
had asked, on the machine where widths 32 and 64 are the only thing anybody can ask on.

### Five instructions and two loops that had never run

`f_sub`, `f_div`, `f_negate`, `i_sub` and `u_div` were added on 2026-08-18 for an activation and for
the arithmetic that says which of a batch a lane is working on. A week later they had **one consumer
between them**: `tests/instructions.rs`, which builds one module and hands it to `spirv-val`.
`Kernel::repeat_rolled` and `repeat_rolled_many`, added the next day, had their unit tests and
`tests/control_flow.rs` and no dispatch anywhere.

Both passed `every_public_operation_has_a_consumer_outside_its_own_file`, because a test *is* a
consumer — deliberately, and for a good reason stated where the check is written. The gap is one the
check was never asked to close: an operation whose only consumer is the test written to satisfy it.

**And the validator is a weaker witness here than anywhere else in this tree.** `decisions/DR-0001`
already says why: an opcode number read wrong assembles into a *different well-formed instruction*.
`OpFNegate` at the wrong number produces a module `spirv-val` accepts. The only thing that can tell
a negation from something else is an answer.

`runner/src/kernels/unrun.rs` — the module named for precisely this category — has four now. Each was
built so its reference is exact rather than approximate, which is the bar `notes/NEXT.md` sets when it
refuses to fuzz `sqrt` and `exp` for want of one:

* **centre and scale** — `-((x - 8) / 4)`, an integer centre and a power-of-two divisor, so every
  step is representable and the comparison is on bits with no epsilon.
* **remainder** — `x - (x / 7) * 7`. Seven and not eight, so `OpUDiv` is a division rather than a
  shift the driver folds away.
* **a rolled loop that reaches a buffer** — `DR-0010`'s kernel, reading block `counter * 64` each
  trip. A loop that re-read block zero every trip emits a valid module and returns a plausible
  number.
* **two running totals from one pass** — the workload `repeat_rolled_many`'s own doc comment names.
  It stores the *difference* of the two, so a second phi wired to the first collapses the answer to
  zero rather than passing.

All four were **checked by breaking them**: an `f_sub` written as `f_add`, an `i_sub` as `i_add`, a
loop whose every trip reads block zero, a second phi reading the first. Four mutations, four red
tests, each for its own reason.

They declare no subgroup capability and go through `whole_subgroup!`, so they run at every width and
the per-width skip counts in `ci.yml` did not move. That was designed in rather than lucky: a test
that skips at 4, 8 and 16 would have changed three numbers in a matrix and taught nobody anything.

### The mutation gate had no configuration here, and four commits had not been through it

`noha.yaml` is excluded by this machine's global gitignore — policy, deliberate, and the reason
`tests/integrity.rs` skips four checks rather than panicking on a clone. What the policy says is that
the file is never *committed*. It does not say the file should be *missing*, and it was.

So the strongest instrument this repository has — `notes/CLAIMS.md` rates it first, at 100% over 651
mutants — could not run, and had not run over the four commits between 2026-08-16 and 2026-08-19.

Rebuilt by **generating the source list from the tree minus `NOT_MUTATED`**, which is the only honest
way to do it: a hand-typed list that nothing compares against reality is the exact drift those tests
exist to catch, and `noha.yaml` listing 33 of 38 sources is the failure `tests/integrity.rs` was
written after. 94 targets, 15 excused FFI files, all 17 integrity checks running with no skips.

One thing was easy to get wrong and is worth stating: `test:` has to be the **whole workspace**. A
mutant in `runner/src` is killed only by the device suite, and a root `cargo test` runs the emitter
alone — so every runner mutant would survive, and the emitter's real score would be buried under
them. A gate that reds for the wrong reason teaches everyone to ignore red, which is the argument
`ci.yml` already makes about `session.rs` and a shared runner's wall clock.

The diff-scoped run over all four commits and the audit's own work: **10 viable mutants, 10 killed,
no survivors**.

### What the four have in common, which is the actual finding

None of them was a claim without an instrument. Every one was an instrument that had stopped
reaching its subject:

| instrument | still ran | had stopped touching |
| --- | --- | --- |
| `tests/documented.rs` | every push | any character outside ASCII |
| `spirv-val` | in CI | this machine, for 631 modules |
| `every_public_operation_has_a_consumer` | every push | whether the consumer runs |
| the mutation gate | in principle | this checkout, for four commits |

This project is unusually good at asking whether a claim has something behind it. What it had not
asked is whether the things behind its claims still touch what they are supposed to — a different
question, and on this evidence the one with more in it.

**And a candidate the audit could not close.** Nothing here checks that a test which is supposed to
run actually ran. `ci.yml`'s lavapipe job asserts a skip count per width and is the only place in the
tree that does; it is also why the CI numbers are trustworthy and the workstation's were not. 631
skips on a workstation looked exactly like a clean run, and a `--nocapture` flag added months ago for
this precise reason was not enough, because nobody reads 631 lines of a passing run.
