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
generates mutants for `encode.rs` and a little of `module/`, and **none at all** for `spec.rs`,
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


## Specialization constants save 1% — 2026-08-12

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

## Holding a reduction's pipelines is worth 5× — 2026-08-12

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
