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
throughput that eleven instructions cost nearly what one does, and at one dot product per element
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

## A second binding costs about 330 µs, and the submission is a fifth of it

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
