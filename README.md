# simdr

**SIMD on the GPU, in Rust, with an empty dependency table.**

`Simd<T, N>` semantics — splat, elementwise arithmetic, reductions, shuffles, votes — lowered onto
SPIR-V subgroup instructions by an emitter that writes the binary format itself. No `build.rs`, no
shader compiler, no `unsafe`, and no crates.

```rust
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::F32;

// A compute kernel: read 32 floats per subgroup, sum across the lanes, write the total.
let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2))?;   // subgroup 32, 64 invocations
let value = kernel.load::<32>(0)?;
let total = kernel.lanes()?.reduce_sum(value)?;
kernel.store_scalar(1, total)?;

let spirv: Vec<u32> = kernel.finish()?;   // hand these words to vkCreateShaderModule
```

Those lines name no opcode, no reduction shape and no cluster size. Change the `32` to `8` and the
same source emits a clustered reduce running four vectors at once; change it to `128` and it
strip-mines four elements per lane and folds them before the subgroup step. Picking between those
three is what the library does.

---

## What is here

| | |
| --- | --- |
| `simdr` | The emitter. Zero dependencies, `#![forbid(unsafe_code)]`, no panics on any input. |
| `runner` | Runs the output on a real GPU through `ash`. Separate crate, because Vulkan is FFI and FFI is `unsafe`. |
| `cli` | `simdr probe`, which tells you the subgroup width you cannot build without, and `simdr list`, which names every device that could run compute work. |

The arrow points one way. Nothing in the emitter can reach the runner.

### Layers

```
spec/      Khronos' numbers — opcodes, capabilities, enumerants, GLSL.std.450
module/    A SPIR-V module being assembled: types, constants, blocks, phis, subgroup ops, atomics
lanes/     Simd<T, N> semantics: mappings, reductions, shuffles, votes, loops, branches,
           min/max/clamp, shifts, and the packed integer dot product
kernel/    The buffer interface, one axis or two, workgroup shared memory, the barrier,
           and atomic scatter
encode.rs  Words out: the little-endian stream, literal strings, instruction headers
decode.rs  Reading a module back, which is how the tests inspect what was emitted
half.rs    f32 ↔ f16, because Rust has no stable f16 and this crate has nothing to borrow one from
```

The four with a slash are directories and the three without are files — `encode.rs` used to be
missing from this list entirely, and the other two were drawn as directories they are not.

### Start here

```
simdr probe
```

`decisions/DR-0002` makes the subgroup width an argument so that nobody can forget it exists,
which left no way to *ask* what it is without writing a program. That is what the command is for.
It also reports which subgroup features the device offers, which narrow element types it can both
compute in and hold in a buffer, and what its memory types look like.

`simdr list` names every device. That matters more than it sounds: this machine turned out to have
two, at two different subgroup widths, and for a month only one of them had ever been built for.

---

## The three lane mappings

`N` is fixed when you write the code. The subgroup width is fixed by the hardware. They meet at
build time, because the three rows below are three *different instruction sequences* rather than one
instruction with a parameter — and no value arriving later can add instructions that were never
emitted. `Lanes::new` therefore takes the width, and the caller reads it off the device
(`gpu.limits().subgroup_size`). See `decisions/DR-0002`.

That reason is narrower than the one this file used to give. `ClusterSize` *can* be deferred to
pipeline creation — a specialization constant is a constant instruction, the validator accepts one
there and an RTX 4080 runs it at 4, 8 and 16 from a single module. What cannot be deferred is the
choice of *shape*. `decisions/DR-0005` records the experiment; DR-0002 carries the correction.

| `N` vs width | mapping | a reduction | a scan | a shuffle |
| --- | --- | --- | --- | --- |
| equal | `WholeSubgroup` | one subgroup instruction | one subgroup instruction | one instruction |
| a divisor | `Clusters` | one clustered instruction, several vectors at once | a `log2(N)`-step ladder | a butterfly or a broadcast, inside the vector |
| a multiple | `Strips` | `strips - 1` scalar ops, then one subgroup instruction | one scan per strip, carrying a running total | one instruction per strip |

Anything else — 12 lanes on a 32-wide subgroup — has no mapping and is refused by name.

**The clustered row's last column was empty until it was read rather than trusted.** All four
shuffles refused a vector narrower than the subgroup, on the grounds that its lanes are shared with
other vectors. That is true of the two *shifts* and false of the other two: a butterfly with
`mask < N` cannot leave an aligned run of `N` lanes, and a broadcast reads
`(lane & !(N - 1)) + source`, which `OpGroupNonUniformShuffle` will take because its id may differ
per invocation. So the mapping that exists to run four small vectors at once can now swizzle them,
and `kernels::butterfly_cluster_sum` — four independent trees, checked against the one
`ClusteredReduce` that computes the same thing — is what that buys.

**All three scan, and the clustered one is the expensive row.** SPIR-V has a `ClusteredReduce` and
no clustered scan, so `Lanes::prefix_sum` builds a Hillis-Steele ladder instead: `log2(N)` steps of
shuffle, compare and select, against one instruction for the other two rows.

The cheap alternative is a subgroup-wide scan minus each cluster's starting offset — three
instructions rather than a dozen — and it is not taken, because in floating point it subtracts a
large running total back off itself and loses exactly the low bits the scan just accumulated. Same
reason the exclusive scan is its own group operation and not `inclusive - own`.

The ladder's mask needs the invocation's position inside its cluster, and that number is
`SubgroupLocalInvocationId` — which `Lanes` **declares for itself**, the first time something asks.
A kernel that only scales still declares no `Input` variable and no `GroupNonUniform` capability;
`decisions/DR-0007` has why that is the shape, and why the entry point's interface list is data
until the module is finished rather than words written once.

## Crossing between subgroups

Every subgroup instruction stops at the subgroup. A workgroup holds several of them, and combining
them needs shared memory and a barrier:

```rust
let mine = kernel.lanes()?.reduce_sum(value)?;   // one total per subgroup
let shared = kernel.shared(64)?;
let slot = kernel.local_index();
kernel.store_shared(shared, slot, mine)?;        // no two invocations collide
kernel.barrier()?;                               // reached by all of them

// Slots 0, w, 2w … are build-time constants, so every invocation reads the same
// places, runs the same instructions, and computes the same answer. No divergence.
let total = kernel.load_shared(shared, 0)?;
```

`Gpu::sum` used to end with two floats coming home for the host to add, because there was no way
to combine two subgroups on the device. It reads one number now.

### The prefix sum, where the same handover gets harder

A reduction throws away all but one number and a scan keeps them all, which makes the scan the
stricter test of the two. A reduction sums the same set whatever order the lanes are in, so a
mapping that pairs the wrong lanes still returns the right total; a scan gets a different number at
almost every position and still ends on the same grand total — so `runner/tests/scan.rs` compares
every element rather than the last.

Which subgroups come "before mine" differs per lane, and the obvious spelling is a loop bounded by
this lane's subgroup index — a loop that runs a different number of times per lane, which is the
divergence DR-0003 refuses. It is a fixed number of `OpSelect` steps instead, one per subgroup in
the workgroup, each adding that subgroup's total **or not**:

```rust
// 1 step at width 64, 15 at width 4 — fixed when the module is built, so every
// invocation runs all of them and the select is what makes the answer differ.
let after = kernel.module().binary(op::U_GREATER_THAN, boolean, slot, boundary)?;
let with = kernel.module().binary(T::ADD, element, offset, theirs)?;
offset = kernel.module().select(element, after, with, offset)?;
```

`kernels::scan::scan_workgroup` scans one workgroup — 64 elements. Longer than that is
`Gpu::scanner`, which is the same idea applied to itself: cut the input into blocks, scan each,
scan the block totals to find what each block owes the ones before it, and pay it. The block totals
are themselves an array needing a scan, so past 64 blocks the same three steps run again one level
up.

```rust
let mut scanner = gpu.scanner(1 << 20)?;   // three levels, seven dispatches
let running = scanner.scan(&input)?;       // one submission

gpu.scan(&input)?;                         // the same, building and dropping it all per call
```

**2²⁰ elements is three levels**, and the count is decided in one place and cross-checked against
the dispatches actually recorded. The offsets a block owes are an *exclusive* scan — SPIR-V has the
operation, and computing it as `inclusive - own` instead would lose precisely the low bits the scan
had just accumulated.

`Gpu::scanner_of` fuses an elementwise map into the chain's first pass, so the running total of
f(x) crosses the bus once instead of three times: **2.0–3.0×**, falling as the input grows because
what is removed is two crossings of the buffer and the scan grows faster than they do.
`runner/examples/scanner.rs` prints both tables.

## Branches are uniform or they are refused

A branch takes a `Uniform`, and only a vote produces one. A subgroup instruction inside a divergent
branch answers for whichever lanes happen to be running, which has no `Simd` meaning at all. Per-lane
conditionals have a different spelling — `select`, which computes both sides and picks. See
`decisions/DR-0003`.

```rust
let over = lanes.any_uniform(lanes.greater_than(value, limit)?)?;

// A value that survives the merge, through an OpPhi. Exactly one arm runs.
let answer = lanes.choose_uniform(
    over,
    element,
    |lanes| lanes.reduce_sum(value),
    |lanes| lanes.reduce_max(value),
)?;
```

---

## How this is known to work

Validity is not correctness. A kernel can satisfy the validator down to the last rule and still
compute the wrong number, so there are seven layers and each has caught something the ones above it
did not.

| Layer | What it is | What it caught |
| --- | --- | --- |
| **Unit tests** | 348 in the emitter, decoding what was emitted; 740 across the workspace | Everything cheap |
| **`spirv-val`** | Khronos' validator, at `--target-env vulkan1.1` | `OpLoopMerge` in the wrong position — a unit test asserted "merge before branch" and passed while the comparison sat between them. And, the first time it was pointed at `Lanes::dot_unsigned`, that `OpUDot` had been emitted with a **signed** result type: invalid SPIR-V in a shipped public method that had no caller, no unit test and no validator coverage. It is also the only layer that can see an entry point whose interface omits a built-in the body loads: drop that one line and 19 of 20 modules are rejected while all three devices go on returning the right answers |
| **Execution** | Real dispatches on a real GPU, against CPU references | A missing staging write: every computing kernel returned garbage and the empty-kernel test still passed |
| **Other widths** | The same suite at **4, 8, 16, 32 and 64** lanes, across three devices | Ten tests that had conflated "32 lanes" with "the subgroup", four of which could not build at all because a vote has no clustered form. Then, at 8: a fuzzer generating shuffles that leave the subgroup, and three tests assuming uninitialised device memory is zero. Then, at 4: `kernels::scale` — *the control kernel* — reading and writing eight times its buffer, which had been undefined behaviour returning zeros at width 8 for a day before it became an access violation at 4. And that was not the last of them: eleven more of the same shape were still there four days later, found by the row below rather than by running at 4 again |
| **Differential fuzzing** | Generated programs across **all eight** element types, ending in a reduction *or a scan*, at all three mappings, each interpreted on the CPU and compared exactly | `reduce_min` folding strips with a *maximum* — right for every mapping but the strip-mined one, so hand tests never saw it. Then, the day the clustered scans stopped being refused and started being generated: an integrated AMD driver that **faults inside `vkCreateComputePipelines`** on a module `spirv-val` accepts and two other implementations run correctly |
| **Mutation coverage** | `noha prober` over the emitter and the runner's pure half | Eight real gaps in one night, five of them in the *fuzzer* — including a generated program that dispatched nothing and therefore agreed with everything. Later, fifteen more in the half-float rounding path, which an *exhaustive* round-trip test could not reach because it only ever fed `from_f32` values that came from a half — and those never round |
| **Dispatch bounds** | `dispatch::extent` reads the workgroup size, the element stride and the strip count out of the module and refuses a dispatch that cannot fit | **Eleven tests reading past their input**, across five files. Each paired a kernel built for 32 lanes with a buffer of one workgroup — correct on the two GPUs here, and an eighth of what the kernel reads at four lanes. Every one of them had been passing on lavapipe by getting the first sixty-four elements right and going off the end for the rest |

Each row was added because the ones above it were green while something was wrong. What that does
*not* mean is that the list is finished — the last three rows were added within four days of each
other and every one found something on the day it arrived. The newest found eleven.

### `--target-env` is not optional

Left off, `spirv-val` checks the *universal* SPIR-V environment, which is far laxer than any real
consumer: it accepted a `GLCompute` entry point with no `LocalSize`, because that requirement is
Vulkan's rather than SPIR-V's. A validator run against the wrong environment is a validator that
agrees with you.

### `--workspace` is not optional either

This is a root package with a member, so plain `cargo test` runs six suites of nineteen. The
mutation gate's kill command was missing the flag, so the whole execution and fuzzing layer sat
outside it and the score was measured over the tests that happened to be in scope. Every command
in `noha.yaml` names `--workspace` now.

### What guards the guard

Twelve mutants reproduced. **Eight were real gaps and four were equivalent mutants; nine of the
twelve were in the fuzzer or its CPU reference** — the machinery that exists to check everything
else. **None was in the emitter.**

That is not luck. The emitter has four other layers watching it and the checking machinery had
none, which is the ordinary way a suite rots: whatever is furthest from the thing under test is
furthest from anybody looking. A fuzzer that has stopped exploring still reports thousands of
agreements, and the number goes up.

They came in one family. Whatever a fuzzer's coverage claim rests on has to be *asserted*, not
inferred from the fact that it produced a lot of agreements:

- It must **explore** — a degraded random stream generates the same few programs forever.
- It must reach its **operands** — every operation appeared while the butterfly only ever used
  distance zero.
- It must actually **run** — a program dispatching zero workgroups agrees with everything.
- Its rules must hold **both ways** — narrow programs were checked to have no shuffle; wide ones
  were never checked to have one.

Each had to be said separately, because each was invisible from the others.

### A scoped run is not the gate

The gate is normally run with `NOHA_ONLY` naming the files a piece of work touched. Running it one
file at a time afterwards turned up **two survivors the batched runs over the same files had
scored 100%**. Both were real:

- `24 - byte * 8` → `24 + byte * 8` in the written-out dot product. The shift counts become 32, 40
  and 48; SPIR-V leaves a shift past the operand width undefined, this device masks it to five
  bits, and the kernel read bytes 0, 3, 2, 1 — a *permutation*. Every test summed the squares of
  all four, and a sum does not care about order, so two GPU kernels and a host reference agreed
  exactly on the wrong bytes. **An operation that folds N things symmetrically cannot test how the
  N were chosen.**
- `type_int(32, false)` → `true`, in two kernels that built their own index type. Equivalent —
  `OpIAdd` is sign-agnostic. Deleted rather than tested: `Kernel::index_type()` hands back the type
  the kernel already decided on, and there is no longer a sign to get wrong.

And a limit worth knowing: some files generate **no mutants at all**. `runner/src/kernels/plane.rs`
is straight-line module construction with no comparison, boolean or branch, so the gate has nothing
to change and scores it 100% on an empty set. Device tests are what cover those.

### A survivor is only a finding if it reproduces

Survivors were chased down one at a time on 2026-08-12, each applied by hand before being believed.
**Several died instantly** — the probes had timed out under load, and a timeout is recorded as a
survivor. Apply it yourself first.

Of the twelve that did reproduce, **four were equivalent mutants** sitting on branches that could
not be got wrong: `if lanes > subgroup { lanes / subgroup } else { 1 }` gives one either way at the
boundary, and an `unwrap_or` on an index that cannot occur is a default no test can reach. Each
time the fix was to delete the branch rather than write a test that cannot exist — and each time
the code came out simpler.

The other eight were real.

### What none of it caught

`Buffer::write` did not check that the caller's slice fit, and its safety comment explained why it
did not need to: *"this crate always allocates from the same element count it writes"*. That was
true when it was written, because `Gpu::run` was the only caller.

`Session` broke it without touching the file. Its staging buffer is sized to the largest binding
and `Session::write` takes a slice from outside, so a long one would have memcpyd past the end of
a mapping — from safe code, in a crate whose whole claim is that it cannot.

Nothing found it. Not clippy, not the mutation tester, not 353 tests, not the fuzzer. It was found
by re-reading a `SAFETY` comment while checking something else. **A safety argument that names the
current set of callers has an expiry date and does not say when.**

### The paperwork is checked too

`tests/integrity.rs` reads `noha.yaml`, **which is deliberately not in this repository** — a global
gitignore keeps the local verification toolchain out of every repo on this machine, and a *global*
exclusion is invisible from inside a working tree, where the file is present and `git status` is
clean. It is a price, it is chosen, and it is written down rather than worked around.

That used to mean four of those tests **panicked** on any clone, which is the same failure the file
exists to catch: a check that reported green for a reason that did not travel. They skip loudly now,
naming themselves, the way the GPU harness reports a missing device. `cargo test --workspace` from a
clone passes.

What still runs without the config is the more interesting half, because the excuse is "this file is
FFI, so it contains `unsafe`" and **both directions of that are checkable**:

- every excused file still exists and still contains the `unsafe` that excused it — an expired
  excuse costs coverage;
- every file containing `unsafe` **is** excused — unsafe code left inside the gate costs the
  mutation run itself, because a mutant that passes a wrong handle or frees twice kills the process
  instead of failing a test. That direction was missing, and it is the rule this project had already
  applied by hand three times: `dispatch/step.rs` split out of `chain.rs`, `reduction/plan.rs` out of
  `held.rs`, `step::upload_bytes` out of `dispatch/upload.rs` — each so a decision would sit in a
  file with no `unsafe` and therefore inside the gate. Nothing enforced the shape until now;
- the emitter still declares `#![forbid(unsafe_code)]`, which is the one line the whole arrangement
  rests on;
- and every `Thing::member` written in backticks in `decisions/` still exists in the source.

With the config present it also compares the mutation tool's source list against the tree in both
directions. All of it because hand-maintained lists had drifted while reporting green.

---

## Running it

```powershell
cargo test --workspace                       # everything; GPU tests skip loudly with no device
cargo clippy --workspace --all-targets -- -D warnings

$env:RUSTDOCFLAGS = "-D warnings"            # a broken doc link is a build failure
cargo doc --workspace --no-deps

$env:SPIRV_VAL = "path\to\spirv-val.exe"     # or install where tests/common looks
$env:SIMDR_FUZZ_ROUNDS = "6000"              # search harder
$env:NOHA_JOBS = "4"                         # the mutation gate drives the GPU; do not oversubscribe
```

**Rust 1.88 or newer**, which is where `if let` chains stabilised. That number is checked rather
than declared: `.github/workflows/ci.yml` builds the workspace with exactly it, because the version
written there before was the one that happened to be installed and excluded nine releases of users
for no reason.

### What CI runs, and what it cannot

Formatting, clippy at `-D warnings`, the emitter's suite against `spirv-val`, the integrity checks,
and the **whole runner suite on lavapipe at widths 4, 8 and 16** — Mesa's CPU implementation needs
no GPU, so most of the layers above travel to a shared runner.

Three things do not, and the workflow lists them rather than leaving them to be assumed: **widths 32
and 64**, which need the two devices in this machine; the **mutation gate**, whose configuration is
deliberately not in this repository; and every **measurement** in `notes/FINDINGS.md`, none of which
means anything on a shared runner. A green run there is the part that travels, not the whole suite.

**`--workspace` on all of them.** This is a root package with a member, so leaving it off runs six
suites of nineteen — which is how the mutation gate came to be measuring a fraction of the suite
for weeks.

A machine with no Vulkan device is a normal state for the suite to find. Those tests print
`SKIPPED` with a reason rather than passing quietly — a skipped correctness test that looks green
is worse than a red one.

### Examples

```powershell
cargo run --release --example bench    -p runner  # what each mapping costs
cargo run --release --example latency  -p runner  # one answer, waited on, versus batched
cargo run --release --example overhead -p runner  # where the time in a round trip actually goes
cargo run --release --example memtypes -p runner  # which memory the staging path gets
cargo run --release --example resident -p runner  # where three simultaneous buffers land
cargo run --release --example nnue     -p runner  # a chess engine's NNUE layer, at its real size
cargo run --release --example sweep    -p runner  # working-set sweep, with spreads
cargo run --release --example narrow   -p runner  # i8 and i16 against i32, at the same element count
cargo run --release --example specialize -p runner # emitting a module against building a pipeline
cargo run --release --example reducer   -p runner  # a reduction that keeps its pipelines
cargo run --release --example dot       -p runner  # OpSDot against the eleven instructions it replaces
cargo run --release --example plane     -p runner  # what a second dispatch axis costs, against what it was confounded with
cargo run --release --example occupancy -p runner  # how many subgroups a workgroup should hold, swept
```

To run any of it against a different device:

```powershell
simdr list                      # NVIDIA GeForce RTX 4080 / AMD Radeon(TM) Graphics
$env:SIMDR_DEVICE = "radeon"    # substring, case-insensitive
cargo test --workspace          # the whole suite, now on a 64-wide subgroup
```

And against a CPU implementation, which needs no GPU at all. Mesa's lavapipe reports a subgroup
width of 8 by default:

```powershell
$env:VK_ICD_FILENAMES = "H:\tools\mesa\msvc\lvp_icd.x86_64.json"
cargo test --workspace
```

**And at 4 and 16**, which no hardware here offers — llvmpipe's subgroup width is its vector width
divided by 32, and that is an environment variable. `minSubgroupSize` equals `maxSubgroupSize` at
each setting, so the width is pinned rather than a default the driver may vary:

```powershell
$env:LP_NATIVE_VECTOR_WIDTH = "128"      # subgroup 4;  512 gives 16
cargo test -p runner -- --test-threads=1
```

`--test-threads=1` is not optional at 128 or 512: lavapipe is unstable there under concurrent
devices, and about 40% of parallel runs report a disagreement that does not reproduce. The default
256-bit build has no such problem. `notes/FINDINGS.md` has the evidence, including what rules our
own code out.

The build is [pal1000/mesa-dist-win](https://github.com/pal1000/mesa-dist-win) — take the **msvc**
release and copy `vulkan_lvp.dll` and `lvp_icd.x86_64.json` out of `x64`. The mingw build's DLL
would not load here.

---

## Asking more than once

`Gpu::run` allocates three buffers, builds a pipeline, submits three times and throws it all away.
That is the right shape for a test and the wrong one for anything that asks repeatedly: measured
on an RTX 4080, an *empty* kernel over a 256-byte buffer costs ~875 µs a round trip against 0.8 µs
for the dispatch itself. Allocating and freeing one buffer is ~310 µs whatever its size.

`Gpu::session` pays that once:

```rust
let mut session = gpu.session(&spirv, &[input_len, output_len])?;
session.write(0, &words)?;
session.dispatch(workgroups, 1)?;
let answer = session.read(1, output_len)?;
```

Measured at **52× faster per dispatch** than rebuilding everything on the RTX 4080, and **5×** on
the integrated Radeon in the same machine. It does not make the kernel faster and it does not
remove the host copies — it removes the setup the measurement said was there to remove.

A full-buffer reduction is a chain of a dozen pipelines rather than one, and `Gpu::reducer` is the
same idea applied to all of them: **11.2×** over 8 192 elements and **5.6×** over 2²⁰.

The chain itself is five dispatches over 2²⁰ and three over 8 192, folding sixteen elements into
one at each level rather than two. That is a quarter of the passes halving needed — and worth
about 8%, which is a quarter of what its arithmetic promised. `notes/FINDINGS.md` has both halves.

```rust
let mut reducer = gpu.reducer(8_192)?;   // every pipeline built once
let total = reducer.sum(&input)?.total;  // and again, and again
```

The two numbers are why the test that guards this asserts a factor of three. Ten passed comfortably
for as long as there was one device to run it on, and was a property of that device wearing the
costume of a property of sessions.

### Where the rest of a large reduction goes

Measured rather than argued, each row a difference between two calls that differ in one thing —
`runner/examples/reducer.rs`, RTX 4080, 2²⁰ elements, a 1762 µs call:

| | per call | share |
| --- | --- | --- |
| the chained steps — one barrier each, nothing copied | 76 µs | 19% |
| the input's four megabytes | 205 µs | 52% |
| one submission and its fence | 56 µs | 14% |
| **accounted for** | **337 µs** | **86%** |
| *(a second and third submission, no longer paid)* | *126 µs* | *30%* |
| *(a whole-buffer download, no longer paid)* | *697 µs* | *164%* |
| *(an `f32` → `u32` copy of the input, no longer paid)* | *583 µs* | *137%* |

**That table came to 52% of the call for weeks**, and every one of the three bracketed rows was
hiding in the missing half. Two of its rows were absent because the *measurement* skipped them, not
because the call did: the upload row hoisted the `f32` → `u32` conversion out of its own timed loop,
and the per-step row is a difference between two chains, so anything paid once per call cancels out
of it exactly.

Made to add up, it named three things in a row:

| 2²⁰ elements | `Reducer::sum` |
| --- | --- |
| where the day started | ~1930 µs |
| reading one word home instead of the whole buffer | ~1140 µs |
| writing the caller's floats straight into the mapping | ~548 µs |
| recording the copies inside the chain's submission, one instead of three | ~424 µs |
| folding by sixteen — five dispatches instead of fifteen | ~407 µs |
| writing the input into memory the device can already read | **~280 µs** |

Only one of the five was an algorithm. The others were a download sized to the buffer rather than
the answer; a conversion of bits that were already the right bits; two submissions that existed
only to move bytes between buffers a third submission already touched; and a copy from staging into
the kernel's buffer, on a device where the host could have written that buffer in the first place.

**When a breakdown does not come close to the whole, the gap is the finding.** It has been wrong in
both directions here. It read 52% *under* the call before its missing rows were found; it reads
about **123%** now, because the call got a third shorter while rows timed in isolation did not.
Two of the three are upper bounds by construction — the step row comes from a chain of empty
kernels where a barrier has nothing to overlap with, and it overestimated a real chained step by
about four times. Both facts are printed in the table itself.

That last row is where most of the work went, and it took three attempts to notice. Two changes
shaved the device side — shortening the between-pass copies, then replacing them with a ping-pong
across two descriptor sets — for 85 µs and 32 µs against predictions of 111 and 250. Both missed
the same way: a component was timed *with its barriers included* and then costed as though removing
the component removed the barriers too.

Meanwhile `Reducer::sum` was copying 4 MB home and calling `.first()` on it. A reduction produces
one number. Reading one word instead is **33% of the whole call** on an RTX 4080, 30% on the
integrated Radeon and 25% on lavapipe — every round on every device, which none of the other three
managed, because it removes traffic without adding instructions.

The ping-pong is kept for being shorter code — no copies, no copy lengths, and no class of bug
where a short copy returns the previous call's data — and because on the integrated Radeon it *is*
worth 5.5%.

### Two ways to stop paying for an upload

What was left after all that is the upload, at about 70% of the call, and no device-side change
touches it. There are two ways to not pay it, and this project needed both.

**Write it once instead of twice.** The input went into staging memory and then across into the
buffer the first pass reads. A device that offers memory which is both device-local and
host-coherent lets the host write that buffer directly, and the second move stops existing: **31%
on an RTX 4080, 33% on the integrated Radeon**, over 2²⁰ elements.

Which devices offer it is not something to reason about. The guess written into the first version
of that change — that an integrated part shares its memory and a discrete card cannot — is wrong in
*both* directions on this machine, and `cargo run --example memtypes -p runner` prints why. That
first version was therefore dead code, and it still appeared to save 19 µs consistently, because
the same binary ran first in every round. Reversing the order reversed the result.

It is also a **62% regression** in `Gpu::sum`, one call away, where the buffers are allocated per
call rather than held: allocating out of that memory costs more than the copy it saves. Same three
lines, opposite sign, and the only difference is how often the buffer is made.

**Or don't cross the bus at all.** Σ f(x) is a map and a reduce, and the obvious way to compute it
crosses three times: send the input, run the map, bring the result home, send it back, reduce.

```rust
let square = kernels::square(width)?;                    // out[i] = in[i] * in[i]
let mut reducer = gpu.reducer_of(1 << 20, &square)?;     // the map is the chain's first pass
let norm = reducer.sum(&input)?.total;                   // Σ x², one crossing
```

**2.4× on an RTX 4080 and 2.6× on the integrated Radeon**, over 2²⁰ elements, against the same
route with a held `Session` for the map and a held `Reducer` for the fold — so neither column pays
for allocation or pipeline creation and the only difference is where the intermediate went. The
993 µs saved is the 718 µs download plus the 294 µs upload measured separately in the same file, as
they stood at the time of that run. The upload row has since fallen to ~190 µs for the reason
above, so the saving on the current build is smaller than 993 µs; the multiple has not been
re-measured since, and `cargo run --release --example reducer -p runner` prints today's on whatever
device runs it.

A first attempt reported 2.9× by writing the old route as `gpu.run`, which allocates and builds a
pipeline every call. Give the thing you are replacing every advantage you would give the
replacement.

## One instruction where there were eleven

`VK_KHR_shader_integer_dot_product` sums four 8-bit products in one instruction. What it replaces
is four shifts up, four bitcasts, four shifts down, four multiplies and three adds — and
`runner/tests/dot_product.rs` runs both spellings against each other and against a host reference.

```rust
let packed = kernel.load::<32>(0)?;              // one u32 per lane, four i8 inside it
let totals = lanes.dot_signed(packed, packed)?;  // Vector<I32, 32>
```

**Whether it is worth using depends entirely on the device**, and both of the ones here report it
as accelerated:

| kernel, 262 144 invocations | RTX 4080 | integrated Radeon |
| --- | --- | --- |
| one dot product per element | 1.00× | 1.52× |
| thirty-two per element | 1.18× | **9.08×** |

The first row is memory-bound: the load hides the arithmetic, and eleven instructions cost what one
does. The discrete part has enough integer throughput that even the second row barely moves; the
integrated part does not.

This is **not** a packed lane mapping. A `Simd<u32, N>` is still one `u32` per lane — `OpSDot` is
an operation that reads each of them as four bytes, and `decisions/DR-0004` says why that
distinction is worth keeping.

## Rows and columns

A matrix used to have to linearise itself before it reached a kernel. `Shape::grid` gives the
kernel a second axis, and `Grid` gives the dispatch one:

```rust
// One subgroup across, four invocation rows deep, over a matrix `pitch` elements to the row.
let mut kernel = Kernel::<U32>::new(Shape::grid(width, width, 4, 2))?;
let value = kernel.load_row::<32>(0, pitch)?;      // row × pitch + this invocation's column
let total = kernel.lanes()?.reduce_sum(value)?;    // one total per row, not per matrix
kernel.store_row_scalar(1, pitch, total)?;

gpu.run_grid(&spirv, &input, Grid::new(pitch / width, height / 4))?;
```

The column is the same expression a one-axis kernel uses — the same code, not a second copy written
to agree with it. `load_row_at` takes the row as a value, which is what a bias row or a vertical
stencil needs.

**It costs nothing measurable.** The extra multiply and add hide behind the loads on both hardware
devices — 3.38 µs against 3.38 on the RTX 4080, 42.99 against 42.92 on the integrated Radeon, at
the same invocation count. What it buys is that the caller stops doing the arithmetic by hand, and
hand-done addressing is exactly what ten tests got wrong the first time a second device ran them.

> The first version of that measurement said **2×**, and was comparing a grid four rows deep
> against a one-axis kernel with an eighth of the invocations per workgroup — so it moved the
> occupancy and the address at once. `notes/FINDINGS.md` has the corrected two-by-two, and the
> workgroup-size effect it was confounded with, which is real and points the opposite way on the
> two devices.

## What this is not

- **Not a shader language.** There is no Rust-to-GPU compiler here. You write the kernel against
  `Kernel` and `Lanes`, in Rust, and get SPIR-V words out. If you want to compile arbitrary Rust to
  the GPU, that is [rust-gpu](https://github.com/Rust-GPU/rust-gpu) and it is a much larger thing.
- **Not `core::simd`.** The nearest neighbour on the narrower question — lowering `Simd<T, N>` onto
  subgroup lanes — is [VectorWare](https://www.vectorware.com/blog/simd-on-gpu/), which is building
  it as a compiler backend consuming Rust's own portable SIMD. Their premise is this one: *"a warp
  issues one instruction, and each of its 32 lanes runs that instruction on its own data"*. The
  difference is where the code comes from. They compile the `Simd` you already wrote, so one source
  targets x86-64, Arm and a GPU; here you write against a builder and get words out, which is a
  smaller thing that needs no compiler and works on stable.

  Their post is honest about the same hard part `decisions/DR-0002` is about — what to do when `N`
  is not the width — and describes idling lanes for a smaller `N`. That is the case a
  `ClusteredReduce` exists for, and it is why the three mappings are named rather than implied.
- **Not portable across subgroup widths.** A module is built for one width, deliberately and
  visibly. That is what the hardware is. **Five widths have been run** — 32 on an RTX 4080, 64 on an
  integrated Radeon in the same machine, and 4, 8 and 16 on lavapipe, whose subgroup follows
  llvmpipe's vector width. The execution suite and the fuzzer pass at every one of them.
- **Not fast, as a claim.** Some measurements exist and are in `notes/FINDINGS.md` with their
  spreads. There is **no performance claim about large working sets**: a cliff shows up past about
  50 MB and *three* explanations have now been tested and refuted — L2 capacity, eviction of a
  single allocation, and placement under three simultaneous ones.
- **Not matrices or cooperative matrices.** `i8`, `u8`, `i16`, `u16` and `f16` are here and run on
  every device tried; matrix types are not.
- **Two dispatch axes, not three.** `Shape::grid` and `Grid::new(x, y)` are here; `vkCmdDispatch`'s
  z is always 1. `decisions/DR-0006` has the argument — the term is easy, the *layout* is not, and
  a z count above 1 would silently run every workgroup again over the same elements.
- **Nothing here defers a value.** Specialization constants work end to end and no kernel uses one
  outside the tests and the measurement written for them. That is a conclusion rather than a gap:
  `runner/examples/specialize.rs` timed one module specialized fourteen ways against fourteen
  modules and the difference was **9.7% of the setup**, because a specialization constant is fixed
  *at* pipeline creation and fourteen values still need fourteen pipelines. Keeping the pipelines
  instead is worth 5×, and that is what `Gpu::reducer` does. `decisions/DR-0005` has both tables,
  including the retraction of a first measurement that was wrong by a factor of eight.

## Reading the tree

- `decisions/` — the six decisions that shape everything, and why. `DR-0002` carries a correction
  where a later experiment showed its reasoning was too strong.
- `notes/FINDINGS.md` — what has been learnt, including the retractions, in place and struck
  through rather than deleted.
- `notes/NEXT.md` — what is worth doing next and why, with the measurements that say so.

## License

MIT OR Apache-2.0.
