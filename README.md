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
kernel/    The buffer interface, workgroup shared memory, the barrier, and atomic scatter
half/      f32 ↔ f16, because Rust has no stable f16 and this crate has nothing to borrow one from
decode/    Reading a module back, which is how the tests inspect what was emitted
```

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

| `N` vs width | mapping | what a reduction costs |
| --- | --- | --- |
| equal | `WholeSubgroup` | one subgroup instruction |
| a divisor | `Clusters` | one clustered instruction, several vectors at once |
| a multiple | `Strips` | `strips - 1` scalar ops, then one subgroup instruction |

Anything else — 12 lanes on a 32-wide subgroup — has no mapping and is refused by name.

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
compute the wrong number, so there are six layers and each has caught something the ones above it
did not.

| Layer | What it is | What it caught |
| --- | --- | --- |
| **Unit tests** | 308 in the emitter, decoding what was emitted; 544 across the workspace | Everything cheap |
| **`spirv-val`** | Khronos' validator, at `--target-env vulkan1.1` | `OpLoopMerge` in the wrong position — a unit test asserted "merge before branch" and passed while the comparison sat between them |
| **Execution** | Real dispatches on a real GPU, against CPU references | A missing staging write: every computing kernel returned garbage and the empty-kernel test still passed |
| **Other devices** | The same suite at 64 lanes and at 8, as well as at 32 | Ten tests that had conflated "32 lanes" with "the subgroup", four of which could not build at all because a vote has no clustered form. Then, at 8: a fuzzer generating shuffles that leave the subgroup, and three tests assuming uninitialised device memory is zero |
| **Differential fuzzing** | Generated programs across seven element types, each interpreted on the CPU and compared exactly | `reduce_min` folding strips with a *maximum* — right for every mapping but the strip-mined one, so hand tests never saw it |
| **Mutation coverage** | `noha prober` over the emitter and the runner's pure half | Eight real gaps in one night, five of them in the *fuzzer* — including a generated program that dispatched nothing and therefore agreed with everything. Later, fifteen more in the half-float rounding path, which an *exhaustive* round-trip test could not reach because it only ever fed `from_f32` values that came from a half — and those never round |

Each row was added because the ones above it were green while something was wrong. What that does
*not* mean is that the list is finished — the last two rows were added on 2026-08-12 and both found
something on the day they arrived.

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

`tests/integrity.rs` compares the mutation tool's source list against the tree in both directions,
holds the list of files deliberately *not* mutated with a reason for each, checks that each of
those still contains the `unsafe` that excused it, and extracts every `Thing::member` written in
backticks in `decisions/` and fails when the source no longer defines it. All of it because
hand-maintained lists had drifted while reporting green.

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
```

To run any of it against a different device:

```powershell
simdr list                      # NVIDIA GeForce RTX 4080 / AMD Radeon(TM) Graphics
$env:SIMDR_DEVICE = "radeon"    # substring, case-insensitive
cargo test --workspace          # the whole suite, now on a 64-wide subgroup
```

And against a CPU implementation, which needs no GPU at all. Mesa's lavapipe reports a subgroup
width of 8:

```powershell
$env:VK_ICD_FILENAMES = "H:\tools\mesa\msvc\lvp_icd.x86_64.json"
cargo test --workspace
```

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
same idea applied to all of them: **5.0×** over 8 192 elements, 1.6× over 2²⁰ where the arithmetic
starts to dominate.

```rust
let mut reducer = gpu.reducer(8_192)?;   // every pipeline built once
let total = reducer.sum(&input)?.total;  // and again, and again
```

The two numbers are why the test that guards this asserts a factor of three. Ten passed comfortably
for as long as there was one device to run it on, and was a property of that device wearing the
costume of a property of sessions.

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

## What this is not

- **Not a shader language.** There is no Rust-to-GPU compiler here. You write the kernel against
  `Kernel` and `Lanes`, in Rust, and get SPIR-V words out. If you want to compile arbitrary Rust to
  the GPU, that is [rust-gpu](https://github.com/Rust-GPU/rust-gpu) and it is a much larger thing.
- **Not portable across subgroup widths.** A module is built for one width, deliberately and
  visibly. That is what the hardware is. Three widths have been run: 32 on an RTX 4080, 64 on an
  integrated Radeon in the same machine, and 8 on lavapipe, which runs on the CPU. The execution
  suite and the fuzzer pass on all three. Nothing has run at 4 or 16.
- **Not fast, as a claim.** Some measurements exist and are in `notes/FINDINGS.md` with their
  spreads. There is **no performance claim about large working sets**: a cliff shows up past about
  50 MB and *three* explanations have now been tested and refuted — L2 capacity, eviction of a
  single allocation, and placement under three simultaneous ones.
- **Not matrices or cooperative matrices.** `i8`, `u8`, `i16`, `u16` and `f16` are here and run on
  every device tried; matrix types are not.
- **No multi-dimensional dispatch.** `cmd_dispatch(x, 1, 1)`.
- **Nothing here defers a value.** Specialization constants work end to end and no kernel uses one
  outside the tests and the measurement written for them. That is a conclusion rather than a gap:
  `runner/examples/specialize.rs` timed one module specialized fourteen ways against fourteen
  modules and the difference was **9.7% of the setup**, because a specialization constant is fixed
  *at* pipeline creation and fourteen values still need fourteen pipelines. Keeping the pipelines
  instead is worth 5×, and that is what `Gpu::reducer` does. `decisions/DR-0005` has both tables,
  including the retraction of a first measurement that was wrong by a factor of eight.

## Reading the tree

- `decisions/` — the five decisions that shape everything, and why. `DR-0002` carries a correction
  where a later experiment showed its reasoning was too strong.
- `notes/FINDINGS.md` — what has been learnt, including the retractions, in place and struck
  through rather than deleted.
- `notes/NEXT.md` — what is worth doing next and why, with the measurements that say so.

## License

MIT OR Apache-2.0.
