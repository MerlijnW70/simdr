# simdr

**SIMD on the GPU, in Rust, with an empty dependency table.**

`Simd<T, N>` semantics — splat, elementwise arithmetic, reductions, scans, shuffles, votes —
lowered onto SPIR-V subgroup instructions by an emitter that writes the binary format itself. No
`build.rs`, no shader compiler, no `unsafe`, and no crates.

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

## Start here

```
cargo run -p simdr-cli -- probe
```

You cannot build a module without the subgroup width, so the first thing to do is ask for it. On
this machine, 2026-08-26:

```text
NVIDIA GeForce RTX 4080          subgroup width 32, up to 32 subgroups per workgroup
AMD Radeon(TM) Graphics          subgroup width 64, up to 16 subgroups per workgroup
```

Both report every subgroup feature the lane API can ask for and all six narrow features, which
resolve to five element types usable end to end: `i8`, `u8`, `i16`, `u16` and `f16`.
`simdr probe` also prints which features are missing, which is what tells you a kernel will be
refused at pipeline creation rather than at validation. `simdr list` names every device — this
machine turned out to have two at two different widths, and for a month only one had been built for.

## What you get

| | |
| --- | --- |
| `simdr` | The emitter. Zero dependencies, `#![forbid(unsafe_code)]`, no panics on any input. |
| `runner` | Runs the output on a real GPU through `ash`. A separate crate, because Vulkan is FFI and FFI is `unsafe`. |
| `simdr-cli` | `simdr probe` and `simdr list`. |

The arrow points one way: nothing in the emitter can reach the runner.

**`USING.md` is the shorter document** and the one to read next if you are here to use this rather
than to read it — what the crate promises, what holds each promise up, what it asks of you, and
what is deliberately missing. Its examples compile, so a guide that stops working stops the build.

## The one thing you have to know

`N` is fixed when you write the code and the subgroup width is fixed by the hardware. They meet at
build time, because the rows below are three *different instruction sequences* rather than one
instruction with a parameter — no value arriving later can add instructions that were never
emitted.

| `N` vs width | mapping | a reduction | a scan | a shuffle |
| --- | --- | --- | --- | --- |
| equal | `WholeSubgroup` | one subgroup instruction | one subgroup instruction | one instruction |
| a divisor | `Clusters` | one clustered instruction, several vectors at once | a `log2(N)`-step ladder | a butterfly, a broadcast or a rotate, inside the vector |
| a multiple | `Strips` | `strips - 1` scalar ops, then one subgroup instruction | one scan per strip, carrying a running total | one instruction per strip |

Anything else — 12 lanes on a 32-wide subgroup — has no mapping and is refused by name. So is a
shuffle operand that reaches outside the lanes the vector occupies: `butterfly(value, 4096)` on a
32-wide subgroup builds a module `spirv-val` accepts and every device runs, and the answer is
whatever was in the register.

The clustered scan is the expensive row. SPIR-V has a `ClusteredReduce` and no clustered scan, so
`Lanes::prefix_sum` builds a Hillis-Steele ladder — `log2(N)` steps of shuffle, compare and select.
The cheap alternative subtracts each cluster's offset back off a subgroup-wide scan, and over
floats that loses exactly the low bits the scan just accumulated.

`decisions/DR-0002` is the record. Branches follow the same rule from `DR-0003`: a branch takes the
result of a vote, which is uniform by construction, and a per-lane conditional is `Lanes::select`,
which computes both arms.

## When this pays, and when it does not

**This is for throughput.** A caller that needs one answer before deciding what to ask next belongs
on the CPU, whatever the arithmetic looks like — and the number that decides it is the host round
trip, not the device clock.

`runner/examples/latency.rs`, 2026-08-26:

| held `Session` | per call | per answer | of which the device |
| --- | --- | --- | --- |
| RTX 4080, 2 answers | 105.6 µs | 52.8 µs | 3.0 µs — **2.9%** |
| RTX 4080, 2 048 answers | 124.2 µs | 0.061 µs | 3.5 µs — 2.8% |
| Radeon, 1 answer | 768.1 µs | 768.1 µs | 2.5 µs — **0.3%** |
| Radeon, 1 024 answers | 840.1 µs | 0.820 µs | 11.6 µs — 1.4% |

Two answers cost 105.6 µs and two thousand cost 124.2, so the round trip is a **fixed** cost and
the only lever is how many answers you divide it by. The example prints the break-even for whatever
device runs it — on the RTX 4080, a CPU at 50 ns an answer is never worth leaving, at 100 ns you
need 3 156 independent answers pending, at 1 µs you need 132.

**Independent** is the load-bearing word. `decisions/DR-0008` works a real case through — a chess
engine whose NNUE layer one of these kernels was modelled on — and the answer is *stay on the CPU*.

Rebuilding per call costs more than the work: `Gpu::run` allocates three buffers and a pipeline
every time, which `runner/examples/overhead.rs` puts at **937.5 µs** of fixed cost against 0.7 µs
for the same dispatch amortised over a thousand. Hold a `Session`, or a `Reducer`, or a `Scanner`.

## What the measurements say

All on this machine, 2026-08-26, each from the example named beside it.

| | RTX 4080 | integrated Radeon | from |
| --- | --- | --- | --- |
| `Reducer` against rebuilding, 8 192 elements | 11.4× | — | `examples/reducer.rs` |
| the same over 1 048 576 | 9.2× | — | `examples/reducer.rs` |
| `Simd<i8, 128>` against `Simd<i32, 32>`, 16 777 216 elements | 6.46× | — | `examples/narrow.rs` |
| `OpSDot` against the nineteen instructions it replaces, 32 per element | 1.20× | **9.17×** | `examples/dot.rs` |
| a map fused into a scan, against three host crossings | 2.1–3.1× | — | `examples/scanner.rs` |
| a second dispatch axis, 131 072 invocations | 3.36 µs against 3.34 | — | `examples/plane.rs` |

Nineteen is counted rather than asserted. Decoding both modules and differencing every opcode: the
written-out spelling emits four `OpBitcast`, four `OpShiftLeftLogical`, four
`OpShiftRightArithmetic`, four `OpIMul` and three `OpIAdd` where the packed one emits a single
`OpSDot` — eighteen more instructions over the whole module, once three shift constants are added
and a capability and an extension are dropped.

Two of these are worth reading as a "no". The second dispatch axis buys **no speed** — what it buys
is that a caller with a matrix stops linearising by hand. And the packed dot product is worth 20% on
the discrete card and nine times on the integrated one, so *whether to use it depends on the
device*; both report it as accelerated.

There is **no performance claim about large working sets**. A cliff shows up past about 50 MB and
three explanations have been tested and refuted — L2 capacity, eviction of a single allocation, and
placement under three simultaneous ones. `notes/FINDINGS.md` has the runs.

## Running it

```
cargo test -p simdr                       # the emitter, no device needed
cargo test -p runner -- --test-threads=1  # every kernel on a real GPU
cargo run -p simdr-cli -- probe           # what this device offers
cargo run --release -p runner --example latency   # the round trip, for your machine
```

`SIMDR_DEVICE=radeon` picks a device by substring. `SPIRV_VAL` points at `spirv-val`; unset, the
suite looks on `PATH` and skips loudly if it finds nothing.

## How it is checked

<!--count:test-functions-->905 `#[test]` functions, which is not what `cargo test --workspace`
prints — that number moves with the machine, and this one is a property of the source.
On 2026-08-26: **496 passing in the emitter**, **409 in the runner at subgroup 32 with no skips**,
and **398 at subgroup 64 with 17**, sixteen of which say `written for a 32-wide subgroup`.

Seven layers, each of which has caught something the ones above it did not: unit tests,
`spirv-val` at `--target-env vulkan1.1`, execution against CPU references, the same suite at five
widths, a differential fuzzer over all eight element types, mutation coverage, and a dispatch-bounds
check that refuses a kernel reading past its binding. `notes/CLAIMS.md` is the honest inventory —
what is checked, what is not, and which numbers rest on a manual run.

## What this is not

- **Not a shader language.** You write against `Kernel` and `Lanes` in Rust and get SPIR-V words
  out. Compiling arbitrary Rust to a GPU is [rust-gpu](https://github.com/Rust-GPU/rust-gpu), which
  is a much larger thing and pins nightly.
- **Not `core::simd`.** The nearest neighbour on the same narrow question is
  [VectorWare](https://www.vectorware.com/blog/simd-on-gpu/), building it as a compiler backend that
  consumes Rust's own portable SIMD. They compile the `Simd` you already wrote; here you write
  against a builder, which is smaller, needs no compiler and works on stable.
- **Not portable across subgroup widths.** A module is built for one width, deliberately and
  visibly. Five have been run — 32 and 64 on the two devices here, and 4, 8 and 16 on lavapipe.
- **Not matrices.** `i8`, `u8`, `i16`, `u16` and `f16` are here; cooperative matrix types are not.
- **Two dispatch axes, not three.** `vkCmdDispatch`'s z is always 1, and `Grid` has no field to set
  — a z count above 1 would silently run every workgroup again over the same elements.

## Reading the tree

- `USING.md` — the guide, and the shortest way in.
- `decisions/` — the <!--count:decisions-->10 records, each one a measurement, a decision, the
  route it rejected and the figure that killed it, and what the numbers do not establish.
- `notes/FINDINGS.md` — what has been learnt, retractions in place rather than deleted.
- `notes/NEXT.md` — what is worth doing next, with the measurements that say so.

## License

MIT OR Apache-2.0.
