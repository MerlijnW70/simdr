# Using simdr

What this crate promises somebody outside this repository, what holds each promise up, and what it
asks of you in return.

**It has no external callers yet.** That is the honest state and it shapes this document: everything
below is either enforced by something in the tree or marked as not. `notes/CLAIMS.md` is the same
exercise turned inward, and this is the outward half — the first piece of work here whose absence
nothing in the tree could detect.

## What it is

An **emitter**. Words in, words out: it builds SPIR-V compute modules and hands you a `Vec<u32>` to
give to `vkCreateShaderModule`. It does not open a device, allocate a buffer or dispatch anything.

```rust
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::F32;

// A compute kernel: read 32 floats per subgroup, sum across the lanes, write the total.
let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2))?;
let value = kernel.load::<32>(0)?;
let total = kernel.lanes()?.reduce_sum(value)?;
kernel.store_scalar(1, total)?;

let spirv: Vec<u32> = kernel.finish()?;
# Ok::<(), simdr::lanes::LaneError>(())
```

Running the result is Vulkan's job. `runner/` is this repository's own harness for that — a
workspace member so the tests can reach a device, `publish = false`, and **not** something to depend
on. The arrow points `runner -> simdr` and never back.

## What it promises, and what holds each promise up

| promise | what enforces it |
| --- | --- |
| **No dependencies.** Ever. | `[dependencies]` is empty in `Cargo.toml`, and `noha gate`'s boundary check audits the import graph of `src/` — 56 checks |
| **No `unsafe`.** | `#![forbid(unsafe_code)]` in `src/lib.rs`. The compiler, not a review |
| **No input makes it panic.** Every failure you can provoke is a `Result`. | `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]`, and every refusal is a `LaneError` variant named for what it refused |
| **The opcodes are Khronos', not remembered.** | `decisions/DR-0001`, and `spirv-val` over the kernel library at five widths plus 232 generated modules — a wrong number that makes an invalid module is caught, and `tests/integrity.rs` makes sure every number declared is one something emits |
| **The modules are legal SPIR-V.** | `spirv-val --target-env vulkan1.1`, in the suite, before any device sees them |
| **The behaviour is right.** | Two GPUs and a software driver at widths 4, 8, 16, 32 and 64, plus a differential fuzzer: <!--count:fuzz-operations-->23 generated operations across eight element domains, answered by a device and by an independent CPU reference. 256 rounds a domain on every push, 8 000 nightly |
| **Every branch is reached by some test.** | A mutation gate over the whole tree: 651 mutants over 93 targets, 100% |
| **Every public operation has a consumer.** | `tests/integrity.rs`, which also fails on an opcode nothing emits and a pipeline builder that dispatches without a bound |
| **The documentation does not drift.** | `tests/documented.rs` — every number these documents state, every file they name and every `Type::member` they name is resolved against the tree |

**What none of that promises.** That the API is the right shape, that it is stable, or that it does
what your workload needs. It has had one caller — this repository — and an API with one caller is a
guess that has been checked once.

## What it asks of you

**1. Ask the device for its subgroup width. Do not assume 32.**

```rust
# use simdr::kernel::{Kernel, Shape};
# use simdr::lanes::F32;
# let subgroup = 32;
let mut kernel = Kernel::<F32>::new(Shape::new(subgroup, 64, 2))?;
# Ok::<(), simdr::lanes::LaneError>(())
```

`decisions/DR-0002` is why the width is an argument rather than a default: a module is built *for* a
width, and one built for 32 does the wrong thing on a 64-wide device rather than a slower right
thing. `VkPhysicalDeviceSubgroupProperties::subgroupSize` is where the number comes from.

**2. Handle `LaneError` — a refusal is an answer.** A vector width with no mapping onto the
subgroup, a butterfly distance that leaves it, a lane index outside the vector: each is refused *by
name* and none of them is a bug. `Mapping::of` is the one place the width rule lives.

**3. Declare what you need.** `decisions/DR-0007`: the lane API adds a capability and an entry-point
interface entry when an operation needs one. If you build modules by hand alongside these, that is
the part which is easiest to get wrong and hardest to notice — dropping one line from an interface
list leaves 19 of 20 modules rejected by the validator while every device goes on returning the
right answers.

**4. A branch must be uniform.** `decisions/DR-0003`: a subgroup operation inside a per-lane branch
answers for whichever lanes happened to be running. `Lanes::if_uniform` takes a `Uniform`, which only
the votes produce, so the wrong version does not compile.

**5. Validate in your own CI.** `spirv-val --target-env vulkan1.1` on every module you emit. This is
the one piece of advice worth repeating, because the failure it prevents is invisible without it:
**drivers are lenient about things the validator is not.** An `OpUDot` emitted with a signed result
type ran correctly on two devices for weeks. A store of an `i32` into a `u32` buffer returned 192
correct-looking answers on two devices and an opaque `ERROR_UNKNOWN` on a third.

## What the elementwise surface actually is

Worth stating plainly, because the list is shorter than "SIMD semantics" suggests and the gaps are
decisions rather than oversights. On a `Vector<T, LANES>` you have:

* **arithmetic** — `add`, `mul`, `min`, `max`, `clamp`, `abs`, and the three bit shifts;
* **comparison and choice** — `greater_than`, `equal`, `select`;
* **floats only** — `sqrt`, `inverse_sqrt`, `exp`, `log`, `fma`;
* **across the lanes** — the reductions, the two scans, four shuffles, three votes, the packed
  integer dot products.

**There is no subtraction and no division** — `op.rs` declares no `OpISub`, no `OpFSub` and no
divide of any kind. **And no bitwise operation on a vector**: `OpBitwiseAnd` and `OpBitwiseOr` are
declared and emitted, but for the lane arithmetic inside a rotate and a clustered scan, and no
`Lanes` method offers them to a caller. That is not an omission waiting to be corrected: a
difference is `add(a, mul(b, splat(-1)))`, one extra instruction a driver folds, and no kernel
written here has wanted a name for it. `decisions/DR-0006`'s argument is the general one — *"a term would have no
caller… and an untested term is worse than a missing one"* — and the seven opcodes deleted in one
commit for having no emitter are what it looks like when the rule is applied late instead of early.

Ask for one if you need it. What you will get is the opcode, a bound that is exactly what the
instruction takes, a test at five widths and a place in the differential fuzzer — which is why the
answer is not simply "yes" the moment somebody asks.

## What is deliberately absent

Each of these was asked for, argued, and left out. The reasoning is in `notes/NEXT.md` under
*Deliberately not doing*; the short version:

* **Float-to-integer conversion.** `OpConvertFToS` truncates toward zero, so a caller wanting a
  rounded integer needs `RoundEven` first, and SPIR-V leaves the result **undefined** where the
  value does not fit — a safe API would have to invent a semantics the specification does not have.
* **A third dispatch axis.** `decisions/DR-0006`: `Grid` has no `z` field, so the dispatch cannot be
  written. Nothing here needed three, and an untested term is worse than a missing one.
* **A batching API.** The thing `decisions/DR-0008` says matters most — a round trip is ~100 µs and
  the device's share of it is 2.9% — and still unbuilt, because it has no caller. `notes/FINDINGS.md`
  has the shape it would take.
* **`sqrt`, `exp`, `log` in the differential fuzzer.** GLSL.std.450 specifies them in ULPs of
  tolerance rather than exactly, so a comparison would be two approximations agreeing.

## Stability

**There is none yet, and the version says so.** `0.0.0`, unpublished. What that means concretely:

* **The MSRV is 1.88** and CI holds it there. It is the release where `if let` chains stabilised,
  measured rather than assumed — it read `1.97` once, with a comment claiming it was measured,
  excluding nine releases' worth of callers for no reason.
* **The public surface is checked for consumers, not frozen.** `src/lanes/` declares
  <!--count:lane-operations-->65 public functions and `src/module/op.rs` <!--count:opcodes-->98
  opcodes; both numbers are asserted rather than typed, and both have moved. Seven opcodes were
  deleted in one commit for having no emitter.
* **Bounds tighten.** `Lanes::shift_left` took `T: Element` until it turned out that shifting a
  vector of floats built a module `spirv-val` rejects; it takes `T: Integer` now. That change breaks
  source compatibility for code that was already emitting invalid SPIR-V, and this project will make
  that trade every time.

## If a device disagrees with you

The differential fuzzer is the right shape to report through, because it makes a disagreement
reproducible from a number. `runner/src/fuzz/mod.rs` generates a program from a seed, works out the
answer on the CPU, and compares — so a finding is *a seed, a domain and a subgroup width*, and
re-running those three gives the same program on any machine.

Worth including: the device and driver version, the subgroup width it reported, whether `spirv-val`
accepts the module, and whether a second implementation agrees. That last one has decided every
device-shaped question in this repository so far — including one where the specification was silent
and the hardware was not.
