# simdr

SIMD on the GPU, in Rust, with an empty dependency table.

`Simd<T, N>` semantics — splat, elementwise arithmetic, reductions, scans, shuffles, votes —
lowered onto SPIR-V subgroup instructions by an emitter that writes the binary format itself. No
build script, no shader compiler, no `unsafe`, no dependencies.

```rust
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::F32;

let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2))?;
let value = kernel.load::<32>(0)?;
let total = kernel.lanes()?.reduce_sum(value)?;
kernel.store_scalar(1, total)?;

let spirv: Vec<u32> = kernel.finish()?;
```

## Crates

| | |
| --- | --- |
| `simdr` | The emitter. No dependencies, `#![forbid(unsafe_code)]`, every refusal a `Result`. |
| `runner` | Runs the output on a real GPU through `ash`. Not published; Vulkan is FFI and FFI is `unsafe`. |
| `simdr-cli` | `simdr probe` and `simdr list`. |

Nothing in the emitter reaches the runner.

## The rule

A module is built for one subgroup width, and the width comes from the device rather than from an
assumption:

```
cargo run -p simdr-cli -- probe
```

`N` and that width meet at build time, because each case below is a different instruction sequence
rather than one sequence with a parameter. Equal is `WholeSubgroup`; a divisor is `Clusters`; a
multiple is `Strips`. Anything else has no mapping and is refused by name as a `LaneError`, as is a
shuffle operand that reaches outside the lanes the vector occupies.

## Running it

```
cargo test -p simdr                       # the emitter, no device needed
cargo test -p runner -- --test-threads=1  # every kernel on a real GPU
cargo run -p simdr-cli -- probe           # what this device offers
```

`SIMDR_DEVICE` picks a device by substring. `SPIRV_VAL` points at `spirv-val`; unset, the suite
looks on `PATH` and skips loudly if it finds nothing.

## What is in the tree

Every number below is a marker resolved against the source by `tests/documented.rs`. A stale one
fails the build, so nothing here is stated on trust.

| | |
| --- | --- |
| SPIR-V opcodes declared | <!--count:opcodes-->116 |
| lane operations | <!--count:lane-operations-->78 |
| `#[test]` functions | <!--count:test-functions-->925 |
| checks in `tests/integrity.rs` | <!--count:integrity-tests-->17 |
| checks in `tests/documented.rs` | <!--count:documented-tests-->12 |
| counters behind this table | <!--count:counters-->10 |
| CI jobs | <!--count:ci-jobs-->5 |
| element operations the differential fuzzer generates | <!--count:fuzz-operations-->23 |
| examples | <!--count:examples-->17 |
| of those, needing a device | <!--count:device-examples-->16 |

## What this is not

- **Not a shader language.** You write against `Kernel` and `Lanes` in Rust and get SPIR-V words
  out.
- **Not portable across subgroup widths.** A module is built for one width, deliberately.
- **Not matrices.** `i8`, `u8`, `i16`, `u16` and `f16` are here; cooperative matrix types are not.
- **Two dispatch axes, not three.** `vkCmdDispatch`'s z is always 1 and `Grid` has no field to set.

## License

MIT OR Apache-2.0.
