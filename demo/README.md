# demo — procedural worlds on the GPU

A throwaway. Three worlds generated from nothing but the invocation's own coordinates: a
two-octave landscape drawn as a cross-section, a cave system packed one bit per layer, and an
escape-time fractal in fixed point.

The landscape's fine octave ran at a quarter of the resolution first, and the picture came out in
four-column blocks — a detail layer coarser than the thing being drawn is not a detail layer. It is
at full resolution now, and a skyline is a shape a reader can tell is wrong, which a shaded map
from above is not.

```bash
cd demo
cargo run --release      # three pictures and a timing table
cargo test --release     # the same three, held to the number a CPU says they should be
```

## Deleting it

```bash
rm -r demo
# then remove the `exclude = ["demo"]` line from ../Cargo.toml
```

That is the whole of it. `grep -rn demo --include='*.rs' --include='*.toml' ..` outside this
directory returns the one `exclude` line, which is the proof rather than the claim.

**Nothing the engine states about itself moves when this appears or goes.** `tests/documented.rs`
counts `#[test]` functions and examples over a *positive list* of the workspace's own directories,
not over the whole tree — so a scratch crate beside `src/` cannot change a number the README states,
whatever it is called.

| what | why it cannot see this directory |
| --- | --- |
| `cargo test --workspace` at the root | `exclude = ["demo"]` in `../Cargo.toml`, and an empty `[workspace]` here making this its own root |
| CI | builds `-p simdr`, `-p runner` and the root workspace; none of them reaches an excluded directory |
| `tests/integrity.rs` | scans `src/`, `runner/src/` and `cli/src/` by name |
| `tests/documented.rs`'s counters | walk the workspace's own directories, listed |
| `noha.yaml` and the mutation gate | list source files explicitly; nothing here is listed |
| `noha gate`'s zero-dependency boundary | audits `src/` |

The reference checks in `tests/documented.rs` *do* read this directory, and that is the opposite
obligation on purpose: every file path and every `Type::member` named in the prose here is resolved
against the engine, because a sentence about the engine rots wherever it is written. Deleting the
directory takes those sentences with it and leaves nothing owed.

## Why it is a test and not a picture

`notes/FINDINGS.md` records the entry requirement the last sandbox was built to: **a workload is
only a test if something can disagree with it.** It also records that a procedural world was ruled
*out* on exactly that ground — "no reference at all beyond looking right".

That objection is right about float noise and wrong about this. Everything here is **integer
arithmetic**, so the host computes the same number and the comparison is a fact rather than a
tolerance: 16 384 heights, 16 384 cave words — half a million layer bits — and 16 384 escape counts,
all bit-exact, on the RTX 4080, the integrated Radeon and lavapipe at 4, 8 and 16 lanes.

The pictures are a side effect. What is being checked is the mapping, the shifts and the loop.

## What the engine's rules cost

Three of them shaped this code, and none of them is a limitation somebody forgot to lift.

**No per-lane branch.** `decisions/DR-0003` refuses one, and procedural generation is written with
branches everywhere — *if the density is above the threshold, place stone*. Every one of those is a
comparison and a `select` here. That is not a workaround: a divergent branch runs both sides and
masks, so the select is what the hardware was going to do.

**No exclusive-or.** `src/module/op.rs` declares no opcode for one, and the two bitwise operations
it does declare are not on the lane API. So the hash mixes with **multiply, add and shift** — the
usual `h ^= h >> 16` becomes `h += h >> 16`. It mixes less per round and it mixes enough.

**No subtraction and no division.** A difference is `add(a, mul(b, -1))` and a halving is a shift.
Both appear, and the fractal's `zx² − zy²` is the first one.

There is a fourth thing that is not a rule but a shape worth naming: **the coordinates come from the
invocation**. A `Vector<T, LANES>` at `LANES == subgroup` is one element per invocation, so a value
splatted from `Kernel::local_index` is a *different number in every lane* — the lane's own column.
No buffer of coordinates is uploaded and the input is never read.

## What the timing table says, which is not what a demo usually says

```
  world         round trip        host    ratio     dispatch  agreed
  landscape         2.76ms      2.15ms     0.8×      73.44µs  all of them
  caverns           2.84ms      8.43ms     3.0×     126.02µs  all of them
  fractal           3.42ms     41.88ms    12.2×     165.76µs  all of them
```

A million answers each, on an RTX 4080 against one CPU thread.

**The device's own work is 29×, 67× and 253× faster than the host, and the landscape still loses.**
The dispatch column is the arithmetic; the round trip column is what a caller waits for, and it is
~2.8 ms of moving eight megabytes for every one of them. `decisions/DR-0008` priced that gap and
nothing in this directory can move it.

What *does* move is the work per byte returned. All three return the same four bytes per answer and
do two octaves, thirty-two, and forty iterations respectively — so the ratio climbs from 0.8× to
12.2× without the transfer changing at all. **That is the whole lesson of this demo**, and it is the
same boundary DR-0008 drew for a chess engine, redrawn on a workload that looks nothing like one.

And half of that transfer is waste: `Gpu::run_grid` sizes the output from its input, so a generator
that reads nothing still uploads four megabytes of zeros. `notes/NEXT.md`'s *"a buffer the caller
already owns"* is the entry that would fix it, and it is still open for want of a caller — this
directory is one, and a throwaway is not the caller that entry is waiting for.
