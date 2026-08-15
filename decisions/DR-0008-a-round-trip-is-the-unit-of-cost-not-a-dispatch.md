---
id: DR-0008
title: A round trip is the unit of cost, not a dispatch
status: prose-only
---

## The decision

**This project is for throughput, and a workload belongs on it only when the caller can hand it
thousands of independent answers at once.** A caller that needs one answer before it can decide what
to ask next belongs on the CPU, whatever the arithmetic looks like.

The number that decides it is the **host round trip** — submit, wait, copy back — and not the
device's own clock. `runner/examples/latency.rs` measures both and prints them side by side, because
the gap between them *is* the finding.

## The measurement

`cargo run --release -p runner --example latency`, best of three runs of 200 round trips, after a
warm-up. `Gpu::run` rebuilds its pipeline per call; the held `Session` does not, and is the fair
figure for a caller in a loop.

| RTX 4080, subgroup 32 | per call | per answer | of which the device |
| --- | --- | --- | --- |
| 2 answers, built per call | 858 µs | 429 µs | — |
| 2 answers, held session | **100 µs** | 50 µs | 2.9 µs — **2.9%** |
| 2 048 answers, held session | 129 µs | **0.063 µs** | 3.6 µs — 2.8% |

| Integrated Radeon, subgroup 64 | per call | per answer | of which the device |
| --- | --- | --- | --- |
| 1 answer, held session | **779 µs** | 779 µs | 2.5 µs — **0.3%** |
| 1 024 answers, held session | 878 µs | 0.858 µs | 11.4 µs — 1.3% |

**Ninety-seven per cent of a single answer is not computation.** It is the submission, the fence and
the copy back, and no kernel change touches any of it. On the integrated part it is 99.7%.

That is why the batched row costs almost the same as the single row: 100 µs for two answers and
129 µs for two thousand. The round trip is a **fixed** cost, so the only lever a caller has is how
many answers it divides that cost by.

## The break-even, as arithmetic

A device wins only where the CPU would have taken longer than the whole round trip. With a round
trip of `R` and a device per-answer cost of `d`, a CPU costing `c` per answer needs

```text
R / (c - d)
```

independent answers pending before the device is worth asking — and **never**, at any batch size,
when `c ≤ d`. The example prints this table for the machine it runs on:

| a CPU that takes | independent answers needed (RTX 4080) |
| --- | --- |
| 50 ns per answer | **never** — the device is slower per answer too |
| 100 ns | 3 458 |
| 1 µs | 137 |
| 10 µs | 13 |

The word *independent* carries the whole decision. Answers that depend on each other cannot be
batched, however many of them there are in total.

## What settled it: a chess engine, next door

`H:\schaak` is a UCI engine whose NNUE layer this project's `kernels::network::clipped_dot` was
modelled on, so the question was fair rather than rhetorical: can the GPU evaluate its network?

Its own measurements answer twice over, before this table is even consulted.

* `SPEED.md` attributes a node by differential timing: a node is **376 ns** and evaluation is
  **~20%** of it, **78 ns**. So an evaluation that cost *nothing at all* would buy about 25% more
  nodes per second — worth perhaps **15 Elo**, against a 2–3× gap to its reference engines.
* `NNUE.md` records that the network is **not adopted**: distilling the engine's own hand
  evaluation asymptotes to parity with it, and the live evaluation remains the hand one.

And then the arithmetic here. At 78 ns per answer on the CPU against 63 ns per answer on the device,
the break-even is **~9 700 independent evaluations pending at once**. Alpha–beta produces **one**:
a node's score decides whether its siblings are searched at all, which is what pruning *is*. The
network is also far too small to amortise anything — `(768→256)×2→1` makes the output layer a
512-element dot product, tens of nanoseconds of CPU SIMD against a 100 µs round trip.

**Neither is the engine's bottleneck anyway.** Every lever in its own documents bottoms out on
playing games: `NNUE.md` names the WDL reinforcement loop — *"a large, multi-day, self-play-bound
compute project"* — as the only path past parity, and `TUNING.md` records a 342-parameter tune that
improved validation loss and lost **−24.7 Elo** in play, concluding that *"only game-playing strength
is a trustworthy arbiter"*. Self-play is alpha–beta search: branchy, hash-table-bound,
pointer-chasing. The one workload a GPU is worst at.

There is a second, structural refusal underneath the numbers. `schaak` is `#![forbid(unsafe_code)]`
with an empty dependency table, which is in its crate description. `simdr` shares both properties
and only *emits* SPIR-V; `runner` is `ash`, FFI and `unsafe` by necessity, because something has to
run it. Linking `runner` into that engine ends both claims at once.

## What this rules in

Stating the boundary is the useful half, and it is not "GPUs are for big data". It is specific:

* **Batch size beats kernel quality.** A caller who can raise the answers per round trip from 2 to
  2 048 gains 800×; no kernel change in this repository has ever been worth more than 3×.
* **The reduction and scan chains are the right shape**, because they answer one question over a
  whole buffer in one submission — `Gpu::sum` at 11.2× over 8 192 elements, `Gpu::scanner_of` at
  2.0–3.0×, both measured in `notes/FINDINGS.md`.
* **The differential fuzzer is the best workload here**, and it is not a speed one: 30 000 generated
  programs, each answered by a device and by a CPU reference, is throughput-shaped *and* the thing
  that finds emitter bugs.
* **Latency-bound callers get an honest no.** That is worth more than a benchmark they would have
  discovered for themselves after a fortnight's integration.

## Why this is a decision and not a note

Because the pressure to re-open it is predictable and the arithmetic is not. "The kernel is only a
few microseconds" is true and irrelevant — it is 2.9% of what the caller waits. Any future proposal
to put a latency-bound workload here has to move the *round trip*, not the kernel, and nothing in
this repository can: it is the driver, the scheduler and the PCIe crossing.

`runner/examples/latency.rs` is the check. It runs on any device and prints the break-even for that
machine, so the decision can be re-tested rather than re-argued.

## The table it replaced

The first version of that example put a **host** round trip and a **device** timestamp under one
`per answer` heading, and divided only one of them by the answers it produced — so the batched row
read about 500× better than any caller would ever see. It was the right measurement for a kernel
author and the wrong one for the decision above, presented as the same thing.

Fixing it moved the batched figure from `1.9 µs per dispatch` to `129 µs per call`, which is the
number this decision rests on. A figure that reads as evidence and was produced by an instrument
that cannot see the claim is the failure this project keeps finding; this one was in the example
written to settle exactly that question.
