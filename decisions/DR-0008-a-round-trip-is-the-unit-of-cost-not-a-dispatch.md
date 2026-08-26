---
id: DR-0008
title: A round trip is the unit of cost, not a dispatch
status: prose-only
---

## The Measurement

`runner/examples/latency.rs` on 2026-08-26, host wall clock over submit, wait and copy back. On an
RTX 4080 at subgroup 32: two answers with the pipeline rebuilt per call at 886.4 µs, the same two
through a held `Session` at **105.6 µs** of which the device is 3.0 µs — **2.9%** — and 2048 answers
through the same session at 124.2 µs, 0.061 µs an answer, the device 3.5 µs at 2.8%. On the
integrated Radeon at subgroup 64: one answer rebuilt per call at 2160.6 µs, held at **768.1 µs** of
which the device is 2.5 µs — **0.3%** — and 1024 answers at 840.1 µs, 0.820 µs an answer, the device
11.6 µs at 1.4%. Two answers cost 105.6 µs and two thousand cost 124.2, so the round trip is fixed
and the batch is the only lever. The example prints the break-even for the machine it runs on: on
the RTX 4080, never against a CPU at 50 ns an answer, 3156 pending answers at 100 ns, 132 at 1 µs
and 12 at 10 µs.

## The Decision

A workload belongs here when the caller can hand it thousands of independent answers at once, and
one that needs an answer before deciding what to ask next belongs on the CPU whatever its
arithmetic looks like. The number that decides it is the host round trip and not the device clock,
and `runner/examples/latency.rs` prints both side by side so the decision is re-tested rather than
re-argued.

## The Rejected Route

Putting a chess engine's NNUE evaluation here was rejected at ~9 700 independent evaluations needed
against the **one** that alpha-beta produces, since a node's score decides whether its siblings are
searched at all. Its own documents rejected it twice more before this arithmetic: `SPEED.md` puts a
node at 376 ns with evaluation ~20% of it, so an evaluation costing nothing buys about 25% more
nodes and perhaps 15 Elo against a 2–3× gap, and `NNUE.md` records the network as not adopted.

## The Limit

Both figures above are one run of the example on one machine on one day, and the example takes the
best of three runs of 200 round trips rather than reporting a spread — so no variance is recorded
here and none should be read into the third decimal. The break-even table is arithmetic over a CPU
cost the example is told rather than one it measures. Nothing runs this: CI runs none of the
<!--count:examples-->17 examples, by design, because a shared runner's wall clock is not evidence
about a round trip. The engine's figures are that project's and were not re-taken here.
