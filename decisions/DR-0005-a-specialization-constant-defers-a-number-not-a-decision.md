---
id: DR-0005
title: A specialization constant defers a number, not a decision
status: prose-only
---

## The Measurement

`spirv-val --target-env vulkan1.1` accepts a clustered `OpGroupNonUniformIAdd` whose `ClusterSize`
is an `OpSpecConstant` — `tests/deferred.rs`, in
`a_cluster_size_that_is_a_specialization_constant_is_valid_spirv` — and `runner/tests/specialized.rs`
runs one such module at 4, 8 and 16 on an RTX 4080 with the default of 32 still reducing the whole
subgroup. The size **can** be deferred.

`runner/examples/specialize.rs` on that card, a reduction over 1 048 576 elements, which
`reduction::folds` resolves to four folds. Five standalone runs on 2026-08-26: emitting the four
modules 53.8–55.1 µs; a pipeline per fold from one module each **571.6–708.7 µs**; a pipeline per
fold from one specialized module **6 168.3–7 306.2 µs**. Specializing costs between 8.7 and 11.5
times what building a module per fold costs, and the example now reports it as removing −766% of
the setup.

Pipeline creation does not vary with the shape being compiled: at fold factors 2, 4, 8, 16 and 32 —
modules of 258 to 1 278 words — a single-build call measures 565.5–611.4 µs, of which the two
256-byte buffers `probe_pipelines` allocates and frees account for about 566 µs on their own.

## The Decision

A specialization constant may carry any value a kernel needs at pipeline creation and nothing the
emitter has to reason about while building the module. `Lanes::new` still takes the subgroup width,
because the three mappings differ in which instructions are emitted and a value arriving at pipeline
time cannot add instructions that were never written. The cluster size can be deferred; the mapping
cannot.

## The Rejected Route

Deferring constants to cut setup was rejected at 6 168.3–7 306.2 µs against 571.6–708.7 for building
one module per fold: on this driver it costs roughly ten times more rather than less, and a
specialization constant is fixed *at* pipeline creation, so *n* values need *n* pipelines however
few modules they came from. Holding the pipelines instead removes the setup rather than moving it,
at 11.4× over 8 192 elements and 9.2× over 1 048 576 — `runner/examples/reducer.rs`, same day.

## The Limit

**The instrument does not reproduce between harnesses, and no cause has been found.** The same four
builds measure 571.6–708.7 µs from this example and 21 242–23 316 µs from a test binary issuing the
identical `probe_pipelines` calls — thirty times apart, stable to within 10% inside each, in a debug
build and a release build alike. Ruled out by measurement: the fold factor, the module size, drift
across ten rounds in one process, and the build profile. Not ruled out: anything else.

So the *ratio* between the two strategies is what this record rests on, because it is taken inside
one process in one run, and the absolute microseconds are **NOT PORTABLE**. An earlier version of
this record reported the specialized column at 793.0 µs and specializing as removing 9.7% of setup;
five runs today put that column at 6 168.3–7 306.2 µs and the sign the other way. Which of the two
machines-and-days is anomalous is not established, and a figure quoted from either should be
re-taken rather than cited.

Nothing tests what a specialization constant must not do. `Module::spec_constant` returns an `Id`
like any other, so an emitter branching on a default and shipping a module that only appears to
defer the decision would pass every check here.
