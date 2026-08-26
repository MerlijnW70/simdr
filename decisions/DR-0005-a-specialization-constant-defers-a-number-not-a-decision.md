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
subgroup. So the size **can** be deferred.

`runner/examples/specialize.rs` on that card on 2026-08-26, a reduction over 1 048 576 elements,
buffers allocated once so that what is timed is pipelines: emitting the modules 55.9 µs; a pipeline
each from one module per fold **20 770.9 µs**; a pipeline each from one specialized module
**10 658.5 µs**. Specializing removes 48.8% of the setup, and emission is 0.3% of what building the
pipelines costs. `runner/examples/reducer.rs` on the same run holds its pipelines instead and
measures **11.4×** over 8 192 elements at 3 folds and **9.2×** over 1 048 576 at 5.

## The Decision

A specialization constant may carry any value a kernel needs at pipeline creation and nothing the
emitter has to reason about while building the module. `Lanes::new` still takes the subgroup width,
because the three mappings differ in which instructions are emitted — a value arriving at pipeline
time cannot add instructions that were never written. The cluster size can be deferred; the mapping
cannot.

## The Rejected Route

Deferring constants to cut setup was rejected at 10 658.5 µs against 20 770.9, because fourteen
values still need fourteen pipelines however few modules they came from, while holding the
pipelines removes the setup entirely at 11.4× and 9.2×. `kernels::fold_halves_open` and
`Kernel::load_offset_by` are kept at one `OpIAdd` per strip against the baked-in form, because they
are what made the comparison possible.

## The Limit

**The figures this record carried before do not reproduce, and the discrepancy is unexplained.** It
reported a pipeline each from fourteen modules at 809.6 µs and specializing as removing 9.7%; the
run above puts the same column at 20 770.9 µs and 48.8%, a factor of 25.7 on the first number and a
reversed conclusion on the second. No cause was established, no intermediate revision was
bisected, and the example's own output labels its rows "fourteen modules" while stating four folds
for this size — so which count the second table timed is **NOT ESTABLISHED**. The decision above
rests on the ordering of the two strategies, which both runs agree on; the size of the gap between
them does not have one figure.

Nothing tests what a specialization constant must not do. `Module::spec_constant` returns an `Id`
like any other, so an emitter branching on a default and shipping a module that only appears to
defer the decision would pass every check here.
