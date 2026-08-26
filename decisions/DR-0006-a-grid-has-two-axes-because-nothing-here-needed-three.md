---
id: DR-0006
title: A grid has two axes because nothing here needed three
status: prose-only
---

## The Measurement

`runner::Grid` declares two fields, `x` and `y`, and no third. `runner/examples/plane.rs` runs the
same elementwise kernel five ways over the same elements on 2026-08-26: on an RTX 4080 at 131 072
invocations, one axis reads 3.34 µs against the grid's 3.36 at workgroup 32, and 1.69 against 1.68
at workgroup 256; on the integrated Radeon at 262 144 invocations, 39.92 against 39.74 at workgroup
64, and 43.00 against 42.64 at workgroup 512. The workgroup size moves the same kernel from 3.34 µs
to 1.69 on the RTX 4080 and from 39.92 to 43.00 on the Radeon — 1.98× faster on one device and 7.7%
slower on the other, in opposite directions.

## The Decision

`simdr::kernel::Shape` describes a kernel with one axis or with two and `runner::Grid` dispatches
along x and y, with `vkCmdDispatch` always called at a z count of 1. `Shape::new` and `Shape::grid`
are separate constructors, so a kernel with no second axis says so and `Kernel::load_row` on one is
refused as `LaneError::NotAGrid`; a pitch of zero is `LaneError::BadPitch` and a zero row count is
`LaneError::BadRows`.

## The Rejected Route

A third axis was rejected because a z count above 1 today runs every workgroup again over the same
elements — nothing in the emitted address distinguishes them — and `Grid` has no z field, so that
dispatch cannot be written. The addressing for a third term was rejected as **NOT DETERMINED**: a
slice may be a stride or a separate binding, those are different layouts, and no workload here has
had one to choose between.

## The Limit

The two axes buy no speed and the measurement says so: the multiply and the add a row costs are
inside the noise on both devices, on kernels that are memory-bound. What the axis buys is that a
caller stops linearising by hand, and that is not a figure. That `pitch` matches the buffer and that
the dispatch covers the matrix are the caller's, and reading past a runtime array is undefined
rather than refused. The workgroup-size effect above is device-specific in direction and belongs to
`WORKGROUP_SIZE` rather than to grids; no third device was measured.
