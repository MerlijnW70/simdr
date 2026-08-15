---
id: DR-0006
title: A grid has two axes because nothing here needed three
status: prose-only
---

## The decision

`simdr::kernel::Shape` describes a kernel with one axis or with two. `runner::Grid` dispatches
along x and y. Neither has a z, and `vkCmdDispatch` is always called with a z count of 1.

`Shape::new` and `Shape::grid` are different constructors rather than one with a defaulted `rows`,
so a kernel that has no second axis says so and a row index on it is refused by name rather than
computed from a component that is always zero.

## Why not three

Vulkan gives three. `WorkgroupId` and `LocalInvocationId` are three-component vectors and the third
component is already loaded — reaching it is one more `OpCompositeExtract` and one more term in the
address. It is not hard, and that is not the argument.

The argument is that a third term would have no caller. Every use this project has had for a second
axis — a matrix, an image, a batch of rows — is two-dimensional, and the addressing for a third is
not obvious from the second: is the buffer `plane × slice + row × pitch + column`, or is a "slice"
a separate binding? Those are different layouts and the right one depends on what is being laid
out. Guessing produces an API that has to be replaced rather than extended.

**And an untested term is worse than a missing one.** A z count above 1 today runs every workgroup
again over the same elements, because nothing in the emitted address distinguishes them — which is
a wrong answer rather than an error. `Grid` has no z field, so that dispatch cannot be written.

## What the second axis actually bought, measured

Not speed. `runner/examples/plane.rs` runs the same elementwise kernel five ways over the same
elements, and the two shapes that differ only in the address come out equal on both hardware
devices:

| 131 072 invocations | one axis | two axes |
| --- | --- | --- |
| RTX 4080, workgroup 32 | 3.38 µs | 3.38 µs |
| RTX 4080, workgroup 256 | 1.66 µs | 1.67 µs |

| 262 144 invocations | one axis | two axes |
| --- | --- | --- |
| integrated Radeon, workgroup 64 | 42.99 µs | 42.92 µs |
| integrated Radeon, workgroup 512 | 48.00 µs | 46.71 µs |

So the multiply and the add the row costs are invisible on a memory-bound kernel, which is what
these are. What the axis buys is that a caller with a matrix stops linearising it by hand — and
hand-linearising is exactly the arithmetic that ten tests got wrong on the second device.

## The measurement that nearly said something false

The first version of that example compared a one-axis kernel of `width` invocations against a grid
`8` rows deep — and reported the grid at **2×**.

That is not the address. A grid `rows` deep has `width × rows` invocations per workgroup, so the
comparison moved the occupancy and the addressing at once, and the occupancy is the whole of the
difference. The example now runs both variables independently, and the two-by-two is above.

The workgroup-size effect is real and is **device-specific in direction**: eight subgroups per
workgroup is 2.04× faster on the RTX 4080 and 12% *slower* on the integrated Radeon. It is written
up in `notes/FINDINGS.md` as a finding about `WORKGROUP_SIZE`, which is where it belongs — it has
nothing to do with grids.

## What is refused

- `Shape::grid(_, _, 0, _)` — `LaneError::BadRows`. A workgroup with no invocations on its second
  axis.
- `Kernel::load_row` on a `Shape::new` kernel — `LaneError::NotAGrid`. There is no row.
- A pitch of zero — `LaneError::BadPitch`. Every row would sit on the address of the first, which
  validates, runs, and returns whichever row was written last.

## What is not checked, and cannot be

That `pitch` matches the buffer, and that the dispatch covers the matrix. Both are the caller's,
for the same reason the one-axis addressing leaves buffer sizing to the caller: a `Kernel` binds a
runtime array whose length is whatever the caller supplies, and reading past it is undefined rather
than an error. See `src/kernel/access.rs` for the same division on one axis.

## How to re-check this

```
cargo test -p simdr --test kernels          # the grid modules, through spirv-val
cargo test -p runner --test plane           # the same modules on a device, against a host reference
cargo run -p runner --release --example plane
```

The device tests run each kernel at four shapes — one row per workgroup, several rows per
workgroup, and a row spanning several workgroups — because a wrong address arithmetic agrees with a
right one on any single shape.

## What enforces this

**The type system.** `runner::Grid` has an `x` and a `y` and no `z`, so the dispatch this decision
refuses cannot be written — there is no field to set. The record says as much above, and it is
literally true rather than a figure of speech.

`noha gate` reports it prose-only because its invariants are about imports. The absence of a struct
field is not something that vocabulary can see, and it is stronger than anything it could.
