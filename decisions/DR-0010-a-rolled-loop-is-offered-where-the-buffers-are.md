---
id: DR-0010
title: A rolled loop is offered where the buffers are, not only where the lanes are
status: prose-only
---

## The Measurement

`src/kernel/mod.rs` builds a rolled loop whose body loads from a buffer and asserts the decoded
module holds one merge, two phis and **one** body — `a_rolled_loop_over_a_kernel_builds_one_body_that_reads_a_buffer`
— and a second test asserts four `OpFAdd` for four carried totals in one body.
`runner/tests/validated.rs` hands `rolled_block_sum` and `rolled_weighted_totals` to `spirv-val` at
the five widths its `WIDTHS` constant names — 4, 8, 16, 32 and 64 — and `runner/tests/unrun.rs`
dispatches both on a device. The block order is exacting: the phis must be first in the header, the
merge second-to-last in its block, and the branch immediately after it, and each of those has been
got wrong here once.

A `Lanes` holds a module and a subgroup width and no bindings — `load_at` and `store_at` hang off
`Kernel` — so a body handed a `Lanes` could compute and could not fetch.

## The Decision

The four-block loop shape is one routine generic over a crate-internal `Emits` trait with one
method, and both `Lanes` and `Kernel<T>` offer `repeat_rolled` over it. What differs is only what
the body is handed. The trait is `pub(crate)` and has one method, so it is not an extension point.

## The Rejected Route

A second hand-written copy on `Kernel` was rejected because the block order above has three
constraints that have each been got wrong once already, and two copies are two chances to get them
wrong and two things to keep in step. A loop whose trip count varies at runtime was rejected as
having no caller: a caller that dispatches one workgroup per strip already carries the runtime part
in its workgroup count, and a per-lane count would diverge, which `DR-0003` refuses.

## The Limit

The emitted module is unchanged by the generic routine and the existing tests assert the lane
version's words instruction for instruction, so what is checked is that the two hosts emit the same
shape — not that either shape is the fastest one. **No timing was taken**: nothing here measures a
rolled loop against an unrolled one at any strip count, so the case for rolling rests on the
caller's buffer bound rather than on a figure. Nothing enforces that the two hosts stay in step,
because there is nothing to enforce — they are one routine.
