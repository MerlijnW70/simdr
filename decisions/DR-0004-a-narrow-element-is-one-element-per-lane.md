---
id: DR-0004
title: A narrow element is one element per lane, not several packed into one
status: prose-only
---

## The Measurement

`runner/examples/narrow.rs` on an RTX 4080 at subgroup 32, a clamp over 16 777 216 elements on
2026-08-26: `Simd<i8, 32>` at 126 µs and 265.7 GB/s, 1.68× the `i32` kernel; `Simd<i8, 128>` — four
strips — at 33 µs and 1023.3 GB/s, **6.46×**; `Simd<i16, 32>` at 127 µs and 530.2 GB/s, 1.67×;
`Simd<i16, 64>` at 66 µs and 1012.3 GB/s, 3.20×; `Simd<i32, 32>` at 212 µs and 633.5 GB/s. Over
1 048 576 elements the three unstripped kernels take **9 µs each** whatever the element width, and
the stripped ones 3 µs and 5 µs. `runner/examples/dot.rs` on the same run puts `OpSDot` against the
written-out form at 1.01× and 1.20× on the RTX 4080 for one and thirty-two products per element at
262 144 invocations, and at 1.50× and **9.17×** on the integrated Radeon at subgroup 64. Both
devices report `integerDotProduct4x8BitPackedSignedAccelerated`, and `simdr probe` reports all six
narrow features on both.

## The Decision

`Simd<i8, 32>` is 32 lanes each holding one `i8`. The element width changes the SPIR-V type and the
buffer's `ArrayStride` through `Element::STRIDE`, and changes nothing else — not the mapping, not
the lane count, not which instruction a reduction reaches. Four `i8` per lane is not offered, and
`OpSDot` reading a `u32` as four bytes is an operation on an operand rather than a fourth mapping.

## The Rejected Route

Packing four `i8` into a lane was rejected at 33 µs against 126, because `Simd<i8, 128>` reaches the
same four bytes per lane as four strips at one instruction each, with no masking and no carry
between elements — a mapping this crate already had. It was rejected a second time on shape: the
three mappings are chosen by `N` against the subgroup width, and a packed one would be chosen by
element width against lane width, so `Simd<i8, 128>` would name two different things.

## The Limit

The 6.46× is not a bandwidth ratio alone: at 16 777 216 elements the `i32` buffers are 64 MB each
and the `i8` ones are not, so some of the factor is residency rather than bytes, and no run
separated the two. At 1 048 576 elements the dispatch is not bandwidth-bound and the narrow types
buy nothing on the unstripped rows. `shaderSubgroupExtendedTypes` leaves no trace in the module, so
a module reducing over `i8` validates and is refused at pipeline creation on a device lacking it;
only `runner/tests/narrow.rs` can see that, and it was not run on a device that lacks it because
neither device here does.
