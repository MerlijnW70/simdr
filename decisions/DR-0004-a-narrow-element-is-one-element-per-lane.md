---
id: DR-0004
title: A narrow element is one element per lane, not several packed into one
status: prose-only
---

## The decision

`Simd<i8, 32>` is **32 lanes each holding one `i8`**, exactly as `Simd<i32, 32>` is 32 lanes each
holding one `i32`. The element's width changes the SPIR-V type and the buffer's `ArrayStride`, and
changes nothing else — not the mapping, not the lane count, not which instruction a reduction
reaches.

The alternative was to pack four `i8` into each 32-bit lane and call that `Simd<i8, 128>`. That is
not offered.

## Why

**The win is memory traffic, and packing is not what buys it.** A buffer of `i8` is a quarter the
size of a buffer of `i32` because the *stride* is one byte, which is true whatever the lanes hold.
Declaring `StorageBuffer8BitAccess` and a stride of 1 is the whole of it.

**Packing would be a fourth mapping.** `decisions/DR-0002` already has three — whole subgroup,
clusters, strips — chosen by comparing `N` against the subgroup width. A packed mapping would be
chosen by comparing the *element width* against the lane width, an orthogonal axis, and every rule
in the lane API would then have to say which of the two it meant. `Simd<i8, 128>` would be four
strips under one reading and one packed subgroup under the other.

**Packed arithmetic is not elementwise arithmetic.** Four `i8` in a lane added with one `OpIAdd`
carry between the elements. Getting that right means either masking after every operation — which
is the four instructions per element that the narrow types exist to avoid — or the integer
dot-product extension, which is a different instruction set with its own device feature.

## What it costs, measured

One element per lane leaves throughput on the table, and the measurement says how much.
`runner/examples/narrow.rs`, RTX 4080, a clamp over 16 777 216 elements:

| kernel | per pass | GB/s | against `i32` |
| --- | --- | --- | --- |
| `Simd<i8, 32>` — one element per lane | 127 µs | 264 | 1.67× |
| `Simd<i8, 128>` — four strips | 33 µs | 1016 | 6.45× |
| `Simd<i16, 32>` — one element per lane | 127 µs | 527 | 1.67× |
| `Simd<i16, 64>` — two strips | 65 µs | 1038 | 3.29× |
| `Simd<i32, 32>` | 213 µs | 630 | 1.00× |

An invocation that loads one byte and one that loads one word cost the same, so a byte-per-lane
kernel runs at a quarter of the achievable rate. **Strip mining is what recovers it** — and strip
mining is a mapping this crate already had, expressed in the lane count rather than in the element
type.

So the packed mapping's benefit is available without the packed mapping: ask for `Simd<i8, 128>`
and each lane holds four elements *as four strips*, one instruction each, no masking, no carry
between them, and the same four bytes move per lane. That is the argument this record rests on,
and it is the reason the measurement is in it.

Two honest qualifications. The 6.45× is not a pure bandwidth ratio: at this size the `i32` buffers
are 64 MB each and land in the regime `notes/NEXT.md` records as unsteady past ~50 MB, while the
`i8` ones do not — some of that factor is cache residency rather than bytes. And at 1 048 576
elements every unstripped row takes the same 9 µs whatever its width, because at that size the
dispatch is not bandwidth-bound at all and the narrow types buy nothing.

## Consequences

- `Element::STRIDE` is the only new number in the type table, and `kernel/binding.rs` decorates
  with it instead of the 4 it used to hard-code.
- Six device features gate this and are reported separately by `simdr probe`, because a device may
  hold any subset. For the 8-bit types: `shaderInt8` for the arithmetic and
  `storageBuffer8BitAccess` for the buffer. For the 16-bit ones: `shaderInt16`, `shaderFloat16` and
  `storageBuffer16BitAccess`. And over all of them, `shaderSubgroupExtendedTypes` for the subgroup
  operations. Both devices in this machine offer all six.
- **The third one leaves no trace in the module.** There is no SPIR-V capability for
  `shaderSubgroupExtendedTypes`, so a module reducing over `i8` is byte-identical to one that
  would run anywhere, validates cleanly, and is refused at pipeline creation on a device without
  it. `runner/tests/narrow.rs` is the only layer that can tell.
- A packed mapping remains possible later. It would be a new `Mapping` variant and a new decision
  record, not a reinterpretation of this one.
