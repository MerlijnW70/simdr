# proeftuin — a sandbox for tools that test the engine

A place to build workloads that put the engine under pressure, without any of them becoming part of
what the engine claims about itself.

## The contract

**Nothing in the engine may depend on anything here.** The arrow points one way, as it does between
`runner` and `simdr`, and for the same reason: a test tool that the thing under test relies on is
not a test tool.

The isolation is structural rather than promised:

| what | why it cannot see this directory |
| --- | --- |
| `cargo test --workspace` at the root | `exclude = ["proeftuin"]` in `../Cargo.toml`, and an empty `[workspace]` here making this its own root |
| CI | builds `-p simdr`, `-p runner` and the root workspace; none of them reaches an excluded directory |
| `tests/integrity.rs` | scans `src/`, `runner/src/` and `cli/src/` by name |
| `noha.yaml` and the mutation gate | list source files explicitly; nothing here is listed |
| `noha gate`'s zero-dependency boundary | audits `src/` |

So this directory may take dependencies, may be messy, and may be thrown away, and none of that
touches the emitter's zero-dependency claim or the suite's numbers.

## Deleting it

```bash
rm -r proeftuin
# then remove the `exclude = ["proeftuin"]` line from ../Cargo.toml
```

That is the whole of it. `grep -rn proeftuin --include='*.rs' --include='*.toml' ..` outside this
directory returns the one `exclude` line, which is the proof rather than the claim.

## What it is for, and why these workloads

A workload is only a test if something can **disagree with it**. Everything this repository has ever
found came from two things disagreeing — a device against a CPU reference, a module against
`spirv-val`, a mutant against a test. So the sandbox is not for demonstrations; it is for workloads
that carry an exact independent oracle.

That rules more out than it sounds. A fluid simulation is float chaos with no cheap exact reference;
a procedural world has no reference at all beyond looking right, and would spend its time fighting
`decisions/DR-0003`, which refuses per-lane branches on purpose. **Quantised integer arithmetic has
an exact reference**, which is why the first tool here is a neural-network layer.

### The gap it aims at, measured

`notes/CLAIMS.md` ends with the class nothing covers: claims about the outside world. Two measured
holes sit inside it.

* **The packed dot products are fuzzed by nothing.** `dot_signed`, `dot_unsigned`, `dot_mixed` and
  `dot_signed_saturating` are absent from the fuzzer's vocabulary, and every kernel that uses one is
  built through `whole_subgroup!` — so they had only ever run as **whole-subgroup vectors**, never
  clustered and never strip-mined. `OpUDot` is the instruction that shipped **invalid**: correct on
  two devices for weeks, caught by the first `spirv-val` run against it. Being valid now says
  nothing about being right, and being right at one mapping says nothing about the other two.
* **The four differ only where it hides.** `OpSDot` and `OpUDot` agree on every byte with its top
  bit clear; `OpSUDot` agrees with both wherever the weights happen to be positive; and
  `OpSDotAccSat` differs from `OpSDot` *only at the overflow*. A corpus of small values proves one
  instruction and reads as proving four.

An earlier version of this section claimed narrow types "reach exactly three operations on a
device". That is true of `kernels::narrow` and **false of the tree** — the fuzzer has `Byte`,
`UnsignedByte`, `Short` and `UnsignedShort` among its domains and dispatches them through
`Gpu::run_bytes` and `run_halves`, so its whole vocabulary runs at 8 and 16 bits. Narrow *execution*
was never the gap. `notes/CLAIMS.md` carries the correction.

A quantised layer is both at once: `u8` activations, `i8` weights, four of them packed to a word,
accumulated in `i32`. Its answer is an integer, so the reference is exact rather than approximate,
and disagreement is a finding rather than a rounding question.

## What the first outing found, which was about the sandbox

The layer stored an `i32` — what a packed dot answers with — into a `u32` buffer. Invalid SPIR-V, and
mine rather than the engine's: `kernels::dot` does the `reinterpret` this had left out.

What it cost is the point:

| device | what it said |
| --- | --- |
| RTX 4080 | **192 of 192 agreed** with the reference |
| integrated Radeon | **192 of 192 agreed** |
| lavapipe | `Vulkan(ERROR_UNKNOWN)`, 192 times, with no indication why |

Two devices ran an illegal module and produced the right numbers. The third refused it and could not
say what was wrong. Nothing in that table points at the store, and the tool had no layer that could —
because it dispatched without validating.

That is `runner/tests/validated.rs`'s opening paragraph happening again, in the sandbox built to test
the thing that paragraph is about: *"drivers are lenient about things the validator is not."*

**So the rule here is the engine's rule.** `check` runs `spirv-val` before it runs anything, using
the emitter's own harness by path rather than a copy, and `Outcome` distinguishes four things a tool
that only counted agreements would have merged:

* `Refused` — the lane API said no. The mapping working.
* `Unsupported` — the device does not offer what the module declares. The device being honest.
* `Invalid` — `spirv-val` rejected it. **This crate's mistake**, and nothing was dispatched.
* `Errored` — the driver took a *validated* module and failed. The device's mistake, and the only
  one of the four worth reporting upstream.

With the `reinterpret` in place all four configurations agree: 192 runs each at subgroup 4, 16, 32
and 64, across all three mappings.

## The second tool: the conversions

`Lanes::convert_u32::<T>` is one method and **five instructions** — `T::FROM_U32` names
`OpCopyObject`, `OpBitcast`, `OpSConvert`, `OpUConvert` or `OpConvertUToF` depending on the target.
`src/lanes/narrow.rs` says why two of those are worth separating: *"`OpUConvert` requires a result
type whose signedness is 0 and `OpSConvert` does not… That is the kind of asymmetry that assembles
cleanly when it is wrong."* None of the three conversions is in the fuzzer's vocabulary.

The probes are **boundaries rather than samples**, because every distinction here lives at one and
nowhere else: `OpSConvert` and `OpUConvert` agree below 128, a bitcast and a numeric conversion agree
below `i32::MAX`, and sign extension only shows where the truncated top bit is set. Twelve values
into six integer targets, 72 conversions, on four devices — all agreeing with a reference written
from the **opcode table** rather than from the method's documentation.

That distinction is the point, and it found something. The method is documented as *"a `u32` value's
number, as a value of `T`"*, and for `i32` the opcode is `OpBitcast` — so `0xFFFF_FFFF` converts to
−1 rather than to 4 294 967 295. The two readings are identical for every value below `i32::MAX`,
which is every loop counter, which is what the method exists for. A reference written from the
sentence would have agreed with the implementation about everything except the answer.

Not a bug: there is no `i32` holding that number, so a bitcast is a defensible choice. It is a
sentence that was true of the inputs anybody draws, which is the shape `notes/FINDINGS.md` catalogues
under *a relationship decided twice* — here between prose and an opcode table. `src/lanes/mod.rs`
carries the table now, and a qualifier on the first sentence.
