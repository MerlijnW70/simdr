# What checks what — the claims, and which of them nothing tests

This repository asserts a great deal in prose: 2 400-odd sentences of the form *every*, *never*,
*the only*, *cannot*, or a number with a unit on it. Its own rule is that a claim nothing checks is
a claim. So this is that rule turned on the documents.

It is an inventory rather than a list of sentences. Enumerating 2 400 assertions would produce
something nobody reads; what is useful is the **class** each belongs to, and whether an instrument
exists that could ever contradict it.

## The part that is checked, and by what

Stated first because the gap is only visible against it.

| what | instrument | reach |
| --- | --- | --- |
| every **branch** in the source | mutation gate | 93 targets, 639 mutants, **100%** |
| every emitted module's **legality** | `spirv-val` | the kernel library at 5 widths, plus 232 generated programs |
| the zero-dependency boundary, the decision records' presence, the fail-closed sites | `noha gate` | 56 + 8 + 93 checks |
| every public operation has a consumer; **every opcode is emitted by something**; every pipeline builder bounds its dispatch; the mutation config matches the tree | `tests/integrity.rs` | 17 tests |
| formatting, lints, **doc links**, the MSRV, the skip counts per width | CI | 5 jobs |
| behaviour | two GPUs and lavapipe | widths 4, 8, 16, 32, 64 |

**The first row is stronger than it looks, and it retires a whole class of question.** A guard nothing
reaches *survives* mutation — flipping its condition changes no observable behaviour. So a gate at
100% is a statement that **every refusal this crate documents is provoked by some test**, and that
every "only if", "unless" and "when the mapping is…" in a doc comment has something behind it. All
13 `LaneError` variants, every width guard, every bound: reached.

That is why the classes below are what they are. What is left unchecked is precisely the claims that
**are not encoded as a branch**.

## 1. Measured numbers — 378 of them, and CI runs none of the sixteen examples

`README.md` carries 55, `notes/FINDINGS.md` 222, `notes/NEXT.md` 101. Every one rests on a manual run
at a moment in time. `.github/workflows/ci.yml` mentions the word *example* zero times.

**It decays immediately, and here is the proof.** The README says the suite is 451 tests in the
emitter and 822 across the workspace. It is **455 and 837** — drifted within the same day the line
was written, by the same hand that wrote it. And that line already carries the scar of the previous
occurrence: *"these were 348 and 740 until somebody counted again"*. Third time.

The class splits three ways and the honest answer differs for each:

* **Counts** — tests, kernels, opcodes, files. Trivially assertable, and the one above has now
  drifted three times. There is no argument for leaving these to prose.
* **Timings and multiples** — `11.2×`, `52×`, `~100 µs`, `376 ns`. CI *cannot* check these and says
  so: a shared runner's wall clock is not evidence, which is why `session.rs` prints its ratio there
  rather than asserting it. What can be checked is that the example which produces the number still
  runs and still prints one. All 16 do today — measured while writing this — and nothing would
  notice if one stopped.
* **Facts about a device** — widths, features, heap sizes. Assertable where a device is present, and
  already are in places.

## 2. Every decision record says of itself that it is unchecked

`noha gate` prints, on every run, eight lines ending `prose-only: recorded, not machine-checked`.
That is honest and it is also too blunt, because several of those decisions **are** structurally
enforced and simply not marked as such:

* **DR-0006** — *a grid has two axes* — is enforced by `Grid` having no `z` field. The record says
  so itself: "that dispatch cannot be written".
* **DR-0002** — *a module is built for a known subgroup width* — is enforced by `LANES` being a const
  generic; a width discovered later cannot reach it.
* **DR-0004** — *a narrow element is one element per lane* — is enforced by there being no packing
  path to take.

The remainder are genuinely prose. **DR-0001** — *the numbers come from the grammar* — could only be
machine-checked against Khronos' grammar JSON, which is not installed on this machine while the tool
that consumes it is. **DR-0008** — *a round trip is the unit of cost* — has a re-runnable check in
`runner/examples/latency.rs`, which is closer to enforced than to prose.

### The marking was tried, and the tool said no — which is the finding

`noha` does have the mechanism. A record may say `status: enforced` with an `invariant:`, and it is
strict about the pairing: *"status `prose-only` forbids an `invariant` — an invariant nobody enforces
is a false promise"*. Four kinds exist: `zero-deps`, `forbid-unsafe`, `sole-use` and `sole-ref`.

**All four operate on the import graph of the audited surface**, and none of the eight decisions is
an import-confinement claim. Marking DR-0007 as `enforced` with a `sole-use` invariant was tried;
the gate accepted the syntax and reported

```text
no audited source imports `require_capability` (the restriction holds vacuously)
```

which is the tool being better than the attempt. A vacuous invariant reads as enforcement and checks
nothing — the precise failure this whole document is about — so the eight stay `prose-only`, and the
blanket turns out to have been accurate.

What was missing was not a status field. It was that a reader could not tell **which** of the eight
had something behind it. Each record now ends with a `## What enforces this` section naming the
artefact and its kind:

| record | what backs it | kind |
| --- | --- | --- |
| DR-0003 — a branch is uniform or refused | `if_uniform` takes a `Uniform`, which only the votes produce | **type** |
| DR-0006 — a grid has two axes | `Grid` has no `z` field | **type** |
| DR-0002 — a known subgroup width | `LANES` is a const generic; `Mapping::of` is the one runtime copy | **type** |
| DR-0004 — one element per lane | no packing path exists to take | **absence** |
| DR-0007 — declares what it needs | `spirv-val`; breaking it leaves 19 of 20 modules rejected | **tested** |
| DR-0008 — a round trip is the unit of cost | `runner/examples/latency.rs`, re-runnable anywhere | **measured** |
| DR-0001 — numbers from the grammar | `spirv-val` catches a wrong number that makes an *invalid* module, and nothing catches one that makes a valid one | **partial** |
| DR-0005 — a constant defers a number | an `Id` is an `Id` | **weakest** |

Three are enforced by the type system and cannot be violated; one by something not existing; one is
tested and was checked by breaking it; one has an instrument and no schedule. **Two are genuinely
thin** — and now they say so where a reader will find it, which is what the blanket could not do.

## 3. Uniqueness and absence — 165 claims, three mechanised

*"The only place that…"*, *"nothing else does…"*, *"no caller wants it"*. These are the claims that
go stale **silently**, because nothing about adding a second copy makes the first one wrong.

Three have been turned into checks, each after it had already failed once: `NO_CONSUMER` (every
public operation is named outside its own file), `NO_DISPATCH` (every pipeline builder bounds what it
dispatches), and `NOT_MUTATED` (every unsafe file is excused by name and still contains unsafe).

The rest are prose, and this class produced two of the three findings the gate turned up this week.
A concrete one still open: **the relationship between a vector's width and the subgroup's is decided
in three places** — `Lanes::mapping` in the emitter, `shape_of` in the fuzzer (which cannot use the
first, because it is a const generic and the fuzzer's width is a runtime value), and a guard in
`kernels/reduce.rs`. Nothing says there are three, so a fourth would be invisible.

Note the asymmetry that makes this class dangerous: a duplicated *branch* has a mutant, so the gate
finds it. A duplicated claim in *prose* has nothing.

## 4. The "done" markers

`notes/NEXT.md` marks some thirty items **done**, each describing behaviour that later work could
undo. Nothing re-reads them. This is the lowest-severity class — the items are mostly narrative —
but it is worth knowing that "done" in that file is a statement about the past.

## 5. Claims about the outside world — the class with the least cover and the highest cost

What SPIR-V permits, and what a driver does with it. Only `spirv-val` and the devices can arbitrate,
and only where a test actually asks them.

**This is where today's worst finding lived.** `Lanes::shift_left` was bounded by `Element`; `F32`
is an `Element`; and a shift of a float built a module the validator rejects. No branch expressed
that claim, so the mutation gate could not see it. No test built such a module, so `spirv-val` was
never asked. It was reachable from safe code with a plausible spelling for as long as it existed.

The sweep that found it covered `src/lanes/`, and established that the three shifts were the only
operations there whose type bound was wider than the instruction allows.

**`src/module/` has now had the same sweep, and it comes out clean — but not for the reason the
first one did.** Every helper at that layer takes an `Id`, so the Rust type system cannot help at
all; the question is instead *where an opcode and a type are chosen independently*. Eighteen raw
`binary`/`unary` calls exist, and each is one of two shapes:

* **derived from `Element`** — `T::ADD`, `T::EQUAL`, `T::GREATER_THAN`, `T::FROM_U32`, the group
  opcodes — where the opcode and the type come from the same trait implementation and cannot
  disagree;
* **fixed to `uint` or `boolean`** — the bitwise lane arithmetic in `shuffle.rs` and `scan.rs`, the
  loop counter's comparison, the bitcasts — where the operand is a lane index or a comparison
  result, never the caller's `T`.

The shifts were the only place a caller-chosen `T` met an opcode SPIR-V restricts. That is now a
type bound, and the sweep is written down here so the next person redoes it rather than repeats it.

**It did leave one gap, and closing it is what the sweep was for.** Bounding the shifts to `Integer`
admits six types; `tests/instructions.rs` validated them at *two*. The four narrow integers — the
ones needing `Int8` or `Int16` declared and a result type whose width the shift must match — had
never been handed to the validator. They are valid, which is the answer rather than the point: the
point is that nothing had asked.

Pulling that thread found the general case. **Every module `spirv-val` had ever seen at 8 or 16 bits
came from `kernels::narrow`**, which reaches seven operations: `add`, `clamp`, `load`, `reduce_sum`,
`splat_bits`, `store`, `store_scalar`. The comparisons, the selects, the extremes, the shuffles, the
votes and the scans all accept a narrow element and were validated at 32 bits and nowhere else —
while narrow is where SPIR-V is fussiest, since the group opcodes differ between the signed and
unsigned forms of one width and `shaderSubgroupExtendedTypes` is a permission no module can declare.

> **Correction, 2026-08-15.** An earlier version of this paragraph said narrow types "reach exactly
> three operations on a device". That is true of `kernels::narrow` and **false of the tree**. The
> differential fuzzer has `Byte`, `UnsignedByte`, `Short` and `UnsignedShort` among its domains and
> dispatches them through `Gpu::run_bytes` and `run_halves` — so its whole vocabulary, nineteen
> operations across all three mappings, runs at 8 and 16 bits for 256 seeds a domain on every
> device. Narrow *execution* was never the gap; narrow **validation** was, and that is what the 65
> modules closed.
>
> The claim was about `kernels::narrow` and got stated about the tree. Which is this document's own
> subject, arriving in this document.

`the_lane_surface_is_valid_for_every_narrow_integer` fills that grid — **65 modules**: five types
across five widths and all three mappings, because a width is not a parameter to these instructions,
it *chooses the instruction sequence*. The same `butterfly` call is one shuffle whole-subgroup, a
masked one clustered and one per strip above that; the same `prefix_sum` is a single instruction, a
Hillis–Steele ladder and a carry between strips.

**The validator earned its place while the test was being written**, twice. `from_lane_value` puts
one id into a *one-strip* vector, so a strip-mined reduction result needs `splat_id` instead — the
lane API refused it by name, `TooManyStrips { strips: 1, limit: 2 }`. And `splat_id::<T>` takes a raw
`Id` and believes the caller about its type, so splatting a **vote's boolean** into a `Vector<T>`
compiles: `spirv-val` rejected it with *"Expected both objects to be of Result Type: Select"*.

That second one is worth keeping as a shape of its own. `splat_id` and `from_lane_value` are the
seam where a type parameter is a **claim rather than a check** — the escape hatch a reduction result
has to come back through — and the validator is the only layer downstream of it.

## What to do about it, worst first

0. **~~The `pub fn` check has a type it does not cover.~~ — done, and it was seven rather than one.**
   `every_public_operation_has_a_consumer_outside_its_own_file` asks it of functions, and an opcode
   is a `pub const`, so the same shape was invisible in the same tree.
   `every_opcode_is_emitted_by_something` asks it now: `F_CONVERT`, `LOGICAL_NOT`,
   `GROUP_NON_UNIFORM_I_MUL` and the four atomic min/max opcodes are declared and emitted by nothing
   — each of them half of an operation nobody had asked for. `notes/FINDINGS.md` records why an
   unemitted number is worse than a missing one: `spirv-val` is what keeps `decisions/DR-0001`
   honest, and it can only check a number that reaches a module.

   **All seven were deleted** rather than left excused, which is the consistent reading of DR-0001.
   The excuse list stays in place at length zero — an exception should be a line somebody writes
   rather than a silence — so the check is now an absolute: each of the **95** opcodes `op.rs`
   declares reaches a module. Reading them out of the grammar again costs a minute on the day
   somebody wants one back, and that is the day it becomes checkable.
1. **Assert the counts.** Trivial, and the one in the README has now been wrong three times. The fix
   is not to bump the number a fourth time — it is to make the test print it.
2. **Sweep `src/module/` for type bounds wider than the instruction allows**, the way `src/lanes/`
   was swept. This is the class that produces invalid modules that run.
3. **Run the examples in CI for liveness**, not for their numbers. One step, and it catches the
   runtime drift that `cargo clippy --all-targets` cannot: it compiles them and never runs them.
4. **Mark the decision records that are enforced**, naming the artefact that enforces each.
5. **Mechanise the mapping-relationship uniqueness** — the one that has now cost three findings.
