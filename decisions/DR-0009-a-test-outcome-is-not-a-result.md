---
id: DR-0009
title: A test outcome is not a Result
status: prose-only
---

## The decision

**A harness that runs generated or swept work reports an `Outcome` with an arm per reason, never a
`Result<bool>` and never a bare pass/fail.** At minimum:

| arm | what it means | whose problem |
| --- | --- | --- |
| `Agreed` / `Ran` | it executed and matched the reference | nobody's |
| `Disagreed` | it executed and did not | the engine's, or the reference's |
| `Refused` | the API declined to build it, by name | nobody's — the refusal working |
| `Unsupported` | the device does not offer what the module declares | nobody's — the device being honest |
| `Invalid` | `spirv-val` rejected it; nothing was dispatched | **the caller's** |
| `Errored` | the driver took a *validated* module and failed | **the device's** |

The last two are the ones a `Result` collapses, and they have different owners. Everything that is
not a failure still has to be **counted and printed**, because coverage lost quietly reads exactly
like coverage held.

## Why this is a decision and not a style note

Because it has already been got wrong twice, in opposite directions, and both cost real time.

**The sandbox, on its first outing.** `proeftuin`'s quantised layer stored an `i32` into a `u32`
buffer — invalid SPIR-V. It had no `Invalid` arm because it did not validate at all, so:

```text
RTX 4080            192 of 192 agreed with the reference
integrated Radeon   192 of 192 agreed
lavapipe            Vulkan(ERROR_UNKNOWN), 192 times, with no indication why
```

Two devices ran an illegal module and returned right-looking numbers; a third refused it and could
not say what was wrong. Nothing in that table points at a store, and no arm of the harness could
have — which is `runner/tests/validated.rs`'s opening paragraph happening inside the sandbox built
to test what that paragraph is about.

**The fuzzer, in the other direction.** `runner::fuzz::Outcome` had this shape first and is where
the pattern comes from: `Agreed`, `Disagreed`, `Refused`, `Unrepresentable` — with the doc on the
last one making the argument explicitly, *"a domain that is refused every round is a domain with no
coverage, and it would otherwise look exactly like a domain that always agreed."* But a dispatch
failure there is a `FuzzError`, which the sweep propagates as a test error. So "this driver cannot
run this module" and "the environment is broken" arrive as the same thing, and the sweep stops
rather than counting.

Two harnesses, one missing the arm that says *my module was wrong* and one missing the arm that says
*your device was*.

## What follows from it

* **Validate before dispatching.** `Invalid` cannot exist as an arm unless something asks, and a
  driver is lenient about exactly what the validator is not. The sandbox reaches the emitter's own
  `tests/common/spirv_val.rs` by `#[path]` rather than copying it.
* **Report the non-failures.** A run that refused everything must be visibly different from one that
  agreed with everything. `proeftuin`'s report ends with *"N of 12 combinations executed here"* for
  that reason, and the tests assert a floor on it — eight of twelve — so a sweep cannot pass by
  proving nothing.
* **A count of agreements is not a result.** The useful number is what did *not* run.

## What enforces this

**Nothing yet, and that is the honest state.** It is a rule two harnesses now follow — one by
construction and one partially — and no check compares a new harness against it. The gate's
invariant vocabulary confines an import graph, which this is not.

What it has instead is a worked failure in each direction, above, and a place to look: `Answer` in
`proeftuin/src/batch.rs` is the fuller of the two, and `runner::fuzz::Outcome` is the one that got
there first.

**Both have moved since this was written, and in the direction the record argues for.** The sandbox
had *three* of these — one per tool, the same four arms with different names — which is the shape
this rule prevents inside a harness and did not prevent between harnesses. They are one generic
`Answer<T>` now, so a fifth reason is one arm rather than three. And `runner::fuzz::Outcome` gained
the arm it was missing from the other end: `ProgramError` splits a refusal into *the width has no
mapping* and *this element type has no such instruction*, which arrived with the first generated
operation that not every domain has.
