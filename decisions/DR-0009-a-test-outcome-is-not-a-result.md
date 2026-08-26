---
id: DR-0009
title: A test outcome is not a Result
status: prose-only
---

## The Measurement

`runner::fuzz::Outcome` carries four arms — `Agreed`, `Disagreed`, `Refused`, `Unrepresentable` —
and `ProgramError` splits a refusal three ways: `Lanes`, `NotInThisDomain` and `ShiftTooFar`. A
dispatch failure is a separate `FuzzError`, whose two arms are `Run` and `ShortInput`, and the sweep
propagates it as a test error rather than counting it.

A quantised layer in the sandbox that has since been deleted stored an `i32` into a `u32` buffer —
invalid SPIR-V — and reported 192 of 192 agreements on an RTX 4080, 192 of 192 on the integrated
Radeon, and `Vulkan(ERROR_UNKNOWN)` 192 times on lavapipe with no indication why. Two devices ran an
illegal module and returned right-looking numbers; the third refused it and could not say what was
wrong. That harness had no arm for *the module was invalid* because it never validated.

## The Decision

A harness that runs generated or swept work reports an outcome with an arm per reason, never a
`Result<bool>` and never a bare pass/fail, and validates before dispatching so that *the caller's
module was wrong* and *the device failed on a valid module* cannot arrive as one thing. Everything
that is not a failure is counted and printed, because coverage lost quietly reads exactly like
coverage held.

## The Rejected Route

Reporting a sweep as `Result<bool>` was rejected at 192 of 192 agreed on two devices over a module
`spirv-val` would have refused — a run in which every number was right and the module was illegal.
Propagating a device failure as an error was rejected at the sweep stopping instead of counting,
which is what `FuzzError` still does.

## The Limit

`Outcome` holds four of the six arms this record argues for: there is no `Invalid` and no `Errored`,
so the two failures that have different owners are still not separated in the type — the rule is
stated here and followed in part. **Nothing checks it.** No test compares a new harness against
this shape, and the gate's vocabulary confines an import graph rather than a type's arms. The
sandbox that supplied half the evidence above is deleted, so its table cannot be re-taken and the
192s are quoted from `notes/FINDINGS.md` rather than re-measured.
