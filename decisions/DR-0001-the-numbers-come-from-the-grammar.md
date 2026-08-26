---
id: DR-0001
title: The numbers come from the grammar, not from memory
status: prose-only
---

## The Measurement

`src/module/op.rs` declares 100 opcodes and the eight files of `src/spec/` declare 78 enumerant
values, held to Khronos' `spirv.core.grammar.json` by 18 tests of which 10 are named
`every_*_matches_the_khronos_grammar`. Those numbers were read against grammar **1.6.7** on
2026-08-11, when `module::op` held ten. Writing the multi-pass reduction that day, `OpUDiv` was
written as 152 from memory and Khronos' assembler answered **134**, which is the value
`src/module/op.rs` carries. `OpTypeInt` is 21 and `OpTypeFloat` is 22, one apart in the same table,
and a module that transposes them declares the wrong type and assembles cleanly.

## The Decision

Every opcode and enumerant value is read out of `spirv.core.grammar.json` or out of `spirv-as`
before it is written down, and the grammar is downloaded when it is needed rather than vendored, so
it is external data and not a dependency. `tests/validated.rs` and `tests/control_flow.rs` hand
every structural shape this crate emits to `spirv-val` at `--target-env vulkan1.1`.

## The Rejected Route

Writing a number from recall was rejected at 152 against 134, one guess against one probe on
2026-08-11. Leaving `--target-env` off the validator was rejected at a `GLCompute` entry point
carrying no `LocalSize`, which the universal environment accepts and `vulkan1.1` refuses, because
the requirement is Vulkan's and not SPIR-V's.

## The Limit

The grammar says what an instruction is and not whether a module is valid, so nothing here catches
a missing capability, an id used before it is defined, or a mismatch between operands. A wrong
number that names another legal instruction in the same position validates, runs, and computes
something else; no layer in this repository can tell that from a right one. The same probe read
`BuiltIn SubgroupId` at 40, and that value is **NOT CHECKABLE HERE**: no file under `src/` names
`SubgroupId`, so the tree holds nothing to compare it against.

And **nothing in this repository consults the grammar at test time.** The file is downloaded
when it is needed and never vendored, so the ten tests named for it are pins written by hand
from a reading taken once. Re-running the recipe is procedural, and no check can tell whether
it was done.
