---
id: DR-0001
title: The numbers come from the grammar, not from memory
status: prose-only
---

## The decision

Every opcode and enumerant value in this crate is read out of Khronos'
`spirv.core.grammar.json` before it is written down. Not from the specification's prose, not from
another project's header, and never from recall.

## Why

This crate's output is a binary format with no redundancy in it. A wrong opcode does not fail to
assemble — it assembles into a different, well-formed instruction. `OpTypeInt` is 21 and
`OpTypeFloat` is 22; transpose them and the module is valid SPIR-V that declares the wrong type,
and the first symptom is a kernel returning plausible nonsense some layers away.

There is no compiler and no type system between us and that mistake. The grammar file is the only
thing that is.

## The recipe

The grammar is **external data and not a dependency** — it is downloaded when it is needed and
never vendored into the tree, for the same reason a conformance corpus is not a crate.

```powershell
$url = "https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/main/include/spirv/unified1/spirv.core.grammar.json"
Invoke-WebRequest -Uri $url -OutFile grammar.json -UseBasicParsing
$g = Get-Content grammar.json -Raw | ConvertFrom-Json
"$($g.major_version).$($g.minor_version).$($g.revision)"          # the version these were read at

# an opcode
($g.instructions | Where-Object { $_.opname -eq 'OpTypeVector' }).opcode

# an enumerant
($g.operand_kinds | Where-Object { $_.kind -eq 'Capability' }).enumerants |
    Where-Object { $_.enumerant -eq 'GroupNonUniformArithmetic' } | Select-Object value
```

Checked against grammar **1.6.7** on 2026-08-11: all ten opcodes then in `module::op`, and every
value in `spec.rs`.

## The second form of the recipe

There is not always a grammar file to hand. Khronos' **assembler** is as authoritative as the JSON
and is already installed beside the validator, so an instruction can be assembled and read back:

```powershell
# %d = OpUDiv %uint %v %sc, in a module small enough to assemble
& spirv-as.exe --target-env vulkan1.1 probe.spvasm -o probe.spv
# then walk the words: (word >> 16) is the count, (word & 0xFFFF) is the opcode
```

This was not a hypothetical. Writing the multi-pass reduction on 2026-08-11 I wrote `OpUDiv = 152`
from memory; the probe said **134**. The guess was wrong, it would have assembled into a different
well-formed instruction, and nothing downstream would have said so.

The same probe confirmed `BuiltIn SubgroupId = 40`, which then went unused — the reduction found a
shape needing no subgroup identity. Both numbers came from the tool rather than from recall, which
is the whole of this record.

## What this does not give us

The grammar says what an instruction *is*, not whether a module is *valid*. Nothing here catches a
missing capability declaration, an id used before it is defined, or a type mismatch between
operands. That needs `spirv-val`, and **it is installed and running**: `tests/validated.rs` and
`tests/control_flow.rs` hand it every structural shape this crate emits, at
`--target-env vulkan1.1`.

Naming the environment is not optional, and finding that out cost a wrong assumption. Left off,
`spirv-val` checks the *universal* environment, which happily accepted a `GLCompute` entry point
with no `LocalSize` — a requirement that is Vulkan's rather than SPIR-V's. A validator run against
the wrong environment is a validator that agrees with you.

Grammar-checked numbers plus a green test suite is still a strictly weaker claim than a validated
module, and a validated module is weaker than one that computed the right answer on a device.
`runner/` is the layer that closes the last of it.

## Consequences

- `spec.rs` carries one test asserting every value it defines. That test is a *pin*, not a
  verification: it stops a value drifting, and it would happily pin a wrong one. Re-run the recipe
  when adding to it.
- Adding an opcode to `module::op` means running the recipe for that opcode. "It is obviously 43"
  is exactly the reasoning this record refuses.

## What enforces this

**A validator, partly, and nothing else.** A wrong opcode number assembles into a *different
instruction*, and `spirv-val` rejects most of those — which is how `OpUDot` with a signed result
type was caught on the first run against it. What it cannot catch is a wrong number that happens to
name another legal instruction in the same position: that module validates, runs, and computes
something else.

So this is **prose with a partial backstop**, and the rule that makes it hold is procedural — read
the number out of `spirv-as` or the grammar, never from memory. `noha gate` is right to call the
record unchecked.
