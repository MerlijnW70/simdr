# Contributing to simdr

This is a SPIR-V emitter and a Vulkan runner for it. The emitter compiles `Simd<T, N>` semantics
down onto subgroup lanes; the runner puts the result on a device. Almost every rule below exists
because something was wrong once and the check is what keeps it fixed.

## The four commands

```sh
cargo fmt --all
cargo clippy --workspace --all-targets      # zero warnings
cargo test --workspace
cargo test --test integrity --test documented
```

CI runs all of them plus a device job on lavapipe at subgroup widths 4, 8 and 16, and an MSRV job
against the version `Cargo.toml` claims. The emitter job needs no GPU; the device job does.

A local `cargo fmt` hook is worth installing, and it is deliberately not committed:

```sh
printf '#!/bin/sh\ncargo fmt --all --check\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The commit history contains one titled *"`cargo fmt` is the first gate and I did not run it"*, and
two commits after it landed unformatted code again. CI catches it, but a gate that fires after a
push teaches nobody anything at the moment they could act on it.

## The emitter takes no dependencies and contains no `unsafe`

`simdr`'s `[dependencies]` is empty and stays empty. `runner` carries `ash` and the Vulkan calls,
and that boundary is what makes the emitter testable without hardware.

`unsafe` is forbidden in the emitter outright — `the_emitter_forbids_unsafe_outright_so_none_of_it_needs_excusing`
asserts it. In the runner, every file containing `unsafe` must be excused in `NOT_MUTATED` with a
reason, and that excuse **expires**: if the `unsafe` goes, the file must return to the mutation
surface. A separate test checks that too.

## `spirv-val` is the oracle, not the unit tests

A unit test decodes a module and agrees it says what the emitter meant. That is not the same as the
module being legal. The audit that started `tests/instructions.rs` found `dot_unsigned` emitting
`OpUDot` with a signed result type — invalid SPIR-V, in a shipped public method, with a unit test
and a device test and no validator.

So: an instruction family that no validated test reaches is not covered. Build the smallest module
that reaches it and hand it to `expect_valid`.

## Every public function needs a consumer outside its own file

`every_public_operation_has_a_consumer_outside_its_own_file` enforces it. A `pub fn` nobody calls is
either dead or missing a test, and the check found an `OpMemoryBarrier` whose semantics Vulkan
forbids sitting there with no caller.

The same rule applies to opcodes. An opcode declared in `src/module/op.rs` and emitted by nothing is
a number `spirv-val` has never checked — either emit it from the lane API, or delete it and read it
out of the grammar again on the day somebody wants it.

## Every number in a document is counted, not typed

The counters in `README.md` carry markers like `<!--count:opcodes-->`, and `tests/documented.rs`
asserts each against the tree. Add a lane operation and the count moves; the suite tells you which
line to change. Do not adjust a number by hand without re-running it.

## Mutation coverage is the gate that a passing suite cannot be

A green suite says a test ran, not that it would fail if the code were wrong. `noha prober` mutates
the code and reports what nothing noticed.

There are three surfaces, because they have different costs:

| surface | kill command | workers |
| --- | --- | --- |
| emitter, the files under `src/` | `cargo test -p simdr` | eight — nothing reaches a card |
| runner, the files under `runner/src/` | the device suite | one, always |
| the command-line binary | its own | — |

**One worker for the runner is not a suggestion.** Two device suites against one card manufacture
kills that belong to the GPU rather than to the mutant, and on at least one host that combination
took a bugcheck.

The three `noha.yaml` files are excluded by many contributors' global gitignore and so are not in
the repository. They are recorded verbatim in the commit that split them, and
`tests/integrity.rs` skips its checks rather than failing when they are absent.

The emitter surface stands at zero survivors. Keep it there: a change that adds a survivor is a
change whose behaviour nothing checks.

When a survivor will not die, read the code before writing another test. Four survivors here were
all the same shape — two operators agreeing on every value the suite happened to pass, `& 0xff`
against `| 0xff`, `>> 24` against `<< 24`. And a survivor that *cannot* be killed usually means the
code is redundant, so deleting it is the fix.

## Comments

`.attest.toml` sets `comments = "strict"`. The source carries no narrative: what the code does is
stated by tests, and why it is that way is stated in the commit message. What survives a strict
file is `// SAFETY:` and the whole argument under it, lint directives, and fenced blocks in doc
comments — `cargo test` compiles and runs those, so they are tests.

If you have written a paragraph explaining a function, it belongs in the commit that introduces it.
Read a few messages here first: they say what was found, what it cost, and what was refused.

## Naming

Tests are named as specifications rather than as labels — `a_rotate_wraps_inside_the_vector_and_leaves_no_lane_undefined`,
not `test_rotate`. A reader scanning the suite reads names and nothing else.

Two instructions that agree on small numbers are named apart, and neither is a default:
`shift_right_logical` and `shift_right_arithmetic` agree on every value whose top bit is clear, and
that is exactly the shape of mistake this crate keeps finding.

## Reporting a bug

The most useful report is a module the emitter produces that `spirv-val` rejects, or a kernel whose
result differs between subgroup widths. Include the width, the driver, and the smallest program
that shows it.

## Licence

By contributing you agree that your work is licensed under MIT or Apache-2.0, at your option.
