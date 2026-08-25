//! The checks that keep this project's own paperwork honest.
//!
//! Everything else in the suite tests the emitter. This tests the things *around* it that claim
//! the emitter is tested — and those had drifted, silently, while reporting green:
//!
//! - `noha.yaml` listed 33 of 38 sources, so five files were never mutated. One of them was
//!   `src/lanes/branch.rs`, the phi and block-tracking code. "100% mutation coverage" was true of
//!   a list that excluded the most dangerous file in the tree.
//! - `decisions/DR-0002` said strip mining "is not built" and named `LaneError::TooWide` as the
//!   error that says so. Strip mining had been built for weeks and that error never existed.
//!   `noha gate` printed a tick beside it, because a decision record is prose and prose is not
//!   checked.
//!
//! Both are the same failure: a hand-maintained list that nothing compares against reality. These
//! tests are that comparison. They are deliberately in the emitter's own suite rather than in a
//! tool's configuration, because a check that lives inside the thing it guards cannot be forgotten
//! when the tool is not run.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The crate root, wherever the test happens to be run from.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The files that exist and are deliberately **not** mutated, each with the reason.
///
/// Every one of them is Vulkan FFI. A mutant there does not compute a wrong number for a test to
/// catch — it passes a wrong handle, frees something twice, or submits a command buffer that was
/// never recorded, and the process dies or the driver does. That is not coverage, it is a crash
/// harness, and `noha`'s job is behavioural coverage.
///
/// The list is here rather than only in `noha.yaml` so that adding an FFI file is a decision
/// somebody writes down. A new file in `runner/src` that is neither mutated nor listed here fails
/// the test below.
const NOT_MUTATED: [(&str, &str); 15] = [
    (
        "runner/src/scan/held.rs",
        "owns buffers and a pipeline per level; its arithmetic lives in scan/plan.rs",
    ),
    (
        "runner/src/dispatch/upload.rs",
        "writes mapped memory; its decision lives in dispatch/step.rs",
    ),
    ("runner/src/buffer.rs", "allocates and maps device memory"),
    ("runner/src/device.rs", "opens the instance and the device"),
    (
        "runner/src/device/narrow.rs",
        "queries physical-device feature chains",
    ),
    (
        "runner/src/lib.rs",
        "destroys the device and instance on drop",
    ),
    (
        "runner/src/dispatch.rs",
        "records and submits command buffers",
    ),
    ("runner/src/dispatch/bindings.rs", "as dispatch.rs"),
    ("runner/src/dispatch/chain.rs", "as dispatch.rs"),
    (
        "runner/src/dispatch/pipeline.rs",
        "creates descriptor sets and pipelines",
    ),
    (
        "runner/src/dispatch/placement.rs",
        "allocates to ask where memory lands",
    ),
    (
        "runner/src/dispatch/session.rs",
        "owns buffers and a pipeline across calls",
    ),
    (
        "runner/src/reduction/held.rs",
        "owns buffers and a chain of pipelines across calls",
    ),
    (
        "runner/src/dispatch/submit.rs",
        "submits and waits on fences",
    ),
    (
        "runner/src/dispatch/timestamps.rs",
        "creates and reads query pools",
    ),
];

/// Public operations with no consumer outside the file that defines them, and why each is allowed.
///
/// **The list this check exists to keep short.** A `pub fn` nothing calls from anywhere else is not
/// dead code — it is *untested* code that reads as dead, and the two are the same thing right up
/// until somebody calls it. This project has been bitten twice:
///
/// - `Lanes::dot_unsigned` emitted `OpUDot` with a **signed** result type for a week. Invalid
///   SPIR-V in a shipped public method with no caller, no unit test of its own and no validator
///   coverage — three layers, and it fell between all of them.
/// - `Module::memory_barrier` emitted an `OpMemoryBarrier` whose semantics Vulkan forbids, and the
///   documentation on `MemorySemantics::None` recommended exactly that mask. Nobody had ever built
///   one, so nobody had ever been told.
///
/// Both were found by asking this question by hand, months apart. Asking it here means it is asked
/// on every run instead of when somebody remembers.
///
/// The entries below are the ones where "nothing calls it" is the right answer, and each says why.
/// [`nothing_is_excused_from_needing_a_consumer_and_has_one`] deletes an excuse that has expired.
const NO_CONSUMER: [(&str, &str); 6] = [
    (
        "require_extension",
        "called by require_capability in the same file, which is the only place that knows a \
         capability needs one; every narrow-type kernel reaches it that way and tests/kernels.rs \
         validates what comes out",
    ),
    (
        "subgroup_f_add",
        "a readable spelling of what the typed path emits through subgroup_reduce; the opcode is \
         validated by every float reduction and the four wrappers are pinned apart by one unit test",
    ),
    ("subgroup_f_max", "as subgroup_f_add"),
    ("subgroup_f_min", "as subgroup_f_add"),
    ("subgroup_i_add", "as subgroup_f_add"),
    (
        "set_f32",
        "tested by a_float_goes_in_as_its_bits in the file that declares it, and it cannot be \
         tested anywhere else: what that test observes is `data`, which is pub(crate), so an \
         integration test has nothing to look at but `len` — and `len` cannot tell a float from \
         an integer. A weaker consumer than another crate's, and not no consumer",
    ),
];

/// Every `.rs` file under `src/` and `runner/src/`, as forward-slashed paths from the root.
fn sources_on_disk() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    walk(&root().join("src"), &mut found);
    walk(&root().join("runner").join("src"), &mut found);
    walk(&root().join("cli").join("src"), &mut found);
    found
}

/// Every `.rs` file in the workspace, tests and examples included.
///
/// Wider than [`sources_on_disk`] on purpose: a public operation reached only by a test *is*
/// reached, and for the emitter a validator test is the most valuable consumer there is. What this
/// must not do is miss a directory — a root left out here would report every operation it consumes
/// as unconsumed, which is a failing direction rather than a silent one.
fn workspace_files() -> BTreeSet<String> {
    let mut found = sources_on_disk();
    for relative in [
        "tests",
        "examples",
        "runner/tests",
        "runner/examples",
        "cli/tests",
    ] {
        walk(&root().join(relative), &mut found);
    }
    found
}

/// Every `pub fn` the workspace declares, as `(name, the file that declares it)`.
///
/// **`runner` as well as `src/` since 2026-08-16, and the widening was the later decision this
/// paragraph used to defer.** It said `runner` is `publish = false` and exists to be consumed by
/// tests, which is true and is not a reason: a public function nobody calls is untested surface
/// whichever crate it sits in, and the crate that dispatches to a device is the one where untested
/// surface reaches a driver.
///
/// It cost three, of a hundred and seventy-four. `Gpu::run_words` was public and named by nothing
/// outside its own file, with `Gpu::run_u32` the public spelling of the same thing — private now.
/// `Gpu::time_specialized` had no caller anywhere, examples included — deleted.
/// `Specialization::set_f32` is excused with its reason above: it *is* tested, in the file that
/// declares it, and cannot be tested anywhere else, because what that test observes is
/// `pub(crate)`.
///
/// **The whole file, tests included, and that is deliberate.** The first version stopped at the
/// first `#[cfg(test)]` on the grounds that a helper inside one is not shipped surface — and a
/// throwaway `pub fn` appended *after* the test module was not flagged, because everything after
/// that marker had become invisible. Every file here puts its tests last, so trimming bought
/// nothing and cost a blind spot exactly where somebody adding code in an unusual place would land.
///
/// The cost of not trimming is that a `pub fn` inside a test module counts as surface. There is one
/// in this tree — `subgroup::test_support::operands_of` — and it is consumed by the sibling files'
/// tests, so it needs no excuse. A future one that is not would ask for a line in [`NO_CONSUMER`],
/// which is a sentence to write rather than a wrong answer.
fn public_functions() -> Vec<(String, String)> {
    let mut found = Vec::new();

    for path in sources_on_disk()
        .iter()
        .filter(|p| p.starts_with("src/") || p.starts_with("runner/src/"))
    {
        let Ok(text) = fs::read_to_string(root().join(path)) else {
            continue;
        };

        for line in text.lines().map(str::trim_start) {
            let Some(rest) = line
                .strip_prefix("pub fn ")
                .or_else(|| line.strip_prefix("pub const fn "))
            else {
                continue;
            };

            // The name ends where the generics or the arguments begin.
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                found.push((name, path.clone()));
            }
        }
    }

    found
}

/// Which files mention each identifier **in code**.
///
/// Tokenised into whole identifiers rather than searched as substrings, which is the difference
/// between `add` being mentioned by `add` and by `padding`.
///
/// Two exclusions, and both are the check being about the right thing:
///
/// - **Comments do not count.** This tree documents its own API heavily and a doc comment naming an
///   operation is prose, not a consumer. Counting it would let a function be "reached" by the
///   sentence explaining that nothing reaches it.
/// - **This file does not count.** [`NO_CONSUMER`] names every operation it excuses, so without
///   this the excuse list would be the consumer that makes each excuse expire — and the check would
///   report itself green for having been written.
fn mentions() -> Vec<(String, BTreeSet<String>)> {
    workspace_files()
        .into_iter()
        .filter(|path| path != "tests/integrity.rs")
        .filter_map(|path| {
            let text = fs::read_to_string(root().join(&path)).ok()?;
            let words = text
                .lines()
                .map(|line| line.trim_start())
                .filter(|line| !line.starts_with("//"))
                .flat_map(|line| line.split(|c: char| !(c.is_alphanumeric() || c == '_')))
                .filter(|word| !word.is_empty())
                .map(str::to_owned)
                .collect();
            Some((path, words))
        })
        .collect()
}

/// Whether anything outside `defined_in` names `function`.
///
/// **A floor rather than a proof, and in the safe direction.** Two files may declare the same
/// method name — `word` is a `pub const fn` on eight `spec` enums — and this cannot tell one from
/// the other, so a reference to any of them counts for all of them. That direction reports a
/// consumer where there may be none, which makes the check weaker and never wrong: it is the same
/// trade `dispatch::extent` makes about a module it cannot read.
fn consumed_outside(
    function: &str,
    defined_in: &str,
    mentions: &[(String, BTreeSet<String>)],
) -> bool {
    mentions
        .iter()
        .any(|(path, words)| path != defined_in && words.contains(function))
}

/// Collect `.rs` paths under `directory`, recursively.
fn walk(directory: &Path, into: &mut BTreeSet<String>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, into);
        } else if path.extension().is_some_and(|extension| extension == "rs")
            && let Ok(relative) = path.strip_prefix(root())
        {
            into.insert(relative.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// The `src/…` entries listed under `sources:` in `noha.yaml`, or `None` if there is no such file.
///
/// **`noha.yaml` is not in the repository and cannot be.** It is the local mutation runner's
/// configuration, and a global ignore excludes it — along with the rest of that toolchain — from
/// every repository on this machine. So a clone has the tests and not the config.
///
/// This used to `expect` the file. Four of the five tests in this binary therefore *panicked* on
/// any clone, including CI and including this machine after a reinstall: the suite was green for a
/// reason that did not travel, which is the exact failure the file's own header is about. A
/// hand-maintained thing that nothing compares against reality — except here the thing was the
/// comparison itself.
///
/// `None` now means "cannot be checked here", and the tests that need it skip loudly rather than
/// passing quietly, the same way `runner`'s harness reports a missing GPU. What can be checked
/// without it is checked unconditionally, and that turned out to be the more interesting half —
/// see [`every_file_with_unsafe_code_in_it_is_excused`].
fn sources_in_config() -> Option<BTreeSet<String>> {
    let text = fs::read_to_string(root().join("noha.yaml")).ok()?;

    Some(
        text.lines()
            .skip_while(|line| line.trim() != "sources:")
            .skip(1)
            // A comment between entries is not the end of the list, but anything that is neither a
            // comment nor an entry is.
            .take_while(|line| line.starts_with("  - ") || line.trim_start().starts_with('#'))
            .filter_map(|line| line.strip_prefix("  - ").map(str::trim))
            .filter(|entry| entry.ends_with(".rs"))
            .map(str::to_owned)
            .collect(),
    )
}

/// The config, or a loud skip — the caller returns when this hands back `None`.
fn config_or_skip(label: &str) -> Option<BTreeSet<String>> {
    let found = sources_in_config();
    if found.is_none() {
        eprintln!("SKIPPED {label}: no noha.yaml here (the mutation runner's config is local)");
    }
    found
}

/// Whether `text` contains an `unsafe` block, function, or impl — as opposed to the *word*.
///
/// A plain search for "unsafe" matches `#![forbid(unsafe_code)]`, the crate docs explaining that
/// Vulkan is FFI and FFI is unsafe, the name of the `unsafe_op_in_unsafe_fn` lint, and every one of
/// the 79 `SAFETY` notes. None of those is unsafe code, and the emitter — which forbids it outright
/// — would match on the attribute that forbids it.
///
/// Comment lines go first and the rest is searched for the three forms that introduce it. Crude,
/// and crude in the safe direction: a false negative here would let an unsafe file through, so the
/// forms are the ones the language actually has rather than a guess at what a file might contain.
fn contains_unsafe_code(text: &str) -> bool {
    text.lines()
        .map(str::trim_start)
        .filter(|line| !line.starts_with("//"))
        .any(|line| {
            line.contains("unsafe {")
                || line.contains("unsafe fn")
                || line.contains("unsafe impl")
                || line.contains("unsafe extern")
        })
}

#[test]
fn every_source_file_is_mutated_or_listed_as_deliberately_not() {
    let on_disk = sources_on_disk();
    let Some(configured) = config_or_skip("every_source_file_is_mutated_or_listed") else {
        return;
    };
    let excused: BTreeSet<String> = NOT_MUTATED
        .iter()
        .map(|&(path, _)| path.to_owned())
        .collect();

    let unaccounted: Vec<&String> = on_disk
        .difference(&configured)
        .filter(|path| !excused.contains(*path))
        .collect();

    assert!(
        unaccounted.is_empty(),
        "these files exist, `noha.yaml` does not list them, and NOT_MUTATED does not excuse \
         them — so no mutant is ever generated for them and the coverage score is over a smaller \
         surface than the code:\n{unaccounted:#?}"
    );
}

#[test]
fn nothing_is_both_mutated_and_excused() {
    // The other way the two lists can disagree. An entry in both means somebody added a file to
    // `noha.yaml` and forgot to take it off the excuse list, and the excuse then reads as true
    // when it is not.
    let Some(configured) = config_or_skip("nothing_is_both_mutated_and_excused") else {
        return;
    };

    let contradictory: Vec<&str> = NOT_MUTATED
        .iter()
        .map(|&(path, _)| path)
        .filter(|path| configured.contains(*path))
        .collect();

    assert!(
        contradictory.is_empty(),
        "listed as not mutated and mutated anyway:\n{contradictory:#?}"
    );
}

#[test]
fn every_excused_file_still_exists_and_still_contains_unsafe() {
    // The excuse is "this is FFI". If a file stops containing `unsafe` it has stopped being FFI,
    // and the reason for excusing it has expired even though the line is still there.
    let on_disk = sources_on_disk();

    for (path, reason) in NOT_MUTATED {
        assert!(
            on_disk.contains(path),
            "{path} is excused from mutation and does not exist ({reason})"
        );

        let text = fs::read_to_string(root().join(path)).expect("readable");
        assert!(
            text.contains("unsafe"),
            "{path} is excused as FFI ({reason}) and contains no `unsafe` any more, \
             so the excuse has expired and it should be mutated"
        );
    }
}

#[test]
fn the_mutation_tool_is_not_pointed_at_files_that_are_gone() {
    let on_disk = sources_on_disk();
    let Some(configured) =
        config_or_skip("the_mutation_tool_is_not_pointed_at_files_that_are_gone")
    else {
        return;
    };

    let missing: Vec<&String> = configured.difference(&on_disk).collect();
    assert!(
        missing.is_empty(),
        "`noha.yaml` lists files that no longer exist:\n{missing:#?}"
    );
}

#[test]
fn the_source_list_is_not_empty_so_this_test_can_fail() {
    // Both tests above pass trivially if the parser returns nothing. This is what says the parser
    // works — the same reason `validated.rs` carries a module the validator must reject.
    //
    // The disk half runs everywhere. It is the one that would catch `walk` silently finding
    // nothing, which would make *every* test in this file vacuous rather than only the ones that
    // read the config.
    assert!(sources_on_disk().len() >= 30);

    let Some(configured) = config_or_skip("the_source_list_is_not_empty") else {
        return;
    };
    assert!(configured.len() >= 30);
}

#[test]
fn every_file_with_unsafe_code_in_it_is_excused() {
    // **The direction that was missing, and the one that runs on a clone.**
    //
    // `every_excused_file_still_exists_and_still_contains_unsafe` checks that each excuse is still
    // deserved. Nothing checked the converse: that a file which *gained* `unsafe` was taken out of
    // the gate. The two failures are not symmetrical. An expired excuse costs coverage; unsafe code
    // inside the gate costs the mutation run itself — a mutant that passes a wrong handle or frees
    // twice kills the process instead of failing a test, and the run reports a crash rather than a
    // survivor.
    //
    // It is also the rule this project keeps applying by hand. `dispatch/step.rs` was split out of
    // `chain.rs`, `reduction/plan.rs` out of `held.rs`, and `step::upload_bytes` out of
    // `dispatch/upload.rs` — every one so that a decision would sit in a file with no `unsafe` in
    // it and therefore inside the gate. Three deliberate splits, and nothing enforcing the shape.
    let excused: BTreeSet<String> = NOT_MUTATED
        .iter()
        .map(|&(path, _)| path.to_owned())
        .collect();

    let unguarded: Vec<String> = sources_on_disk()
        .into_iter()
        .filter(|path| !excused.contains(path))
        .filter(|path| {
            fs::read_to_string(root().join(path)).is_ok_and(|text| contains_unsafe_code(&text))
        })
        .collect();

    assert!(
        unguarded.is_empty(),
        "these files contain unsafe code and are not excused from mutation, so the gate will \
         generate mutants that crash the process rather than fail a test. Either excuse them in \
         NOT_MUTATED with a reason, or — better, and what this project has done three times — \
         split the decision out into a file with no unsafe in it:\n{unguarded:#?}"
    );
}

#[test]
fn the_unsafe_scanner_finds_unsafe_where_there_is_some_and_not_where_there_is_none() {
    // Without this the test above is vacuous: a scanner that always answered `false` would report
    // nothing unguarded and pass for ever. It is the same hole `validated.rs` fills with a module
    // the validator must reject.
    //
    // Both directions, against real files rather than invented strings — the invented ones are
    // what a scanner is accidentally written to match.
    let ffi = fs::read_to_string(root().join("runner/src/dispatch/submit.rs")).expect("readable");
    assert!(
        contains_unsafe_code(&ffi),
        "submit.rs is nothing but unsafe blocks and the scanner cannot see them"
    );

    let emitter = fs::read_to_string(root().join("src/lanes/reduce.rs")).expect("readable");
    assert!(
        !contains_unsafe_code(&emitter),
        "the emitter forbids unsafe, so a hit here is the scanner matching prose"
    );

    // The three shapes that are the word and not the code, each of which appears in this tree.
    assert!(!contains_unsafe_code("#![forbid(unsafe_code)]"));
    assert!(!contains_unsafe_code(
        "//! Vulkan is FFI, and FFI is `unsafe`."
    ));
    assert!(!contains_unsafe_code(
        "    // SAFETY: as above — an unsafe fn forwarding its own"
    ));

    // And the three that are.
    assert!(contains_unsafe_code(
        "        unsafe { device.destroy_fence(fence, None) };"
    ));
    assert!(contains_unsafe_code(
        "pub(crate) unsafe fn destroy(self, gpu: &Gpu) {"
    ));
    assert!(contains_unsafe_code("unsafe impl Send for Handle {}"));
}

#[test]
fn every_public_operation_has_a_consumer_outside_its_own_file() {
    // **The audit that found `OpUDot`, asked on every run instead of when somebody remembers.**
    //
    // It has been done by hand twice, months apart, and found something both times — the second
    // time an `OpMemoryBarrier` whose semantics Vulkan forbids, recommended by this crate's own
    // documentation. What a unit test beside a function establishes is that the emitter agrees with
    // its author about the word stream. Whether that stream is *legal*, and whether the operation
    // does what its name says, are questions only a consumer asks — a kernel, a validator run, a
    // device test.
    //
    // So an operation nothing reaches is not covered by six layers. It is covered by none of them,
    // and it looks identical in the counts to one that is covered by all six.
    let mentions = mentions();
    let excused: BTreeSet<&str> = NO_CONSUMER.iter().map(|&(name, _)| name).collect();

    let unreached: Vec<String> = public_functions()
        .into_iter()
        .filter(|(name, _)| !excused.contains(name.as_str()))
        .filter(|(name, path)| !consumed_outside(name, path, &mentions))
        .map(|(name, path)| format!("{path}: {name}"))
        .collect();

    assert!(
        unreached.is_empty(),
        "these public operations are named by nothing outside the file that declares them, so no \
         kernel, no validator run and no device test reaches any of them — which is the state \
         `Lanes::dot_unsigned` was in while it emitted invalid SPIR-V. Give each one a consumer, \
         or add it to NO_CONSUMER with the reason it does not need one:\n{unreached:#?}"
    );
}

#[test]
fn nothing_is_excused_from_needing_a_consumer_and_has_one() {
    // The other direction, and the one that keeps the list honest rather than growing. An excuse
    // whose operation has since gained a caller is a line that reads as true and is not — the same
    // failure `every_excused_file_still_exists_and_still_contains_unsafe` catches for the FFI list.
    let mentions = mentions();
    let declared = public_functions();

    for (name, reason) in NO_CONSUMER {
        let Some((_, path)) = declared.iter().find(|(declared, _)| declared == name) else {
            panic!("NO_CONSUMER excuses `{name}` ({reason}) and no `pub fn` by that name exists");
        };

        assert!(
            !consumed_outside(name, path, &mentions),
            "`{name}` is excused from needing a consumer ({reason}) and something outside \
             {path} now names it, so the excuse has expired and the line should go"
        );
    }
}

#[test]
fn the_consumer_scanner_finds_one_where_there_is_one_and_not_where_there_is_none() {
    // Without this the check above is vacuous in the worst way: a scanner that answered "consumed"
    // for everything would report nothing unreached and pass for ever, exactly as a validator that
    // never returns `Err` would. Same hole `validated.rs` fills with a module that must be
    // rejected, and `the_unsafe_scanner_finds_unsafe_where_there_is_some` fills for the FFI list.
    //
    // Against real declarations rather than invented strings, for the reason that test gives: the
    // invented ones are what a scanner is accidentally written to match.
    let mentions = mentions();
    let declared = public_functions();

    let find = |wanted: &str| {
        declared
            .iter()
            .find(|(name, _)| name == wanted)
            .map(|(name, path)| consumed_outside(name, path, &mentions))
    };

    assert_eq!(
        find("reduce_sum"),
        Some(true),
        "`Lanes::reduce_sum` is reached by kernels, tests and the fuzzer, and the scanner cannot \
         see any of them"
    );
    assert_eq!(
        find("subgroup_f_min"),
        Some(false),
        "`Module::subgroup_f_min` is named by nothing outside its own file — if that has changed, \
         take it out of NO_CONSUMER; if it has not, the scanner is finding consumers that are not \
         there"
    );

    // And the surface is being read at all. A parser that returned nothing would make both
    // assertions above unreachable and every other check here vacuous.
    assert!(
        declared.len() >= 150,
        "only {} public functions found in src/, which is fewer than this crate has — the \
         declaration parser has stopped matching",
        declared.len()
    );
}

#[test]
fn the_emitter_forbids_unsafe_outright_so_none_of_it_needs_excusing() {
    // Why every entry in NOT_MUTATED is a `runner/` path and none is a `src/` one. The emitter's
    // half of the coverage claim rests on this attribute, and an attribute is one line somebody can
    // delete — so it is asserted rather than assumed, and asserting it costs nothing.
    let lib = fs::read_to_string(root().join("src").join("lib.rs")).expect("the emitter has a lib");

    assert!(
        lib.contains("#![forbid(unsafe_code)]"),
        "the emitter no longer forbids unsafe, so `unsafe` can appear anywhere in `src/` — and \
         every file there is inside the mutation gate, which is the arrangement the test above \
         exists to prevent"
    );
}

/// Files that build a pipeline and dispatch nothing through it.
///
/// Every other one owes a bound: see the test below for why the pipeline is the family being asked.
const NO_DISPATCH: [(&str, &str); 1] = [(
    "runner/src/dispatch/pipeline.rs",
    "`probe_pipelines` builds pipelines to time creation and destroys them without submitting, \
     so there is no dispatch to bound",
)];

/// Whether any *code* line of `path` contains `needle`.
///
/// Comment lines are skipped for the reason [`mentions`] skips them: a bound named only in prose is
/// a claim about the file rather than a check inside it — and this file's own prose names both of
/// the things it searches for.
fn code_names(path: &str, needle: &str) -> bool {
    let Ok(text) = fs::read_to_string(root().join(path)) else {
        return false;
    };
    text.lines()
        .map(str::trim_start)
        .filter(|line| !line.starts_with("//"))
        .any(|line| line.contains(needle))
}

/// Every source file that builds a `Pipeline`, and whether it also bounds a dispatch.
///
/// The **call** rather than the two words apart. A set of words would match any file that imports
/// `Pipeline` and calls some other `new`, which is a false positive that demands a bound of a file
/// that dispatches nothing — and the excuse list is exactly where a false positive would be parked
/// and forgotten.
fn pipeline_builders() -> Vec<(String, bool)> {
    sources_on_disk()
        .into_iter()
        .filter(|path| code_names(path, "Pipeline::new"))
        .map(|path| {
            let bounds = code_names(&path, ".overrun(") || code_names(&path, ".overrun_uniform(");
            (path, bounds)
        })
        .collect()
}

#[test]
fn every_file_that_builds_a_pipeline_bounds_what_it_dispatches() {
    // **The half of the "where is this called from" audit that had no mechanism.**
    // `notes/NEXT.md` records the shape: `Gpu`'s dispatch family had six members and one bound
    // check, and nothing would have said so, because all six were consumed and each was reached by
    // its own tests. What was missing was a notion of *family* — a set of operations that owe the
    // same check.
    //
    // `Pipeline::new` is one, and it is the one that matters: every dispatch in this crate goes
    // through it, and it is handed both halves of the question at once — the module, and how much
    // of each buffer the shader may see. So a file that builds a pipeline and never mentions a
    // bound is a door that was added without one, which is exactly the state `run_bound`,
    // `Session::dispatch`, `run_chain`, the reducer and the scanner were all in.
    //
    // A floor rather than a proof, in the direction that costs coverage rather than truth: it asks
    // whether the file names the check, not whether every path through it reaches one.
    // `runner/tests/bounds.rs` asks the sharper question of each door by dispatching past its
    // buffers, and this is what notices a *seventh* door appearing.
    let excused: BTreeSet<&str> = NO_DISPATCH.iter().map(|&(path, _)| path).collect();

    let unbounded: Vec<String> = pipeline_builders()
        .into_iter()
        .filter(|(path, bounds)| !bounds && !excused.contains(path.as_str()))
        .map(|(path, _)| path)
        .collect();

    assert!(
        unbounded.is_empty(),
        "these files build a compute pipeline and never mention `overrun`, so whatever they \
         dispatch is unbounded — which is undefined behaviour on a buffer, and has shown up here \
         as an access violation on one device and plausible wrong numbers on another. Bound the \
         dispatch, or add the file to NO_DISPATCH with the reason it submits nothing:\n{unbounded:#?}"
    );
}

#[test]
fn nothing_is_excused_from_bounding_a_dispatch_and_bounds_one() {
    // The other direction, as everywhere else here: an excuse that has stopped being true reads
    // exactly like one that is.
    let builders = pipeline_builders();

    for (path, reason) in NO_DISPATCH {
        let Some((_, bounds)) = builders.iter().find(|(built, _)| built == path) else {
            panic!("NO_DISPATCH excuses {path} ({reason}) and nothing there builds a pipeline");
        };

        assert!(
            !bounds,
            "{path} is excused from bounding a dispatch ({reason}) and now names `overrun`, so \
             the excuse has expired and the line should go"
        );
    }
}

#[test]
fn the_pipeline_scanner_finds_the_doors_that_are_there() {
    // Without this the check above is vacuous in the worst way: a scanner matching nothing reports
    // every file as compliant and reads as a clean run. The six doors are named here so that a
    // rename which makes the scanner blind fails rather than passes.
    let builders = pipeline_builders();
    let found: BTreeSet<&str> = builders.iter().map(|(path, _)| path.as_str()).collect();

    for door in [
        "runner/src/dispatch.rs",
        "runner/src/dispatch/bindings.rs",
        "runner/src/dispatch/chain.rs",
        "runner/src/dispatch/session.rs",
        "runner/src/reduction/held.rs",
        "runner/src/scan/held.rs",
    ] {
        assert!(
            found.contains(door),
            "{door} dispatches and the scanner did not see it build a pipeline, so this check is \
             passing over the thing it exists to watch"
        );
    }

    // And that the two halves are told apart rather than both answered yes.
    assert!(
        builders
            .iter()
            .any(|(path, bounds)| path == "runner/src/dispatch/pipeline.rs" && !bounds),
        "the excused file reads as bounded, so the scanner cannot tell a bound from its absence"
    );
}

/// Opcodes declared in `src/module/op.rs` that nothing emits.
///
/// **It is empty, and that is the state to keep it in.** Seven entries lived here for about an hour:
/// `F_CONVERT`, `LOGICAL_NOT`, `GROUP_NON_UNIFORM_I_MUL` and the four atomic minimum and maximum
/// opcodes, each of them half of an operation nobody had asked for. They were deleted rather than
/// excused, so every number this file holds now reaches a module.
///
/// **`decisions/DR-0001` is why that was the right way round.** The rule is that every opcode was
/// read out of Khronos' grammar rather than remembered, and what keeps it true is that a wrong
/// number produces a module `spirv-val` rejects. A number nothing emits is a copy of the grammar
/// with **no check behind it**: it can be wrong for as long as it exists, and whoever reaches for it
/// first inherits the mistake along with the convenience. Deleting costs a doc comment and a minute
/// of `spirv-as` on the day somebody wants it back — which is the day it becomes checkable.
///
/// The list stays because an exception should be a line somebody writes rather than a silence, and
/// because [`nothing_is_excused_from_being_emitted_and_is`] expires one the moment its opcode gains
/// an emitter.
const NO_EMITTER: [(&str, &str); 0] = [];

/// Where the opcode numbers live.
const OPCODES: &str = "src/module/op.rs";

/// Every `pub const NAME: u16` that file declares.
fn declared_opcodes() -> Vec<String> {
    let Ok(text) = fs::read_to_string(root().join(OPCODES)) else {
        return Vec::new();
    };

    text.lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("pub const ")?;
            let name = rest.split(':').next()?.trim();
            (!name.is_empty()).then(|| name.to_owned())
        })
        .collect()
}

#[test]
fn every_opcode_is_emitted_by_something() {
    // **The kind the consumer audit could not see.** Its sibling asks whether every `pub fn` is
    // named outside the file that declares it, and found an `OpMemoryBarrier` whose semantics
    // Vulkan forbids sitting there with no caller. An opcode is a `pub const`, so the same shape in
    // the same tree was invisible to it — and there were **seven**, found by a sandbox looking for
    // something else.
    //
    // A dead opcode is not merely unused. `spirv-val` is what keeps `decisions/DR-0001`'s promise
    // honest, and it can only check a number that reaches a module.
    let mentions = mentions();
    let excused: BTreeSet<&str> = NO_EMITTER.iter().map(|&(name, _)| name).collect();

    let unemitted: Vec<String> = declared_opcodes()
        .into_iter()
        .filter(|name| !excused.contains(name.as_str()))
        .filter(|name| !consumed_outside(name, OPCODES, &mentions))
        .collect();

    assert!(
        unemitted.is_empty(),
        "these opcodes are declared in {OPCODES} and emitted by nothing, so no module contains \
         them and `spirv-val` has never checked the number. Emit them, or delete them and read the \
         number out of the grammar again when it is wanted:\n{unemitted:#?}"
    );
}

#[test]
fn nothing_is_excused_from_being_emitted_and_is() {
    // The other direction, as everywhere here: an excuse whose opcode has since gained an emitter
    // is a line that reads as true and is not.
    let mentions = mentions();
    let declared = declared_opcodes();

    for (name, reason) in NO_EMITTER {
        assert!(
            declared.iter().any(|held| held == name),
            "NO_EMITTER excuses `{name}` ({reason}) and {OPCODES} declares no such opcode"
        );
        assert!(
            !consumed_outside(name, OPCODES, &mentions),
            "`{name}` is excused from being emitted ({reason}) and something now names it, so the \
             excuse has expired and the line should go"
        );
    }
}

#[test]
fn the_opcode_scanner_finds_the_numbers_that_are_there() {
    // Without this the check above is vacuous in the worst way: a scanner that parsed nothing would
    // report every opcode as emitted and read as a clean run.
    let declared = declared_opcodes();

    assert!(
        declared.len() > 90,
        "only {} opcodes parsed out of {OPCODES}, so the scanner is reading the wrong shape",
        declared.len()
    );
    for expected in ["I_ADD", "GROUP_NON_UNIFORM_I_ADD", "BITCAST", "S_CONVERT"] {
        assert!(
            declared.iter().any(|name| name == expected),
            "{expected} is declared and the scanner did not see it"
        );
    }

    // And that it tells an emitted opcode from an unemitted one, rather than answering the same
    // either way. The negative used to be `F_CONVERT`, which was dead; now that nothing here is,
    // it has to be a name no opcode has — which is the shape a *new* dead one would arrive in.
    let mentions = mentions();
    assert!(consumed_outside("I_ADD", OPCODES, &mentions));
    assert!(!consumed_outside(
        "OP_NO_SUCH_INSTRUCTION",
        OPCODES,
        &mentions
    ));
}

/// Comparisons of a lane count against a subgroup width, and why each is not a copy of the rule.
///
/// **This is the check `notes/NEXT.md` asked for and the class `notes/FINDINGS.md` named.** A
/// relationship decided in two places is not two spellings of one rule. It is two rules that agree
/// on the inputs anybody draws and diverge on the ones nobody does — and this project has paid for
/// that exact shape three times:
///
/// | where | how it was written | how it diverged |
/// | --- | --- | --- |
/// | `interpret::strips_of` | `if lanes > subgroup` | both arms give one at equal widths, so nothing could tell them apart |
/// | `fuzz::generate` | `lanes < subgroup` | called a whole-subgroup vector clustered, deleting the only `OpPhi` finish's coverage |
/// | `kernels::reduce` | `LANES > subgroup` | refused a cluster exactly the subgroup's width, which is a whole-subgroup vector |
///
/// All three are gone, routed through `Mapping::of`. What no check asserted until now is the
/// absence of a **fourth**, and the reason that matters is the asymmetry recorded with the class: a
/// duplicated *branch* has a mutant, so `noha` finds it — both live ones were found that way — but
/// the coverage guarding the original cannot see the copy, and a copy that never diverges on a
/// power of two is invisible until somebody draws seven lanes on an eight-wide subgroup.
///
/// `Mapping::of` decides by **divisibility**. Every one of the three copies decided by
/// **comparison**, which is why they agreed with it for months. So what this looks for is a
/// comparison, and every entry below has to say why it is not a decision.
///
/// **A floor rather than a proof**, in the direction that costs coverage rather than truth: it
/// matches the two spellings this codebase uses for those quantities, so a copy written `n > w`
/// walks past it. Sharpening that means resolving what a binding *means*, which is a type checker
/// rather than a grep. What it does buy is that the fourth copy cannot arrive in the vocabulary the
/// first three arrived in.
const LANE_WIDTH_COMPARISONS: [(&str, &str, &str); 2] = [
    (
        "src/lanes/mapping.rs",
        "lanes == subgroup",
        "the rule itself. `Mapping::of`'s first arm, and the one comparison here that is allowed \
         to be a decision — everything else in this table exists to point at it",
    ),
    (
        "runner/src/fuzz/generate.rs",
        "lanes > subgroup",
        "an assertion *about* `Mapping::of`, not a second copy of it: the test asks the rule for a \
         mapping and then checks that the only refusal it produces is the one a wide vector can \
         produce. It reads the rule's answer rather than recomputing it, which is the distinction \
         this whole check is drawing",
    ),
];

/// The comparison operators a copy of the mapping rule has ever been written with.
///
/// `==` is here beside the orderings because the rule's *first* arm is an equality — a copy that
/// says `lanes == subgroup` somewhere else is deciding `WholeSubgroup` for itself.
const COMPARISONS: [&str; 5] = ["<=", ">=", "==", "<", ">"];

/// Rust source with its line comments and string literals removed.
///
/// Both have to go, and for opposite reasons. The doc comments on `Mapping::of` and on the two
/// files above *quote* the copies they replaced — `lanes < subgroup`, `LANES > subgroup` — because
/// that history is the argument for the rule existing. A check that read them would fire on its own
/// documentation. And `generate/coverage.rs` carries `lanes == subgroup` inside a panic message,
/// which is prose that happens to live in quotes.
///
/// Block comments are not handled. Nothing in this tree uses them, and a stripper that got them
/// half right would be a second rule about what a comment is.
fn code_only(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut previous = '\0';

    for character in line.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            previous = character;
            continue;
        }
        if character == '"' {
            in_string = true;
            previous = character;
            continue;
        }
        if character == '/' && previous == '/' {
            out.pop();
            break;
        }
        out.push(character);
        previous = character;
    }

    out
}

/// The last identifier in `text`, ignoring trailing space and punctuation.
fn last_word(text: &str) -> String {
    text.trim_end()
        .rsplit(|character: char| !character.is_alphanumeric() && character != '_')
        .next()
        .unwrap_or_default()
        .to_owned()
}

/// The first identifier in `text`, ignoring leading space.
fn first_word(text: &str) -> String {
    text.trim_start()
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .next()
        .unwrap_or_default()
        .to_owned()
}

/// Every lane-against-width comparison in the workspace, as `(file, the comparison)`.
fn lane_width_comparisons() -> Vec<(String, String)> {
    const LANES: [&str; 2] = ["lanes", "LANES"];
    const WIDTHS: [&str; 3] = ["subgroup", "width", "WIDTH"];

    let mut found = Vec::new();
    for path in workspace_files() {
        // This file is skipped: the table above quotes the very comparisons it excuses, so a check
        // that read itself would report its own excuses as findings.
        if path == "tests/integrity.rs" {
            continue;
        }
        let Ok(text) = fs::read_to_string(root().join(&path)) else {
            continue;
        };

        for line in text.lines() {
            let code = code_only(line);
            for operator in COMPARISONS {
                for (index, _) in code.match_indices(operator) {
                    let (left, rest) = code.split_at(index);
                    let right = &rest[operator.len()..];
                    // `<=` and `>=` contain `<` and `>`, so a bare-operator match whose next
                    // character is `=` is the same comparison counted a second time.
                    if operator.len() == 1 && right.starts_with('=') {
                        continue;
                    }

                    let left_word = last_word(left);
                    let right_word = first_word(right);
                    let lane_then_width = LANES.contains(&left_word.as_str())
                        && WIDTHS.contains(&right_word.as_str());
                    let width_then_lane = WIDTHS.contains(&left_word.as_str())
                        && LANES.contains(&right_word.as_str());

                    if lane_then_width || width_then_lane {
                        found.push((path.clone(), format!("{left_word} {operator} {right_word}")));
                    }
                }
            }
        }
    }
    found
}

#[test]
fn no_second_copy_of_the_mapping_rule() {
    for (path, comparison) in lane_width_comparisons() {
        let excused = LANE_WIDTH_COMPARISONS
            .iter()
            .any(|(file, text, _)| *file == path && *text == comparison);

        assert!(
            excused,
            "{path} decides `{comparison}` for itself.\n\n\
             `simdr::lanes::Mapping::of(lanes, subgroup)` is where that relationship is decided, \
             and it decides it by divisibility rather than by comparison — so this line and the \
             rule agree on every power of two and disagree on, say, seven lanes over eight. Three \
             copies of exactly this shape have been found and removed; the table on \
             LANE_WIDTH_COMPARISONS says what each one cost.\n\n\
             Call `Mapping::of` and match on what it returns. If this really is an assertion about \
             the rule rather than a copy of it, add it to that table with the reason."
        );
    }
}

#[test]
fn every_excused_lane_width_comparison_still_exists() {
    // The direction that keeps the table honest rather than growing, the same way
    // `nothing_is_excused_from_needing_a_consumer_and_has_one` does for the other list. An excuse
    // whose line has since been rewritten reads as a live exception and guards nothing.
    let found = lane_width_comparisons();

    for (path, comparison, reason) in LANE_WIDTH_COMPARISONS {
        assert!(
            found
                .iter()
                .any(|(file, text)| file == path && text == comparison),
            "{path} no longer contains `{comparison}`, so this excuse has expired and should be \
             deleted: {reason}"
        );
    }
}

#[test]
fn the_scanner_can_tell_a_comparison_from_a_comment_about_one() {
    // The blind spot this check would have had, asserted rather than assumed. The first two lines
    // are what it must find; the next three are its own documentation and a panic message, which it
    // must not.
    assert_eq!(
        code_only("        if lanes == subgroup {").trim(),
        "if lanes == subgroup {"
    );
    assert_eq!(
        code_only("    let clustered = lanes < subgroup;").trim(),
        "let clustered = lanes < subgroup;"
    );

    assert_eq!(
        code_only("/// `lanes < subgroup`, and `kernels::reduce` with `LANES > subgroup`").trim(),
        ""
    );
    assert_eq!(
        code_only("        // No `lanes < subgroup` here, though that is what this arm means.")
            .trim(),
        ""
    );
    assert_eq!(
        code_only("        \"the pool for `lanes == subgroup` is empty\",").trim(),
        ","
    );

    // A comparison with a comment after it: the code survives, the comment does not.
    assert_eq!(
        code_only("    if lanes > subgroup { // a wide vector").trim(),
        "if lanes > subgroup {"
    );

    // And the scanner reads what the stripper leaves. `>=` must not also be counted as `>`.
    let doubled = COMPARISONS
        .iter()
        .filter(|operator| "lanes >= subgroup".contains(**operator))
        .count();
    assert_eq!(
        doubled, 2,
        "`>=` contains `>`, which the scanner has to allow for"
    );
}
