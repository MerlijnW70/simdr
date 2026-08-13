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

/// Every `.rs` file under `src/` and `runner/src/`, as forward-slashed paths from the root.
fn sources_on_disk() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    walk(&root().join("src"), &mut found);
    walk(&root().join("runner").join("src"), &mut found);
    walk(&root().join("cli").join("src"), &mut found);
    found
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
