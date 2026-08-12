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
const NOT_MUTATED: [(&str, &str); 13] = [
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

/// The `src/…` entries listed under `sources:` in `noha.yaml`.
fn sources_in_config() -> BTreeSet<String> {
    let text = fs::read_to_string(root().join("noha.yaml")).expect("noha.yaml is readable");

    text.lines()
        .skip_while(|line| line.trim() != "sources:")
        .skip(1)
        // A comment between entries is not the end of the list, but anything that is neither a
        // comment nor an entry is.
        .take_while(|line| line.starts_with("  - ") || line.trim_start().starts_with('#'))
        .filter_map(|line| line.strip_prefix("  - ").map(str::trim))
        .filter(|entry| entry.ends_with(".rs"))
        .map(str::to_owned)
        .collect()
}

#[test]
fn every_source_file_is_mutated_or_listed_as_deliberately_not() {
    let on_disk = sources_on_disk();
    let configured = sources_in_config();
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
    let configured = sources_in_config();

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
    let configured = sources_in_config();

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
    assert!(sources_in_config().len() >= 30);
    assert!(sources_on_disk().len() >= 30);
}
