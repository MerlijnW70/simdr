//! The decision records, checked against the code they describe.
//!
//! A decision record is prose and most of it cannot be checked. What *can* be is that when it
//! names a Rust path, that path still exists — and that half had rotted: `DR-0002` said strip
//! mining "is not built" and named an error to prove it, when strip mining had been built for
//! weeks and the error had never existed under that name. `noha gate` printed a tick beside the
//! file throughout, because its decision check reads the front matter and not the claims.
//!
//! # Backticks are the claim
//!
//! Code spelled as code asserts this crate defines it, and is checked here. A dead name being
//! discussed in prose is not — which is what lets a retraction name what it retracts without the
//! check mistaking the obituary for a promise. `DR-0002` relies on exactly that.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The crate root, wherever the test happens to be run from.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.rs` file under the three source trees.
fn sources_on_disk() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for tree in [
        root().join("src"),
        root().join("runner").join("src"),
        root().join("cli").join("src"),
    ] {
        walk(&tree, &mut found);
    }
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

/// Every `Thing::member` written inside backticks in the decision records.
///
/// Deliberately narrow. A decision record is prose and most of it cannot be checked; what *can* be
/// checked is that when it names a Rust path, that path still exists. `OpTypeInt` has no `::` and
/// is not a claim about this crate, so it is not matched.
fn paths_named_in_decisions() -> BTreeSet<String> {
    let mut named = BTreeSet::new();
    let Ok(entries) = fs::read_dir(root().join("decisions")) else {
        return named;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|extension| extension == "md") {
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            for token in backticked(&text) {
                if let Some(last) = rust_path_tail(&token) {
                    named.insert(last);
                }
            }
        }
    }
    named
}

/// The contents of every single-backtick span in `text`.
fn backticked(text: &str) -> Vec<String> {
    // Split on backticks: odd-numbered pieces are the spans between them. Fenced blocks make some
    // of those spans large and multi-line, which the path test below simply will not match.
    text.split('`')
        .skip(1)
        .step_by(2)
        .map(str::to_owned)
        .collect()
}

/// The final segment of `token`, if `token` is a `::`-separated Rust path.
///
/// Returns `None` for anything else — plain words, prose, code blocks, `spirv-val`.
fn rust_path_tail(token: &str) -> Option<String> {
    // Strip a call or generic suffix so `Lanes::new()` and `Simd::<T>` still resolve.
    let cleaned: String = token
        .chars()
        .take_while(|character| {
            character.is_alphanumeric() || *character == '_' || *character == ':'
        })
        .collect();

    let segments: Vec<&str> = cleaned.split("::").collect();
    if segments.len() < 2 {
        return None;
    }

    let last = segments.last()?;
    let plausible = !last.is_empty()
        && last
            .chars()
            .all(|character| character.is_alphanumeric() || character == '_')
        && last
            .chars()
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic() || first == '_');

    plausible.then(|| (*last).to_owned())
}

/// Every `.rs` file under `src/`, concatenated.
fn all_source_text() -> String {
    sources_on_disk()
        .iter()
        .filter_map(|relative| fs::read_to_string(root().join(relative)).ok())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn a_decision_record_does_not_name_something_that_no_longer_exists() {
    // What would have caught `LaneError::TooWide` on the day it was renamed: the record said the
    // error existed, `noha gate` ticked it, and nothing looked.
    let text = all_source_text();
    let orphans: Vec<String> = paths_named_in_decisions()
        .into_iter()
        // Paths into the standard library or another crate are not this crate's to keep alive.
        .filter(|name| !is_word_in(&text, name))
        .collect();

    assert!(
        orphans.is_empty(),
        "the decision records name these, and nothing in `src/` defines them any more. Either the \
         record is stale or the code lost something it promised:\n{orphans:#?}"
    );
}

/// Whether `needle` appears in `haystack` as a whole word.
fn is_word_in(haystack: &str, needle: &str) -> bool {
    let boundary = |character: char| !character.is_alphanumeric() && character != '_';

    haystack.match_indices(needle).any(|(at, _)| {
        let before = haystack[..at].chars().next_back().is_none_or(boundary);
        let after = haystack[at + needle.len()..]
            .chars()
            .next()
            .is_none_or(boundary);
        before && after
    })
}

#[test]
fn the_decision_records_name_enough_to_be_worth_checking() {
    // The teeth test again: if the extractor found nothing, the check above passes on an empty
    // set and proves the parser ran rather than that the records are true.
    let named = paths_named_in_decisions();
    assert!(
        named.len() >= 5,
        "only {} path(s) extracted from the decision records, which is too few for the check \
         above to mean anything: {named:#?}",
        named.len()
    );
}

#[test]
fn the_word_match_respects_boundaries() {
    // `is_word_in` is what decides whether a record is stale, so its own edges are pinned here
    // rather than trusted.
    assert!(is_word_in("enum LaneError { NoMapping }", "NoMapping"));
    assert!(!is_word_in(
        "enum LaneError { NoMappingAtAll }",
        "NoMapping"
    ));
    assert!(!is_word_in("let x = my_NoMapping;", "NoMapping"));
    assert!(is_word_in("Mapping::Strips { size }", "Strips"));
    assert!(!is_word_in("", "Strips"));
}

#[test]
fn the_path_extractor_takes_paths_and_leaves_prose() {
    assert_eq!(
        rust_path_tail("LaneError::TooWide").as_deref(),
        Some("TooWide")
    );
    assert_eq!(rust_path_tail("Lanes::new()").as_deref(), Some("new"));
    assert_eq!(
        rust_path_tail("crate::lanes::Lanes").as_deref(),
        Some("Lanes")
    );

    assert_eq!(rust_path_tail("OpTypeInt"), None, "no path separator");
    assert_eq!(rust_path_tail("spirv-val"), None, "not an identifier");
    assert_eq!(rust_path_tail("spirv.core.grammar.json"), None);
}
