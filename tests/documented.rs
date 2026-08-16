//! What this repository's prose says about itself, checked against the repository.
//!
//! Three claims, and only the first needed a marker invented for it: the **numbers** the documents
//! state, the **files** they name, and the **members** they name. The last two need nothing new,
//! because this tree already quotes what it means — a file, a type, an instruction — and leaves
//! ordinary prose unquoted. **The backticks were already the markup**; nothing read them.
//!
//! `notes/CLAIMS.md` opens by counting 378 measured numbers across the documents and observing that
//! nothing checks any of them. Most of that class genuinely cannot be checked here — a shared
//! runner's wall clock is not evidence, which is why `session.rs` prints its ratio under CI rather
//! than asserting it. But one part of it can be, and it is the part that has actually drifted:
//!
//! > The README says the suite is 451 tests in the emitter and 822 across the workspace. It is 455
//! > and 837 — drifted within the same day the line was written, by the same hand that wrote it.
//!
//! Third occurrence, which is where a rule stops being a lapse and starts being a design fault. The
//! README's answer was to stop writing the number down. That works for one sentence and does not
//! scale: a document that may not state a number cannot describe the thing it documents.
//!
//! So the number stays in the prose and gains a marker, and this file resolves the marker against
//! the tree:
//!
//! ```text
//! `op.rs` declares <!--count:opcodes-->95 numbers
//! ```
//!
//! The comment renders as nothing, the sentence reads as it did, and the digits after it are now a
//! claim with an instrument behind it. Bumping the number by hand is still possible and now fails.
//!
//! **Three rules, and the third is the one that took a deletion to learn.** A marked number must be
//! right; a marker must name a counter that exists, so a typo fails rather than passing quietly; and
//! every counter must be stated by at least one document. That last is the opcode rule one level up
//! — a number nothing emits is a copy of the grammar with no check behind it, and a counter no
//! document states is a check with nothing to check.
//!
//! **What is deliberately not here.** `noha`'s 93 targets and 639 mutants, every timing, every
//! multiple. Those are a tool's output at a moment, not a property of the tree, and the two that
//! could be derived from `noha.yaml` would need a second copy of its parser — the duplication this
//! suite exists to catch.
//!
//! # The files and the members, which needed no marker at all
//!
//! A path in backticks must name a file that exists, by suffix — the prose says `scan/plan.rs` and
//! means `runner/src/scan/plan.rs`, which is a habit worth keeping rather than three hundred
//! sentences to rewrite. A `Type::member` in backticks must name something this tree declares, where
//! the type is one this tree declares; `f32::MAX` and `Vec::with_capacity` are nobody's business
//! here and are skipped because `f32` and `Vec` are not declared in it.
//!
//! Both are floors rather than proofs, in the safe direction. Two files may share a tail and this
//! cannot tell them apart; a member is checked against every name the tree declares rather than
//! against its own type's. That direction reports a reference as good where a reader might land one
//! file over, which makes the check weaker and never wrong — the trade `consumed_outside` makes in
//! `tests/integrity.rs`, for the reason it gives there.
//!
//! **The member check is the one this repository was owed.** `decisions/DR-0002` said strip mining
//! "is not built" and named `LaneError::TooWide` as the error that says so. Strip mining had been
//! built for weeks and that error was never written — and `noha gate` printed a tick beside the
//! record, because a decision record is prose and prose is not checked. It is now.
//!
//! What both checks needed was one honest exception apiece, because prose legitimately names things
//! that are not here: a file in the sibling project, and four members that were **deleted** — two of
//! them in the same sentence that records the deletion. [`NOT_IN_THE_TREE`] and [`GONE`] hold those
//! by name with a reason, and an expiry test fails if an excused name comes back.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The crate root, wherever the test happens to be run from.
///
/// Deliberately a second copy of `tests/integrity.rs`'s: it is one expression of a compiler macro,
/// and the alternative — a `tests/common/` module — would put shared state between two suites that
/// are meant to be readable alone.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// What each counter counts, and where the true number comes from.
///
/// `None` means the input is not present and the count is unknowable here rather than zero — the
/// same distinction `config_or_skip` makes in `tests/integrity.rs`, and for the same reason: a
/// missing file reported as `0` would fail a document that is telling the truth.
type Counter = (&'static str, fn() -> Option<usize>);

/// Every number a document may state about this repository.
///
/// Adding a row is half a change. The other half is a sentence somewhere that states it, without
/// which [`every_counter_is_stated_by_some_document`] fails — because a counter nothing reads is the
/// shape the seven dead opcodes had.
const COUNTERS: [Counter; 10] = [
    ("opcodes", || count_lines("src/module/op.rs", "pub const ")),
    ("lane-operations", || Some(lane_operations())),
    ("test-functions", || Some(test_functions())),
    ("decisions", || Some(decision_records().len())),
    ("integrity-tests", || {
        count_lines("tests/integrity.rs", "#[test]")
    }),
    ("documented-tests", || {
        count_lines("tests/documented.rs", "#[test]")
    }),
    ("ci-jobs", ci_jobs),
    ("fuzz-operations", fuzz_operations),
    ("examples", || Some(examples().len())),
    ("device-examples", || {
        Some(
            examples()
                .iter()
                .filter(|path| path.starts_with("runner/"))
                .count(),
        )
    }),
];

/// How many lines of `path` begin with `prefix`, ignoring indentation.
fn count_lines(path: &str, prefix: &str) -> Option<usize> {
    let text = fs::read_to_string(root().join(path)).ok()?;
    Some(
        text.lines()
            .filter(|line| line.trim_start().starts_with(prefix))
            .count(),
    )
}

/// How many public functions the lane API declares.
///
/// The surface `notes/NEXT.md` measures the fuzzer's vocabulary against. Counted rather than
/// written down because that comparison is the one this project keeps returning to, and a stale
/// denominator makes a growing numerator look like progress.
fn lane_operations() -> usize {
    let mut found = 0;
    walk(&root().join("src").join("lanes"), &mut |path, _| {
        if let Ok(text) = fs::read_to_string(path) {
            found += text
                .lines()
                .filter(|line| {
                    let bare = line.trim_start();
                    line.starts_with("    ")
                        && (bare.starts_with("pub fn ") || bare.starts_with("pub const fn "))
                })
                .count();
        }
    });
    found
}

/// The decision records, by file name.
fn decision_records() -> BTreeSet<String> {
    let Ok(entries) = fs::read_dir(root().join("decisions")) else {
        return BTreeSet::new();
    };

    entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            (name.starts_with("DR-") && name.ends_with(".md")).then_some(name)
        })
        .collect()
}

/// The jobs CI runs, which is not the number of runs — the device job is a matrix over three widths.
///
/// Counted from each file's `jobs:` line down so that the `push:` and `pull_request:` under `on:`
/// are not mistaken for two more; both sit at the same indentation and only their position tells
/// them apart.
///
/// **Across every workflow rather than out of `ci.yml`.** This read one file until the scheduled
/// fuzz sweep arrived in a second one, which would have been a job nothing counted — the blind spot
/// this whole file exists to close, in the check that closes it.
fn ci_jobs() -> Option<usize> {
    let mut found = 0;
    let mut any = false;

    walk(&root().join(".github").join("workflows"), &mut |path, _| {
        let Ok(text) = fs::read_to_string(path) else {
            return;
        };
        any = true;
        found += text
            .lines()
            .skip_while(|line| line.trim_end() != "jobs:")
            .filter(|line| {
                let Some(rest) = line.strip_prefix("  ") else {
                    return false;
                };
                let Some(name) = rest.strip_suffix(':') else {
                    return false;
                };
                !name.is_empty()
                    && name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            })
            .count();
    });

    any.then_some(found)
}

/// How many operations the fuzzer's generator can draw.
///
/// Read from `EVERY_KIND`'s declared length, which the compiler checks against its entries and
/// which a test beside it holds equal to the union of the pools. So this is a number with two
/// checks already behind it, and stating it is the third — `notes/NEXT.md` measures the fuzzer's
/// reach against the lane API's surface, and a stale numerator makes a growing one look like
/// progress in the wrong direction.
fn fuzz_operations() -> Option<usize> {
    let text = fs::read_to_string(root().join("runner/src/fuzz/generate/vocabulary.rs")).ok()?;
    let rest = text.split("const EVERY_KIND: [Kind; ").nth(1)?;
    rest.chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// The directories this workspace is made of.
///
/// **A list of what the workspace *is*, rather than the whole tree minus whatever should not be in
/// it.** The two counters below used to walk everything and skip a named sandbox, and the name in
/// the skip was itself a reference to a directory meant to be deletable. A positive list needs no
/// such name: a scratch crate that appears beside `src/` has its own `Cargo.toml`, is excluded from
/// this workspace, and cannot move a number these documents state — whatever it is called and
/// however long it stays.
///
/// The reference checks are the other way round on purpose and say so: they read every file in the
/// tree, because a sentence about this repository rots wherever it is written.
const WORKSPACE: [&str; 5] = ["src", "tests", "examples", "runner", "cli"];

/// Run `visit` over every file of the workspace's own directories.
fn in_the_workspace(visit: &mut impl FnMut(&Path, &str)) {
    for directory in WORKSPACE {
        walk(&root().join(directory), visit);
    }
}

/// Every example in the workspace, found by looking for directories named `examples` rather than by
/// listing the two that exist — a third crate's would otherwise be uncounted and nothing would say.
fn examples() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    in_the_workspace(&mut |path, relative| {
        if path.extension().is_some_and(|extension| extension == "rs")
            && path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|name| name == "examples")
        {
            found.insert(relative.to_owned());
        }
    });
    found
}

/// How many `#[test]` functions the workspace has written down.
///
/// **Not the number `cargo test --workspace` prints**, and the difference is the reason this counter
/// can exist at all. That number moves with the machine: doctests are counted separately, a device
/// test skips where there is no device, and a `cfg` can take a module out. This one is a property of
/// the source, so it is the same on every machine and it fails when it changes.
///
/// It walks [`WORKSPACE`] rather than the tree, which is what keeps it stable while a throwaway
/// crate comes and goes beside it — see `notes/FINDINGS.md` for the deletion that taught the
/// difference.
///
/// The README stopped writing its test count after the third drift. This is the count it can write
/// again — a narrower quantity, named exactly, with an instrument.
fn test_functions() -> usize {
    let mut found = 0;
    in_the_workspace(&mut |path, _| {
        if path.extension().is_some_and(|extension| extension == "rs")
            && let Ok(text) = fs::read_to_string(path)
        {
            found += text
                .lines()
                .filter(|line| line.trim_start() == "#[test]")
                .count();
        }
    });
    found
}

/// Every markdown file in the tree, so a marker cannot be placed somewhere unread.
///
/// The rules below are *per marker found*, which is what let a deletable directory carry markers
/// without owing anything: removing it took its markers with it and left no hole.
fn documents() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    walk(&root(), &mut |path, relative| {
        if path.extension().is_some_and(|extension| extension == "md") {
            found.insert(relative.to_owned());
        }
    });
    found
}

/// Visit every file under `directory`, skipping build output and git's own storage.
fn walk(directory: &Path, visit: &mut impl FnMut(&Path, &str)) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();

        if path.is_dir() {
            if name != "target" && name != ".git" {
                walk(&path, visit);
            }
        } else if let Ok(relative) = path.strip_prefix(root()) {
            visit(&path, &relative.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// The opening of a marker. Split so that this file's own explanation of the syntax is not a marker.
const OPENS: &str = "<!--count:";

/// Every marked number in `text`, as `(counter, the number stated, the line it is on)`.
///
/// A marker whose digits are missing is reported as `None` rather than skipped, because
/// `<!--count:opcodes-->` followed by prose is a sentence that has lost its number — which is the
/// failure this file is about, arriving in its most convincing disguise.
///
/// Digits may carry the thin spaces this repository writes large numbers with (`65 536`), which are
/// stripped before parsing. That is why the scan cannot be a `split` on the closing marker.
fn marks(text: &str) -> Vec<(String, Option<usize>, usize)> {
    let mut found = Vec::new();

    for (number, line) in text.lines().enumerate() {
        let mut rest = line;
        while let Some(at) = rest.find(OPENS) {
            let after = &rest[at + OPENS.len()..];
            let Some(close) = after.find("-->") else {
                break;
            };

            let counter = after[..close].trim().to_owned();
            let digits: String = after[close + 3..]
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == ' ' || *c == '\u{a0}')
                .filter(char::is_ascii_digit)
                .collect();

            found.push((counter, digits.parse().ok(), number + 1));
            rest = &after[close + 3..];
        }
    }

    found
}

/// The counter of that name, if this file declares one.
fn counter(name: &str) -> Option<Counter> {
    COUNTERS.iter().copied().find(|(known, _)| *known == name)
}

#[test]
fn every_marked_number_is_the_number_that_is_there() {
    let mut wrong = Vec::new();

    for path in documents() {
        let Ok(text) = fs::read_to_string(root().join(&path)) else {
            continue;
        };

        for (name, stated, line) in marks(&text) {
            let Some((_, count)) = counter(&name) else {
                continue; // Named by the test below, which is the one that should say so.
            };
            let Some(truth) = count() else {
                eprintln!("SKIPPED {path}:{line} — nothing to count `{name}` from");
                continue;
            };

            match stated {
                Some(number) if number == truth => {}
                Some(number) => wrong.push(format!(
                    "{path}:{line} says {name} is {number}, it is {truth}"
                )),
                None => wrong.push(format!("{path}:{line} marks {name} and states no number")),
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "the documents state numbers the tree does not have:\n  {}\n\n\
         The marker is there so this fails instead of being noticed a week later. Fix the prose \
         unless the count itself is the surprise — in which case the tree changed and nobody said.",
        wrong.join("\n  ")
    );
}

#[test]
fn every_marker_names_a_counter_that_exists() {
    let known: Vec<&str> = COUNTERS.iter().map(|(name, _)| *name).collect();
    let mut unknown = Vec::new();

    for path in documents() {
        let Ok(text) = fs::read_to_string(root().join(&path)) else {
            continue;
        };

        for (name, _, line) in marks(&text) {
            if counter(&name).is_none() {
                unknown.push(format!("{path}:{line} marks `{name}`"));
            }
        }
    }

    assert!(
        unknown.is_empty(),
        "markers naming nothing this file counts:\n  {}\n\nKnown counters: {}\n\n\
         A misspelled marker would otherwise render as an ordinary number and be checked by nobody \
         — which is the state every marked number was in before this file existed.",
        unknown.join("\n  "),
        known.join(", ")
    );
}

#[test]
fn every_counter_is_stated_by_some_document() {
    let mut stated = BTreeSet::new();

    for path in documents() {
        if let Ok(text) = fs::read_to_string(root().join(&path)) {
            stated.extend(marks(&text).into_iter().map(|(name, _, _)| name));
        }
    }

    let unread: Vec<&str> = COUNTERS
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| !stated.contains(*name))
        .collect();

    assert!(
        unread.is_empty(),
        "counters no document states: {}\n\n\
         This is the rule that emptied `NO_EMITTER`: a number nothing emits is a copy of the \
         grammar with no check behind it, and a counter nothing states is a check with nothing to \
         check. State it or delete it.",
        unread.join(", ")
    );
}

#[test]
fn the_marker_scanner_finds_the_numbers_that_are_there() {
    // Assembled rather than written out, so that this file is not itself a document full of markers
    // — `every_marker_names_a_counter_that_exists` walks markdown, but the habit is worth keeping.
    let open = |name: &str| format!("{OPENS}{name}-->");
    let text = format!(
        "declares {}95 numbers, over {}17 tests\n\
         a plain 42 with no marker at all\n\
         {}and then prose\n\
         {}65 536 patterns\n",
        open("opcodes"),
        open("integrity-tests"),
        open("decisions"),
        open("nothing-counts-this"),
    );

    let found = marks(&text);

    assert_eq!(
        found,
        vec![
            ("opcodes".to_owned(), Some(95), 1),
            ("integrity-tests".to_owned(), Some(17), 1),
            ("decisions".to_owned(), None, 3),
            ("nothing-counts-this".to_owned(), Some(65_536), 4),
        ],
        "the scan has to find two on one line, read a thin space through, \
         and report a marker with no number rather than passing it over"
    );

    // And the teeth: a number nobody marked is not a claim, or every timing in the tree would be one.
    assert!(marks("the suite is 837 tests").is_empty());

    assert!(counter("opcodes").is_some());
    assert!(counter("no-such-counter").is_none());
}

#[test]
fn every_counter_counts_something_that_is_there() {
    let missing: Vec<&str> = COUNTERS
        .iter()
        .filter(|(_, count)| count().is_none_or(|number| number == 0))
        .map(|(name, _)| *name)
        .collect();

    assert!(
        missing.is_empty(),
        "counters returning nothing: {}\n\n\
         Every input here is committed, so `None` means a path moved and the counter now measures \
         a file that is not there. It would have gone on reporting `0`, or skipping, and the \
         documents would have agreed with it.",
        missing.join(", ")
    );
}

/// The section every decision record ends with, naming the artefact that backs it.
const ENFORCED_BY: &str = "## What enforces this";

#[test]
fn every_decision_record_says_what_enforces_it() {
    let mut silent = Vec::new();

    for name in decision_records() {
        let Ok(text) = fs::read_to_string(root().join("decisions").join(&name)) else {
            silent.push(name);
            continue;
        };
        if !text.contains(ENFORCED_BY) {
            silent.push(name);
        }
    }

    assert!(
        silent.is_empty(),
        "decision records with no `{ENFORCED_BY}` section: {}\n\n\
         `noha gate` prints `prose-only: recorded, not machine-checked` beside all of them, which \
         is true and too blunt — three are enforced by the type system and one by something not \
         existing. The section is where a record says which it is, and a record without one leaves \
         a reader unable to tell a promise from a guarantee.",
        silent.join(", ")
    );
}

#[test]
fn the_claims_table_names_every_decision_record() {
    let text = fs::read_to_string(root().join("notes").join("CLAIMS.md")).unwrap_or_default();
    assert!(
        !text.is_empty(),
        "notes/CLAIMS.md is unreadable, and it is the document this whole file is a response to"
    );

    // The records are numbered, and the number is what the table cites — `DR-0003 — a branch is…`.
    let cited: BTreeSet<String> = text
        .split("DR-")
        .skip(1)
        .map(|rest| rest.chars().take_while(char::is_ascii_digit).collect())
        .filter(|number: &String| number.len() == 4)
        .map(|number| format!("DR-{number}"))
        .collect();

    let uncited: Vec<String> = decision_records()
        .iter()
        .map(|name| name[..7].to_owned())
        .filter(|number| !cited.contains(number))
        .collect();

    assert!(
        uncited.is_empty(),
        "decision records `notes/CLAIMS.md` does not mention: {}\n\n\
         That document's whole subject is which claims have an instrument behind them, and its \
         table of what backs each record is hand-written. A tenth record would have been absent \
         from it silently — the same shape as the five files `noha.yaml` was not mutating.",
        uncited.join(", ")
    );
}

/// Files this repository's prose names that are deliberately **not** in the repository.
///
/// One entry, and it is the verification toolchain: `noha.yaml` and `.noha/` are ignored globally
/// on this machine and never committed, so a check that asserted their presence would fail in CI on
/// the one file that decides what the mutation gate covers. Naming it here is the same trade
/// [`NOT_MUTATED`] makes in `tests/integrity.rs` — an absence somebody wrote down rather than a
/// silence.
///
/// [`NOT_MUTATED`]: https://example.invalid
const NOT_IN_THE_TREE: [(&str, &str); 4] = [
    (
        "noha.yaml",
        "the mutation config, globally ignored and never committed",
    ),
    (".noha/", "the mutation tool's cache, likewise"),
    (
        "notes/SPEED.md",
        "a document in the sibling project this one borrowed a measurement habit from",
    ),
    (
        "runner/src/reduction.rs",
        "a path `notes/FINDINGS.md` names in the past tense, in the sentence recording that it          moved — prose about a file that stopped existing is the one kind this check cannot read",
    ),
];

/// Whether `token` reads like a path into this repository.
///
/// Deliberately narrow: a suffix this tree actually holds, and either a directory in it or a name a
/// reader would only write about because it is here. `f32::MAX` and `OpShiftLeftLogical` are not
/// paths and a check that thought they were would have to be argued with rather than fixed.
fn looks_like_a_path(token: &str) -> bool {
    const SUFFIXES: [&str; 5] = [".rs", ".md", ".toml", ".yml", ".yaml"];

    // A glob is a sentence about a set of files rather than a name of one, and `tests/*.rs` is
    // how two of them say so.
    if token.starts_with("..")
        || token.starts_with('/')
        || token.contains("://")
        || token.contains('*')
    {
        return false;
    }
    SUFFIXES.iter().any(|suffix| token.ends_with(suffix)) && token.contains('/')
}

/// Every backtick-quoted span of `text`, with the line it sits on.
///
/// Backticks rather than every word, and that is the whole of why this check can exist without an
/// allowlist. This repository quotes what it means — a file, a type, an instruction — and leaves
/// ordinary prose unquoted, so the quoting *is* the marker. Nothing new had to be written into the
/// documents to make them checkable.
fn quoted(text: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();

    for (number, line) in text.lines().enumerate() {
        let mut rest = line;
        while let Some(open) = rest.find('`') {
            let after = &rest[open + 1..];
            let Some(close) = after.find('`') else {
                break;
            };
            let span = after[..close].trim();
            if !span.is_empty() {
                found.push((number + 1, span.to_owned()));
            }
            rest = &after[close + 1..];
        }
    }

    found
}

/// Every quoted span in every document and every comment, as `(path, line, span)`.
///
/// **Comments as well as markdown**, because the drift this catches has landed in both. A doc
/// comment naming a file that moved is exactly as wrong as a README naming one, and rustdoc checks
/// only the links — an intra-doc link is verified and a backtick is decoration.
///
/// **It reaches every file, including any the workspace excludes**, which is the opposite of what
/// the counters do and deliberately so. A count the workspace states must not depend on a directory
/// that can be deleted; a reference is checked where it is written, and prose about this repository
/// rots wherever it lives. The sandbox that made the distinction concrete is gone, and
/// `notes/FINDINGS.md` records how the deletion went: the code left cleanly and the *citations of
/// it* did not, which is what this check exists to catch.
fn claims() -> Vec<(String, usize, String)> {
    let mut found = Vec::new();

    walk(&root(), &mut |path, relative| {
        let markdown = path.extension().is_some_and(|extension| extension == "md");
        let rust = path.extension().is_some_and(|extension| extension == "rs");
        if !markdown && !rust {
            return;
        }
        let Ok(text) = fs::read_to_string(path) else {
            return;
        };

        let scanned = if markdown {
            text.clone()
        } else {
            text.lines()
                .map(str::trim_start)
                .filter(|line| line.starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n")
        };

        // Rust files lose their line numbers to that filter, and a wrong line is worse than none:
        // it sends a reader to an unrelated statement. So a comment's claim is reported by file.
        for (line, span) in quoted(&scanned) {
            found.push((relative.to_owned(), if markdown { line } else { 0 }, span));
        }
    });

    found
}

/// Every file in the workspace, by its path from the root.
fn every_file() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    walk(&root(), &mut |_, relative| {
        found.insert(relative.to_owned());
    });
    found
}

/// Whether some file in the tree is named by `span`.
///
/// **A suffix match, because that is how this repository writes.** The prose says `scan/plan.rs`
/// and `fuzz/mod.rs` and means `runner/src/scan/plan.rs` and `runner/src/fuzz/mod.rs` — a habit
/// worth keeping, since a sentence about a file reads better than a sentence about a path. So the
/// check follows the convention instead of asking three hundred sentences to change.
///
/// It is a floor rather than a proof, in the safe direction: two files could share a tail and this
/// cannot tell them apart. That direction says a path exists where the reader might land on the
/// wrong one, which makes the check weaker and never wrong — the same trade `consumed_outside`
/// makes in `tests/integrity.rs`.
fn some_file_is_called(span: &str, files: &BTreeSet<String>) -> bool {
    files
        .iter()
        .any(|path| path == span || path.ends_with(&format!("/{span}")))
}

#[test]
fn every_path_the_prose_names_is_a_path_that_is_there() {
    let files = every_file();
    let mut missing = Vec::new();

    for (path, line, span) in claims() {
        if !looks_like_a_path(&span) {
            continue;
        }
        if NOT_IN_THE_TREE.iter().any(|(name, _)| span == *name) {
            continue;
        }
        if !some_file_is_called(&span, &files) {
            missing.push(format!("{path}:{line} names `{span}`"));
        }
    }

    assert!(
        missing.is_empty(),
        "the prose names files that are not there:\n  {}\n\n\
         A moved file leaves every sentence about it pointing at nothing, and nothing else here \
         notices: rustdoc checks intra-doc links and a backtick is decoration. `decisions/DR-0002` \
         spent weeks naming an error that never existed, which is this shape one level up.",
        missing.join("\n  ")
    );
}

#[test]
fn every_decision_record_the_prose_cites_is_a_record_that_exists() {
    let numbered: BTreeSet<String> = decision_records()
        .iter()
        .map(|name| name[..7].to_owned())
        .collect();

    let mut dangling = Vec::new();
    walk(&root(), &mut |path, relative| {
        if !path
            .extension()
            .is_some_and(|extension| extension == "md" || extension == "rs")
        {
            return;
        }
        let Ok(text) = fs::read_to_string(path) else {
            return;
        };

        for rest in text.split("DR-").skip(1) {
            let number: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if number.len() == 4 && !numbered.contains(&format!("DR-{number}")) {
                dangling.push(format!("{relative} cites DR-{number}"));
            }
        }
    });

    dangling.sort_unstable();
    dangling.dedup();
    assert!(
        dangling.is_empty(),
        "decisions cited by number that no record answers to:\n  {}\n\n\
         A record is cited by number in fifty places and renamed in one, and every citation still \
         reads like a reference.",
        dangling.join("\n  ")
    );
}

/// Members the prose names that this repository does not declare, and should not.
///
/// **Two kinds, and both are prose doing its job.** Four were *deleted*, and a note recording a
/// deletion has to name what was deleted — two of these say so in the same sentence the check
/// objects to: *"`Module::f_ord_greater_than` … **deleted**"* and *"`Reduction::tail` is gone"*.
/// The fifth never existed at all.
///
/// That fifth is why this whole file exists. `decisions/DR-0002` said strip mining "is not built"
/// and named `LaneError::TooWide` as the error that says so; strip mining had been built for weeks
/// and that error was never written. Three test files quote it now as the failure that motivated
/// them, so the name has to survive here — and the test below makes sure it survives as an
/// *absence*, which is the only way to quote a mistake without re-making it.
const GONE: [(&str, &str); 6] = [
    (
        "LaneError::TooWide",
        "never existed; `decisions/DR-0002` named it and `tests/integrity.rs` was written because of it",
    ),
    (
        "Module::f_ord_greater_than",
        "deleted — a second spelling of what `Lanes::greater_than` emits through `Element::GREATER_THAN`",
    ),
    (
        "Gpu::probe_pipeline",
        "deleted — it allocated to time an allocation, so it measured itself",
    ),
    (
        "Reduction::tail",
        "deleted with the reduction chain's tail pass",
    ),
    (
        "Pass::writing",
        "deleted when the between-pass copy was shortened",
    ),
    (
        "Gpu::time_specialized",
        "deleted with the widening: a public timing path with no caller anywhere",
    ),
];

/// Every name this repository declares, and which of them are types.
///
/// Two sets from one pass: the types a `Type::member` span may be about, and every name any of them
/// could resolve to. Enum variants are collected by tracking the block, because a variant is the one
/// kind of name with no keyword in front of it — and `LaneError::TooWide` is precisely the mistake
/// that needs a variant list to catch.
fn declared() -> (BTreeSet<String>, BTreeSet<String>) {
    let mut types = BTreeSet::new();
    let mut names = BTreeSet::new();

    // The sandbox is read here as well as in `claims`, and for the matching reason: its prose names
    // its own `Outcome` and `Checked` as freely as it names this crate's, and a name the check
    // cannot see reads to it exactly like a name that does not exist. Widening what *counts as
    // declared* can only ever excuse more, so a deleted sandbox cannot fail anything by leaving.
    walk(&root(), &mut |path, _| {
        if !path.extension().is_some_and(|extension| extension == "rs") {
            return;
        }
        let Ok(text) = fs::read_to_string(path) else {
            return;
        };

        let mut inside_a_body = false;
        for line in text.lines() {
            // Every enum and struct here is declared at the left margin, so its closing brace is
            // too. That is what makes tracking the block one boolean instead of a brace counter.
            //
            // **Fields as well as variants**, because a field is a member a reader writes about the
            // same way. `Reference::exact` and `Reduction::total` were both reported as undeclared
            // by a version that collected only variants — two sentences that were right, failed by
            // a check that was not.
            if inside_a_body {
                if line == "}" {
                    inside_a_body = false;
                } else if line.trim_start().starts_with(|c: char| c.is_alphabetic()) {
                    names.insert(word_of(line.trim_start().trim_start_matches("pub ")));
                }
                continue;
            }

            // Every modifier a declaration can wear before its keyword. `unsafe` is on this list
            // because leaving it off reported thirty of `runner`'s own Vulkan methods as undeclared
            // — the check finding its own blind spot, in the half of the tree that has one.
            let mut bare = line.trim_start();
            for modifier in [
                "pub(crate) ",
                "pub(super) ",
                "pub ",
                "unsafe ",
                "async ",
                "default ",
            ] {
                bare = bare.strip_prefix(modifier).unwrap_or(bare);
            }

            for keyword in ["enum ", "struct ", "trait ", "type "] {
                if let Some(rest) = bare.strip_prefix(keyword) {
                    types.insert(word_of(rest));
                    names.insert(word_of(rest));
                    // Only a declaration at the left margin opens a body this can track, and only
                    // one that ends in a brace has a body at all — `pub struct Rng { … }` does and
                    // `pub struct Id(u32);` does not.
                    if matches!(keyword, "enum " | "struct ")
                        && !line.starts_with(char::is_whitespace)
                        && line.trim_end().ends_with('{')
                    {
                        inside_a_body = true;
                    }
                }
            }
            for keyword in ["fn ", "const fn ", "const ", "static "] {
                if let Some(rest) = bare.strip_prefix(keyword) {
                    names.insert(word_of(rest));
                }
            }
        }
    });

    (types, names)
}

/// The identifier `text` opens with.
fn word_of(text: &str) -> String {
    text.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

#[test]
fn every_member_the_prose_names_is_a_name_this_repository_declares() {
    let (types, names) = declared();
    let mut dangling = Vec::new();

    for (path, line, span) in claims() {
        // `LaneError::TooManyStrips { .. }` and `Domain::bits()` are both spans about one member.
        let head: &str = span
            .split([' ', '(', '<', '{', ','])
            .next()
            .unwrap_or_default();

        let parts: Vec<String> = head.split("::").map(word_of).collect();
        for pair in parts.windows(2) {
            let (owner, member) = (&pair[0], &pair[1]);
            let span = format!("{owner}::{member}");
            if member.is_empty()
                || !types.contains(owner)
                || names.contains(member)
                || GONE.iter().any(|(name, _)| *name == span)
            {
                continue;
            }
            dangling.push(format!("{path}:{line} names `{span}`"));
        }
    }

    dangling.sort_unstable();
    dangling.dedup();
    assert!(
        dangling.is_empty(),
        "the prose names members that nothing declares:\n  {}\n\n\
         `decisions/DR-0002` said strip mining \"is not built\" and named `LaneError::TooWide` as \
         the error that says so. Strip mining had been built for weeks and that error never \
         existed, and `noha gate` printed a tick beside the record — because a decision record is \
         prose and prose is not checked. This is that, checked.",
        dangling.join("\n  ")
    );
}
#[test]
fn nothing_excused_as_gone_has_come_back() {
    let (_, names) = declared();

    let returned: Vec<&str> = GONE
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| {
            name.rsplit("::")
                .next()
                .is_some_and(|member| names.contains(member))
        })
        .collect();

    assert!(
        returned.is_empty(),
        "excused as deleted and declared again: {}

         An excuse that outlives its reason is the drift this file is about, one level up: the          prose would go on describing a deletion that has been undone, and the check that should          have said so is the one holding the excuse.",
        returned.join(", ")
    );
}
