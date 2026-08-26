//! What this repository's prose says about itself, checked against the repository.
//!
//! Four claims, and only the first needed a marker invented for it: the **numbers** the documents
//! state, the **files** they name, the **members** they name, and — added last, after it had already
//! gone wrong — the **text** they are written in. The middle two need nothing new,
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
//!
//! # And the text itself, which all three of those can be true of a ruined document
//!
//! The three checks above read digits and backticked names. Both are ASCII. So a document may rot
//! in every character they do not touch and pass all of them — which `README.md` did, for nine days
//! and eight commits, after one edit re-encoded it and turned every em dash, every micro sign and
//! every ratio's multiplication sign into three characters of rubbish.
//!
//! [`every_document_is_the_text_it_was_written_as`] is the fourth claim. It undoes the round trip
//! that causes the damage rather than matching a list of known-bad sequences: take each character
//! back to the byte Windows-1252 would have decoded it from, and ask whether those bytes are a
//! character. When they are, nobody typed them. That covers the damage this repository has not met
//! yet as well as the ten shapes it has, and it reaches `.rs` as readily as `.md` — a doc comment
//! carries the same punctuation as a paragraph, and the front page was only the file that happened
//! to be open.

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
const COUNTERS: [Counter; 11] = [
    // **The registry counts itself, including this row.** `notes/CLAIMS.md` stated the size of
    // this array in prose and was wrong by three within a day of writing it, in the paragraph
    // arguing that a number in a document should carry an instrument. This is that instrument, and
    // it is the smallest one here: a counter is a claim about the tree, and how many claims there
    // are is one too.
    ("counters", || Some(COUNTERS.len())),
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
/// The four sections a decision record is made of, in the order it makes them.
///
/// **This replaced a check for one heading, `## What enforces this`, and the replacement is
/// stricter rather than looser.** That heading was where a record said whether it was held up by
/// the type system, by a validator, or by nothing — the difference between a promise and a
/// guarantee. The form below keeps that distinction and spreads it over two required sections
/// instead of one: `The Rejected Route` carries the figure that killed the alternative, and
/// `The Limit` says what the measurement does not establish, which is where "nothing checks this"
/// now has to be written down.
///
/// Four required headings is four times the structure the old check asked for, and the order is
/// part of it: a record that states its decision before its measurement is arguing rather than
/// reporting.
const SECTIONS: [&str; 4] = [
    "## The Measurement",
    "## The Decision",
    "## The Rejected Route",
    "## The Limit",
];

#[test]
fn every_decision_record_is_made_of_the_four_sections_in_order() {
    let mut wrong = Vec::new();

    for name in decision_records() {
        let Ok(text) = fs::read_to_string(root().join("decisions").join(&name)) else {
            wrong.push(format!("{name}: unreadable"));
            continue;
        };

        let mut at = 0;
        for section in SECTIONS {
            match text[at..].find(section) {
                Some(found) => at += found + section.len(),
                None => {
                    let anywhere = text.contains(section);
                    wrong.push(format!(
                        "{name}: {section} {}",
                        if anywhere {
                            "is out of order"
                        } else {
                            "is missing"
                        }
                    ));
                    break;
                }
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "decision records that are not the four sections in order:\n  {}\n\n\
         The measurement comes first because the decision is supposed to follow from it. \
         `The Rejected Route` names what was not built and the figure that decided it, and \
         `The Limit` says what the numbers above do not establish — including, where it is true, \
         that nothing checks the decision at all.",
        wrong.join("\n  ")
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

/// What Windows-1252 makes of the thirty-two bytes where it is not Latin-1.
///
/// A byte from `0xA0` up decodes to the character of its own number, so only this range needs a
/// table. The five Windows-1252 leaves undefined — `0x81`, `0x8D`, `0x8F`, `0x90`, `0x9D` — are
/// written as themselves, which is what the editor that did the damage did with them: it is the
/// Latin-1 fallback every real implementation of this codepage carries, and a repair that omitted
/// it would leave five bytes it could not name.
const CP1252_HIGH: [char; 32] = [
    '\u{20AC}', '\u{81}', '\u{201A}', '\u{0192}', '\u{201E}', '\u{2026}', '\u{2020}', '\u{2021}',
    '\u{02C6}', '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\u{8D}', '\u{017D}', '\u{8F}',
    '\u{90}', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}', '\u{2022}', '\u{2013}', '\u{2014}',
    '\u{02DC}', '\u{2122}', '\u{0161}', '\u{203A}', '\u{0153}', '\u{9D}', '\u{017E}', '\u{0178}',
];

/// The character Windows-1252 decodes `byte` to.
fn cp1252(byte: u8) -> char {
    match byte {
        0x80..=0x9F => CP1252_HIGH
            .get(usize::from(byte - 0x80))
            .copied()
            .unwrap_or(char::REPLACEMENT_CHARACTER),
        _ => char::from(byte),
    }
}

/// The byte Windows-1252 would have decoded to `character`, if any did.
fn byte_of(character: char) -> Option<u8> {
    let code = u32::from(character);
    if (0xA0..=0xFF).contains(&code) {
        return u8::try_from(code).ok();
    }
    CP1252_HIGH
        .iter()
        .position(|&mapped| mapped == character)
        .and_then(|index| u8::try_from(0x80 + index).ok())
}

/// Every run in `line` that is UTF-8 read as Windows-1252, as `(what is there, what it was)`.
///
/// The damage is a round trip: a document's UTF-8 bytes handed to something that believed they were
/// Windows-1252, and the characters that came out written back as UTF-8. It is detected by undoing
/// exactly that — take each character back to the byte it would have decoded from, and ask whether
/// the run those bytes form is a character. When it is, the run is not text anybody typed.
///
/// **A run is claimed only when every one of its bytes fits.** A lead byte needs continuation bytes
/// after it in the right count, and the whole must decode; anything short of that is left alone.
/// That direction makes the scan a floor rather than a proof — a damaged sequence whose bytes happen
/// not to form a character would be missed — and never lets it rewrite something a person wrote.
///
/// Runs cannot cross a line, because a newline is ASCII and no continuation byte decodes to one.
fn damage(line: &str) -> Vec<(String, String)> {
    let characters: Vec<char> = line.chars().collect();
    let mut found = Vec::new();
    let mut index = 0;

    while index < characters.len() {
        let width = match characters.get(index).copied().and_then(byte_of) {
            Some(0xC2..=0xDF) => 2,
            Some(0xE0..=0xEF) => 3,
            Some(0xF0..=0xF4) => 4,
            _ => {
                index += 1;
                continue;
            }
        };

        let Some(run) = characters.get(index..index + width) else {
            index += 1;
            continue;
        };
        let Some(bytes) = run
            .iter()
            .copied()
            .map(byte_of)
            .collect::<Option<Vec<u8>>>()
        else {
            index += 1;
            continue;
        };
        if !bytes
            .iter()
            .skip(1)
            .all(|byte| (0x80..=0xBF).contains(byte))
        {
            index += 1;
            continue;
        }

        match std::str::from_utf8(&bytes) {
            Ok(original) => {
                found.push((run.iter().collect(), original.to_owned()));
                index += width;
            }
            Err(_) => index += 1,
        }
    }

    found
}

/// Every file in the workspace whose text this suite is willing to read.
///
/// Wider than [`documents`] deliberately. The damage below arrived in a markdown file, and nothing
/// about the editor that caused it cares which extension it is saving — a doc comment carries the
/// same em dashes as a paragraph, and `README.md` was only the file that happened to be open.
fn text_files() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    walk(&root(), &mut |path, relative| {
        let readable = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| matches!(extension, "md" | "rs" | "toml" | "yml"));
        if readable {
            found.insert(relative.to_owned());
        }
    });
    found
}

#[test]
fn every_document_is_the_text_it_was_written_as() {
    let mut damaged = Vec::new();

    for name in text_files() {
        // **A file that is not UTF-8 at all fails here rather than being skipped.** Skipping it
        // would be this check's own version of the bug it exists for: the worst-damaged file in the
        // tree is the one `read_to_string` cannot read, and `continue` would pass over it in
        // silence while reporting on the ones that are merely wrong.
        let text = match fs::read_to_string(root().join(&name)) {
            Ok(text) => text,
            Err(complaint) => {
                damaged.push(format!("{name}: is not readable as UTF-8 — {complaint}"));
                continue;
            }
        };
        for (number, line) in text.lines().enumerate() {
            for (there, was) in damage(line) {
                damaged.push(format!(
                    "{name}:{}: {:?} was written as {was:?}",
                    number + 1,
                    there
                ));
            }
        }
    }

    assert!(
        damaged.is_empty(),
        "text that is UTF-8 read as Windows-1252:\n  {}\n\n\
         `README.md` sat like this for nine days and eight commits. One edit re-encoded it — 136 \
         lines in a diff whose message was about something else — and every em dash, every `us` \
         with a micro sign on it and every ratio's multiplication sign became three characters of \
         rubbish on the repository's front page.\n\n\
         Nothing here could see it. The counters check digits and the reference checks check \
         backticked names, and both of those are ASCII: a document can rot in every character \
         those two do not read and pass this suite. That is `notes/CLAIMS.md`'s own subject — a \
         claim nothing checks — arriving in the shape it warns about, which is why the check is \
         here rather than a note asking somebody to look.",
        damaged.join("\n  ")
    );
}

#[test]
fn the_damage_scanner_finds_it_where_there_is_some_and_not_where_there_is_none() {
    // Built rather than typed. A literal example of the damage would *be* damage, in a file the
    // test above reads — the check would fail on its own teeth. `OPENS` is split one screen up for
    // the same reason, and this is that rule for a second kind of markup.
    let damaged: String = "\u{2014}\u{00B5}\u{2192}"
        .as_bytes()
        .iter()
        .map(|&byte| cp1252(byte))
        .collect();

    let found = damage(&damaged);
    let recovered: String = found.iter().map(|(_, was)| was.as_str()).collect();
    assert_eq!(
        recovered,
        "\u{2014}\u{00B5}\u{2192}",
        "the scanner found {} runs in three characters' worth of damage and read them back as \
         {recovered:?} — a scanner that cannot recover what it detects would report the repair as \
         complete while the file still said something else",
        found.len()
    );

    // And the other half, which is the one that matters: ordinary prose must survive it untouched.
    // Every character below is one this repository writes on purpose, and each is exactly what a
    // damaged run decodes *to* — so a scan that could not tell the two apart would eat the
    // documents it is meant to protect.
    for intact in [
        "a round trip \u{2014} the unit of cost",
        "127 \u{00B5}s per pass, 6.45\u{00D7} against `i32`",
        "f32 \u{2194} f16, and 2\u{00B2}\u{2074} exactly",
        "\u{03A3} over the strips, then \u{2192} the buffer",
    ] {
        assert!(
            damage(intact).is_empty(),
            "the scanner called {intact:?} damaged, and it is what this repository writes"
        );
    }
}

#[test]
fn no_file_opens_with_a_byte_order_mark() {
    let mut marked = Vec::new();

    for name in text_files() {
        let Ok(bytes) = fs::read(root().join(&name)) else {
            continue;
        };
        if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
            marked.push(name);
        }
    }

    assert!(
        marked.is_empty(),
        "files opening with a byte order mark: {}\n\n\
         Three bytes that render as nothing and mean UTF-8, which every file here already is. It is \
         checked because of what it indicates rather than what it costs: `README.md` grew one on \
         the same save that mangled 136 of its lines, and it is the fingerprint of the editor that \
         did it. A file that gains one has been round-tripped through something that has an opinion \
         about encodings, and the line above is what that opinion does to the rest of the text.",
        marked.join(", ")
    );
}

/// Nouns that name a **standing set** in this tree, and the counter that knows its size.
///
/// [`every_marked_number_is_the_number_that_is_there`] reads the digits behind a marker. This reads
/// the other half of the same problem: a count written as an English word, which no marker can
/// precede — a marker resolves to digits — and which drifts at exactly the same speed.
///
/// Both failures it was written after were that shape. `notes/CLAIMS.md` said *"Seven counters
/// exist"* in the very paragraph arguing that there is "no argument for leaving these to prose",
/// while stating the right number two hundred lines further down; and `README.md` and
/// `notes/CLAIMS.md` said seven and eight decision records where the directory holds ten. Neither
/// was reachable by anything here, because both numbers are spelled.
///
/// # Why these three nouns and not the ones with more hits
///
/// The vocabulary was chosen by measuring the documents rather than by taste. `tests`,
/// `operations`, `opcodes` and `jobs` sit beside a spelled numeral forty-odd times between them,
/// and nearly every one is an *account of something that happened* — "Eleven tests reading past
/// their input", "seven opcodes deleted in one commit", "all ten opcodes then in `module::op`".
/// Those sentences are true permanently. Checking them against today's tree would fail a document
/// that is telling the truth, which is the gate `ci.yml` describes as teaching everybody to ignore
/// red.
///
/// So the rule is not "spelled numbers are forbidden". It is that a noun belongs here when it names
/// a set that **exists now** and whose size this file can ask the tree for. Counters, decision
/// records and examples do. Tests that once failed do not.
const SPELLED_SETS: [(&str, &str); 3] = [
    ("counters", "counters"),
    ("decisions", "decisions"),
    ("examples", "examples"),
];

/// The words a count may be written as, up to the largest any of these sets is likely to reach.
const NUMERALS: [(&str, usize); 20] = [
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
    ("eight", 8),
    ("nine", 9),
    ("ten", 10),
    ("eleven", 11),
    ("twelve", 12),
    ("thirteen", 13),
    ("fourteen", 14),
    ("fifteen", 15),
    ("sixteen", 16),
    ("seventeen", 17),
    ("eighteen", 18),
    ("nineteen", 19),
    ("twenty", 20),
];

/// A newline, spelled as its byte so that no escape has to survive being written here.
const NEWLINE: u8 = 10;

/// `text` with every code span blanked, newlines flattened to spaces and the ASCII lowered.
///
/// **Backticks are the other check's territory, and reading them here was a false positive on the
/// first run.** `notes/FINDINGS.md` says "the one `decisions/DR-0004` rests on", where the path
/// tokenises to `decisions` and lands next to `one` — a claim that this repository holds one
/// decision record, which nobody wrote. A backticked token is a *name*, and names are what
/// [`every_member_the_prose_names_is_a_name_this_repository_declares`] reads.
///
/// Toggling on each backtick handles fenced blocks for free: three backticks flip the state an odd
/// number of times, so a fence opens into the blanked state and its closing fence leaves it.
///
/// Blanked rather than removed, because every transform here is byte-for-byte length-preserving —
/// which is what lets a failure below quote a line number a reader can open.
fn prose_of(text: &str) -> Vec<u8> {
    let mut flat = Vec::with_capacity(text.len());
    let mut in_code = false;

    for byte in text.bytes() {
        if byte == b'`' {
            in_code = !in_code;
            flat.push(b' ');
        } else if in_code || byte == NEWLINE {
            flat.push(b' ');
        } else {
            flat.push(byte.to_ascii_lowercase());
        }
    }

    flat
}

/// Every maximal run of word bytes in `flat`, as `(offset, word)`.
///
/// Tokenised rather than searched for as `"seven counters"`, and the difference is a loophole this
/// check had for as long as it took to write the paragraph announcing it: `*sixteen* examples` in
/// `notes/NEXT.md` slipped straight past a scan for the two words with one space between them.
/// Markdown puts emphasis, backticks and line breaks between a number and its noun, and none of
/// that makes the sentence any less of a claim.
fn words_of(flat: &[u8]) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    let mut at = 0;

    while at < flat.len() {
        if !is_word_byte(flat, at) {
            at += 1;
            continue;
        }
        let from = at;
        while at < flat.len() && is_word_byte(flat, at) {
            at += 1;
        }
        found.push((from, String::from_utf8_lossy(&flat[from..at]).into_owned()));
    }

    found
}

/// Whether the bytes between two words end the sentence, so the pair is not a phrase.
///
/// Without this, "...was seven. Examples of this..." reads as a claim about how many examples
/// there are. A count and its noun belong to each other only inside one sentence.
fn separated(flat: &[u8], from: usize, to: usize) -> bool {
    flat[from..to]
        .iter()
        // No newline arm: `flat` replaced every one with a space before this is reached.
        .any(|byte| matches!(byte, b'.' | b'!' | b'?' | b':' | b';' | b'|'))
}

/// Whether the byte at `at` is part of a word.
fn is_word_byte(text: &[u8], at: usize) -> bool {
    text.get(at)
        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
}

#[test]
fn every_spelled_count_of_a_standing_set_is_the_number_that_is_there() {
    let mut wrong: Vec<String> = Vec::new();

    for path in documents() {
        let Ok(text) = fs::read_to_string(root().join(&path)) else {
            continue;
        };

        let flat = prose_of(&text);

        let words = words_of(&flat);
        for pair in words.windows(2) {
            let (at, spelled) = (&pair[0].0, &pair[0].1);
            let (noun_at, noun) = (&pair[1].0, &pair[1].1);

            let Some((_, stated)) = NUMERALS.iter().find(|(word, _)| word == spelled) else {
                continue;
            };
            let Some((counter, _)) = SPELLED_SETS.iter().find(|(_, set)| set == noun) else {
                continue;
            };
            if separated(&flat, at + spelled.len(), *noun_at) {
                continue;
            }

            let Some(truth) = COUNTERS
                .iter()
                .find(|(name, _)| name == counter)
                .and_then(|(_, count)| count())
            else {
                continue;
            };
            if *stated == truth {
                continue;
            }

            let line = text[..*at].matches('\n').count() + 1;
            wrong.push(format!(
                "{path}:{line} says {spelled} {noun}, there are {truth}"
            ));
        }
    }

    assert!(
        wrong.is_empty(),
        "counts written as words that the tree does not agree with:
  {}

         A marker resolves to digits, so it cannot stand in front of one of these — write the          number, or say something that is not a count. A document describing a wrong count is an          instance of one: describe it rather than quoting it, the way the mojibake note does.",
        wrong.join("
  ")
    );
}
