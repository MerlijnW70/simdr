//! ```text
//! `op.rs` declares <!--count:opcodes-->95 numbers
//! ```

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

type Counter = (&'static str, fn() -> Option<usize>);

const COUNTERS: [Counter; 10] = [
    ("counters", || Some(COUNTERS.len())),
    ("opcodes", || count_lines("src/module/op.rs", "pub const ")),
    ("lane-operations", || Some(lane_operations())),
    ("test-functions", || Some(test_functions())),
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

fn count_lines(path: &str, prefix: &str) -> Option<usize> {
    let text = fs::read_to_string(root().join(path)).ok()?;
    Some(
        text.lines()
            .filter(|line| line.trim_start().starts_with(prefix))
            .count(),
    )
}

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

fn fuzz_operations() -> Option<usize> {
    let text = fs::read_to_string(root().join("runner/src/fuzz/generate/vocabulary.rs")).ok()?;
    let rest = text.split("const EVERY_KIND: [Kind; ").nth(1)?;
    rest.chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

const WORKSPACE: [&str; 5] = ["src", "tests", "examples", "runner", "cli"];

fn in_the_workspace(visit: &mut impl FnMut(&Path, &str)) {
    for directory in WORKSPACE {
        walk(&root().join(directory), visit);
    }
}

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

fn documents() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    walk(&root(), &mut |path, relative| {
        if path.extension().is_some_and(|extension| extension == "md") {
            found.insert(relative.to_owned());
        }
    });
    found
}

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

const OPENS: &str = "<!--count:";

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
                continue;
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

const NOT_IN_THE_TREE: [(&str, &str); 2] = [
    (
        "noha.yaml",
        "the mutation config, globally ignored and never committed",
    ),
    (".noha/", "the mutation tool's cache, likewise"),
];

fn looks_like_a_path(token: &str) -> bool {
    const SUFFIXES: [&str; 5] = [".rs", ".md", ".toml", ".yml", ".yaml"];

    if token.starts_with("..")
        || token.starts_with('/')
        || token.contains("://")
        || token.contains('*')
    {
        return false;
    }
    SUFFIXES.iter().any(|suffix| token.ends_with(suffix)) && token.contains('/')
}

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

        for (line, span) in quoted(&scanned) {
            found.push((relative.to_owned(), if markdown { line } else { 0 }, span));
        }
    });

    found
}

fn every_file() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    walk(&root(), &mut |_, relative| {
        found.insert(relative.to_owned());
    });
    found
}

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
         notices: rustdoc checks intra-doc links and a backtick is decoration.",
        missing.join("\n  ")
    );
}

const GONE: [(&str, &str); 6] = [
    (
        "LaneError::TooWide",
        "never existed; `tests/integrity.rs` was written because of it",
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

fn declared() -> (BTreeSet<String>, BTreeSet<String>) {
    let mut types = BTreeSet::new();
    let mut names = BTreeSet::new();

    walk(&root(), &mut |path, _| {
        if !path.extension().is_some_and(|extension| extension == "rs") {
            return;
        }
        let Ok(text) = fs::read_to_string(path) else {
            return;
        };

        let mut inside_a_body = false;
        for line in text.lines() {
            if inside_a_body {
                if line == "}" {
                    inside_a_body = false;
                } else if line.trim_start().starts_with(|c: char| c.is_alphabetic()) {
                    names.insert(word_of(line.trim_start().trim_start_matches("pub ")));
                }
                continue;
            }

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
         `LaneError::TooWide` was named as the error that refuses strip mining. Strip mining \
         had been built for weeks and that error never existed. This is that, checked.",
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

const CP1252_HIGH: [char; 32] = [
    '\u{20AC}', '\u{81}', '\u{201A}', '\u{0192}', '\u{201E}', '\u{2026}', '\u{2020}', '\u{2021}',
    '\u{02C6}', '\u{2030}', '\u{0160}', '\u{2039}', '\u{0152}', '\u{8D}', '\u{017D}', '\u{8F}',
    '\u{90}', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}', '\u{2022}', '\u{2013}', '\u{2014}',
    '\u{02DC}', '\u{2122}', '\u{0161}', '\u{203A}', '\u{0153}', '\u{9D}', '\u{017E}', '\u{0178}',
];

fn cp1252(byte: u8) -> char {
    match byte {
        0x80..=0x9F => CP1252_HIGH
            .get(usize::from(byte - 0x80))
            .copied()
            .unwrap_or(char::REPLACEMENT_CHARACTER),
        _ => char::from(byte),
    }
}

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
         those two do not read and pass this suite, which is why this check is here.",
        damaged.join("\n  ")
    );
}

#[test]
fn the_damage_scanner_finds_it_where_there_is_some_and_not_where_there_is_none() {
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

const SPELLED_SETS: [(&str, &str); 2] = [("counters", "counters"), ("examples", "examples")];

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

const NEWLINE: u8 = 10;

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

fn separated(flat: &[u8], from: usize, to: usize) -> bool {
    flat[from..to]
        .iter()
        .any(|byte| matches!(byte, b'.' | b'!' | b'?' | b':' | b';' | b'|'))
}

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
