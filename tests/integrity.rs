use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

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

fn sources_on_disk() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    walk(&root().join("src"), &mut found);
    walk(&root().join("runner").join("src"), &mut found);
    walk(&root().join("cli").join("src"), &mut found);
    found
}

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

fn consumed_outside(
    function: &str,
    defined_in: &str,
    mentions: &[(String, BTreeSet<String>)],
) -> bool {
    mentions
        .iter()
        .any(|(path, words)| path != defined_in && words.contains(function))
}

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

fn sources_in_config() -> Option<BTreeSet<String>> {
    const CONFIGS: [(&str, &str); 3] = [
        ("noha.yaml", ""),
        ("runner/noha.yaml", "runner/"),
        ("cli/noha.yaml", "cli/"),
    ];

    let mut found = BTreeSet::new();
    let mut any = false;
    for (path, prefix) in CONFIGS {
        let Ok(text) = fs::read_to_string(root().join(path)) else {
            continue;
        };
        any = true;
        found.extend(
            text.lines()
                .skip_while(|line| line.trim() != "sources:")
                .skip(1)
                .take_while(|line| line.starts_with("  - ") || line.trim_start().starts_with('#'))
                .filter_map(|line| line.strip_prefix("  - ").map(str::trim))
                .filter(|entry| entry.ends_with(".rs"))
                .map(|entry| format!("{prefix}{entry}")),
        );
    }
    any.then_some(found)
}

fn config_or_skip(label: &str) -> Option<BTreeSet<String>> {
    let found = sources_in_config();
    if found.is_none() {
        eprintln!("SKIPPED {label}: no noha.yaml here (the mutation runner's config is local)");
    }
    found
}

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
    assert!(sources_on_disk().len() >= 30);

    let Some(configured) = config_or_skip("the_source_list_is_not_empty") else {
        return;
    };
    assert!(configured.len() >= 30);
}

#[test]
fn every_file_with_unsafe_code_in_it_is_excused() {
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

    assert!(!contains_unsafe_code("#![forbid(unsafe_code)]"));
    assert!(!contains_unsafe_code(
        "//! Vulkan is FFI, and FFI is `unsafe`."
    ));
    assert!(!contains_unsafe_code(
        "    // SAFETY: as above — an unsafe fn forwarding its own"
    ));

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

    assert!(
        declared.len() >= 150,
        "only {} public functions found in src/, which is fewer than this crate has — the \
         declaration parser has stopped matching",
        declared.len()
    );
}

#[test]
fn the_emitter_forbids_unsafe_outright_so_none_of_it_needs_excusing() {
    let lib = fs::read_to_string(root().join("src").join("lib.rs")).expect("the emitter has a lib");

    assert!(
        lib.contains("#![forbid(unsafe_code)]"),
        "the emitter no longer forbids unsafe, so `unsafe` can appear anywhere in `src/` — and \
         every file there is inside the mutation gate, which is the arrangement the test above \
         exists to prevent"
    );
}

const NO_DISPATCH: [(&str, &str); 1] = [(
    "runner/src/dispatch/pipeline.rs",
    "`probe_pipelines` builds pipelines to time creation and destroys them without submitting, \
     so there is no dispatch to bound",
)];

fn code_names(path: &str, needle: &str) -> bool {
    let Ok(text) = fs::read_to_string(root().join(path)) else {
        return false;
    };
    text.lines()
        .map(str::trim_start)
        .filter(|line| !line.starts_with("//"))
        .any(|line| line.contains(needle))
}

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

    assert!(
        builders
            .iter()
            .any(|(path, bounds)| path == "runner/src/dispatch/pipeline.rs" && !bounds),
        "the excused file reads as bounded, so the scanner cannot tell a bound from its absence"
    );
}

const NO_EMITTER: [(&str, &str); 0] = [];

const OPCODES: &str = "src/module/op.rs";

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

    let mentions = mentions();
    assert!(consumed_outside("I_ADD", OPCODES, &mentions));
    assert!(!consumed_outside(
        "OP_NO_SUCH_INSTRUCTION",
        OPCODES,
        &mentions
    ));
}

/// The operations `prefix_` names, split by whether they leave a lane its own
/// element out.
fn scan_operations() -> (BTreeSet<String>, BTreeSet<String>) {
    let text = fs::read_to_string(root().join("src/lanes/scan.rs")).unwrap_or_default();
    let mut inclusive = BTreeSet::new();
    let mut exclusive = BTreeSet::new();

    for line in text.lines().map(str::trim_start) {
        let Some(rest) = line.strip_prefix("pub fn prefix_") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        match name.strip_suffix("_exclusive") {
            Some(stem) => exclusive.insert(stem.to_owned()),
            None => inclusive.insert(name),
        };
    }

    (inclusive, exclusive)
}

/// The variants of the enum the tour drives its running folds from.
fn tour_running_folds() -> BTreeSet<String> {
    let text = fs::read_to_string(root().join("runner/src/kernels/mod.rs")).unwrap_or_default();
    let Some(body) = text.split("pub enum Running {").nth(1) else {
        return BTreeSet::new();
    };
    let Some(body) = body.split('}').next() else {
        return BTreeSet::new();
    };

    body.lines()
        .map(str::trim)
        .filter_map(|line| line.strip_suffix(','))
        .filter(|name| name.chars().all(char::is_alphanumeric) && !name.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[test]
fn every_scan_has_the_exclusive_twin_its_family_promises() {
    let (inclusive, exclusive) = scan_operations();

    assert!(
        !inclusive.is_empty(),
        "no `prefix_` operation was found at all, so this test is checking nothing"
    );

    let missing: Vec<&String> = inclusive.difference(&exclusive).collect();
    assert!(
        missing.is_empty(),
        "these scans run inclusively and have no exclusive form, so a caller who needs the value \
         before their own lane has to build it by hand: {missing:?}"
    );

    let orphaned: Vec<&String> = exclusive.difference(&inclusive).collect();
    assert!(
        orphaned.is_empty(),
        "these exclusive scans have no inclusive form beside them: {orphaned:?}"
    );
}

#[test]
fn the_tour_knows_every_running_fold_this_crate_can_do() {
    let (inclusive, _) = scan_operations();
    let shown = tour_running_folds();

    assert!(
        !shown.is_empty(),
        "`pub enum Running` was not found in the kernels, so the tour drives its scans some other \
         way and this test no longer watches anything"
    );

    let unshown: Vec<&String> = inclusive.difference(&shown).collect();
    assert!(
        unshown.is_empty(),
        "`runner/examples/show.rs` walks `Running` to print one row per running fold, and these \
         scans have no variant there — so the tour would print a surface smaller than the one \
         this crate has, and would go on doing it silently: {unshown:?}"
    );

    let invented: Vec<&String> = shown.difference(&inclusive).collect();
    assert!(
        invented.is_empty(),
        "`Running` names folds that `src/lanes/scan.rs` does not declare: {invented:?}"
    );
}
