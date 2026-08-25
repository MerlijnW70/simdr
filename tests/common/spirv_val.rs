//! Finding `spirv-val` and running a module through it.
//!
//! **Shared by both crates' test trees**, which is why it is its own file. `simdr`'s tests validate
//! modules they build themselves; `runner`'s validate the kernel library, which `simdr` cannot see
//! because the dependency arrow points the other way. Two copies of this would be two things to
//! keep in step, and the one that got less attention would be the one covering the kernels nobody
//! had validated until now.
//!
//! Nothing here uses either crate — only `std` — so the same file compiles into both test trees.
//! `runner/tests/common/mod.rs` reaches it with a `#[path]`.

#![allow(
    dead_code,
    reason = "each test binary compiles this file and uses a different subset of it"
)]

use std::path::PathBuf;
use std::process::Command;

/// Which validation rules to hold a module to.
///
/// **`--target-env` is not optional, and finding that out cost a wrong assumption.** Left off,
/// `spirv-val` checks the *universal* SPIR-V environment, which is far laxer than any real
/// consumer: it happily accepted a `GLCompute` entry point with no `LocalSize`, because that
/// requirement is Vulkan's rather than SPIR-V's. Every call names an environment, and it is the
/// one the module will actually run under.
pub const VULKAN_1_0: &str = "vulkan1.0";
/// Vulkan 1.1 — the environment for SPIR-V 1.3, and the first with subgroup operations.
pub const VULKAN_1_1: &str = "vulkan1.1";

/// Where to find `spirv-val`, or `None` if it is not installed.
///
/// **Set-and-wrong is not the same as unset, and used to be.** `SPIRV_VAL` is how CI says where the
/// validator is; a path with no file at it returned `None`, which every caller reads as "no
/// validator installed" and skips over. So a typo in that one variable turned off every validation
/// in *both* test trees, and left a green run behind — with the skip lines in `eprintln!`, which
/// `libtest` swallows from a passing test.
///
/// Naming a path is asserting something is at it. Unset it to search the usual place instead.
pub fn validator() -> Option<PathBuf> {
    if let Some(from_env) = std::env::var_os("SPIRV_VAL") {
        let path = PathBuf::from(from_env);
        assert!(
            path.is_file(),
            "SPIRV_VAL points at {path:?} and there is no file there — every validation test would \
             skip over that, and skips are invisible. Unset it to look in the usual place."
        );
        return Some(path);
    }

    // **The fallback was one absolute path on one machine, and that machine had changed.** It read
    // `H:\tools\spirv-tools\install\bin\spirv-val.exe` — a drive that is not mounted here any
    // more — so "look in the usual place" had quietly meant "find nothing" for as long as the
    // letter had been wrong. That is the failure the paragraph above is about, arriving two lines
    // under it: every validation in *both* test trees skipped, and a skip is invisible.
    //
    // `PATH` is the usual place. It is what a machine with the tools installed already answers, it
    // costs the same lookup, and it cannot go stale when a drive is remounted or a checkout moves.
    // `SPIRV_VAL` still wins where it is set, because CI names the path it installed to.
    let name = if cfg!(windows) {
        "spirv-val.exe"
    } else {
        "spirv-val"
    };
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

/// Write `module` out and hand it to `spirv-val`, returning the tool's complaint if it had one.
///
/// Panicking here is correct: a harness that cannot write a temporary file or spawn a process has
/// a broken environment, which is a different thing from a module being invalid.
pub fn validate(words: &[u32], label: &str, target_env: &str) -> Result<(), String> {
    let Some(tool) = validator() else {
        eprintln!("SKIPPED {label}: spirv-val not found (set SPIRV_VAL)");
        return Ok(());
    };

    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }

    // The label reaches a filename, so anything that is not a filename is replaced rather than
    // trusted — the kernel names below are built from type names and widths, and `<` and `>` are
    // not characters a path may hold on Windows.
    let stem: String = label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();

    let path = std::env::temp_dir().join(format!("simdr-{stem}.spv"));
    std::fs::write(&path, &bytes).expect("the temp directory is writable");

    let output = Command::new(&tool)
        .arg("--target-env")
        .arg(target_env)
        .arg(&path)
        .output()
        .expect("spirv-val is executable");

    if output.status.success() {
        return Ok(());
    }

    Err(format!(
        "spirv-val rejected {label}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

/// Validate, and fail the calling test with the validator's own words if it objects.
pub fn expect_valid(words: &[u32], label: &str, target_env: &str) {
    if let Err(complaint) = validate(words, label, target_env) {
        panic!("{complaint}");
    }
}
