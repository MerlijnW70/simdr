//! Shared harness for the integration tests: finding `spirv-val` and running it.
//!
//! Not a test binary of its own — Cargo compiles `tests/*.rs` as separate binaries and leaves
//! directories alone, so each of those declares `mod common;` and gets its own copy. That is also
//! why the allow below is needed: a helper only one of them uses is dead code in the other.

#![allow(
    dead_code,
    reason = "each test binary compiles this file and uses a different subset of it"
)]

use simdr::encode;
use simdr::module::{BuildError, Id, Module, Section, Version, op};
use simdr::spec::{
    AddressingModel, Capability, ExecutionMode, ExecutionModel, FunctionControl, MemoryModel,
};
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
pub fn validator() -> Option<PathBuf> {
    if let Some(from_env) = std::env::var_os("SPIRV_VAL") {
        let path = PathBuf::from(from_env);
        return path.is_file().then_some(path);
    }

    let fallback = PathBuf::from(r"H:\tools\spirv-tools\install\bin\spirv-val.exe");
    fallback.is_file().then_some(fallback)
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

    let path = std::env::temp_dir().join(format!("simdr-{label}.spv"));
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

/// A compute module with everything but its body, plus the id its entry point will use.
///
/// A module missing its capability or its execution mode is rejected for *that*, which would tell
/// us nothing about the part under test — so every test starts from a shape that already passes.
pub fn compute_skeleton(version: Version) -> Result<(Module, Id), BuildError> {
    let mut module = Module::new(version);
    let main = module.alloc_id()?;
    module.name(main, "main")?;

    module.emit(
        Section::Capability,
        op::CAPABILITY,
        &[Capability::Shader.word()],
    )?;
    module.emit(
        Section::MemoryModel,
        op::MEMORY_MODEL,
        &[AddressingModel::Logical.word(), MemoryModel::Glsl450.word()],
    )?;

    let mut entry = vec![ExecutionModel::GlCompute.word(), main.word()];
    encode::literal_string(&mut entry, "main");
    module.emit(Section::EntryPoint, op::ENTRY_POINT, &entry)?;

    module.emit(
        Section::ExecutionMode,
        op::EXECUTION_MODE,
        &[main.word(), ExecutionMode::LocalSize.word(), 64, 1, 1],
    )?;

    Ok((module, main))
}

/// Close `module` off with an empty definition of `main`.
pub fn finish_empty_body(module: &mut Module, main: Id) -> Result<(), BuildError> {
    let void = module.type_void()?;
    let signature = module.type_function(void, &[])?;

    module.begin_function(void, main, FunctionControl::None, signature)?;
    module.label()?;
    module.return_void()?;
    module.end_function()
}
