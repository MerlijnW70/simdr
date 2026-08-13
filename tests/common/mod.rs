//! Shared harness for the emitter's integration tests: the module skeletons they start from.
//!
//! Not a test binary of its own — Cargo compiles `tests/*.rs` as separate binaries and leaves
//! directories alone, so each of those declares `mod common;` and gets its own copy. That is also
//! why the allow below is needed: a helper only one of them uses is dead code in the other.
//!
//! The half that runs `spirv-val` lives in `spirv_val.rs` beside this, because `runner`'s tests
//! need it too and cannot reach anything in here — this file builds `simdr` modules, and `runner`
//! validating its own kernel library has no use for a skeleton.

#![allow(
    dead_code,
    unused_imports,
    reason = "each test binary compiles this file and uses a different subset of it — and a               re-export nobody in *this* binary names is an unused import rather than dead code,               which is a second lint saying the same thing about the same arrangement"
)]

// Named for the tool rather than for what it does, because `validator()` is one of the functions
// it exports and a module of the same name shadows it at every use site.
mod spirv_val;
pub use spirv_val::{VULKAN_1_0, VULKAN_1_1, expect_valid, validate, validator};

use simdr::encode;
use simdr::module::{BuildError, Id, Module, Section, Version, op};
use simdr::spec::{
    AddressingModel, Capability, ExecutionMode, ExecutionModel, FunctionControl, MemoryModel,
};

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
