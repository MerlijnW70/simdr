#![allow(
    dead_code,
    unused_imports,
    reason = "each test binary compiles this file and uses a different subset of it — and a               re-export nobody in *this* binary names is an unused import rather than dead code,               which is a second lint saying the same thing about the same arrangement"
)]

mod spirv_val;
pub use spirv_val::{VULKAN_1_0, VULKAN_1_1, expect_valid, validate, validator};

use simdr::encode;
use simdr::module::{BuildError, Id, Module, Section, Version, op};
use simdr::spec::{
    AddressingModel, Capability, ExecutionMode, ExecutionModel, FunctionControl, MemoryModel,
};

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

    module.entry_point(ExecutionModel::GlCompute, main, "main")?;

    module.emit(
        Section::ExecutionMode,
        op::EXECUTION_MODE,
        &[main.word(), ExecutionMode::LocalSize.word(), 64, 1, 1],
    )?;

    Ok((module, main))
}

pub fn finish_empty_body(module: &mut Module, main: Id) -> Result<(), BuildError> {
    let void = module.type_void()?;
    let signature = module.type_function(void, &[])?;

    module.begin_function(void, main, FunctionControl::None, signature)?;
    module.label()?;
    module.return_void()?;
    module.end_function()
}
