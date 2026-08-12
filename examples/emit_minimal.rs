//! Emit the minimal compute module and write it to the path given as the first argument.
//!
//! This exists to be fed to `spirv-val`. It is the shortest path from "the tests are green" to
//! "Khronos agrees", and those are very different claims.

use simdr::encode;
use simdr::module::{Module, Section, Version, op};
use simdr::spec::{
    AddressingModel, Capability, ExecutionMode, ExecutionModel, FunctionControl, MemoryModel,
};
use std::io::Write as _;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut module = Module::new(Version::V1_0);

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
        &[main.word(), ExecutionMode::LocalSize.word(), 1, 1, 1],
    )?;

    let void = module.type_void()?;
    let signature = module.type_function(void, &[])?;

    module.emit(
        Section::Function,
        op::FUNCTION,
        &[
            void.word(),
            main.word(),
            FunctionControl::None.word(),
            signature.word(),
        ],
    )?;
    let entry_block = module.alloc_id()?;
    module.emit(Section::Function, op::LABEL, &[entry_block.word()])?;
    module.emit(Section::Function, op::RETURN, &[])?;
    module.emit(Section::Function, op::FUNCTION_END, &[])?;

    let path = std::env::args()
        .nth(1)
        .ok_or("usage: emit_minimal <path>")?;
    let bytes = module.to_bytes();
    std::fs::File::create(&path)?.write_all(&bytes)?;

    eprintln!("wrote {} bytes to {path}", bytes.len());
    Ok(())
}
