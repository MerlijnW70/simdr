//! The SPIR-V vocabulary: which word means `Shader`, which means `StorageBuffer`.
//!
//! Every number here was read out of Khronos' machine-readable grammar
//! (`spirv.core.grammar.json`, 1.6.7) rather than transcribed from prose — see
//! `decisions/DR-0001` for why, and for how to re-check them. A wrong number produces a module
//! that assembles cleanly and means something else, which is the most expensive kind of bug this
//! crate can have.
//!
//! [`Glsl`] is the exception and reads from a second grammar,
//! `extinst.glsl.std.450.grammar.json` — an extended instruction set numbers its instructions in
//! its own space, so the two files disagree about what any given number means.
//!
//! Split by what each group of values is *for* rather than alphabetically, and each file carries
//! the test that pins its own numbers. One list of a hundred assertions told a reader nothing
//! about which of them mattered where.

mod capability;
mod control;
mod execution;
mod extended;
mod group;
mod memory;
mod packed;

pub use self::capability::Capability;
pub use self::control::{LoopControl, SelectionControl};
pub use self::execution::{
    AddressingModel, ExecutionMode, ExecutionModel, FunctionControl, MemoryModel,
};
pub use self::extended::Glsl;
pub use self::group::{GroupOperation, Scope};
pub use self::memory::{BuiltIn, Decoration, MemorySemantics, StorageClass};
pub use self::packed::PackedVectorFormat;
