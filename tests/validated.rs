//! Khronos' validator, run against the structural shapes this crate emits.
//!
//! Everything in `src/` checks that the encoder does what *we* think SPIR-V is. This file and
//! `kernels.rs` are the only things that check we are right. They shell out to `spirv-val`, an
//! external tool and deliberately not a dependency — the same reason a conformance corpus is not
//! a crate.
//!
//! Point `SPIRV_VAL` at the binary, or install it where `common::validator` looks. When neither
//! finds it the tests report themselves as skipped rather than passing quietly: a validator you
//! can silently not run is not a gate.

mod common;

use common::{
    VULKAN_1_0, VULKAN_1_1, compute_skeleton, expect_valid, finish_empty_body, validate, validator,
};
use simdr::module::{Module, Section, Version, op};
use simdr::spec::{
    AddressingModel, Capability, ExecutionModel, FunctionControl, MemoryModel, StorageClass,
};

#[test]
fn the_minimal_compute_module_is_valid_spirv() {
    let (mut module, main) = compute_skeleton(Version::V1_0).expect("built");
    finish_empty_body(&mut module, main).expect("built");

    expect_valid(&module.finish(), "minimal", VULKAN_1_0);
}

#[test]
fn a_module_carrying_every_scalar_type_and_constant_is_valid() {
    let (mut module, main) = compute_skeleton(Version::V1_0).expect("built");

    module.constant_u32(7).expect("7u32");
    module.constant_i32(-1).expect("-1i32");
    module.constant_f32(1.5).expect("1.5f32");
    module.constant_bool(true).expect("true");
    module.constant_bool(false).expect("false");

    let float = module.type_float(32).expect("f32");
    module.type_vector(float, 4).expect("vec4");
    module
        .type_pointer(StorageClass::Workgroup, float)
        .expect("pointer");

    finish_empty_body(&mut module, main).expect("built");

    expect_valid(&module.finish(), "scalars", VULKAN_1_0);
}

#[test]
fn asking_for_the_same_type_repeatedly_still_yields_a_valid_module() {
    // The dedup exists because a duplicate declaration is a *validation* failure, not a size
    // problem. This is the test that says so: ask twenty times, and let spirv-val judge.
    let (mut module, main) = compute_skeleton(Version::V1_0).expect("built");

    for _ in 0..20 {
        let float = module.type_float(32).expect("f32");
        module.type_vector(float, 4).expect("vec4");
        module.constant_f32(0.0).expect("0.0");
    }

    finish_empty_body(&mut module, main).expect("built");

    expect_valid(&module.finish(), "deduped", VULKAN_1_0);
}

#[test]
fn two_structurally_identical_structs_are_two_valid_types() {
    // The other half of the dedup rule: §2.8 makes aggregates the exception, so these must *not*
    // be merged, and a module carrying both must still validate.
    let (mut module, main) = compute_skeleton(Version::V1_0).expect("built");

    let float = module.type_float(32).expect("f32");
    let first = module.type_struct(&[float]).expect("first struct");
    let second = module.type_struct(&[float]).expect("second struct");

    assert_ne!(first, second, "aggregates are not interned");

    finish_empty_body(&mut module, main).expect("built");

    expect_valid(&module.finish(), "twin-structs", VULKAN_1_0);
}

#[test]
fn a_module_declaring_the_subgroup_capabilities_is_valid_at_spirv_1_3() {
    // Nothing uses them here. This pins that the capability words themselves are right, which is
    // the half of DR-0001 that reading the grammar cannot cover: the grammar says what the number
    // is, and only the validator says the number is accepted where we put it.
    let (mut module, main) = compute_skeleton(Version::V1_3).expect("built");

    for capability in [
        Capability::GroupNonUniform,
        Capability::GroupNonUniformVote,
        Capability::GroupNonUniformArithmetic,
        Capability::GroupNonUniformBallot,
        Capability::GroupNonUniformShuffle,
        Capability::GroupNonUniformShuffleRelative,
        Capability::GroupNonUniformClustered,
    ] {
        module
            .emit(Section::Capability, op::CAPABILITY, &[capability.word()])
            .expect("fits");
    }

    finish_empty_body(&mut module, main).expect("built");

    expect_valid(&module.finish(), "subgroup-capabilities", VULKAN_1_1);
}

/// The gate's own test: a module that *should* be refused, is.
///
/// Without this, every other test here could be passing because `validate` never returns `Err` —
/// a green suite that proves the harness runs, not that the modules are good. §2.16.2 requires a
/// `GLCompute` entry point to declare `LocalSize`, so leaving it out is a failure the validator
/// must catch and we can predict.
#[test]
fn a_compute_entry_point_without_a_workgroup_size_is_refused() {
    if validator().is_none() {
        eprintln!("SKIPPED teeth: spirv-val not found (set SPIRV_VAL)");
        return;
    }

    let mut module = Module::new(Version::V1_0);
    let main = module.alloc_id().expect("%1");

    module
        .emit(
            Section::Capability,
            op::CAPABILITY,
            &[Capability::Shader.word()],
        )
        .expect("fits");
    module
        .emit(
            Section::MemoryModel,
            op::MEMORY_MODEL,
            &[AddressingModel::Logical.word(), MemoryModel::Glsl450.word()],
        )
        .expect("fits");

    module
        .entry_point(ExecutionModel::GlCompute, main, "main")
        .expect("fits");

    // The OpExecutionMode that would go here is deliberately absent.

    let void = module.type_void().expect("void");
    let signature = module.type_function(void, &[]).expect("signature");
    module
        .begin_function(void, main, FunctionControl::None, signature)
        .expect("fits");
    module.label().expect("fits");
    module.return_void().expect("fits");
    module.end_function().expect("fits");

    assert!(
        validate(&module.finish(), "teeth", VULKAN_1_0).is_err(),
        "spirv-val accepted a compute entry point with no LocalSize, so this gate proves nothing"
    );
}
