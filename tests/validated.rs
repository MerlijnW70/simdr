mod common;

use common::{
    VULKAN_1_0, VULKAN_1_1, compute_skeleton, expect_valid, finish_empty_body, validate, validator,
};
use simdr::module::{Module, Section, Version, op};
use simdr::spec::{
    AddressingModel, Capability, ExecutionModel, FunctionControl, MemoryModel, Scope, StorageClass,
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
    let (mut module, main) = compute_skeleton(Version::V1_0).expect("built");

    let float = module.type_float(32).expect("f32");
    let first = module.type_struct(&[float]).expect("first struct");
    let second = module.type_struct(&[float]).expect("second struct");

    assert_ne!(first, second, "aggregates are not interned");

    finish_empty_body(&mut module, main).expect("built");

    expect_valid(&module.finish(), "twin-structs", VULKAN_1_0);
}

#[test]
fn the_three_subgroup_operations_nothing_else_emits_are_valid_spirv() {
    let (mut module, main) = compute_skeleton(Version::V1_3).expect("built");

    module
        .require_capability(Capability::GroupNonUniform)
        .expect("basic");
    module
        .require_capability(Capability::GroupNonUniformBallot)
        .expect("ballot");

    let boolean = module.type_bool().expect("bool");
    let float = module.type_float(32).expect("f32");
    let scope = module.scope(Scope::Subgroup).expect("scope");
    let value = module.constant_f32(1.5).expect("1.5");
    let lane = module.constant_u32(3).expect("3");

    let void = module.type_void().expect("void");
    let signature = module.type_function(void, &[]).expect("signature");
    module
        .begin_function(void, main, FunctionControl::None, signature)
        .expect("function");
    module.label().expect("entry");

    module.subgroup_elect(boolean, scope).expect("elect");
    module
        .subgroup_broadcast(float, scope, value, lane)
        .expect("broadcast");
    module
        .subgroup_broadcast_first(float, scope, value)
        .expect("broadcast first");

    module.return_void().expect("return");
    module.end_function().expect("end");

    expect_valid(&module.finish(), "subgroup-unreached", VULKAN_1_1);
}

#[test]
fn a_module_declaring_the_subgroup_capabilities_is_valid_at_spirv_1_3() {
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
