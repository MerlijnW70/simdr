//! Values that arrive at pipeline creation, validated.
//!
//! Split from `kernels.rs`, which validates kernels that compute something with everything already
//! decided. These leave a number open until `vkCreateComputePipeline` — a specialization constant,
//! or an address offset built from one — and that is a different question for a validator:
//! `OpSpecConstant` has to sit in the constants section, `OpSpecConstantOp` has to name an opcode
//! it is allowed to fold, and a `ClusterSize` that is one of them has to still count as a constant
//! instruction.
//!
//! `decisions/DR-0005` is what these were written to answer.

mod common;

use common::{VULKAN_1_1, expect_valid};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{F32, U32};

/// A 32-wide subgroup, 64 invocations, two buffers — the shape every kernel here shares.
fn shape() -> Shape {
    Shape::new(32, 64, 2)
}
#[test]
fn a_kernel_with_a_specialization_constant_is_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let raised = {
        let mut lanes = kernel.lanes().expect("lanes");
        let element = lanes.type_of::<U32>().expect("u32");
        let addend = lanes
            .module()
            .spec_constant(element, 1, 0)
            .expect("declared");
        let addend = lanes.splat_id::<U32, 32>(addend).expect("splat");
        lanes.add(value, addend).expect("added")
    };
    kernel.store(1, raised).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-spec",
        VULKAN_1_1,
    );
}

#[test]
fn a_derived_specialization_constant_is_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let raised = {
        let mut lanes = kernel.lanes().expect("lanes");
        let element = lanes.type_of::<U32>().expect("u32");
        let base = lanes
            .module()
            .spec_constant(element, 3, 0)
            .expect("declared");
        let two = lanes.module().constant_u32(2).expect("2");
        let doubled = lanes
            .module()
            .spec_constant_op(element, simdr::module::op::I_MUL, &[base, two])
            .expect("derived");
        let doubled = lanes.splat_id::<U32, 32>(doubled).expect("splat");
        lanes.add(value, doubled).expect("added")
    };
    kernel.store(1, raised).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-spec-derived",
        VULKAN_1_1,
    );
}

#[test]
fn a_cluster_size_that_is_a_specialization_constant_is_valid_spirv() {
    // The question `notes/NEXT.md` asked. `ClusterSize` must come from a *constant instruction*,
    // and `OpSpecConstant` is one — so the specification appears to permit this, and the validator
    // is the first of the two authorities that can be asked. The second is a device, in
    // `runner/tests/specialized.rs`, and it agrees.
    use simdr::module::{Reduction, op};
    use simdr::spec::{Capability, Scope};

    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let element = kernel.element();

    let total = {
        let module = kernel.module();
        module
            .require_capability(Capability::GroupNonUniformArithmetic)
            .expect("declared");
        module
            .require_capability(Capability::GroupNonUniformClustered)
            .expect("declared");
        let size = module.spec_constant(element, 8, 0).expect("cluster size");
        let scope = module.scope(Scope::Subgroup).expect("subgroup");
        module
            .subgroup_reduce(
                op::GROUP_NON_UNIFORM_I_ADD,
                element,
                scope,
                Reduction::Clustered { size },
                value.id(),
            )
            .expect("reduced")
    };
    kernel.store_scalar(1, total).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-spec-cluster",
        VULKAN_1_1,
    );
}

#[test]
fn a_kernel_over_a_64_wide_subgroup_is_valid_too() {
    // This crate has no device in it, so a 64-wide module is checked here and run in `runner`,
    // where the integrated Radeon reports that width. It was written before there was one.
    let mut kernel = Kernel::<F32>::new(Shape::new(64, 64, 2)).expect("built");
    let value = kernel.load::<16>(0).expect("loaded");
    let total = kernel
        .lanes()
        .expect("lanes")
        .reduce_sum(value)
        .expect("sum");
    kernel.store_scalar(1, total).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-wide-subgroup", VULKAN_1_1);
}

#[test]
fn a_boolean_specialization_constant_is_valid_spirv() {
    // **The one specialization shape nothing reached.** A sweep of the public surface for
    // operations with no consumer found `Module::spec_constant_bool`: it had a unit test in
    // `module/specialize.rs` and no validator behind it, which is the state `OpUDot` was in when
    // it turned out to be invalid SPIR-V.
    //
    // It is also the shape most likely to be wrong, because it is the one that is *not* like the
    // others: the default decides the **opcode** — `OpSpecConstantTrue` against
    // `OpSpecConstantFalse` — where every other specialization constant carries its default as an
    // operand. A `SpecId` decoration on the wrong kind of instruction is exactly the sort of thing
    // only a validator says anything about.
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let raised = {
        let mut lanes = kernel.lanes().expect("lanes");
        let element = lanes.type_of::<U32>().expect("u32");

        // Both opcodes in one module, so a mistake in either is visible and neither can pass by
        // being absent.
        let enabled = lanes.module().spec_constant_bool(true, 0).expect("true");
        let disabled = lanes.module().spec_constant_bool(false, 1).expect("false");

        let one = lanes.module().constant_u32(1).expect("1");
        let zero = lanes.module().constant_u32(0).expect("0");
        let chosen = lanes
            .module()
            .select(element, enabled, one, zero)
            .expect("chosen");
        let other = lanes
            .module()
            .select(element, disabled, one, zero)
            .expect("other");
        let addend = lanes.module().i_add(element, chosen, other).expect("sum");

        let addend = lanes.splat_id::<U32, 32>(addend).expect("splat");
        lanes.add(value, addend).expect("added")
    };
    kernel.store(1, raised).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-spec-bool",
        VULKAN_1_1,
    );
}
