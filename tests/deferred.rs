mod common;

use common::{VULKAN_1_1, expect_valid};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{F32, U32};

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
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let raised = {
        let mut lanes = kernel.lanes().expect("lanes");
        let element = lanes.type_of::<U32>().expect("u32");

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
