//! One instruction family at a time, validated.
//!
//! Split from `kernels.rs`, which builds kernels that compute something. These build the smallest
//! module that reaches a particular instruction, because a self-audit on 2026-08-12 asked which
//! public operations appear in no test that runs `spirv-val` and found **fifteen** — the whole
//! shuffle and vote surface among them.
//!
//! They had unit tests, which decode the module and agree it says what the emitter meant, and they
//! had device tests, which run it. Neither is a validator: a surplus capability declaration fails
//! at pipeline creation and a missing one may simply work on the driver that was tried.
//!
//! It found one on its first run. `Lanes::dot_unsigned` had been emitting `OpUDot` with a signed
//! result type — invalid SPIR-V in a shipped public method with no caller, no unit test of its own
//! and no validator coverage.
//!
//! `runner/src/kernels` has executable kernels for most of these and the emitter cannot reach them
//! — the dependency arrow points one way on purpose. So they are built again here, against the
//! only oracle this crate has of its own.

mod common;

use common::{VULKAN_1_1, expect_valid};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{F32, U32};

/// A 32-wide subgroup, 64 invocations, two buffers — the shape every kernel here shares.
fn shape() -> Shape {
    Shape::new(32, 64, 2)
}
// ---------------------------------------------------------------------------------------------
// The operations the validator had never seen
//
// A self-audit on 2026-08-12 asked which public lane and kernel operations appear in *no* test
// that runs `spirv-val`, and found fifteen — the whole shuffle and vote surface among them. They
// had unit tests, which decode the module and agree it says what the emitter meant, and they had
// device tests, which run it. Neither is a validator: a surplus capability declaration fails at
// pipeline creation and a missing one may simply work on the driver that was tried.
//
// `runner/src/kernels` has executable kernels for all of these, and the emitter cannot reach them
// — the dependency arrow points one way on purpose. So they are built again here, against the
// only oracle this crate has of its own.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_votes_and_the_ballot_are_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let low = {
        let mut lanes = kernel.lanes().expect("lanes");
        let limit = lanes.splat_bits::<U32, 32>(7).expect("limit");
        let above = lanes.greater_than(value, limit).expect("compared");

        // `all_uniform` as well as `any_uniform`: they are different opcodes and only one of them
        // had ever reached the validator.
        let every = lanes.all_uniform(above).expect("all");
        let mask = lanes.ballot(above).expect("ballot");

        // A ballot is a `uvec4`; a 32-wide subgroup fits in its first component.
        let uint = lanes.type_of::<U32>().expect("u32");
        let first = lanes
            .module()
            .composite_extract(uint, mask, &[0])
            .expect("component");

        let element = kernel.element();
        kernel
            .lanes()
            .expect("lanes")
            .choose_uniform(
                every,
                element,
                |lanes| lanes.from_lane_value::<U32, 1>(first).map(|v| v.id()),
                |_| Ok(first),
            )
            .expect("chosen")
    };

    kernel.store_scalar(1, low).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-votes",
        VULKAN_1_1,
    );
}

#[test]
fn the_shuffles_are_valid_spirv() {
    // Broadcast, shift-up and shift-down are three different opcodes and three different
    // capability requirements from the butterfly the loop test already covered.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let mixed = {
        let mut lanes = kernel.lanes().expect("lanes");
        let from_first = lanes.broadcast(value, 0).expect("broadcast");
        let earlier = lanes.shift_up(value, 1).expect("up");
        let later = lanes.shift_down(value, 1).expect("down");

        let pair = lanes.add(earlier, later).expect("added");
        lanes.add(pair, from_first).expect("added")
    };

    kernel.store(1, mixed).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-shuffles",
        VULKAN_1_1,
    );
}

#[test]
fn a_minimum_reduction_is_valid_spirv() {
    // Its strip fold goes through compare-and-select, like the maximum, and it is a different
    // group opcode.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<64>(0).expect("loaded");
    let smallest = kernel
        .lanes()
        .expect("lanes")
        .reduce_min(value)
        .expect("min");
    kernel.store_scalar(1, smallest).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-min",
        VULKAN_1_1,
    );
}

#[test]
fn the_unsigned_and_mixed_dot_products_are_valid_spirv() {
    // `OpSDot` was validated; `OpUDot` and `OpSUDot` were not. All three need the same extension
    // and capabilities, and a wrong signedness operand is exactly what a validator checks.
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let packed = kernel.load::<32>(0).expect("loaded");

    let totals = {
        let mut lanes = kernel.lanes().expect("lanes");
        // `OpUDot` yields an unsigned sum and `OpSUDot` a signed one, so the two are brought to
        // the same type before they are added. That difference is the bug this test found.
        let unsigned = lanes.dot_unsigned(packed, packed).expect("udot");
        let mixed = lanes.dot_mixed(packed, packed).expect("sudot");
        let mixed = lanes.reinterpret(mixed).expect("bits");
        lanes.add(unsigned, mixed).expect("added")
    };

    kernel.store(1, totals).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-dots",
        VULKAN_1_1,
    );
}

#[test]
fn the_shifts_are_valid_spirv() {
    // The two right shifts differ only on values with the top bit set, so a validator run is the
    // cheapest thing that distinguishes the instructions at all.
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let shifted = {
        let mut lanes = kernel.lanes().expect("lanes");
        let by = lanes.splat_bits::<U32, 32>(3).expect("three");

        let up = lanes.shift_left(value, by).expect("left");
        let down = lanes.shift_right_logical(up, by).expect("logical");
        let signed = lanes.reinterpret_unsigned(down).expect("as signed");
        let arithmetic = lanes
            .shift_right_arithmetic(signed, by)
            .expect("arithmetic");
        lanes.reinterpret(arithmetic).expect("back")
    };

    kernel.store(1, shifted).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-shifts",
        VULKAN_1_1,
    );
}

#[test]
fn the_rest_of_the_extended_set_is_valid_spirv() {
    // `sqrt` and `clamp` were covered; `log`, `fma`, `inverse_sqrt` and `max` were not. Each is a
    // different instruction number in GLSL.std.450 and `fma` takes three operands rather than one.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let shaped = {
        let mut lanes = kernel.lanes().expect("lanes");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        let safe = lanes.max(value, one).expect("max");
        let logged = lanes.log(safe).expect("log");
        let inverted = lanes.inverse_sqrt(safe).expect("rsqrt");
        lanes.fma(logged, inverted, one).expect("fma")
    };

    kernel.store(1, shaped).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-extended-rest",
        VULKAN_1_1,
    );
}

#[test]
fn an_offset_arriving_at_pipeline_time_is_valid_spirv() {
    // `load_offset_by` builds a different address expression from `load_offset` — one `OpIAdd` per
    // strip against a folded constant — and the offset is a specialization constant, which is the
    // combination `kernels::fold_halves_open` uses and nothing here had validated.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let uint = kernel.index_type();
    let offset = kernel
        .module()
        .spec_constant(uint, 0, 0)
        .expect("spec constant");

    let near = kernel.load::<32>(0).expect("loaded");
    let far = kernel.load_offset_by::<32>(0, offset).expect("loaded");
    let folded = kernel
        .lanes()
        .expect("lanes")
        .add(near, far)
        .expect("added");
    kernel.store(1, folded).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-open-offset",
        VULKAN_1_1,
    );
}

#[test]
fn a_write_to_a_slot_the_data_chose_is_valid_spirv() {
    // The escape hatch under `atomic_add_at`, used on its own: an access chain whose index is a
    // loaded value rather than the invocation index.
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let pointer = kernel.element_pointer_to(1, value.id()).expect("pointer");
    kernel.module().store(pointer, value.id()).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-scatter-pointer",
        VULKAN_1_1,
    );
}
