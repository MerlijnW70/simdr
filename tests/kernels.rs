//! Whole kernels, validated end to end.
//!
//! `validated.rs` checks structural shapes; this checks modules that would actually compute
//! something if a device ran them. Nothing here executes — validity is not correctness, and the
//! `runner` workspace member is what closes that gap.
//!
//! These used to hand-build sixty lines of buffer interface each, duplicating what
//! `runner/src/kernels/mod.rs` also had. [`Kernel`] owns that now, so what is left is the shape of
//! each kernel and the question of whether the validator accepts it.
//!
//! # What is next door
//!
//! This file grew to 848 lines and was cut along the questions its tests ask, not by size:
//!
//! - `deferred.rs` — kernels with a value left open until pipeline creation.
//! - `grids.rs` — kernels with a second axis.
//! - `instructions.rs` — the smallest module that reaches one instruction family, which is a
//!   different question from whether a whole kernel is valid, and the one that caught `OpUDot`.
//!
//! What is left here is kernels that compute something with everything already decided: the
//! reductions, the scan, the loop, every element type including the narrow ones, the extended
//! instruction set, the dot product and the atomics.

mod common;

use common::{VULKAN_1_1, expect_valid};
use simdr::half;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, F16, F32, I8, I16, I32, U8, U16, U32};

/// A 32-wide subgroup, 64 invocations, two buffers — the shape every kernel here shares.
fn shape() -> Shape {
    Shape::new(32, 64, 2)
}

#[test]
fn an_elementwise_kernel_is_valid_spirv() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let doubled = {
        let mut lanes = kernel.lanes().expect("lanes");
        let two = lanes.splat_bits::<F32, 32>(2.0_f32.to_bits()).expect("two");
        lanes.mul(value, two).expect("scaled")
    };
    kernel.store(1, doubled).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-elementwise", VULKAN_1_1);
}

#[test]
fn a_full_width_reduction_is_valid_spirv() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let total = kernel
        .lanes()
        .expect("lanes")
        .reduce_sum(value)
        .expect("sum");
    kernel.store_scalar(1, total).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-reduce", VULKAN_1_1);
}

#[test]
fn a_clustered_reduction_is_valid_spirv() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<8>(0).expect("loaded");
    let total = kernel
        .lanes()
        .expect("lanes")
        .reduce_sum(value)
        .expect("sum");
    kernel.store_scalar(1, total).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-clustered", VULKAN_1_1);
}

#[test]
fn a_strip_mined_reduction_is_valid_spirv() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<128>(0).expect("loaded");
    assert_eq!(value.strip_count(), 4);

    let total = kernel
        .lanes()
        .expect("lanes")
        .reduce_sum(value)
        .expect("sum");
    kernel.store_scalar(1, total).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-strips", VULKAN_1_1);
}

#[test]
fn a_maximum_reduction_is_valid_spirv() {
    // Its strip fold goes through compare-and-select rather than a core max opcode, so it is a
    // different instruction sequence from every other reduction here.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<64>(0).expect("loaded");
    let largest = kernel
        .lanes()
        .expect("lanes")
        .reduce_max(value)
        .expect("max");
    kernel.store_scalar(1, largest).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-max", VULKAN_1_1);
}

#[test]
fn a_prefix_sum_is_valid_spirv() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let running = kernel
        .lanes()
        .expect("lanes")
        .prefix_sum(value)
        .expect("scan");
    kernel.store(1, running).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-scan", VULKAN_1_1);
}

#[test]
fn the_three_operations_an_audit_found_unreachable_are_valid_spirv() {
    // **The check that has caught this class before.** `Lanes::dot_unsigned` emitted `OpUDot` with
    // a signed result type for a week, and it was a public method with no caller, no unit test and
    // no validator coverage — three layers, and it fell between all of them. An audit of the
    // public surface found three more in the same state: the atomic exchange, the atomic load, and
    // the vote about a value. Whether they were right was not known until this ran.
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let index = kernel.load::<32>(0).expect("loaded");
    let seven = kernel.module().constant_u32(7).expect("7");
    let slot = kernel.local_index();
    let out = kernel.element_pointer_to(1, slot).expect("pointer");

    let displaced = kernel
        .atomic_exchange_at(1, index.id(), seven)
        .expect("exchanged");
    let read = kernel.atomic_load_at(1, displaced).expect("read");

    // The vote drives a real branch, which is the whole of what it is for: `if_uniform` takes a
    // `Uniform`, and until `all_equal` there was no way to make one out of a *value*.
    let agreed = kernel
        .lanes()
        .expect("lanes")
        .all_equal_uniform(index)
        .expect("voted");
    kernel
        .lanes()
        .expect("lanes")
        .if_uniform(agreed, |lanes| Ok(lanes.module().store(out, read)?))
        .expect("branched");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-audit-three", VULKAN_1_1);
}

#[test]
fn a_clustered_scan_is_valid_spirv() {
    // **The one the validator has to see.** The clustered ladder reaches for
    // `SubgroupLocalInvocationId` while the body is being built, which is long after the entry
    // point was declared — and a built-in the body loads but the interface does not name is
    // exactly what `spirv-val` rejects and what every driver here runs anyway.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<8>(0).expect("loaded");
    let running = kernel
        .lanes()
        .expect("lanes")
        .prefix_sum(value)
        .expect("scan");
    kernel.store(1, running).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-scan-clusters", VULKAN_1_1);
}

#[test]
fn a_clustered_exclusive_scan_is_valid_spirv() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<8>(0).expect("loaded");
    let running = kernel
        .lanes()
        .expect("lanes")
        .prefix_sum_exclusive(value)
        .expect("scan");
    kernel.store(1, running).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-scan-clusters-exclusive", VULKAN_1_1);
}

#[test]
fn an_unrolled_loop_is_valid_spirv() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let total = {
        let mut lanes = kernel.lanes().expect("lanes");
        lanes
            .repeat(5, value.id(), |lanes, carried, step| {
                let held = lanes.from_lane_value::<F32, 32>(carried)?;
                let partner = lanes.butterfly(held, 1 << step)?;
                Ok(lanes.add(held, partner)?.id())
            })
            .expect("repeated")
    };
    kernel.store_scalar(1, total).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-unrolled", VULKAN_1_1);
}

#[test]
fn every_element_type_produces_a_valid_kernel() {
    // The three take different opcodes end to end, so each is its own module to validate.
    let mut floats = Kernel::<F32>::new(shape()).expect("built");
    let value = floats.load::<32>(0).expect("loaded");
    let total = floats
        .lanes()
        .expect("lanes")
        .reduce_sum(value)
        .expect("sum");
    floats.store_scalar(1, total).expect("stored");
    expect_valid(
        &floats.finish().expect("finished"),
        "kernel-f32",
        VULKAN_1_1,
    );

    let mut signed = Kernel::<I32>::new(shape()).expect("built");
    let value = signed.load::<32>(0).expect("loaded");
    let largest = signed
        .lanes()
        .expect("lanes")
        .reduce_max(value)
        .expect("max");
    signed.store_scalar(1, largest).expect("stored");
    expect_valid(
        &signed.finish().expect("finished"),
        "kernel-i32",
        VULKAN_1_1,
    );

    let mut unsigned = Kernel::<U32>::new(shape()).expect("built");
    let value = unsigned.load::<32>(0).expect("loaded");
    let largest = unsigned
        .lanes()
        .expect("lanes")
        .reduce_max(value)
        .expect("max");
    unsigned.store_scalar(1, largest).expect("stored");
    expect_valid(
        &unsigned.finish().expect("finished"),
        "kernel-u32",
        VULKAN_1_1,
    );
}

#[test]
fn a_kernel_calling_the_extended_instruction_set_is_valid_spirv() {
    // The import is the part a validator has an opinion about: `OpExtInst` naming an id that is
    // not an imported set, or an import placed outside its section, are both rejected here and
    // nowhere in the unit tests — those only read back what was emitted.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let shaped = {
        let mut lanes = kernel.lanes().expect("lanes");
        let low = lanes.splat_bits::<F32, 32>(0.0_f32.to_bits()).expect("low");
        let high = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("high");
        let bounded = lanes.clamp(value, low, high).expect("clamped");
        lanes.sqrt(bounded).expect("root")
    };
    kernel.store(1, shaped).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-extended", VULKAN_1_1);
}

#[test]
fn an_integer_clamp_and_magnitude_are_valid_spirv() {
    // A separate module from the float one: `SClamp` and `SAbs` are different instructions taking
    // a different result type, and a validator checks the type against the instruction.
    let mut kernel = Kernel::<I32>::new(shape()).expect("built");
    let value = kernel.load::<64>(0).expect("loaded");
    let bounded = {
        let mut lanes = kernel.lanes().expect("lanes");
        let magnitude = lanes.abs(value).expect("abs");
        let low = lanes.splat_bits::<I32, 64>(0).expect("low");
        let high = lanes.splat_bits::<I32, 64>(127).expect("high");
        lanes.clamp(magnitude, low, high).expect("clamped")
    };
    kernel.store(1, bounded).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-extended-integer", VULKAN_1_1);
}

/// Every narrow element type, elementwise, validated.
///
/// One module each rather than one shared: the capability, the stride and the constant's literal
/// all differ per type, and a single kernel would prove whichever one it happened to use.
#[test]
fn every_narrow_element_type_produces_a_valid_kernel() {
    fn elementwise<T: Element>(label: &str, constant: u32) {
        let mut kernel = Kernel::<T>::new(shape()).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");
        let doubled = {
            let mut lanes = kernel.lanes().expect("lanes");
            let addend = lanes.splat_bits::<T, 32>(constant).expect("constant");
            lanes.add(value, addend).expect("added")
        };
        kernel.store(1, doubled).expect("stored");

        expect_valid(&kernel.finish().expect("finished"), label, VULKAN_1_1);
    }

    elementwise::<I8>("kernel-i8", 0xff);
    elementwise::<U8>("kernel-u8", 3);
    elementwise::<I16>("kernel-i16", 0xffff);
    elementwise::<U16>("kernel-u16", 300);
    elementwise::<F16>("kernel-f16", u32::from(half::from_f32(1.5)));
}

#[test]
fn a_narrow_reduction_is_valid_spirv() {
    // The subgroup instructions over an 8-bit type. SPIR-V has no capability that says a device
    // supports this — Vulkan's `shaderSubgroupExtendedTypes` gates it and leaves no trace in the
    // module — so the validator accepting this says nothing about whether a device will run it,
    // and `runner/tests/narrow.rs` is where that question is asked.
    let mut kernel = Kernel::<I8>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let total = kernel
        .lanes()
        .expect("lanes")
        .reduce_sum(value)
        .expect("sum");
    kernel.store_scalar(1, total).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-i8-sum",
        VULKAN_1_1,
    );
}

#[test]
fn a_narrow_kernel_declares_the_stride_its_type_occupies() {
    // Not a validity question — a validator has no opinion about a stride of 4 on a buffer of
    // bytes, and a device reading one would return every fourth element. The decoration is the
    // whole of the memory-traffic claim, so it is asserted rather than assumed.
    use simdr::decode;
    use simdr::module::op;
    use simdr::spec::Decoration;

    fn stride_of<T: Element>() -> Option<u32> {
        let kernel = Kernel::<T>::new(shape()).expect("built");
        let words = kernel.finish().expect("finished");

        decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::DECORATE)
            .find(|instruction| {
                instruction.operands().get(1).copied() == Some(Decoration::ArrayStride.word())
            })
            .and_then(|instruction| instruction.operands().get(2).copied())
    }

    assert_eq!(stride_of::<I8>(), Some(1));
    assert_eq!(stride_of::<U8>(), Some(1));
    assert_eq!(stride_of::<I16>(), Some(2));
    assert_eq!(stride_of::<F16>(), Some(2));
    assert_eq!(
        stride_of::<F32>(),
        Some(4),
        "and the wide ones are unchanged"
    );
}

#[test]
fn a_narrow_conversion_from_a_loop_counter_is_valid_spirv() {
    // `OpSConvert` and `OpUConvert` differ only in the signedness of the result type, and the
    // validator is the only thing that checks the pairing. A signed kernel and an unsigned one
    // reach different opcodes from the same source line.
    fn counted<T: Element>(label: &str) {
        let mut kernel = Kernel::<T>::new(shape()).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");

        let total = {
            let mut lanes = kernel.lanes().expect("lanes");
            let element = lanes.type_of::<T>().expect("the element type");
            lanes
                .repeat_rolled(3, element, value.id(), |lanes, held, step| {
                    let converted = lanes.convert_u32::<T>(step)?;
                    let carried = lanes.from_lane_value::<T, 32>(held)?;
                    let addend = lanes.from_lane_value::<T, 32>(converted)?;
                    Ok(lanes.add(carried, addend)?.id())
                })
                .expect("looped")
        };
        kernel.store_scalar(1, total).expect("stored");

        expect_valid(&kernel.finish().expect("finished"), label, VULKAN_1_1);
    }

    counted::<I8>("kernel-i8-convert");
    counted::<U8>("kernel-u8-convert");
    counted::<F16>("kernel-f16-convert");
}

#[test]
fn an_integer_dot_product_is_valid_spirv() {
    // Two capabilities and an extension between them, and the validator is what says all three are
    // present. It is also the only thing that checks the result type is wide enough — four 8-bit
    // products do not fit in an 8-bit result and `OpSDot` says so.
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let packed = kernel.load::<32>(0).expect("loaded");
    let totals = {
        let mut lanes = kernel.lanes().expect("lanes");
        let products = lanes.dot_signed(packed, packed).expect("dot");
        lanes.reinterpret(products).expect("back to u32")
    };
    kernel.store(1, totals).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-dot",
        VULKAN_1_1,
    );
}

#[test]
fn a_saturating_dot_product_chain_is_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let packed = kernel.load::<32>(0).expect("loaded");
    let totals = {
        let mut lanes = kernel.lanes().expect("lanes");
        let zero = lanes.splat_bits::<simdr::lanes::I32, 32>(0).expect("zero");
        let first = lanes
            .dot_signed_saturating(packed, packed, zero)
            .expect("first");
        let second = lanes
            .dot_signed_saturating(packed, packed, first)
            .expect("second");
        lanes.reinterpret(second).expect("back to u32")
    };
    kernel.store(1, totals).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-dot-saturating",
        VULKAN_1_1,
    );
}

#[test]
fn an_atomic_scatter_is_valid_spirv() {
    // Where the validator earns its place here: the scope and the semantics are ids of constants,
    // and an atomic naming a literal where an id belongs is rejected for a type mismatch rather
    // than assembling into something else.
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let bin = {
        let mut lanes = kernel.lanes().expect("lanes");
        let ceiling = lanes.splat_bits::<U32, 32>(7).expect("7");
        lanes.min(value, ceiling).expect("clamped")
    };
    let one = kernel.module().constant_u32(1).expect("1");
    kernel.atomic_add_at(1, bin.id(), one).expect("scattered");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-atomic",
        VULKAN_1_1,
    );
}

#[test]
fn an_atomic_increment_is_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    kernel
        .atomic_increment_at(1, value.id())
        .expect("incremented");

    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-atomic-increment",
        VULKAN_1_1,
    );
}
