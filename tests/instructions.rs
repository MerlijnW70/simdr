mod common;

use common::{VULKAN_1_1, expect_valid, validate, validator};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, F32, I8, I16, I32, Integer, Lanes, U8, U16, U32};
use simdr::module::op;

fn shape() -> Shape {
    Shape::new(32, 64, 2)
}

#[test]
fn the_votes_and_the_ballot_are_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let low = {
        let mut lanes = kernel.lanes().expect("lanes");
        let limit = lanes.splat_bits::<U32, 32>(7).expect("limit");
        let above = lanes.greater_than(value, limit).expect("compared");

        let every = lanes.all_uniform(above).expect("all");
        let mask = lanes.ballot(above).expect("ballot");

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
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let packed = kernel.load::<32>(0).expect("loaded");

    let totals = {
        let mut lanes = kernel.lanes().expect("lanes");
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
fn the_float_to_integer_conversions_are_valid_spirv() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let counted = {
        let mut lanes = kernel.lanes().expect("lanes");
        let low = lanes
            .splat_bits::<F32, 32>(0.0_f32.to_bits())
            .expect("nought");
        let high = lanes
            .splat_bits::<F32, 32>(255.0_f32.to_bits())
            .expect("full");

        let bounded = lanes.clamp(value, low, high).expect("clamp");
        let unsigned = lanes.to_u32(bounded).expect("to u32");
        let signed = lanes.to_i32(bounded).expect("to i32");
        let same = lanes.reinterpret(signed).expect("as bits");
        let difference = lanes.xor(unsigned, same).expect("xor");
        let signed = lanes.reinterpret_unsigned(difference).expect("as signed");
        lanes.to_f32(signed).expect("to f32")
    };

    kernel.store(1, counted).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-float-to-integer",
        VULKAN_1_1,
    );
}

#[test]
fn the_rest_of_the_extended_set_is_valid_spirv() {
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

#[test]
fn the_operations_a_second_sweep_found_unreached_are_valid_spirv() {
    use simdr::spec::{MemorySemantics, Scope};

    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let shaped = {
        let mut lanes = kernel.lanes().expect("lanes");
        let curved = lanes.exp(value).expect("exp");

        let limit = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("limit");
        let over = lanes.greater_than(curved, limit).expect("compared");
        let vote = lanes.any_uniform(over).expect("voted");

        let element = lanes.type_of::<F32>().expect("f32");
        let otherwise = lanes.reduce_sum(curved).expect("sum");
        lanes
            .if_uniform_value(vote, element, otherwise, |lanes| lanes.reduce_max(curved))
            .expect("chose")
    };

    let slot = kernel.local_index();
    let pointer = kernel.element_pointer_to(1, slot).expect("pointer");
    let scope = kernel.module().scope(Scope::Device).expect("scope");
    let relaxed = kernel
        .module()
        .memory_semantics(MemorySemantics::None)
        .expect("relaxed");
    let ordered = kernel
        .module()
        .memory_semantics(MemorySemantics::AcquireReleaseBuffer)
        .expect("acquire-release");

    kernel
        .module()
        .atomic_store(pointer, scope, relaxed, shaped)
        .expect("stored atomically");
    kernel
        .module()
        .memory_barrier(scope, ordered)
        .expect("ordered");

    expect_valid(
        &kernel.finish().expect("finished"),
        "instructions-unreached",
        VULKAN_1_1,
    );
}

#[test]
fn a_memory_barrier_that_orders_nothing_is_refused() {
    if validator().is_none() {
        eprintln!("SKIPPED barrier-teeth: spirv-val not found (set SPIRV_VAL)");
        return;
    }

    use simdr::spec::{MemorySemantics, Scope};

    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let scope = kernel.module().scope(Scope::Device).expect("scope");
    let relaxed = kernel
        .module()
        .memory_semantics(MemorySemantics::None)
        .expect("relaxed");
    kernel
        .module()
        .memory_barrier(scope, relaxed)
        .expect("emitted");

    let words = kernel.finish().expect("finished");
    let outcome = validate(&words, "barrier-relaxed", VULKAN_1_1);

    let message = outcome.expect_err("a relaxed memory barrier is not valid SPIR-V");
    assert!(
        message.contains("MemoryBarrier"),
        "refused for something other than the barrier: {message}"
    );
}

fn shifts_are_valid_for<T: Integer>(name: &str) {
    let mut kernel = Kernel::<T>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let shifted = {
        let mut lanes = kernel.lanes().expect("lanes");
        let by = lanes.splat_bits::<U32, 32>(3).expect("three");

        let up = lanes.shift_left(value, by).expect("left");
        let down = lanes.shift_right_logical(up, by).expect("logical");
        lanes.shift_right_arithmetic(down, by).expect("arithmetic")
    };

    kernel.store(1, shifted).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        &format!("kernel-shifts-{name}"),
        VULKAN_1_1,
    );
}

#[test]
fn the_shifts_are_valid_for_every_integer_they_accept() {
    shifts_are_valid_for::<U32>("u32");
    shifts_are_valid_for::<I32>("i32");
    shifts_are_valid_for::<U8>("u8");
    shifts_are_valid_for::<I8>("i8");
    shifts_are_valid_for::<U16>("u16");
    shifts_are_valid_for::<I16>("i16");
}

fn the_lane_surface_is_valid_for<T: Element, const LANES: u32>(name: &str, width: u32) {
    let mut kernel = Kernel::<T>::new(Shape::new(width, 64, 2)).expect("built");
    let value = kernel.load::<LANES>(0).expect("loaded");

    let (folded, scanned) = {
        let mut lanes = kernel.lanes().expect("lanes");
        let one = lanes.splat_bits::<T, LANES>(1).expect("one");
        let two = lanes.splat_bits::<T, LANES>(2).expect("two");

        let sum = lanes.add(value, one).expect("add");
        let product = lanes.mul(sum, two).expect("mul");
        let smaller = lanes.min(product, two).expect("min");
        let larger = lanes.max(smaller, one).expect("max");
        let bounded = lanes.clamp(larger, one, two).expect("clamp");

        let above = lanes.greater_than(bounded, one).expect("greater");
        let picked = lanes.select(above, bounded, one).expect("select");
        let same = lanes.equal(picked, one).expect("equal");
        let either = lanes.select(same, picked, two).expect("select again");

        let partner = lanes.butterfly(either, 1).expect("butterfly");
        let first = lanes.broadcast(partner, 0).expect("broadcast");
        let up = lanes.shift_up(first, 1).expect("shift up");
        let down = lanes.shift_down(up, 1).expect("shift down");
        let rotated = lanes.rotate_up(down, 1).expect("rotate");

        let scanned = lanes.prefix_sum(rotated).expect("scan");
        let total = lanes.reduce_sum(scanned).expect("sum");
        let biggest = lanes.reduce_max(scanned).expect("max");
        let smallest = lanes.reduce_min(scanned).expect("min");

        let total = lanes.from_lane_value::<T, LANES>(total).expect("as vector");
        let biggest = lanes
            .from_lane_value::<T, LANES>(biggest)
            .expect("as vector");
        let smallest = lanes
            .from_lane_value::<T, LANES>(smallest)
            .expect("as vector");

        let folded = lanes.add(total, biggest).expect("add");
        let folded = lanes.add(folded, smallest).expect("add");
        (folded, scanned)
    };

    let combined = kernel
        .lanes()
        .expect("lanes")
        .add(folded, scanned)
        .expect("add");
    kernel.store(1, combined).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        &format!("kernel-surface-{name}-{width}"),
        VULKAN_1_1,
    );
}

fn the_clustered_surface_is_valid_for<T: Element, const LANES: u32>(name: &str, width: u32) {
    let mut kernel = Kernel::<T>::new(Shape::new(width, 64, 2)).expect("built");
    let value = kernel.load::<LANES>(0).expect("loaded");

    let folded = {
        let mut lanes = kernel.lanes().expect("lanes");

        let partner = lanes.butterfly(value, 1).expect("butterfly");
        let first = lanes.broadcast(partner, 0).expect("broadcast");
        let rotated = lanes.rotate_up(first, 1).expect("rotate");
        let scanned = lanes.prefix_sum(rotated).expect("scan");

        let total = lanes.reduce_sum(scanned).expect("sum");
        let biggest = lanes.reduce_max(scanned).expect("max");
        let smallest = lanes.reduce_min(scanned).expect("min");

        let total = lanes.from_lane_value::<T, LANES>(total).expect("as vector");
        let biggest = lanes
            .from_lane_value::<T, LANES>(biggest)
            .expect("as vector");
        let smallest = lanes
            .from_lane_value::<T, LANES>(smallest)
            .expect("as vector");

        let folded = lanes.add(total, biggest).expect("add");
        let folded = lanes.add(folded, smallest).expect("add");
        lanes.add(folded, scanned).expect("add")
    };

    kernel.store(1, folded).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        &format!("kernel-cluster-{name}-{width}"),
        VULKAN_1_1,
    );
}

fn the_stripped_surface_is_valid_for<T: Element, const LANES: u32>(name: &str, width: u32) {
    let mut kernel = Kernel::<T>::new(Shape::new(width, 64, 2)).expect("built");
    let value = kernel.load::<LANES>(0).expect("loaded");

    let folded = {
        let mut lanes = kernel.lanes().expect("lanes");
        let one = lanes.splat_bits::<T, LANES>(1).expect("one");

        let partner = lanes.butterfly(value, 1).expect("butterfly");
        let up = lanes.shift_up(partner, 1).expect("shift up");
        let down = lanes.shift_down(up, 1).expect("shift down");
        let scanned = lanes.prefix_sum(down).expect("scan");

        let _ = lanes.all_equal(scanned).expect("all equal");

        let total = lanes.reduce_sum(scanned).expect("sum");
        let total = lanes.splat_id::<T, LANES>(total).expect("as vector");

        let folded = lanes.add(total, one).expect("add");
        let folded = lanes.add(folded, scanned).expect("add");
        let above = lanes.greater_than(folded, one).expect("greater");
        lanes.select(above, folded, total).expect("select")
    };

    kernel.store(1, folded).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        &format!("kernel-strip-{name}-{width}"),
        VULKAN_1_1,
    );
}

macro_rules! at_every_width {
    ($type:ty, $name:literal) => {
        the_lane_surface_is_valid_for::<$type, 4>($name, 4);
        the_lane_surface_is_valid_for::<$type, 8>($name, 8);
        the_lane_surface_is_valid_for::<$type, 16>($name, 16);
        the_lane_surface_is_valid_for::<$type, 32>($name, 32);
        the_lane_surface_is_valid_for::<$type, 64>($name, 64);

        the_clustered_surface_is_valid_for::<$type, 4>($name, 8);
        the_clustered_surface_is_valid_for::<$type, 8>($name, 16);
        the_clustered_surface_is_valid_for::<$type, 16>($name, 32);
        the_clustered_surface_is_valid_for::<$type, 32>($name, 64);

        the_stripped_surface_is_valid_for::<$type, 8>($name, 4);
        the_stripped_surface_is_valid_for::<$type, 16>($name, 8);
        the_stripped_surface_is_valid_for::<$type, 32>($name, 16);
        the_stripped_surface_is_valid_for::<$type, 64>($name, 32);
    };
}

#[test]
fn the_lane_surface_is_valid_for_every_narrow_integer() {
    at_every_width!(U32, "u32");
    at_every_width!(U8, "u8");
    at_every_width!(I8, "i8");
    at_every_width!(U16, "u16");
    at_every_width!(I16, "i16");
}

#[test]
fn subtract_divide_and_negate_are_valid_spirv() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let element = kernel.element();
    let held = kernel.load::<32>(0).expect("loaded");
    let value = held.id();

    let one = F32::constant_from_bits(kernel.module(), 1.0_f32.to_bits()).expect("one");
    let flipped = kernel.module().f_negate(element, value).expect("negated");
    let raised = {
        let mut lanes = kernel.lanes().expect("lanes");
        let vector = lanes
            .from_lane_value::<F32, 32>(flipped)
            .expect("a lane value");
        lanes.exp::<32>(vector).expect("exp").id()
    };
    let below = kernel.module().f_add(element, one, raised).expect("added");
    let gated = kernel
        .module()
        .f_div(element, value, below)
        .expect("divided");
    let left = kernel
        .module()
        .f_sub(element, value, gated)
        .expect("subtracted");

    kernel
        .lanes()
        .expect("lanes")
        .from_lane_value::<F32, 32>(left)
        .and_then(|vector| kernel.store::<32>(1, vector))
        .expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "subtract, divide and negate", VULKAN_1_1);
}

#[test]
fn integer_subtract_and_divide_are_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let uint = kernel.index_type();
    let held = kernel.load::<32>(0).expect("loaded");
    let value = held.id();

    let by = kernel.module().constant_u32(7).expect("a divisor");
    let over = kernel.module().u_div(uint, value, by).expect("divided");
    let back = kernel.module().i_mul(uint, over, by).expect("multiplied");
    let left = kernel
        .module()
        .i_sub(uint, value, back)
        .expect("subtracted");

    kernel
        .lanes()
        .expect("lanes")
        .from_lane_value::<U32, 32>(left)
        .and_then(|vector| kernel.store::<32>(1, vector))
        .expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "integer subtract and divide", VULKAN_1_1);
}

#[test]
fn the_elementwise_arithmetic_is_valid_spirv_over_floats() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let total = {
        let mut lanes = kernel.lanes().expect("lanes");
        let two = lanes.splat_bits::<F32, 32>(2.0_f32.to_bits()).expect("two");

        let difference = lanes.sub(value, two).expect("sub");
        let quotient = lanes.div(difference, two).expect("div");
        let negated = lanes.neg(quotient).expect("neg");

        lanes.reduce_sum(negated).expect("summed")
    };

    kernel.store_scalar(1, total).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-arithmetic-f32",
        VULKAN_1_1,
    );
}

#[test]
fn the_elementwise_arithmetic_is_valid_spirv_over_both_integer_families() {
    let signed = {
        let mut kernel = Kernel::<I32>::new(shape()).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");

        let total = {
            let mut lanes = kernel.lanes().expect("lanes");
            let three = lanes.splat_bits::<I32, 32>(3).expect("three");

            let difference = lanes.sub(value, three).expect("sub");
            let quotient = lanes.div(difference, three).expect("div");
            let negated = lanes.neg(quotient).expect("neg");

            lanes.reduce_sum(negated).expect("summed")
        };

        kernel.store_scalar(1, total).expect("stored");
        kernel.finish().expect("finished")
    };

    let unsigned = {
        let mut kernel = Kernel::<U32>::new(shape()).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");

        let total = {
            let mut lanes = kernel.lanes().expect("lanes");
            let three = lanes.splat_bits::<U32, 32>(3).expect("three");

            let difference = lanes.sub(value, three).expect("sub");
            let quotient = lanes.div(difference, three).expect("div");

            lanes.reduce_sum(quotient).expect("summed")
        };

        kernel.store_scalar(1, total).expect("stored");
        kernel.finish().expect("finished")
    };

    expect_valid(&signed, "kernel-arithmetic-i32", VULKAN_1_1);
    expect_valid(&unsigned, "kernel-arithmetic-u32", VULKAN_1_1);
}

#[test]
fn the_whole_comparison_set_is_valid_spirv_and_selects_on_every_one() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let picked = {
        let mut lanes = kernel.lanes().expect("lanes");
        let one = lanes.splat_bits::<F32, 32>(1.0_f32.to_bits()).expect("one");

        let mut kept = value;
        for compare in [
            Lanes::less_than::<F32, 32>,
            Lanes::less_equal::<F32, 32>,
            Lanes::greater_than::<F32, 32>,
            Lanes::greater_equal::<F32, 32>,
            Lanes::equal::<F32, 32>,
            Lanes::not_equal::<F32, 32>,
        ] {
            let predicate = compare(&mut lanes, kept, one).expect("compared");
            kept = lanes.select(predicate, kept, one).expect("selected");
        }

        lanes.reduce_sum(kept).expect("summed")
    };

    kernel.store_scalar(1, picked).expect("stored");
    expect_valid(
        &kernel.finish().expect("finished"),
        "kernel-comparisons",
        VULKAN_1_1,
    );
}

#[test]
fn the_bitwise_family_is_valid_spirv_over_both_integer_families() {
    let signed = {
        let mut kernel = Kernel::<I32>::new(shape()).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");

        let total = {
            let mut lanes = kernel.lanes().expect("lanes");
            let mask = lanes.splat_bits::<I32, 32>(0b1010).expect("mask");

            let kept = lanes.and(value, mask).expect("and");
            let joined = lanes.or(kept, mask).expect("or");
            let flipped = lanes.xor(joined, mask).expect("xor");
            let complemented = lanes.not(flipped).expect("not");

            lanes.reduce_sum(complemented).expect("summed")
        };

        kernel.store_scalar(1, total).expect("stored");
        kernel.finish().expect("finished")
    };

    let unsigned = {
        let mut kernel = Kernel::<U32>::new(shape()).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");

        let total = {
            let mut lanes = kernel.lanes().expect("lanes");
            let mask = lanes.splat_bits::<U32, 32>(0b1010).expect("mask");

            let kept = lanes.and(value, mask).expect("and");
            let joined = lanes.or(kept, mask).expect("or");
            let flipped = lanes.xor(joined, mask).expect("xor");
            let complemented = lanes.not(flipped).expect("not");

            lanes.reduce_sum(complemented).expect("summed")
        };

        kernel.store_scalar(1, total).expect("stored");
        kernel.finish().expect("finished")
    };

    expect_valid(&signed, "kernel-bitwise-i32", VULKAN_1_1);
    expect_valid(&unsigned, "kernel-bitwise-u32", VULKAN_1_1);
}

#[test]
fn the_bitwise_reductions_and_the_product_are_valid_spirv() {
    let unsigned = {
        let mut kernel = Kernel::<U32>::new(shape()).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");

        let total = {
            let mut lanes = kernel.lanes().expect("lanes");
            let all = lanes.reduce_and(value).expect("and");
            let any = lanes.reduce_or(value).expect("or");
            let parity = lanes.reduce_xor(value).expect("xor");
            let product = lanes.reduce_product(value).expect("product");

            let uint = lanes.type_of::<U32>().expect("u32");
            let mut running = all;
            for next in [any, parity, product] {
                running = lanes
                    .module()
                    .binary(op::BITWISE_XOR, uint, running, next)
                    .expect("combined");
            }
            running
        };

        kernel.store_scalar(1, total).expect("stored");
        kernel.finish().expect("finished")
    };

    let float = {
        let mut kernel = Kernel::<F32>::new(shape()).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");
        let product = kernel
            .lanes()
            .expect("lanes")
            .reduce_product(value)
            .expect("product");
        kernel.store_scalar(1, product).expect("stored");
        kernel.finish().expect("finished")
    };

    expect_valid(&unsigned, "kernel-reduce-bitwise", VULKAN_1_1);
    expect_valid(&float, "kernel-reduce-product", VULKAN_1_1);
}
