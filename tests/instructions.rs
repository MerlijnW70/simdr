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

use common::{VULKAN_1_1, expect_valid, validate, validator};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, F32, I8, I16, I32, Integer, U8, U16, U32};

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

#[test]
fn the_operations_a_second_sweep_found_unreached_are_valid_spirv() {
    // **The sweep that found `OpUDot`, run again mechanically.** Asking which public functions have
    // no reference outside the file that defines them turned up six more, and four of them emit
    // instructions no other test in this tree reaches:
    //
    // * `Lanes::exp` — one of the extended set nothing composed.
    // * `Lanes::if_uniform_value` — the one-armed branch that *yields*, where `if_uniform` runs a
    //   body for its effects. Its `OpPhi` names two predecessors where the two-armed form names
    //   one each, which is the half a validator has an opinion about.
    // * `Module::atomic_store` — the only atomic with no result id, and the last one still in the
    //   state an earlier audit found the exchange and the load in.
    // * `Module::memory_barrier` — `OpMemoryBarrier`, which orders without waiting. Its own
    //   documentation says it is rarely what a caller wants, and nothing here wanted it.
    //
    // All four in one module: an instruction that is only valid in company is a thing this suite
    // has met before, and a module carrying one of the four proves nothing about the others.
    use simdr::spec::{MemorySemantics, Scope};

    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let shaped = {
        let mut lanes = kernel.lanes().expect("lanes");
        let curved = lanes.exp(value).expect("exp");

        // The vote is what makes a `Uniform`, and `if_uniform_value` is the only thing that takes
        // one and hands a value back.
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

    // The atomic store, and a barrier ordering it. Both take their scope and their semantics as
    // *ids of constants* rather than literals, which is the trap this crate documents in three
    // places and which assembles cleanly when it is wrong.
    //
    // **The two take different masks, and that is the finding this test produced.** `Relaxed` is
    // legal on the atomic and forbidden on the barrier — see the refusal below.
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
    // **What the test above found the first time it ran.** `Module::memory_barrier` had no caller,
    // no kernel and no validator behind it, and `MemorySemantics::None` — which this crate's own
    // documentation recommended as "the honest mask" for an operation that publishes nothing — is
    // `Relaxed`, which `VUID-StandaloneSpirv-MemorySemantics-10869` forbids on `OpMemoryBarrier`
    // specifically. The same mask on the atomic two lines above is perfectly legal, which is what
    // makes it easy to get wrong.
    //
    // Asserted as a *refusal* rather than fixed and forgotten, for the reason
    // `validated.rs::a_compute_entry_point_without_a_workgroup_size_is_refused` exists: without
    // this, the test above could pass because the validator never says no. The emitter cannot
    // refuse it — the semantics arrive as the id of a constant, and this layer cannot ask what
    // value that constant holds — so the boundary is stated here and in the two doc comments.
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

    // **On the instruction's name, not on the diagnostic's wording.** Two `spirv-val` builds refuse
    // this and describe it differently — the one in Ubuntu's `spirv-tools` cites
    // `VUID-StandaloneSpirv-OpMemoryBarrier-04732` and says "Memory Semantics", a newer one cites
    // `VUID-StandaloneSpirv-MemorySemantics-10869` and says "MemorySemantics". Asserting on the
    // second spelling passed here and failed in CI, which is this suite's own lesson arriving from
    // the other side: a test that pins a detail it is not about is a test about the wrong thing.
    //
    // `MemoryBarrier` is in both, and it is the claim — this module is refused, and refused *for
    // the barrier* rather than for something else that happens to be wrong with it.
    let message = outcome.expect_err("a relaxed memory barrier is not valid SPIR-V");
    assert!(
        message.contains("MemoryBarrier"),
        "refused for something other than the barrier: {message}"
    );
}

/// The three shifts, at one element type.
///
/// Generic because the point is the *type*: `Integer` admits six and the test above reached two of
/// them. A shift is an integer instruction whatever the width, but "whatever the width" is a claim
/// about SPIR-V, and only the validator can settle it — an 8-bit `OpShiftRightArithmetic` is a
/// different question from a 32-bit one, and asking it costs one line per type.
fn shifts_are_valid_for<T: Integer>(name: &str) {
    let mut kernel = Kernel::<T>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let shifted = {
        let mut lanes = kernel.lanes().expect("lanes");
        // The amount is a `u32` whatever the value's width — SPIR-V lets the two differ, and this
        // is where that is checked rather than assumed.
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
    // **An operation validated at a third of the types it takes.** `the_shifts_are_valid_spirv`
    // above builds a `U32` kernel and reaches `I32` through a bitcast; the four narrow integers
    // were never handed to the validator at all, and they are the ones that need `Int8` or `Int16`
    // declared and a result type of a width the shift has to match.
    //
    // Found by sweeping `src/module/` for opcodes paired with a type by hand, which is where the
    // float-shift bug came from one layer up: the fix bounded the shifts to `Integer`, and nothing
    // then asked whether every `Integer` actually works.
    shifts_are_valid_for::<U32>("u32");
    shifts_are_valid_for::<I32>("i32");
    shifts_are_valid_for::<U8>("u8");
    shifts_are_valid_for::<I8>("i8");
    shifts_are_valid_for::<U16>("u16");
    shifts_are_valid_for::<I16>("i16");
}

/// The breadth of the lane API, at one element type, handed to the validator.
///
/// **Narrow types were validated across seven operations and accept the lot.** Everything in
/// `Lanes` is bounded by `Element`, and `I8`, `U8`, `I16` and `U16` are `Element`s — but every
/// module `spirv-val` had ever seen at one of those widths came from `kernels::narrow`, which
/// reaches `add`, `clamp`, `load`, `reduce_sum`, `splat_bits`, `store` and `store_scalar`. The
/// comparisons, the selects, the extremes, the shuffles, the votes and the scans were validated at
/// 32 bits and nowhere else.
///
/// That is the shape the shifts had, one type bound up: an operation checked at a third of what it
/// accepts. And narrow is exactly where SPIR-V is fussiest — `Int8` and `Int16` have to be
/// declared, the group opcodes differ between signed and unsigned, and a result type's width has to
/// follow its operands'.
///
/// One module rather than one per operation, deliberately: they compose, so a single stream reaches
/// all of them and any one being wrong fails the same run.
///
/// # One axis, and the other one is stated rather than implied
///
/// This sweeps the **element type** at a single 32-wide subgroup, because that is the axis that was
/// missing. It does not sweep the *width*: `Kernel` takes the subgroup in its `Shape` and `LANES` is
/// a const generic, so varying both would mean a macro over the pairs. `runner/tests/validated.rs`
/// covers widths 4 to 64 for the narrow kernels it has — which are the seven operations above.
///
/// So the honest reading is that the **type × operation** grid is filled here and the
/// **type × width** grid is not. A narrow butterfly on a four-wide subgroup is a clustered shuffle
/// and is still validated nowhere.
fn the_lane_surface_is_valid_for<T: Element>(name: &str) {
    let mut kernel = Kernel::<T>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");

    let (folded, scanned) = {
        let mut lanes = kernel.lanes().expect("lanes");
        let one = lanes.splat_bits::<T, 32>(1).expect("one");
        let two = lanes.splat_bits::<T, 32>(2).expect("two");

        // Elementwise: the arithmetic, and the three extended instructions whose GLSL number
        // differs between the signed and unsigned forms of the same width.
        let sum = lanes.add(value, one).expect("add");
        let product = lanes.mul(sum, two).expect("mul");
        let smaller = lanes.min(product, two).expect("min");
        let larger = lanes.max(smaller, one).expect("max");
        let bounded = lanes.clamp(larger, one, two).expect("clamp");

        // A comparison — `OpSGreaterThan` or `OpUGreaterThan`, and the choice is the element's —
        // then a select on it, and the equality that is one instruction for every integer.
        let above = lanes.greater_than(bounded, one).expect("greater");
        let picked = lanes.select(above, bounded, one).expect("select");
        let same = lanes.equal(picked, one).expect("equal");
        let either = lanes.select(same, picked, two).expect("select again");

        // Across lanes: four different shuffle opcodes and three capabilities between them.
        let partner = lanes.butterfly(either, 1).expect("butterfly");
        let first = lanes.broadcast(partner, 0).expect("broadcast");
        let up = lanes.shift_up(first, 1).expect("shift up");
        let down = lanes.shift_down(up, 1).expect("shift down");
        let rotated = lanes.rotate_up(down, 1).expect("rotate");

        // A scan and the three reductions, which are where `shaderSubgroupExtendedTypes` lives —
        // the permission with no capability in the module to declare it.
        let scanned = lanes.prefix_sum(rotated).expect("scan");
        let total = lanes.reduce_sum(scanned).expect("sum");
        let biggest = lanes.reduce_max(scanned).expect("max");
        let smallest = lanes.reduce_min(scanned).expect("min");

        let total = lanes.from_lane_value::<T, 32>(total).expect("as vector");
        let biggest = lanes.from_lane_value::<T, 32>(biggest).expect("as vector");
        let smallest = lanes.from_lane_value::<T, 32>(smallest).expect("as vector");

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
        &format!("kernel-surface-{name}"),
        VULKAN_1_1,
    );
}

#[test]
fn the_lane_surface_is_valid_for_every_narrow_integer() {
    // The four the shift sweep turned up as never validated beyond a handful of operations. `U32`
    // is here as the control: it is the width everything else is checked at, so a failure in one of
    // the four and not in it is a *narrow* problem rather than a broken test.
    the_lane_surface_is_valid_for::<U32>("u32");
    the_lane_surface_is_valid_for::<U8>("u8");
    the_lane_surface_is_valid_for::<I8>("i8");
    the_lane_surface_is_valid_for::<U16>("u16");
    the_lane_surface_is_valid_for::<I16>("i16");
}
