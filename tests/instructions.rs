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
/// # Both axes, and the mapping is what makes the second one matter
///
/// The first version of this swept the element type at one 32-wide subgroup and said so. The width
/// is swept too now, because a width is not a parameter to these instructions — it *chooses the
/// instruction sequence*. The same `butterfly` call is one shuffle whole-subgroup, a masked shuffle
/// clustered, and one per strip above that; the same `prefix_sum` is a single instruction, a
/// Hillis–Steele ladder, and a carry between strips.
///
/// So the grid is type × width × **mapping**, and the three mappings have three bodies below rather
/// than one with a flag, because the difference between them is the content: a clustered vector may
/// not shift or vote, and a strip-mined one may not rotate.
///
/// `LANES` is a const generic and the subgroup is a runtime number, which is why the width axis went
/// unswept for so long — there is no loop to write. `at_every_width!` is the macro over the pairs
/// that fills it.
fn the_lane_surface_is_valid_for<T: Element, const LANES: u32>(name: &str, width: u32) {
    let mut kernel = Kernel::<T>::new(Shape::new(width, 64, 2)).expect("built");
    let value = kernel.load::<LANES>(0).expect("loaded");

    let (folded, scanned) = {
        let mut lanes = kernel.lanes().expect("lanes");
        let one = lanes.splat_bits::<T, LANES>(1).expect("one");
        let two = lanes.splat_bits::<T, LANES>(2).expect("two");

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

/// The cross-lane operations a vector **narrower** than the subgroup may have.
///
/// A separate body rather than a flag on the one above, because the difference *is* the content: a
/// clustered vector shares its subgroup with others, so the shifts are refused by name — the lanes
/// below a cluster's first belong to a neighbour — and so are the votes, which answer for every
/// vector in the subgroup at once.
///
/// What is left is the three that stay inside a cluster: a butterfly whose mask cannot leave it, a
/// broadcast of the cluster's own position, and a rotate that wraps. Plus the reductions and the
/// scan, which reach `ClusteredReduce` and the Hillis–Steele ladder — two instruction sequences
/// that exist for no other mapping.
///
/// The elementwise operations are not repeated here. They are per-strip and the mapping cannot
/// reach them, so sweeping them again would add modules and no question.
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

/// The cross-lane operations a **strip-mined** vector may have.
///
/// Wider than the subgroup, so each lane holds several elements and every shuffle applies per strip.
/// The rotate is the one refused here — it would move elements *between* strips, which is a
/// different algorithm rather than a different operand — and the votes fold their strips together,
/// which is a step the other two mappings do not have.
///
/// The scan is the interesting one: it carries a running total from each strip to the next, and that
/// carry exists at no other mapping.
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

        // The vote that folds its strips: `all_equal` is two questions over a strip-mined vector,
        // and one over any other.
        // Emitted and left unused, which is enough: the question here is whether the instruction
        // and its strip fold are *legal*, and an unread result is as validated as a read one. It
        // cannot be folded into the value either way — a vote answers with a **bool**, and the
        // combination below is a `T`.
        let _ = lanes.all_equal(scanned).expect("all equal");

        // **`splat_id`, not `from_lane_value`.** A reduction answers with one id, and a strip-mined
        // vector holds several elements per lane — so putting that id back into a vector means
        // repeating it across the strips. `from_lane_value` builds a *one-strip* vector and is
        // refused here by name, `TooManyStrips { strips: 1, limit: 2 }`, which is the mapping being
        // right about a distinction the other two do not have. The first draft of this used it, and
        // the refusal is what said so.
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

/// Every surface, at every width a device here reports.
///
/// `LANES` is a const generic and the subgroup is a runtime number, so the pairs are written out.
/// That is the whole reason the width axis went unswept: there is no loop to write, and a macro over
/// the pairs is the only shape that fills it.
macro_rules! at_every_width {
    ($type:ty, $name:literal) => {
        // Whole-subgroup: `LANES` equals the width, and everything is legal.
        the_lane_surface_is_valid_for::<$type, 4>($name, 4);
        the_lane_surface_is_valid_for::<$type, 8>($name, 8);
        the_lane_surface_is_valid_for::<$type, 16>($name, 16);
        the_lane_surface_is_valid_for::<$type, 32>($name, 32);
        the_lane_surface_is_valid_for::<$type, 64>($name, 64);

        // Clustered: half the width, so two vectors share every subgroup. A one-wide subgroup has
        // no cluster to make, which is why this starts at 8.
        the_clustered_surface_is_valid_for::<$type, 4>($name, 8);
        the_clustered_surface_is_valid_for::<$type, 8>($name, 16);
        the_clustered_surface_is_valid_for::<$type, 16>($name, 32);
        the_clustered_surface_is_valid_for::<$type, 32>($name, 64);

        // Strip-mined: twice the width, so each lane holds two elements.
        the_stripped_surface_is_valid_for::<$type, 8>($name, 4);
        the_stripped_surface_is_valid_for::<$type, 16>($name, 8);
        the_stripped_surface_is_valid_for::<$type, 32>($name, 16);
        the_stripped_surface_is_valid_for::<$type, 64>($name, 32);
    };
}

#[test]
fn the_lane_surface_is_valid_for_every_narrow_integer() {
    // The four the shift sweep turned up as never validated beyond a handful of operations. `U32`
    // is here as the control: it is the type everything else is checked at, so a failure in one of
    // the four and not in it is a *narrow* problem rather than a broken test.
    //
    // **Both axes now.** The first version swept the type at one width and said so; this fills the
    // grid — five widths, three mappings, five types, and the mappings are what makes the width
    // matter. A clustered butterfly is a masked shuffle, a strip-mined scan carries between strips,
    // and a whole-subgroup one is a single instruction. Same call, three instruction sequences.
    at_every_width!(U32, "u32");
    at_every_width!(U8, "u8");
    at_every_width!(I8, "i8");
    at_every_width!(U16, "u16");
    at_every_width!(I16, "i16");
}

#[test]
fn subtract_divide_and_negate_are_valid_spirv() {
    // Three instructions core SPIR-V has and this crate did not, added because an activation needs
    // all three: `silu(x)` is `x / (1 + exp(-x))`. They were read out of the grammar at 1.6.7 by
    // the DR-0001 recipe — 131, 136 and 127 — and a wrong number would assemble into a different
    // well-formed instruction, so a validator run is the only thing between them and a kernel that
    // returns plausible nonsense.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let element = kernel.element();
    let held = kernel.load::<32>(0).expect("loaded");
    let value = held.id();

    let one = F32::constant_from_bits(kernel.module(), 1.0_f32.to_bits()).expect("one");
    let flipped = kernel.module().f_negate(element, value).expect("negated");
    let raised = {
        let mut lanes = kernel.lanes().expect("lanes");
        let vector = lanes.from_lane_value::<F32, 32>(flipped).expect("a lane value");
        lanes.exp::<32>(vector).expect("exp").id()
    };
    let below = kernel.module().f_add(element, one, raised).expect("added");
    let gated = kernel.module().f_div(element, value, below).expect("divided");
    let left = kernel.module().f_sub(element, value, gated).expect("subtracted");

    kernel
        .lanes()
        .expect("lanes")
        .from_lane_value::<F32, 32>(left)
        .and_then(|vector| kernel.store::<32>(1, vector))
        .expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "subtract, divide and negate", VULKAN_1_1);
}
