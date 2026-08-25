//! Control flow, validated end to end.
//!
//! The half of [`kernels.rs`](kernels) whose shapes the validator has the strongest opinions
//! about: a merge instruction must be second-to-last in its block, an `OpPhi` must be first in
//! its own and must name blocks that really branch there, and a loop needs all four of its parts.
//! Every one of those has been got wrong here at least once, and every time it was `spirv-val`
//! rather than a unit test that said so.

mod common;

use common::{VULKAN_1_1, expect_valid};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, F32, U32};

/// A 32-wide subgroup, 64 invocations, two buffers — the shape every kernel here shares.
fn shape() -> Shape {
    Shape::new(32, 64, 2)
}

#[test]
fn a_uniform_branch_is_valid_spirv() {
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    kernel.store(1, value).expect("stored");

    {
        let mut lanes = kernel.lanes().expect("lanes");
        let limit = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("limit");
        let above = lanes.greater_than(value, limit).expect("compared");
        let over = lanes.any_uniform(above).expect("voted");

        lanes
            .if_uniform(over, |lanes| {
                let two = lanes.splat_bits::<F32, 32>(2.0_f32.to_bits())?;
                lanes.mul(value, two)?;
                Ok(())
            })
            .expect("branched");
    }

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-branch", VULKAN_1_1);
}

#[test]
fn a_rolled_loop_is_valid_spirv() {
    // The four-block shape with its two phis, which is the part of this crate most likely to be
    // subtly malformed — and the only judge of that is the validator.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let element = kernel.element();

    let doubled = {
        let mut lanes = kernel.lanes().expect("lanes");
        lanes
            .repeat_rolled(8, element, value.id(), |lanes, carried, _| {
                let held = lanes.from_lane_value::<F32, 32>(carried)?;
                Ok(lanes.add(held, held)?.id())
            })
            .expect("looped")
    };
    kernel.store_scalar(1, doubled).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-loop", VULKAN_1_1);
}

#[test]
fn a_value_carried_out_of_a_branch_is_valid_spirv() {
    // Two arms, each ending in a subgroup reduction, joined by an `OpPhi`. The rules the validator
    // enforces here are the ones easiest to get wrong by hand: the phi must be the first
    // instruction of the merge block, and every predecessor it names must actually branch there.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let element = kernel.element();

    let answer = {
        let mut lanes = kernel.lanes().expect("lanes");
        let limit = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("limit");
        let above = lanes.greater_than(value, limit).expect("compared");
        let over = lanes.any_uniform(above).expect("voted");

        lanes
            .choose_uniform(
                over,
                element,
                |lanes| lanes.reduce_sum(value),
                |lanes| lanes.reduce_max(value),
            )
            .expect("chosen")
    };
    kernel.store_scalar(1, answer).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-phi", VULKAN_1_1);
}

#[test]
fn a_branch_nested_inside_a_branch_is_valid_spirv() {
    // The case `current_block` exists for: the outer phi names the inner *merge* block, not the
    // block the outer arm opened. Naming the wrong one is a dominance failure, and this is what
    // says so.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let element = kernel.element();

    let answer = {
        let mut lanes = kernel.lanes().expect("lanes");
        let limit = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("limit");
        let above = lanes.greater_than(value, limit).expect("compared");
        let over = lanes.any_uniform(above).expect("voted");

        lanes
            .choose_uniform(
                over,
                element,
                |lanes| {
                    lanes.choose_uniform(
                        over,
                        element,
                        |lanes| lanes.reduce_sum(value),
                        |lanes| lanes.reduce_max(value),
                    )
                },
                |lanes| lanes.reduce_max(value),
            )
            .expect("chosen")
    };
    kernel.store_scalar(1, answer).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-nested-phi", VULKAN_1_1);
}

#[test]
fn a_pairwise_fold_across_an_offset_is_valid_spirv() {
    // Two reads of the same binding at different addresses, which is the whole of a halving pass.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let near = kernel.load::<32>(0).expect("loaded");
    let far = kernel.load_offset::<32>(0, 4096).expect("loaded");
    let folded = kernel
        .lanes()
        .expect("lanes")
        .add(near, far)
        .expect("added");
    kernel.store(1, folded).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-fold", VULKAN_1_1);
}

#[test]
fn a_workgroup_handover_through_shared_memory_is_valid_spirv() {
    // Shared memory, a barrier, and reads at constant indices. The validator has opinions about
    // all three: the array's length must be a constant *id*, the variable's storage class must
    // agree with its pointer type, and the barrier's scopes are ids rather than literals.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let mine = kernel
        .lanes()
        .expect("lanes")
        .reduce_sum(value)
        .expect("summed");

    let shared = kernel.shared(64).expect("declared");
    let slot = kernel.local_index();
    kernel.store_shared(shared, slot, mine).expect("stored");
    kernel.barrier().expect("barrier");

    let first = kernel.load_shared(shared, 0).expect("read");
    let second = kernel.load_shared(shared, 32).expect("read");
    let element = kernel.element();
    let total = kernel
        .module()
        .binary(simdr::module::op::F_ADD, element, first, second)
        .expect("added");
    kernel.store_scalar(1, total).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "workgroup-handover",
        VULKAN_1_1,
    );
}

#[test]
fn a_branch_inside_a_loop_is_valid_spirv() {
    // The loop's body now opens blocks of its own, so its bookkeeping — the copy into the phi's
    // promised name, then the branch to the continue target — lands in the selection's merge block
    // rather than in the body block the loop opened. Structured control flow is a tree, and this
    // is the first place the tree goes more than two deep.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes().expect("lanes");
        let limit = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("limit");
        let above = lanes.greater_than(value, limit).expect("compared");
        let over = lanes.any_uniform(above).expect("voted");

        lanes
            .repeat_rolled(4, element, value.id(), |lanes, carried, _| {
                let held = lanes.from_lane_value::<F32, 32>(carried)?;
                lanes.choose_uniform(
                    over,
                    element,
                    |lanes| Ok(lanes.add(held, held)?.id()),
                    |lanes| Ok(lanes.mul(held, held)?.id()),
                )
            })
            .expect("looped")
    };
    kernel.store_scalar(1, total).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "branch-in-loop",
        VULKAN_1_1,
    );
}

#[test]
fn a_loop_inside_a_branch_is_valid_spirv() {
    // The other nesting. The taken arm finishes in the loop's merge block, so the selection's phi
    // has to name *that* — naming the block the arm opened is a dominance failure, and this is
    // what says so.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes().expect("lanes");
        let limit = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("limit");
        let above = lanes.greater_than(value, limit).expect("compared");
        let over = lanes.any_uniform(above).expect("voted");

        lanes
            .choose_uniform(
                over,
                element,
                |lanes| {
                    lanes.repeat_rolled(4, element, value.id(), |lanes, carried, _| {
                        let held = lanes.from_lane_value::<F32, 32>(carried)?;
                        Ok(lanes.add(held, held)?.id())
                    })
                },
                |_| Ok(value.id()),
            )
            .expect("chosen")
    };
    kernel.store_scalar(1, total).expect("stored");

    expect_valid(
        &kernel.finish().expect("finished"),
        "loop-in-branch",
        VULKAN_1_1,
    );
}

#[test]
fn a_rolled_loop_reading_its_counter_is_valid_spirv() {
    let mut kernel = Kernel::<U32>::new(shape()).expect("built");
    let value = kernel.load::<32>(0).expect("loaded");
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes().expect("lanes");
        lanes
            .repeat_rolled(8, element, value.id(), |lanes, carried, iteration| {
                let held = lanes.from_lane_value::<U32, 32>(carried)?;
                let step = lanes.from_lane_value::<U32, 32>(iteration)?;
                Ok(lanes.add(held, step)?.id())
            })
            .expect("looped")
    };
    kernel.store_scalar(1, total).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(&words, "kernel-loop-counter", VULKAN_1_1);
}

#[test]
fn a_rolled_loop_over_a_kernel_may_read_its_buffers() {
    // **The whole reason this sits beside `Lanes::repeat_rolled`.** A body handed a `Lanes` can
    // compute and cannot fetch — a `Lanes` holds a module and a width and no bindings — so the one
    // shape that most wants a rolled loop, a reduction over a run too long to unroll, had nowhere
    // to live. This is that shape: eight values summed a trip at a time.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let element = kernel.element();
    let nought = F32::constant_from_bits(kernel.module(), 0.0_f32.to_bits()).expect("nought");

    let total = kernel
        .repeat_rolled(8, element, nought, |kernel, carried, counter| {
            let value = kernel.load_at(0, counter)?;
            Ok(kernel.module().f_add(element, carried, value)?)
        })
        .expect("looped");

    let at = kernel.module().constant_u32(0).expect("at");
    kernel.store_at(1, at, total).expect("stored");

    let words = kernel.finish().expect("finished");
    expect_valid(
        &words,
        "a_rolled_loop_over_a_kernel_may_read_its_buffers",
        VULKAN_1_1,
    );
}

#[test]
fn a_rolled_loop_of_no_trips_is_the_value_it_started_with() {
    // Nought trips is a legal ask — a caller whose run turned out to be empty — and it emits no
    // loop at all rather than one that runs once.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let element = kernel.element();
    let nought = F32::constant_from_bits(kernel.module(), 0.0_f32.to_bits()).expect("nought");

    let total = kernel
        .repeat_rolled(0, element, nought, |_, _, _| {
            panic!("a body that cannot run")
        })
        .expect("looped");
    assert_eq!(total, nought, "the value it started with, and no loop");

    let at = kernel.module().constant_u32(0).expect("at");
    kernel.store_at(1, at, total).expect("stored");
    let words = kernel.finish().expect("finished");
    expect_valid(
        &words,
        "a_rolled_loop_of_no_trips_is_the_value_it_started_with",
        VULKAN_1_1,
    );
}

#[test]
fn a_rolled_loop_carrying_several_totals_is_valid_spirv() {
    // The shape a weighted sum over several vectors wants: read the input once, keep a running
    // total apiece. Carrying one value would be one loop a total and one read of the same data
    // apiece, which is a bandwidth problem that does not show in the answer.
    //
    // Four phis at the header rather than two, all of them before the merge declaration — the
    // arrangement the validator has the strongest opinion about, and the reason this is here rather
    // than only in a unit test.
    let mut kernel = Kernel::<F32>::new(shape()).expect("built");
    let element = kernel.element();
    let nought = F32::constant_from_bits(kernel.module(), 0.0_f32.to_bits()).expect("nought");

    let out = kernel
        .repeat_rolled_many(8, element, &[nought; 3], |kernel, carried, counter| {
            let value = kernel.load_at(0, counter)?;
            carried
                .iter()
                .map(|one| Ok(kernel.module().f_add(element, *one, value)?))
                .collect()
        })
        .expect("looped");

    let at = kernel.module().constant_u32(0).expect("at");
    kernel.store_at(1, at, out[0]).expect("stored");
    let words = kernel.finish().expect("finished");
    expect_valid(
        &words,
        "a_rolled_loop_carrying_several_totals_is_valid_spirv",
        VULKAN_1_1,
    );
}
