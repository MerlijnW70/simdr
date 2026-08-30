mod common;

use common::{VULKAN_1_1, expect_valid};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, F32, U32};

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
