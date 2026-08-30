mod common;

use common::{VULKAN_1_1, expect_valid};
use runner::kernels;
use simdr::lanes::{F16, F32, I8, I16, I32, U8, U16, U32};

const WIDTHS: [u32; 5] = [4, 8, 16, 32, 64];

fn valid_at(built: Result<Vec<u32>, simdr::lanes::LaneError>, kernel: &str, width: u32) {
    match built {
        Ok(words) => expect_valid(&words, &format!("{kernel}-{width}"), VULKAN_1_1),
        Err(refused) => eprintln!("  {kernel} at {width}: not built — {refused}"),
    }
}

#[test]
fn the_elementwise_kernels_are_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(kernels::scale(width, 2.0), "scale", width);
        valid_at(kernels::square(width), "square", width);
        valid_at(kernels::empty(width), "empty", width);
        valid_at(
            kernels::lane_affine_whole(width),
            "lane_affine_whole",
            width,
        );
        valid_at(kernels::lane_affine::<32>(width), "lane_affine-32", width);
    }
}

#[test]
fn the_reductions_are_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(
            kernels::reduce::lane_sum_whole::<F32>(width),
            "lane_sum",
            width,
        );
        valid_at(
            kernels::reduce::lane_max_whole::<F32>(width),
            "lane_max",
            width,
        );
        valid_at(
            kernels::reduce::workgroup_sum::<F32>(width),
            "workgroup_sum",
            width,
        );
        valid_at(
            kernels::reduce::butterfly_tree_sum(width),
            "butterfly_tree",
            width,
        );
        valid_at(
            kernels::reduce::butterfly_pair_sum(width, 1),
            "butterfly_pair",
            width,
        );
        for cluster in [2_u32, 4, 8] {
            valid_at(
                kernels::reduce::butterfly_cluster_sum(width, cluster),
                &format!("butterfly_cluster-{cluster}"),
                width,
            );
        }
        valid_at(
            kernels::reduce::fold_halves(width, 64),
            "fold_halves",
            width,
        );
        valid_at(
            kernels::reduce::fold_halves_open(width),
            "fold_halves_open",
            width,
        );
        valid_at(
            kernels::reduce::dot_product_whole::<F32>(width, 0),
            "dot_product",
            width,
        );
    }
}

#[test]
fn every_fold_factor_the_reducer_can_choose_is_valid() {
    for width in WIDTHS {
        for factor in [2_u32, 4, 8, 16] {
            valid_at(
                kernels::reduce::fold_by(width, factor, 64),
                &format!("fold_by-{factor}"),
                width,
            );
        }
    }
}

#[test]
fn the_scan_is_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(
            kernels::scan::scan_workgroup::<F32>(width),
            "scan_workgroup",
            width,
        );
        valid_at(
            kernels::scan::scan_blocks::<F32>(width),
            "scan_blocks",
            width,
        );
        valid_at(
            kernels::scan::scan_blocks_exclusive::<F32>(width),
            "scan_blocks_exclusive",
            width,
        );
        valid_at(
            kernels::scan::scan_workgroup_exclusive::<F32>(width),
            "scan_workgroup_exclusive",
            width,
        );
        valid_at(
            kernels::scan::scan_strips::<64>(width),
            "scan_strips-64",
            width,
        );
        valid_at(
            kernels::scan::scan_strips::<128>(width),
            "scan_strips-128",
            width,
        );
        for cluster in [2_u32, 4, 8, 16] {
            valid_at(
                kernels::scan::scan_clusters(width, cluster),
                &format!("scan_clusters-{cluster}"),
                width,
            );
            valid_at(
                kernels::scan::scan_clusters_exclusive(width, cluster),
                &format!("scan_clusters_exclusive-{cluster}"),
                width,
            );
        }
        valid_at(
            kernels::scan::add_offsets::<F32>(width),
            "add_offsets",
            width,
        );
    }
}

#[test]
fn the_narrow_element_types_are_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(
            kernels::narrow::narrow_sum_whole::<I8>(width),
            "narrow_sum-i8",
            width,
        );
        valid_at(
            kernels::narrow::narrow_sum_whole::<U8>(width),
            "narrow_sum-u8",
            width,
        );
        valid_at(
            kernels::narrow::narrow_sum_whole::<I16>(width),
            "narrow_sum-i16",
            width,
        );
        valid_at(
            kernels::narrow::narrow_sum_whole::<U16>(width),
            "narrow_sum-u16",
            width,
        );
        valid_at(
            kernels::narrow::narrow_sum_whole::<F16>(width),
            "narrow_sum-f16",
            width,
        );

        valid_at(
            kernels::narrow::narrow_add::<I8, 32>(width, 1),
            "narrow_add-i8",
            width,
        );
        valid_at(
            kernels::narrow::narrow_add::<F16, 32>(width, 1),
            "narrow_add-f16",
            width,
        );
        valid_at(
            kernels::narrow::narrow_clamp::<I8, 32>(width, 64, 0, 7),
            "narrow_clamp-i8",
            width,
        );
    }
}

#[test]
fn the_integer_reductions_are_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(
            kernels::reduce::lane_sum_whole::<I32>(width),
            "lane_sum-i32",
            width,
        );
        valid_at(
            kernels::reduce::lane_sum_whole::<U32>(width),
            "lane_sum-u32",
            width,
        );
        valid_at(
            kernels::reduce::lane_max_whole::<I32>(width),
            "lane_max-i32",
            width,
        );
        valid_at(
            kernels::reduce::lane_max_whole::<U32>(width),
            "lane_max-u32",
            width,
        );
        valid_at(
            kernels::reduce::dot_product_whole::<U32>(width, 0),
            "dot_product-u32",
            width,
        );
        valid_at(
            kernels::reduce::dot_product_whole::<I32>(width, 0),
            "dot_product-i32",
            width,
        );
    }
}

#[test]
fn the_packed_dot_product_kernels_are_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(kernels::dot::packed_dot(width), "packed_dot", width);
        valid_at(kernels::dot::unpacked_dot(width), "unpacked_dot", width);
        valid_at(kernels::dot::mixed_dot(width, 0), "mixed_dot", width);
        valid_at(
            kernels::dot::byte_component(width, 0),
            "byte_component",
            width,
        );
        valid_at(
            kernels::dot::repeated_packed_dot(width, 2),
            "repeated_packed",
            width,
        );
        valid_at(
            kernels::dot::repeated_unpacked_dot(width, 2),
            "repeated_unpacked",
            width,
        );
    }
}

#[test]
fn the_control_flow_kernels_are_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(kernels::control::any_above(width, 0.5), "any_above", width);
        valid_at(
            kernels::control::scale_if_any_above(width, 0.5),
            "scale_if_any",
            width,
        );
        valid_at(
            kernels::control::branch_only(width, 0.5),
            "branch_only",
            width,
        );
        valid_at(
            kernels::control::sum_or_max(width, 0.5),
            "sum_or_max",
            width,
        );
        valid_at(
            kernels::control::rolled_counter_sum(width, 3),
            "rolled_counter",
            width,
        );
        valid_at(
            kernels::control::branch_in_loop(width, 3, 0.5),
            "branch_in_loop",
            width,
        );
        valid_at(
            kernels::control::loop_in_branch(width, 3, 0.5),
            "loop_in_branch",
            width,
        );
        valid_at(
            kernels::control::rolled_doubling(width, 3),
            "rolled_doubling",
            width,
        );
    }
}

#[test]
fn the_two_axis_kernels_are_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(
            kernels::plane::row_scale(width, 64, 2, 3),
            "row_scale",
            width,
        );
        valid_at(
            kernels::plane::flat_scale(width, 64, 3),
            "flat_scale",
            width,
        );
        valid_at(kernels::plane::row_sum(width, 2), "row_sum", width);
        valid_at(kernels::plane::row_bias(width, 64, 2), "row_bias", width);
        valid_at(kernels::plane::row_index(width, 64, 2), "row_index", width);
    }
}

#[test]
fn the_atomic_kernels_are_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(
            kernels::scatter::histogram(width, 64, 8),
            "histogram",
            width,
        );
        valid_at(
            kernels::scatter::histogram_incrementing(width, 64, 8),
            "histogram_incrementing",
            width,
        );
        valid_at(kernels::scatter::claim_slots(width), "claim_slots", width);
        valid_at(
            kernels::scatter::exchange_chain(width),
            "exchange_chain",
            width,
        );
        valid_at(
            kernels::scatter::atomic_gather(width),
            "atomic_gather",
            width,
        );
        valid_at(
            kernels::unrun::subgroup_agrees(width),
            "subgroup_agrees",
            width,
        );
        valid_at(
            kernels::unrun::subgroup_agrees_wide::<64>(width),
            "subgroup_agrees_wide-64",
            width,
        );
        valid_at(
            kernels::unrun::subgroup_agrees_wide::<128>(width),
            "subgroup_agrees_wide-128",
            width,
        );
        valid_at(kernels::unrun::equals(width, 3), "equals", width);
        for cluster in [2_u32, 8, 32] {
            valid_at(
                kernels::unrun::rotate_in_cluster(width, cluster, 3),
                &format!("rotate-{cluster}"),
                width,
            );
        }

        valid_at(
            kernels::unrun::centre_and_scale(width, 8.0, 4.0),
            "centre_and_scale",
            width,
        );
        valid_at(kernels::unrun::remainder(width, 7), "remainder", width);
        valid_at(
            kernels::unrun::rolled_block_sum(width, 4),
            "rolled_block_sum",
            width,
        );
        valid_at(
            kernels::unrun::rolled_weighted_totals(width, 4),
            "rolled_weighted_totals",
            width,
        );
    }
}

#[test]
fn the_extended_instruction_kernels_are_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(kernels::extended::root::<32>(width), "root", width);
        valid_at(
            kernels::extended::fused_square::<32>(width),
            "fused_square",
            width,
        );
        valid_at(
            kernels::extended::magnitude::<I32, 32>(width),
            "magnitude-i32",
            width,
        );
        valid_at(
            kernels::extended::clamped::<F32, 32>(width, 0, 1_f32.to_bits()),
            "clamped",
            width,
        );
        valid_at(
            kernels::extended::larger::<F32, 32>(width, 0.5_f32.to_bits()),
            "larger",
            width,
        );
        valid_at(
            kernels::extended::smaller::<F32, 32>(width, 0.5_f32.to_bits()),
            "smaller",
            width,
        );
    }
}

#[test]
fn the_specialized_kernels_are_valid_before_anything_is_specialized() {
    for width in WIDTHS {
        valid_at(
            kernels::specialized::specialized_add::<F32, 32>(width, 1_f32.to_bits()),
            "specialized_add",
            width,
        );
        valid_at(
            kernels::specialized::specialized_affine::<32>(width, 2, 1),
            "specialized_affine",
            width,
        );
    }
}

#[test]
fn the_network_kernels_are_valid_at_every_width() {
    for width in WIDTHS {
        valid_at(
            kernels::network::clipped_dot::<32>(width, 255, 64),
            "clipped_dot",
            width,
        );
        valid_at(
            kernels::network::clipped_dot_split::<32>(width, 255),
            "clipped_dot_split",
            width,
        );
        valid_at(
            kernels::network::unclipped_dot::<32>(width, 0),
            "unclipped_dot",
            width,
        );
    }
}

#[test]
fn the_occupancy_kernels_are_valid_at_every_workgroup_size_they_sweep() {
    for width in WIDTHS {
        for workgroup in [64_u32, 128, 256] {
            valid_at(
                kernels::occupancy::sized_repeated_scale(width, workgroup, 2, 2),
                &format!("sized_repeated_scale-{workgroup}"),
                width,
            );
            valid_at(
                kernels::occupancy::sized_lane_sum(width, workgroup),
                &format!("sized_lane_sum-{workgroup}"),
                width,
            );
        }
    }
}

#[test]
fn the_validator_rejects_a_module_this_suite_has_broken_on_purpose() {
    let Some(tool) = common::validator() else {
        eprintln!("SKIPPED negative control: spirv-val not found (set SPIRV_VAL)");
        return;
    };
    eprintln!("validator: {}", tool.display());

    let mut words = kernels::empty(32).expect("built");
    assert!(
        common::validate(&words, "control-intact", VULKAN_1_1).is_ok(),
        "the module has to be valid before breaking it proves anything"
    );

    words.pop();
    assert!(
        common::validate(&words, "control-truncated", VULKAN_1_1).is_err(),
        "a truncated module was accepted — the suite above cannot fail"
    );
}

#[test]
fn all_three_lane_mappings_are_valid_at_every_width() {
    macro_rules! mappings_at {
        ($width:expr, $($lanes:literal),+) => {
            $(
                valid_at(
                    kernels::reduce::lane_sum::<F32, $lanes>($width),
                    concat!("lane_sum-", stringify!($lanes)),
                    $width,
                );
                valid_at(
                    kernels::reduce::lane_max::<F32, $lanes>($width),
                    concat!("lane_max-", stringify!($lanes)),
                    $width,
                );
                valid_at(
                    kernels::narrow::narrow_sum::<I8, $lanes>($width),
                    concat!("narrow_sum-i8-", stringify!($lanes)),
                    $width,
                );
            )+
        };
    }

    for width in WIDTHS {
        mappings_at!(width, 2, 4, 8, 16, 32, 64, 128);
    }

    for width in WIDTHS {
        for cluster in [2_u32, 4, 8, 16] {
            valid_at(
                kernels::broadcast_in_cluster::<F32>(width, cluster, 1),
                &format!("broadcast_in_cluster-{cluster}"),
                width,
            );
        }
    }
}
