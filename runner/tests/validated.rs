//! Every kernel the library exposes, handed to `spirv-val` at every width.
//!
//! **This layer did not exist, and its absence had already cost something.** `spirv-val` runs in
//! the emitter's own test tree over modules those tests build. Everything in `runner::kernels` went
//! straight to a driver, and `simdr`'s tests cannot reach it even in principle — the dependency
//! arrow points `runner -> simdr`, so the crate holding the validator harness is the one that
//! cannot see the kernels.
//!
//! Drivers are lenient about things the validator is not. `dot_unsigned` emitted `OpUDot` with a
//! signed result type and ran correctly on two devices for weeks; the very first `spirv-val` run
//! against it caught it. Every kernel here had the same standing to be wrong in the same way.
//!
//! # Why every width and not one
//!
//! The width picks the lane mapping, and the mappings are three *different instruction sequences*.
//! A kernel valid as a whole-subgroup reduce says nothing about the clustered form of itself, and
//! `ClusterSize` has its own validation rules — it must be a power of two, at least one, and no
//! larger than the subgroup. Validating at 32 alone would leave the other two shapes unchecked.
//!
//! # What this does not do
//!
//! It does not run anything. A module can validate perfectly and compute the wrong number, which is
//! what `execution.rs`, `lanes.rs`, `scan.rs` and the fuzzer are for. This says only that what the
//! emitter produced is a legal module for the environment it will be handed to.

mod common;

use common::{VULKAN_1_1, expect_valid};
use runner::kernels;
use simdr::lanes::{F16, F32, I8, I16, I32, U8, U16, U32};

/// The widths a device has actually reported here: 32 on an RTX 4080, 64 on an integrated Radeon,
/// and 4, 8 and 16 on lavapipe, whose subgroup follows llvmpipe's vector width.
const WIDTHS: [u32; 5] = [4, 8, 16, 32, 64];

/// Validate `words`, labelled by kernel and width, or report why the kernel could not be built.
///
/// **A kernel that refuses a width is not a failure.** `Lanes` refuses a lane count that has no
/// mapping onto the subgroup, by name, and that refusal is the design working. What would be a
/// failure is a kernel that builds and does not validate, so a build error is reported and skipped
/// while a validation error fails the test.
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
    // `folds()` picks the widest factor leaving a whole workgroup, so the factor is not a fixed
    // number and a module is built for whichever one the element count implies. Each of them is a
    // different unrolled loop in the kernel, and validating one says nothing about the rest.
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
    // The newest kernel, and the one with the most shapes: the cross-subgroup combine is 15 selects
    // at width 4 and none at all at 64, so the two ends of the range are barely the same module.
    for width in WIDTHS {
        valid_at(
            kernels::scan::scan_workgroup::<F32>(width),
            "scan_workgroup",
            width,
        );
        // Three bindings and a store at a runtime index, neither of which any other kernel here
        // has — `OpAccessChain` into a storage buffer with a non-constant index is exactly the
        // shape a validator has rules about.
        valid_at(
            kernels::scan::scan_blocks::<F32>(width),
            "scan_blocks",
            width,
        );
    }
}

#[test]
fn the_narrow_element_types_are_valid_at_every_width() {
    // Where a capability is easiest to get wrong: a narrow subgroup operation needs
    // `GroupNonUniformArithmetic` *and* the extended-types feature, and the module has to declare
    // what it uses. A missing declaration is exactly the class `spirv-val` catches and a permissive
    // driver does not.
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
    // Signed and unsigned take *different opcodes* for the same arithmetic — `OpGroupNonUniformSMax`
    // against `UMax` — and a dot product takes a different one again. `OpUDot` with a signed result
    // type is the invalid module this project shipped, so the unsigned forms are here by name.
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
    // Block structure is what `spirv-val` is strictest about: a merge must precede its branch, a
    // phi must name the block each value came from, and a loop needs a continue target. Every one
    // of those has been got wrong in this repository at least once.
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
    }
}

#[test]
fn the_extended_instruction_kernels_are_valid_at_every_width() {
    // GLSL.std.450 lands as `OpExtInst`, which names an instruction *set* by id — a module that
    // uses one without importing it is invalid and a driver may not care.
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
    // A specialization constant is a *constant instruction* with a default, so the module has to be
    // valid on its own — the value arrives at pipeline creation and validation happens before that.
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
    // These take a workgroup size rather than assuming the constant, which is the one place a
    // `LocalSize` other than 64 gets emitted — and `LocalSize` is what the validator checks a
    // `GLCompute` entry point for.
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
    // **Without this, the fourteen tests above could all be vacuous.** `validate` skips — and
    // returns `Ok` — when `spirv-val` is not installed, which is the right behaviour for a machine
    // that lacks it and the wrong thing to have go unnoticed. So one module is broken deliberately
    // and the harness has to object to it.
    //
    // Dropping the final word leaves the last instruction claiming a length it no longer has, which
    // is a parse failure rather than a rule violation: it fails at the first thing the tool does,
    // so it cannot be mistaken for a rule that happens to be lenient.
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
    // **The claim this file's header makes, made true.** Most of the tests above take the
    // `_whole` forms, where the lane count is the subgroup width and the mapping is always
    // `WholeSubgroup`. That leaves two of the three shapes unvalidated, and they are the two with
    // rules of their own: a clustered reduce carries a `ClusterSize` operand which must be a power
    // of two, at least one, and no larger than the subgroup, and a strip-mined one emits a
    // different instruction sequence entirely.
    //
    // `LANES` is a const generic, so the instantiations are written out. Against the five widths
    // this covers every combination of the three mappings the library can produce — a divisor, an
    // equal, and a multiple — plus the counts that have no mapping at all and are refused by name.
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
}
