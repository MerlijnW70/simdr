#![allow(clippy::expect_used)]

use super::{Bounds, Overrun, element_bytes, invocations, local_size};
use crate::dispatch::{Grid, Specialization};
use crate::kernels;
use simdr::lanes::{F32, I8};
use simdr::module::op;
use simdr::spec::ExecutionMode;

fn per_invocation(spirv: &[u32]) -> u64 {
    Bounds::of(spirv, &Specialization::none()).elements_per_invocation()
}

#[test]
fn the_workgroup_size_is_read_out_of_the_module_the_emitter_built() {
    let spirv = kernels::empty(32).expect("built");
    assert_eq!(
        local_size(&spirv),
        Some([u64::from(kernels::WORKGROUP_SIZE), 1, 1])
    );
}

#[test]
fn a_module_with_no_execution_mode_at_all_reports_nothing() {
    assert_eq!(local_size(&[]), None);
    assert_eq!(local_size(&[0x0723_0203, 0x0001_0300, 0, 1, 0]), None);
}

fn module_with_execution_mode(mode: u32, sizes: [u32; 3]) -> Vec<u32> {
    let mut words = vec![0x0723_0203, 0x0001_0300, 0, 1, 0];
    words.push((6 << 16) | u32::from(op::EXECUTION_MODE));
    words.push(1);
    words.push(mode);
    words.extend_from_slice(&sizes);
    words
}

#[test]
fn an_execution_mode_that_is_not_local_size_declares_no_workgroup() {
    let hint = module_with_execution_mode(ExecutionMode::LocalSize.word() + 1, [2, 3, 4]);
    assert_eq!(local_size(&hint), None);

    let real = module_with_execution_mode(ExecutionMode::LocalSize.word(), [2, 3, 4]);
    assert_eq!(local_size(&real), Some([2, 3, 4]));
}

#[test]
fn the_three_axes_of_a_workgroup_are_multiplied_and_not_divided() {
    let module = module_with_execution_mode(ExecutionMode::LocalSize.word(), [2, 3, 4]);
    let axes = local_size(&module).expect("declared");

    assert_eq!(invocations(Grid::linear(1), axes), 24, "2 * 3 * 4");
    assert_ne!(invocations(Grid::linear(1), axes), 0, "2 / 3 / 4");
}

#[test]
fn invocations_multiply_both_axes_by_the_workgroup() {
    assert_eq!(invocations(Grid::linear(1), [64, 1, 1]), 64);
    assert_eq!(invocations(Grid::linear(16), [64, 1, 1]), 1024);
    assert_eq!(invocations(Grid::new(4, 4), [64, 1, 1]), 1024);

    assert_eq!(invocations(Grid::new(4, 4), [64, 2, 1]), 2048);
}

#[test]
fn the_product_is_computed_wide_enough_not_to_wrap() {
    assert_eq!(invocations(Grid::linear(1 << 20), [1 << 16, 1, 1]), 1 << 36);
    assert_eq!(
        (1_u32 << 20).wrapping_mul(1 << 16),
        0,
        "the narrow product this avoids"
    );
}

#[test]
fn a_dispatch_that_matches_its_buffer_fits_and_one_word_more_does_not() {
    let bounds = Bounds::of(&kernels::empty(32).expect("built"), &Specialization::none());
    let workgroup = kernels::WORKGROUP_SIZE as usize;

    assert!(bounds.fits(Grid::linear(1), workgroup));
    assert!(bounds.fits(Grid::linear(1), workgroup + 1));
    assert!(
        !bounds.fits(Grid::linear(1), workgroup - 1),
        "one word short of a workgroup is one invocation with nowhere to write"
    );
    assert!(!bounds.fits(Grid::linear(2), workgroup));
}

#[test]
fn the_strip_count_is_read_back_out_of_the_module() {
    for (width, lanes, strips) in [(32_u32, 32_u32, 1_u64), (32, 128, 4), (64, 128, 2)] {
        let spirv = match lanes {
            32 => kernels::reduce::lane_sum::<F32, 32>(width),
            _ => kernels::reduce::lane_sum::<F32, 128>(width),
        }
        .expect("built");

        assert_eq!(
            per_invocation(&spirv),
            strips,
            "{lanes} lanes on a {width}-wide subgroup"
        );
    }
}

#[test]
fn a_kernel_that_touches_one_element_per_invocation_reports_one() {
    assert_eq!(per_invocation(&kernels::empty(32).expect("built")), 1);
    assert_eq!(per_invocation(&kernels::scale(32, 2.0).expect("built")), 1);
}

#[test]
fn a_module_with_no_workgroup_arithmetic_reports_one_rather_than_nothing() {
    assert_eq!(per_invocation(&[]), 1);
}

#[test]
fn a_strip_mined_kernel_needs_more_buffer_than_its_invocation_count() {
    let bounds = Bounds::of(
        &kernels::reduce::lane_sum::<F32, 128>(32).expect("built"),
        &Specialization::none(),
    );
    let workgroup = kernels::WORKGROUP_SIZE as usize;

    assert!(bounds.fits(Grid::linear(1), workgroup * 4));
    assert!(
        !bounds.fits(Grid::linear(1), workgroup),
        "one element per invocation is what this kernel does not do"
    );
    assert!(!bounds.fits(Grid::linear(1), workgroup * 4 - 1));
}

#[test]
fn the_strip_count_multiplies_the_requirement_and_does_not_replace_it() {
    let bounds = Bounds::of(
        &kernels::reduce::lane_sum::<F32, 128>(32).expect("built"),
        &Specialization::none(),
    );
    let workgroup = kernels::WORKGROUP_SIZE as usize;

    assert!(bounds.fits(Grid::linear(2), workgroup * 8));
    assert!(
        !bounds.fits(Grid::linear(2), workgroup * 4),
        "twice the workgroups needs twice the buffer"
    );
}

#[test]
fn the_width_four_bug_this_check_was_written_for_is_now_refused() {
    let spirv = kernels::lane_affine::<32>(4).expect("built");
    let bounds = Bounds::of(&spirv, &Specialization::none());
    let workgroup = kernels::WORKGROUP_SIZE as usize;

    assert_eq!(
        per_invocation(&spirv),
        8,
        "32 lanes on a four-wide subgroup is eight elements each"
    );
    assert!(
        !bounds.fits(Grid::linear(1), workgroup),
        "a buffer of one element per invocation is an eighth of what this reads"
    );
    assert!(bounds.fits(Grid::linear(1), workgroup * 8));
}

#[test]
fn a_constant_offset_past_the_run_is_read_out_of_the_folded_address() {
    let offset = kernels::WORKGROUP_SIZE * 8;
    let spirv = kernels::network::clipped_dot::<256>(32, offset, 255).expect("built");
    let bounds = Bounds::of(&spirv, &Specialization::none());

    assert_eq!(
        per_invocation(&spirv),
        8,
        "256 lanes on a 32-wide subgroup is eight strips"
    );
    assert_eq!(
        bounds.offset(),
        u64::from(offset),
        "the strip term is subtracted back off, leaving what the caller asked for"
    );
}

#[test]
fn a_kernel_reading_past_its_run_needs_a_buffer_past_its_run() {
    let offset = kernels::WORKGROUP_SIZE * 8;
    let bounds = Bounds::of(
        &kernels::network::clipped_dot::<256>(32, offset, 255).expect("built"),
        &Specialization::none(),
    );
    let run = kernels::WORKGROUP_SIZE as usize * 8;

    assert!(
        !bounds.fits(Grid::linear(1), run),
        "the run alone leaves nothing for the half this kernel reads past it"
    );
    assert!(!bounds.fits(Grid::linear(1), run + offset as usize - 1));
    assert!(
        bounds.fits(Grid::linear(1), run + offset as usize),
        "the run and the offset is exactly what it touches"
    );
}

#[test]
fn a_kernel_that_offsets_into_nothing_reports_no_offset() {
    assert_eq!(
        Bounds::of(&kernels::empty(32).expect("built"), &Specialization::none()).offset(),
        0
    );
    assert_eq!(
        Bounds::of(
            &kernels::reduce::lane_sum::<F32, 128>(32).expect("built"),
            &Specialization::none()
        )
        .offset(),
        0,
        "four strips of address arithmetic, and not one element past the run"
    );
}

#[test]
fn a_plane_is_measured_by_its_pitch_and_not_by_its_invocations() {
    let width = 32_usize;
    let pitch = 4096;
    let bounds = Bounds::of(
        &kernels::row_scale(32, pitch as u32, 1, 3).expect("built"),
        &Specialization::none(),
    );
    let grid = Grid::new(1, 64);

    let reached = 63 * pitch + width;
    assert!(bounds.fits(grid, reached));
    assert!(
        !bounds.fits(grid, reached - 1),
        "one element short of the last row's own columns"
    );
    assert!(
        !bounds.fits(grid, 64 * width),
        "the invocation product is what this used to compare, and it is 2 048 of 258 080"
    );
}

#[test]
fn a_plane_the_dispatch_covers_whole_agrees_with_the_invocation_product() {
    let width = 32;
    let pitch = width * 3;
    let bounds = Bounds::of(
        &kernels::row_scale(width, pitch, 1, 3).expect("built"),
        &Specialization::none(),
    );

    let grid = Grid::new(pitch / width, 8);
    let whole = 8 * pitch as usize;

    assert!(bounds.fits(grid, whole));
    assert!(!bounds.fits(grid, whole - 1));
}

#[test]
fn a_grid_more_than_one_row_deep_finds_its_row_among_the_other_sums() {
    let width = 32;
    let pitch = width * 4;
    let bounds = Bounds::of(
        &kernels::row_scale(width, pitch, 2, 3).expect("built"),
        &Specialization::none(),
    );
    let grid = Grid::new(1, 4);

    let reached = 7 * pitch as usize + width as usize;
    assert!(bounds.fits(grid, reached));
    assert!(!bounds.fits(grid, reached - 1));
    assert!(
        !bounds.fits(grid, (width * 4 * 2) as usize),
        "the invocation reading is 256 of 928, and a row this deep must not fall back to it"
    );
}

#[test]
fn the_pitch_is_the_constant_beside_the_row_and_not_the_largest_one_nearby() {
    let width = 32;
    let pitch = width;
    let bounds = Bounds::of(
        &kernels::row_scale(width, pitch, 64, 3).expect("built"),
        &Specialization::none(),
    );
    let grid = Grid::new(1, 2);

    let reached = 127 * pitch as usize + width as usize;
    assert!(
        bounds.fits(grid, reached),
        "the pitch is 32, and reading the 64 beside it would ask for twice this"
    );
    assert!(!bounds.fits(grid, reached - 1));
}

fn with_every_sum_twice(spirv: &[u32]) -> Vec<u32> {
    let mut words = spirv[..5].to_vec();
    let mut next = spirv[3];
    let mut at = 5;

    while at < spirv.len() {
        let count = (spirv[at] >> 16) as usize;
        if count == 0 || at + count > spirv.len() {
            break;
        }
        let instruction = &spirv[at..at + count];
        words.extend_from_slice(instruction);

        if (spirv[at] & 0xffff) as u16 == op::I_ADD && count == 5 {
            let mut copy = instruction.to_vec();
            copy[2] = next;
            next += 1;
            words.extend_from_slice(&copy);
        }
        at += count;
    }

    words[3] = next;
    words
}

#[test]
fn two_terms_of_the_rows_shape_give_no_row_rather_than_the_first_one() {
    let width = 32;
    let pitch = width * 4;
    let spirv = kernels::row_scale(width, pitch, 2, 3).expect("built");
    let grid = Grid::new(1, 4);

    let fallback = (width * 4 * 2) as usize;
    assert!(
        !Bounds::of(&spirv, &Specialization::none()).fits(grid, fallback),
        "the unambiguous module reads its pitch and asks for 928"
    );
    assert!(
        Bounds::of(&with_every_sum_twice(&spirv), &Specialization::none()).fits(grid, fallback),
        "two rows to choose between is no row, and no row is no pitch"
    );
}

fn with_a_second_sum_on_each_base(spirv: &[u32]) -> Vec<u32> {
    let mut words = spirv[..5].to_vec();
    let mut next = spirv[3];
    let mut at = 5;

    while at < spirv.len() {
        let count = (spirv[at] >> 16) as usize;
        if count == 0 || at + count > spirv.len() {
            break;
        }
        let instruction = &spirv[at..at + count];
        words.extend_from_slice(instruction);

        if (spirv[at] & 0xffff) as u16 == op::I_ADD && count == 5 {
            let mut copy = instruction.to_vec();
            copy[2] = next;
            copy[4] = copy[3];
            next += 1;
            words.extend_from_slice(&copy);
        }
        at += count;
    }

    words[3] = next;
    words
}

#[test]
fn a_sum_on_the_rows_base_that_is_not_on_the_lane_is_not_a_second_row() {
    let width = 32;
    let pitch = width * 4;
    let spirv = kernels::row_scale(width, pitch, 2, 3).expect("built");
    let grid = Grid::new(1, 4);
    let fallback = (width * 4 * 2) as usize;

    assert!(
        !Bounds::of(
            &with_a_second_sum_on_each_base(&spirv),
            &Specialization::none()
        )
        .fits(grid, fallback),
        "a sum over the row's base is not a row, and the pitch is still 128"
    );
}

fn with_a_constant_added_off_the_lane(spirv: &[u32]) -> Vec<u32> {
    let integer = super::decode::body(spirv).find_map(|instruction| {
        match (instruction.opcode(), instruction.operands()) {
            (op::I_ADD, [kind, ..]) => Some(*kind),
            _ => None,
        }
    });
    let (Some(integer), Some(constant)) = (integer, largest_constant(spirv, integer)) else {
        return spirv.to_vec();
    };

    let mut words = spirv[..5].to_vec();
    let fresh = spirv[3];
    let mut spliced = false;
    let mut at = 5;

    while at < spirv.len() {
        let count = (spirv[at] >> 16) as usize;
        if count == 0 || at + count > spirv.len() {
            break;
        }
        let instruction = &spirv[at..at + count];

        if !spliced && (spirv[at] & 0xffff) as u16 == op::ACCESS_CHAIN && count == 6 {
            words.extend_from_slice(&[
                (5 << 16) | u32::from(op::I_ADD),
                integer,
                fresh,
                instruction[5],
                constant,
            ]);
            let mut chain = instruction.to_vec();
            chain[5] = fresh;
            words.extend_from_slice(&chain);
            spliced = true;
        } else {
            words.extend_from_slice(instruction);
        }
        at += count;
    }

    words[3] = fresh + 1;
    words
}

fn largest_constant(spirv: &[u32], kind: Option<u32>) -> Option<u32> {
    super::decode::body(spirv)
        .filter_map(
            |instruction| match (instruction.opcode(), instruction.operands()) {
                (op::CONSTANT, [declared, id, literal]) if Some(*declared) == kind => {
                    Some((*literal, *id))
                }
                _ => None,
            },
        )
        .max()
        .map(|(_, id)| id)
}

#[test]
fn a_constant_added_off_the_lane_is_not_the_lanes_offset() {
    let spirv = kernels::reduce::lane_sum::<F32, 128>(32).expect("built");

    assert_eq!(
        Bounds::of(&spirv, &Specialization::none()).offset(),
        0,
        "a reduction reads no further than its run"
    );
    assert_eq!(
        Bounds::of(
            &with_a_constant_added_off_the_lane(&spirv),
            &Specialization::none()
        )
        .offset(),
        0,
        "and a constant added off the lane does not make one"
    );
}

fn without_execution_mode(spirv: &[u32]) -> Vec<u32> {
    let mut words = spirv[..5].to_vec();
    let mut at = 5;
    while at < spirv.len() {
        let count = (spirv[at] >> 16) as usize;
        if count == 0 || at + count > spirv.len() {
            break;
        }
        if (spirv[at] & 0xffff) as u16 != op::EXECUTION_MODE {
            words.extend_from_slice(&spirv[at..at + count]);
        }
        at += count;
    }
    words
}

#[test]
fn a_module_that_declares_no_workgroup_size_is_not_divided_by_it() {
    let stripped =
        without_execution_mode(&kernels::reduce::lane_sum::<F32, 128>(32).expect("built"));

    assert_eq!(local_size(&stripped), None, "the mode is the thing removed");

    let bounds = Bounds::of(&stripped, &Specialization::none());
    assert!(
        bounds.fits(Grid::linear(1 << 20), 1),
        "nothing can be claimed about a module with no workgroup size"
    );
    assert_eq!(bounds.overrun(Grid::linear(1 << 20), &[1, 1]), None);
}

#[test]
fn an_undecodable_module_is_let_through_rather_than_refused() {
    assert!(Bounds::of(&[], &Specialization::none()).fits(Grid::linear(1 << 20), 1));
    assert_eq!(
        Bounds::of(&[], &Specialization::none()).overrun(Grid::linear(1 << 20), &[1, 1]),
        None
    );
}

#[test]
fn the_stride_is_the_one_the_element_type_needs() {
    assert_eq!(element_bytes(&kernels::empty(32).expect("built")), Some(4));

    let bytes = kernels::narrow::narrow_add::<I8, 32>(32, 1).expect("built");
    assert_eq!(element_bytes(&bytes), Some(1));
}

#[test]
fn a_byte_kernel_fills_a_word_with_four_invocations_rather_than_one() {
    let bounds = Bounds::of(
        &kernels::narrow::narrow_add::<I8, 32>(32, 1).expect("built"),
        &Specialization::none(),
    );
    let workgroup = kernels::WORKGROUP_SIZE as usize;

    assert!(
        bounds.fits(Grid::linear(2), workgroup / 2),
        "128 byte-writing invocations fit in 32 words"
    );
    assert!(
        !bounds.fits(Grid::linear(2), workgroup / 2 - 1),
        "and do not fit in 31"
    );
}

#[test]
fn a_word_kernel_and_a_byte_kernel_disagree_by_exactly_four() {
    let words = Bounds::of(&kernels::empty(32).expect("built"), &Specialization::none());
    let bytes = Bounds::of(
        &kernels::narrow::narrow_add::<I8, 32>(32, 1).expect("built"),
        &Specialization::none(),
    );
    let workgroup = kernels::WORKGROUP_SIZE as usize;

    assert!(!words.fits(Grid::linear(4), workgroup));
    assert!(bytes.fits(Grid::linear(4), workgroup));
}

#[test]
fn a_partly_filled_word_still_needs_the_whole_word() {
    let bounds = Bounds::of(
        &kernels::narrow::narrow_add::<I8, 32>(32, 1).expect("built"),
        &Specialization::none(),
    );
    let three = Grid::linear(3);
    let invocations = 3 * kernels::WORKGROUP_SIZE as usize;

    assert_eq!(invocations, 192, "three workgroups of 64");
    assert!(bounds.fits(three, 48), "192 bytes is exactly 48 words");
    assert!(!bounds.fits(three, 47));
}

#[test]
fn each_binding_is_measured_against_its_own_size() {
    let bounds = Bounds::of(
        &kernels::reduce::lane_sum::<F32, 128>(32).expect("built"),
        &Specialization::none(),
    );
    let workgroup = kernels::WORKGROUP_SIZE as usize;

    assert_eq!(
        bounds.overrun(Grid::linear(1), &[workgroup * 4, workgroup]),
        None
    );
    assert_eq!(
        bounds.overrun(Grid::linear(1), &[workgroup, workgroup]),
        Some(Overrun {
            binding: Some(0),
            needed: workgroup * 4,
            held: workgroup,
        }),
        "the input is four strips and only the input is"
    );
    assert_eq!(
        bounds.overrun(Grid::linear(1), &[workgroup * 4, workgroup - 1]),
        Some(Overrun {
            binding: Some(1),
            needed: workgroup,
            held: workgroup - 1,
        }),
        "and the output is still checked, against its own size"
    );
}

#[test]
fn a_binding_addressed_by_workgroup_rather_than_by_invocation_is_left_alone() {
    let spirv = kernels::scan::scan_blocks::<F32>(32).expect("built");
    let bounds = Bounds::of(&spirv, &Specialization::none());
    let workgroup = kernels::WORKGROUP_SIZE as usize;

    assert_eq!(
        bounds.overrun(Grid::linear(4), &[workgroup * 4, workgroup * 4, 4]),
        None,
        "four blocks, four totals"
    );
    assert_eq!(
        bounds.overrun(Grid::linear(4), &[workgroup, workgroup * 4, 4]),
        Some(Overrun {
            binding: Some(0),
            needed: workgroup * 4,
            held: workgroup,
        }),
        "and the per-invocation bindings are checked as before"
    );
}

#[test]
fn a_binding_with_no_size_given_is_not_checked() {
    let bounds = Bounds::of(
        &kernels::reduce::lane_sum::<F32, 128>(32).expect("built"),
        &Specialization::none(),
    );

    assert_eq!(bounds.overrun(Grid::linear(1), &[]), None);
    assert_eq!(
        bounds.overrun(Grid::linear(1), &[kernels::WORKGROUP_SIZE as usize * 4]),
        None,
        "binding 1 has no entry, so it is not judged"
    );
}

fn needed(bounds: &Bounds) -> usize {
    (0..4096_usize)
        .find(|&words| bounds.overrun_uniform(Grid::linear(1), words).is_none())
        .unwrap_or(usize::MAX)
}

#[test]
fn an_open_offset_is_counted_at_the_value_the_pipeline_carries() {
    let spirv = kernels::reduce::fold_halves_open(32).expect("built");

    let unset = Bounds::of(&spirv, &Specialization::none());
    let set = Bounds::of(
        &spirv,
        &Specialization::none().set(kernels::FOLD_HALF_SPEC_ID, 64),
    );

    assert_eq!(
        needed(&set).saturating_sub(needed(&unset)),
        64,
        "specializing the offset to 64 makes this kernel reach 64 elements further, and the bound \\
         has to need exactly that many more"
    );
}

#[test]
fn an_unset_constant_falls_back_to_the_modules_own_default() {
    let spirv = kernels::reduce::fold_halves_open(32).expect("built");

    let unset = Bounds::of(&spirv, &Specialization::none());
    let zeroed = Bounds::of(
        &spirv,
        &Specialization::none().set(kernels::FOLD_HALF_SPEC_ID, 0),
    );

    assert_eq!(needed(&unset), needed(&zeroed));
}

#[test]
fn a_constant_this_module_does_not_declare_is_not_counted() {
    let spirv = kernels::reduce::fold_halves_open(32).expect("built");

    let unset = Bounds::of(&spirv, &Specialization::none());
    let elsewhere = Bounds::of(
        &spirv,
        &Specialization::none().set(kernels::FOLD_HALF_SPEC_ID + 7, 512),
    );

    assert_eq!(needed(&unset), needed(&elsewhere));
}

#[test]
fn a_module_with_no_open_offset_reads_the_same_either_way() {
    let spirv = kernels::reduce::lane_sum::<F32, 32>(32).expect("built");

    let plain = Bounds::of(&spirv, &Specialization::none());
    let offered = Bounds::of(&spirv, &Specialization::none().set(0, 4096));

    assert_eq!(needed(&plain), needed(&offered));
}
