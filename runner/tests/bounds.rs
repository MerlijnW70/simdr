mod common;

use common::{device, elements, ramp};
use runner::kernels::{self, WORKGROUP_SIZE};
use runner::{Error, Grid, Pass, Specialization};

fn as_words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn overran<T: std::fmt::Debug>(outcome: &Result<T, Error>) -> bool {
    matches!(outcome, Err(Error::Overrun { needed, held, .. }) if needed > held)
}

#[test]
fn a_bound_dispatch_wider_than_its_buffers_is_refused() {
    let Some(gpu) = device("bounds-run-bound") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scale(width, 2.0).expect("built");
    let input = as_words(&ramp(WORKGROUP_SIZE as usize));

    assert!(
        gpu.run_bound(&spirv, &[&input], WORKGROUP_SIZE as usize, 1)
            .is_ok()
    );

    for workgroups in [2_u32, 3, 64] {
        let outcome = gpu.run_bound(&spirv, &[&input], WORKGROUP_SIZE as usize, workgroups);
        assert!(
            overran(&outcome),
            "{workgroups} workgroups over one workgroup's worth of buffer gave {outcome:?}"
        );
    }
}

#[test]
fn a_bound_dispatch_is_measured_against_each_buffer_separately() {
    let Some(gpu) = device("bounds-run-bound-each") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scale(width, 2.0).expect("built");
    let input = as_words(&ramp(2 * WORKGROUP_SIZE as usize));

    assert!(
        gpu.run_bound(&spirv, &[&input], 2 * WORKGROUP_SIZE as usize, 2)
            .is_ok()
    );

    let outcome = gpu.run_bound(&spirv, &[&input], WORKGROUP_SIZE as usize, 2);
    assert!(
        matches!(
            outcome,
            Err(Error::Overrun {
                binding: Some(1),
                ..
            })
        ),
        "a short output binding gave {outcome:?}"
    );
}

#[test]
fn a_session_dispatch_wider_than_its_buffers_is_refused() {
    let Some(gpu) = device("bounds-session") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scale(width, 2.0).expect("built");
    let count = WORKGROUP_SIZE as usize;
    let mut session = gpu.session(&spirv, &[count, count]).expect("opened");

    session.write(0, &as_words(&ramp(count))).expect("uploaded");
    assert!(
        session.dispatch(1, 1).is_ok(),
        "one workgroup is what it holds"
    );

    for workgroups in [2_u32, 5, 32] {
        let outcome = session.dispatch(workgroups, 1);
        assert!(
            overran(&outcome),
            "{workgroups} workgroups over a one-workgroup session gave {outcome:?}"
        );
    }

    assert!(session.dispatch(1, 1).is_ok());
    assert_eq!(session.read(1, count).expect("read back").len(), count);
}

#[test]
fn a_chained_pass_wider_than_the_buffers_is_refused_before_anything_is_submitted() {
    let Some(gpu) = device("bounds-chain") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scale(width, 2.0).expect("built");
    let input = as_words(&ramp(2 * WORKGROUP_SIZE as usize));

    let fitting = [Pass::new(&spirv, 2), Pass::new(&spirv, 1)];
    assert!(gpu.run_chain(&fitting, &input).is_ok());

    let overrunning = [Pass::new(&spirv, 2), Pass::new(&spirv, 3)];
    let outcome = gpu.run_chain(&overrunning, &input);
    assert!(
        overran(&outcome),
        "a chain whose second pass overruns gave {outcome:?}"
    );
}

#[test]
fn a_held_reduction_checks_every_stage_it_plans() {
    let Some(gpu) = device("bounds-reducer") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED bounds-reducer: no subgroup arithmetic reported");
        return;
    }

    for elements in [128_usize, 256, 1024, 4096] {
        let mut reducer = gpu.reducer(elements).expect("built");
        let input = ramp(elements);
        let total: f32 = input.iter().sum();

        assert_eq!(
            reducer.sum(&input).expect("summed").total,
            total,
            "a reduction over {elements} elements"
        );
    }
}

#[test]
fn a_held_scan_checks_every_pass_it_records() {
    let Some(gpu) = device("bounds-scanner") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED bounds-scanner: no subgroup arithmetic reported");
        return;
    }

    for elements in [WORKGROUP_SIZE as usize, 4096, 1 << 16] {
        let mut scanner = gpu.scanner(elements).expect("built");
        let input = vec![1.0_f32; elements];
        let scanned = scanner.scan(&input).expect("scanned");

        let last = scanned.last().copied().expect("an answer");
        assert_eq!(
            last, elements as f32,
            "a scan over {elements} elements ends on the grand total"
        );
    }
}

#[test]
fn a_kernel_this_cannot_read_is_let_through_rather_than_refused() {
    let Some(gpu) = device("bounds-unreadable") else {
        return;
    };

    let spirv = kernels::empty(gpu.limits().subgroup_size).expect("built");
    let mut session = gpu.session(&spirv, &[8, 8]).expect("opened");

    assert!(session.dispatch(64, 1).is_ok());
}

#[test]
fn a_kernel_reading_past_its_run_is_measured_against_what_it_reads() {
    let Some(gpu) = device("bounds-offset") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    if width != 32 {
        eprintln!("SKIPPED bounds-offset: clipped_dot is written for a 32-wide subgroup");
        return;
    }

    let run = WORKGROUP_SIZE as usize * 8;
    let spirv = kernels::network::clipped_dot::<256>(width, run as u32, 255).expect("built");
    if !common::runnable(&gpu, "bounds-offset", &[&spirv]) {
        return;
    }

    let whole = vec![0_u32; run * 2];
    assert!(
        gpu.run_u32(&spirv, &whole, 1).is_ok(),
        "the run and the offset together is what this kernel touches"
    );

    let short = vec![0_u32; run * 2 - 1];
    let outcome = gpu.run_u32(&spirv, &short, 1);
    assert!(
        overran(&outcome),
        "a buffer one element short of the offset read gave {outcome:?}"
    );

    let run_only = vec![0_u32; run];
    let outcome = gpu.run_u32(&spirv, &run_only, 1);
    assert!(
        overran(&outcome),
        "a buffer of exactly the run gave {outcome:?}, and the weights are the other half"
    );
}

#[test]
fn a_plane_narrower_than_its_rows_is_measured_by_the_pitch() {
    let Some(gpu) = device("bounds-pitch") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let pitch = width * 8;
    let height = 4;

    let grid = Grid::new(1, height);
    let spirv = kernels::row_scale(width, pitch, 1, 3).expect("built");
    if !common::runnable(&gpu, "bounds-pitch", &[&spirv]) {
        return;
    }

    let reached = ((height - 1) * pitch + width) as usize;
    let mut session = gpu.session(&spirv, &[reached, reached]).expect("opened");
    assert!(
        session.dispatch_grid(grid, 1).is_ok(),
        "the last row's own columns are what this reaches"
    );

    let product = (width * height) as usize;
    let mut session = gpu.session(&spirv, &[product, product]).expect("opened");
    let outcome = session.dispatch_grid(grid, 1);
    assert!(
        overran(&outcome),
        "a buffer of the columns dispatched, with no room for the pitch between rows, gave {outcome:?}"
    );
}

#[test]
fn an_offset_chosen_when_the_pipeline_is_built_is_bounded_like_any_other() {
    let Some(gpu) = device("bounds-specialized") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let Ok(spirv) = kernels::reduce::fold_halves_open(width) else {
        eprintln!("SKIPPED bounds-specialized: no mapping for this width");
        return;
    };

    let run = elements(width, width);
    let input: Vec<u32> = (0..run).map(|index| (index as f32).to_bits()).collect();

    assert!(
        gpu.run_specialized(&spirv, &input, 1, &Specialization::none())
            .is_ok(),
        "the default offset is zero, so this reads its own run twice and fits"
    );

    let refused = gpu.run_specialized(
        &spirv,
        &input,
        1,
        &Specialization::none().set(kernels::FOLD_HALF_SPEC_ID, run as u32),
    );
    assert!(
        matches!(refused, Err(Error::Overrun { .. })),
        "an offset of a whole workgroup reads a workgroup past the end and has to be refused, \\
         and the number that says so is in the pipeline rather than in the module: {refused:?}"
    );

    let room: Vec<u32> = (0..run * 2).map(|index| (index as f32).to_bits()).collect();
    assert!(
        gpu.run_specialized(
            &spirv,
            &room,
            1,
            &Specialization::none().set(kernels::FOLD_HALF_SPEC_ID, run as u32),
        )
        .is_ok(),
        "the same offset over a buffer twice as long fits, so the refusal above is about the reach \\
         rather than about the constant being set at all"
    );
}
