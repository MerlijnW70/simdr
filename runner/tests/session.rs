mod common;

use common::{device, grouped_sums, ramp, runnable};
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;
use std::time::Instant;

fn as_words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn a_session_gives_the_same_answer_as_a_fresh_run() {
    let Some(gpu) = device("session-agrees") else {
        return;
    };
    let limits = gpu.limits().clone();

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    let spirv = kernels::lane_sum_whole::<F32>(width).expect("built");
    if !runnable(&gpu, "session-agrees", &[&spirv]) {
        return;
    }

    let once = gpu.run(&spirv, &input, 1).expect("dispatched");

    let mut session = gpu.session(&spirv, &[count, count]).expect("opened");
    session.write(0, &as_words(&input)).expect("uploaded");
    session.dispatch(1, 1).expect("dispatched");
    let held = session.read(1, count).expect("read back");

    let held: Vec<f32> = held.into_iter().map(f32::from_bits).collect();
    assert_eq!(held, once);
    assert_eq!(held, grouped_sums(count, width as usize));
}

#[test]
fn a_session_reused_does_not_return_the_first_answer_again() {
    let Some(gpu) = device("session-reuse") else {
        return;
    };
    let limits = gpu.limits().clone();

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let spirv = kernels::lane_sum_whole::<F32>(width).expect("built");
    if !runnable(&gpu, "session-reuse", &[&spirv]) {
        return;
    }

    let mut session = gpu.session(&spirv, &[count, count]).expect("opened");

    let mut seen = Vec::new();
    for scale in [1.0_f32, 2.0, 7.0] {
        let input: Vec<f32> = ramp(count).iter().map(|value| value * scale).collect();

        session.write(0, &as_words(&input)).expect("uploaded");
        session.dispatch(1, 1).expect("dispatched");
        let output = session.read(1, count).expect("read back");

        let expected: Vec<f32> = grouped_sums(count, width as usize)
            .iter()
            .map(|total| total * scale)
            .collect();
        let actual: Vec<f32> = output.into_iter().map(f32::from_bits).collect();

        assert_eq!(actual, expected, "at scale {scale}");
        seen.push(actual.first().copied().unwrap_or_default());
    }

    seen.dedup();
    assert_eq!(seen.len(), 3);
}

#[test]
fn a_session_answers_far_faster_than_rebuilding_everything() {
    let Some(gpu) = device("session-speed") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = as_words(&ramp(count));
    let spirv = kernels::empty(width).expect("built");
    let trips = 50;

    gpu.run_u32(&spirv, &input, 1).expect("warm");
    let started = Instant::now();
    for _ in 0..trips {
        gpu.run_u32(&spirv, &input, 1).expect("dispatched");
    }
    let per_run = started.elapsed() / trips;

    let mut session = gpu.session(&spirv, &[count, count]).expect("opened");
    session.dispatch(1, 1).expect("warm");
    let started = Instant::now();
    for _ in 0..trips {
        session.dispatch(1, 1).expect("dispatched");
    }
    let per_dispatch = started.elapsed() / trips;

    eprintln!(
        "session: {:.1} us per dispatch against {:.1} us per run ({:.0}x)",
        per_dispatch.as_secs_f64() * 1e6,
        per_run.as_secs_f64() * 1e6,
        per_run.as_secs_f64() / per_dispatch.as_secs_f64()
    );

    if std::env::var_os("CI").is_some() {
        eprintln!(
            "SKIPPED session-speed ratio: CI is set, and a shared runner's wall clock is not \
             evidence about setup cost. The measurement above still ran."
        );
        return;
    }

    assert!(
        per_dispatch * 3 < per_run,
        "a held pipeline was not even three times faster than rebuilding one, \
         which means the setup cost this type exists to remove is not there"
    );
}

#[test]
fn writing_more_than_the_buffer_holds_is_refused_rather_than_overflowing() {
    let Some(gpu) = device("session-overflow") else {
        return;
    };

    let spirv = kernels::empty(gpu.limits().subgroup_size).expect("built");
    let mut session = gpu.session(&spirv, &[64, 64]).expect("opened");

    let far_too_many = vec![0xAAAA_AAAA_u32; 16_384];
    assert!(matches!(
        session.write(0, &far_too_many),
        Err(runner::Error::TooLarge { .. })
    ));

    assert!(matches!(
        session.read(1, 16_384),
        Err(runner::Error::TooLarge { .. })
    ));

    session.write(0, &[1, 2, 3]).expect("still usable");
    session.dispatch(1, 1).expect("still usable");
    assert_eq!(session.read(1, 64).expect("still usable").len(), 64);
}

#[test]
fn writing_nothing_leaves_the_binding_alone() {
    let Some(gpu) = device("session-empty-write") else {
        return;
    };

    let spirv = kernels::empty(gpu.limits().subgroup_size).expect("built");
    let mut session = gpu.session(&spirv, &[8, 8]).expect("opened");

    session.write(0, &[0xFFFF_FFFF; 8]).expect("first");
    session.write(1, &[0x1111_1111; 8]).expect("marker");

    session.write(1, &[]).expect("empty write");

    assert_eq!(
        session.read(1, 8).expect("read back"),
        vec![0x1111_1111_u32; 8],
        "an empty write left something behind"
    );
}

#[test]
fn writing_past_a_small_binding_is_refused_even_though_staging_would_hold_it() {
    let Some(gpu) = device("session-binding-bound") else {
        return;
    };

    let spirv = kernels::empty(gpu.limits().subgroup_size).expect("built");
    let mut session = gpu.session(&spirv, &[64, 4_096]).expect("opened");

    let five_hundred = vec![7_u32; 500];
    assert!(
        matches!(
            session.write(0, &five_hundred),
            Err(runner::Error::TooLarge {
                words: 500,
                capacity: 64
            })
        ),
        "500 words into a 64-word binding was accepted"
    );

    session.write(1, &five_hundred).expect("fits binding 1");

    assert!(matches!(
        session.read(0, 500),
        Err(runner::Error::TooLarge { .. })
    ));
    assert_eq!(session.read(1, 500).expect("fits").len(), 500);
}

#[test]
fn a_session_with_no_bindings_is_refused() {
    let Some(gpu) = device("session-refusal") else {
        return;
    };

    let spirv = kernels::empty(gpu.limits().subgroup_size).expect("built");
    assert!(gpu.session(&spirv, &[]).is_err());
}

#[test]
fn reading_or_writing_a_binding_that_does_not_exist_is_refused() {
    let Some(gpu) = device("session-bounds") else {
        return;
    };

    let spirv = kernels::empty(gpu.limits().subgroup_size).expect("built");
    let mut session = gpu.session(&spirv, &[64, 64]).expect("opened");

    assert_eq!(session.bindings(), 2);
    assert!(session.write(2, &[1, 2, 3]).is_err());
    assert!(session.read(2, 4).is_err());
}
