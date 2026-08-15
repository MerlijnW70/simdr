//! Buffers and a pipeline held across dispatches.
//!
//! Two things have to be true and they pull against each other: a session must give the *same*
//! answer as a fresh `Gpu::run`, and it must be very much faster. A cache that returns stale data
//! would satisfy the second on its own.

mod common;

use common::{device, grouped_sums, ramp};
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::F32;
use std::time::Instant;

/// Words in, words out, for a kernel that reads binding 0 and writes binding 1.
fn as_words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn a_session_gives_the_same_answer_as_a_fresh_run() {
    let Some(gpu) = device("session-agrees") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED session-agrees: no subgroup arithmetic reported");
        return;
    }

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let input = ramp(count);
    let spirv = kernels::lane_sum_whole::<F32>(width).expect("built");

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
    // The failure a cache would have. Three different inputs through one session, each answered
    // on its own terms.
    let Some(gpu) = device("session-reuse") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED session-reuse: no subgroup arithmetic reported");
        return;
    }

    let width = limits.subgroup_size;
    let count = WORKGROUP_SIZE as usize;
    let spirv = kernels::lane_sum_whole::<F32>(width).expect("built");
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

    // And the three genuinely differed, so none of this passed by returning a constant.
    seen.dedup();
    assert_eq!(seen.len(), 3);
}

#[test]
fn a_session_answers_far_faster_than_rebuilding_everything() {
    // The whole reason the type exists. Measured rather than asserted in the abstract: allocating
    // and freeing a buffer costs ~310 us on this hardware whatever its size, and `run` does three
    // of them plus a pipeline on every call.
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

    // **A ratio is a measurement, and measurements do not travel.** `.github/workflows/ci.yml`
    // lists three things a shared runner cannot answer for, and the third is *every measurement* —
    // while running this one. It failed there at 2.3× on lavapipe at width 4, against a bar of
    // three, having passed the run before: which is what a contended virtual machine does to two
    // wall-clock numbers whose ratio is the assertion.
    //
    // The comment this replaces is the whole argument, one size down. The bar was **ten**, which is
    // a comfortable margin on the discrete GPU this was written on — 52× measured — and fails on
    // the integrated part in the same machine at 5×. "Ten was a property of one device dressed up
    // as a property of sessions." Three is a property of *two* devices dressed up the same way, and
    // a third machine said so.
    //
    // So the number is still printed everywhere, because it is the honest half of a benchmark
    // inside a test suite — and it is asserted only where a timing means something. Reported
    // loudly when it is not, the way this suite reports a missing device: a skipped check that
    // looks green is worse than a red one.
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

/// The hole a `Session` opened, and closed.
///
/// `Buffer::write` used to assume the caller's slice fit, and its comment said why: "this crate
/// always allocates from the same element count it writes". True while `Gpu::run` was the only
/// caller. A session's staging buffer is sized to its *largest* binding and `Session::write` takes
/// a slice from outside, so a long one memcpyd past the end of a mapping — from safe code, in a
/// crate whose whole claim is that it cannot.
#[test]
fn writing_more_than_the_buffer_holds_is_refused_rather_than_overflowing() {
    let Some(gpu) = device("session-overflow") else {
        return;
    };

    let spirv = kernels::empty(gpu.limits().subgroup_size).expect("built");
    let mut session = gpu.session(&spirv, &[64, 64]).expect("opened");

    // Sixteen thousand words into a buffer that holds sixty-four.
    let far_too_many = vec![0xAAAA_AAAA_u32; 16_384];
    assert!(matches!(
        session.write(0, &far_too_many),
        Err(runner::Error::TooLarge { .. })
    ));

    // And reading past the end, which is the worse direction: it would have handed back whatever
    // was next in the address space, looking like an answer.
    assert!(matches!(
        session.read(1, 16_384),
        Err(runner::Error::TooLarge { .. })
    ));

    // The session still works afterwards. A refusal is not a poisoning.
    session.write(0, &[1, 2, 3]).expect("still usable");
    session.dispatch(1, 1).expect("still usable");
    assert_eq!(session.read(1, 64).expect("still usable").len(), 64);
}

#[test]
fn writing_nothing_leaves_the_binding_alone() {
    // A zero-byte `vkCmdCopyBuffer` is not allowed, so the size is floored at one word — and
    // copying that word would put whatever staging last held into a binding the caller said
    // nothing about. An empty write has to be a no-op rather than a one-word one.
    let Some(gpu) = device("session-empty-write") else {
        return;
    };

    let spirv = kernels::empty(gpu.limits().subgroup_size).expect("built");
    let mut session = gpu.session(&spirv, &[8, 8]).expect("opened");

    session.write(0, &[0xFFFF_FFFF; 8]).expect("first");
    session.write(1, &[0x1111_1111; 8]).expect("marker");

    // Now write nothing to binding 1. Its contents must not change, and in particular must not
    // pick up the 0xFFFF_FFFF that staging is still holding from the first write.
    session.write(1, &[]).expect("empty write");

    assert_eq!(
        session.read(1, 8).expect("read back"),
        vec![0x1111_1111_u32; 8],
        "an empty write left something behind"
    );
}

/// The subtler half: fitting in *staging* is not the same as fitting in the binding.
///
/// Staging is sized to the largest binding, so a write that overflows a small binding while
/// staying inside staging used to be clamped and reported as success. A short write is a wrong
/// answer arriving later, and this crate refuses rather than truncates everywhere else.
#[test]
fn writing_past_a_small_binding_is_refused_even_though_staging_would_hold_it() {
    let Some(gpu) = device("session-binding-bound") else {
        return;
    };

    let spirv = kernels::empty(gpu.limits().subgroup_size).expect("built");
    // Binding 0 holds 64 words; binding 1 holds 4096, so staging is 4096 wide.
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

    // The same slice into the larger binding is fine, which is what says the bound is per-binding
    // and not a blanket limit.
    session.write(1, &five_hundred).expect("fits binding 1");

    // And reading is bounded the same way round.
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
