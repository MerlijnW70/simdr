//! The dispatch bound, at every entry point that dispatches.
//!
//! `dispatch::extent` reads a module's workgroup size, element stride and address arithmetic and
//! refuses a dispatch that would touch more of a buffer than the buffer holds. It is the seventh
//! layer of the stack `README.md` describes, and it found eleven tests reading past their inputs
//! the first time it ran.
//!
//! **It guarded one of the six ways this crate dispatches.** `Gpu::run` was checked;
//! `Gpu::run_bound`, `Session::dispatch`, `Gpu::run_chain`, `Gpu::reducer` and `Gpu::scanner` were
//! not, and each of them is a way to write past the end of a binding from safe code. A layer that
//! covers one caller is a layer whose absence looks exactly like its presence everywhere else.
//!
//! Each test below asks the same question of a different door: dispatch more workgroups than the
//! buffers can hold, and expect [`runner::Error::Overrun`] rather than a submission. `execution.rs`
//! holds the `Gpu::run` half, where the check already was.
//!
//! # Why a refusal is the right answer and not a clamp
//!
//! A kernel writing past the end of a storage buffer is undefined behaviour, and this project has
//! seen both of its faces: an access violation that killed the process on lavapipe at four lanes,
//! and plausible wrong numbers on a device that let the write land somewhere harmless. Clamping the
//! dispatch instead would produce a *partial* answer that looks like an answer.

mod common;

use common::{device, ramp};
use runner::kernels::{self, WORKGROUP_SIZE};
use runner::{Error, Pass};

/// Words in, for a kernel that reads binding 0 and writes binding 1.
fn as_words(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Whether an outcome is the refusal this file is about, with numbers that make sense.
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

    // One workgroup over one workgroup's worth: exactly full.
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
    // **What a single length could not say.** `run_bound` exists so that buffers can be different
    // sizes — a weight table beside a one-word answer — so the check has to be able to refuse one
    // binding while accepting another. Here the output is the short one.
    let Some(gpu) = device("bounds-run-bound-each") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scale(width, 2.0).expect("built");
    let input = as_words(&ramp(2 * WORKGROUP_SIZE as usize));

    // Both bindings large enough for two workgroups.
    assert!(
        gpu.run_bound(&spirv, &[&input], 2 * WORKGROUP_SIZE as usize, 2)
            .is_ok()
    );

    // The input is still large enough and the output is not, so binding 1 is the one named.
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
    // The one path where the buffers are fixed long before the workgroup count arrives, which is
    // exactly why it needs a check of its own: a session sized for one dispatch is a session a
    // caller can ask for a larger one from.
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

    // A refusal is not a poisoning: the session still runs the dispatch it was sized for.
    assert!(session.dispatch(1, 1).is_ok());
    assert_eq!(session.read(1, count).expect("read back").len(), count);
}

#[test]
fn a_chained_pass_wider_than_the_buffers_is_refused_before_anything_is_submitted() {
    // A chain is where an overrun is hardest to see: every pass reads what the one before it
    // wrote, so a pass that runs off the end corrupts the *next* pass's input and the wrong number
    // arrives a dispatch away from its cause.
    let Some(gpu) = device("bounds-chain") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scale(width, 2.0).expect("built");
    let input = as_words(&ramp(2 * WORKGROUP_SIZE as usize));

    let fitting = [Pass::new(&spirv, 2), Pass::new(&spirv, 1)];
    assert!(gpu.run_chain(&fitting, &input).is_ok());

    // The second pass is the one that does not fit, and it is refused as readily as the first.
    let overrunning = [Pass::new(&spirv, 2), Pass::new(&spirv, 3)];
    let outcome = gpu.run_chain(&overrunning, &input);
    assert!(
        overran(&outcome),
        "a chain whose second pass overruns gave {outcome:?}"
    );
}

#[test]
fn a_held_reduction_checks_every_stage_it_plans() {
    // The reducer decides its own dispatch counts from the element count, so this cannot be
    // provoked from outside — which is the point. The check is there so that a change to
    // `reduction::plan` that dispatched one workgroup too many would be refused at construction
    // rather than folding whatever was past the end of the buffer into the total.
    //
    // What is asserted here is the other direction: every length the reducer accepts still builds.
    // A check that refused a legitimate plan would take the whole type out.
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
    // As above, from the other side of the same machinery: a scanner records seven dispatches at
    // 2²⁰ and each one now passes the bound before it becomes a pipeline.
    let Some(gpu) = device("bounds-scanner") else {
        return;
    };
    if !gpu.limits().subgroup_arithmetic {
        eprintln!("SKIPPED bounds-scanner: no subgroup arithmetic reported");
        return;
    }

    for elements in [WORKGROUP_SIZE as usize, 4096, 1 << 16] {
        let mut scanner = gpu.scanner(elements).expect("built");
        // Ones rather than a ramp: at 2¹⁶ elements a ramp's running total leaves the 2²⁴ an `f32`
        // counts exactly, and what would be under test then is rounding rather than the scan.
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
    // "This runner cannot tell" must never be reported as "your module is wrong". An empty kernel
    // touches no buffer at all, so nothing about it can be measured — and a check that refused it
    // would refuse every module whose addressing it does not recognise.
    let Some(gpu) = device("bounds-unreadable") else {
        return;
    };

    let spirv = kernels::empty(gpu.limits().subgroup_size).expect("built");
    let mut session = gpu.session(&spirv, &[8, 8]).expect("opened");

    // Far more workgroups than eight words could hold, if it wrote anything.
    assert!(session.dispatch(64, 1).is_ok());
}
