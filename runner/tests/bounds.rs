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
//! **And one door was open in a different way.** A dispatch's reach can be widened by a
//! *specialization constant* — a number chosen when the pipeline is created rather than written
//! into the module — and the bound read the module alone, so it counted zero for that term and let
//! the dispatch through. It reads the pipeline's specialization now, and the last test here is what
//! says so: the same module, the same buffer, refused or accepted according to a number that
//! appears nowhere in it.
//!
//! # Why a refusal is the right answer and not a clamp
//!
//! A kernel writing past the end of a storage buffer is undefined behaviour, and this project has
//! seen both of its faces: an access violation that killed the process on lavapipe at four lanes,
//! and plausible wrong numbers on a device that let the write land somewhere harmless. Clamping the
//! dispatch instead would produce a *partial* answer that looks like an answer.

mod common;

use common::{device, elements, ramp};
use runner::kernels::{self, WORKGROUP_SIZE};
use runner::{Error, Grid, Pass, Specialization};

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
    // **Hand-picked, and it has to be.** Every other gate in this suite asks the module what it
    // needs — `common::runnable` reads its `OpCapability` list — but a `Reducer` and a `Scanner`
    // build their modules *inside* themselves, so there is nothing here to ask. Naming the bit is
    // the only option, and it is written down as a limitation rather than left looking like a
    // choice.
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
    // **Hand-picked, and it has to be.** Every other gate in this suite asks the module what it
    // needs — `common::runnable` reads its `OpCapability` list — but a `Reducer` and a `Scanner`
    // build their modules *inside* themselves, so there is nothing here to ask. Naming the bit is
    // the only option, and it is written down as a limitation rather than left looking like a
    // choice.
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

#[test]
fn a_kernel_reading_past_its_run_is_measured_against_what_it_reads() {
    // **The hole this file did not cover, because the check could not see it.**
    // `Kernel::load_offset` reads `in[i + half]`, and the offset was outside `dispatch::extent`
    // entirely: a buffer exactly as long as the run passed, while the kernel read `half` elements
    // past the end of it. Every other test here provokes a refusal by widening the *dispatch*; this
    // one keeps the dispatch at one workgroup and narrows the buffer, which is the only way the
    // offset shows up at all.
    //
    // `clipped_dot` is the kernel that has it for a reason rather than for a test: activations
    // occupy the first `offset` elements of binding 0 and weights the rest, so it reads exactly
    // twice its run and always has.
    let Some(gpu) = device("bounds-offset") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    if width != 32 {
        eprintln!("SKIPPED bounds-offset: clipped_dot is written for a 32-wide subgroup");
        return;
    }

    // One subgroup folds 256 elements in eight strips, and one workgroup is 64 invocations — so
    // the run is 512 elements and the weights are 512 more.
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

    // One element short of that, which is the case that used to run.
    let short = vec![0_u32; run * 2 - 1];
    let outcome = gpu.run_u32(&spirv, &short, 1);
    assert!(
        overran(&outcome),
        "a buffer one element short of the offset read gave {outcome:?}"
    );

    // And the run alone, which is what a caller who had not read the kernel would allocate.
    let run_only = vec![0_u32; run];
    let outcome = gpu.run_u32(&spirv, &run_only, 1);
    assert!(
        overran(&outcome),
        "a buffer of exactly the run gave {outcome:?}, and the weights are the other half"
    );
}

#[test]
fn a_plane_narrower_than_its_rows_is_measured_by_the_pitch() {
    // **The second half of what `dispatch::extent` could not see, and the larger half.** A grid's
    // rows are `pitch` elements apart whether or not the dispatch covers a row, so a kernel reading
    // a narrow slab of a wide matrix reaches its last row `(rows - 1) × pitch` elements in — while
    // the invocation product it used to be measured by counts only the columns dispatched.
    //
    // `plane.rs`'s own header describes exactly this shape and calls it supported: "a buffer whose
    // rows are 4096 long reads a 64-wide slab of it, and `pitch` is 4096". Every grid test in this
    // crate dispatches `pitch / width` workgroups across instead, which covers a whole row — and on
    // that shape the two readings agree exactly, which is why nothing had ever disagreed.
    let Some(gpu) = device("bounds-pitch") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let pitch = width * 8;
    let height = 4;

    // One workgroup across, which is one subgroup of the eight a row holds.
    let grid = Grid::new(1, height);
    let spirv = kernels::row_scale(width, pitch, 1, 3).expect("built");
    if !common::runnable(&gpu, "bounds-pitch", &[&spirv]) {
        return;
    }

    // What the kernel actually reaches: the last row starts at `(height - 1) × pitch`, and it reads
    // one workgroup's columns from there.
    let reached = ((height - 1) * pitch + width) as usize;
    let mut session = gpu.session(&spirv, &[reached, reached]).expect("opened");
    assert!(
        session.dispatch_grid(grid, 1).is_ok(),
        "the last row's own columns are what this reaches"
    );

    // And the invocation product, which is every column this dispatch touches and none of the gaps
    // between the rows they sit on.
    let product = (width * height) as usize;
    let mut session = gpu.session(&spirv, &[product, product]).expect("opened");
    let outcome = session.dispatch_grid(grid, 1);
    assert!(
        overran(&outcome),
        "a buffer of the columns dispatched, with no room for the pitch between rows, gave {outcome:?}"
    );
}

/// An offset chosen at pipeline creation is inside the bound, not outside it.
///
/// `kernels::reduce::fold_halves_open` reads its second operand at an offset held in a
/// specialization constant, so the module has no literal for it and `Bounds` used to count zero.
/// That is the permissive direction: the dispatch reads `offset` elements past its run and the
/// check said it did not.
///
/// **Three dispatches of one module, and the buffer never changes.** Unspecialized the offset is
/// the module's declared default of zero and the read fits; specialized past the end it must be
/// refused; specialized to something that still fits it must not be. A bound that ignored the
/// specialization would accept all three, and a bound that panicked on one would refuse all three.
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

    // Exactly what one workgroup reads with the offset at its default, and not one word more.
    // `fold_halves_open` is built for the device's own width and reads one element per invocation,
    // so its run is a workgroup whatever the width is. Words rather than floats: `run_specialized`
    // takes the buffer as it lies.
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

    // And the other direction, or the test above would pass on a bound that refused everything.
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
