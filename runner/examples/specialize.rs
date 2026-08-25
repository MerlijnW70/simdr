//! What "one module per parameter value" actually costs, and what deferring the value saves.
//!
//! `notes/NEXT.md` put specialization constants on the list on the grounds that `Gpu::sum` builds
//! ten modules for ten fold sizes, and that modules are cheap in bytes but not in *pipeline
//! creation*. That is an argument. This is the measurement, and it separates the three costs a
//! reduction pays before it dispatches anything:
//!
//! - **Emitting** a module — pure computation, no device involved.
//! - **Creating a pipeline** from it — `vkCreateShaderModule` plus `vkCreateComputePipeline`,
//!   which is where the driver compiles the shader.
//! - The same, from **one** module specialized ten different ways.
//!
//! The third is the one the argument rests on. A specialization constant is fixed when the
//! pipeline is created, so ten values still need ten pipelines — what it removes is ten *modules*,
//! not ten compilations. Whether that is most of the cost or a rounding error is the question.

mod common;

use runner::kernels::{self, FOLD_HALF_SPEC_ID};
use runner::{Gpu, Specialization, Timing};
use std::time::Duration;

/// A buffer of a million elements folds fifteen times, which is the shape `Gpu::sum` runs.
const ELEMENTS: usize = 1 << 20;

/// How many times to repeat each measurement.
const REPEATS: u32 = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}\n", gpu.limits().name);

    let folds = runner::reduction::folds(ELEMENTS);
    println!(
        "a reduction over {ELEMENTS} elements is {} folds, so {} modules today",
        folds.len(),
        folds.len()
    );

    // 1. Emitting the modules. No device, no driver — this is the emitter alone.
    let emitting = repeat(|| {
        for fold in &folds {
            kernels::fold_by(width, fold.factor, fold.stride).expect("built");
        }
    })?;

    // 2. One pipeline per module, which is what the chain does now.
    //
    // Batched, so the two buffers a descriptor set needs are allocated once rather than per
    // pipeline. An earlier version of this allocated a pair per call and reported 485 µs where
    // the pipeline itself is nearer 180 — allocation was the larger half of its own measurement.
    let modules: Vec<Vec<u32>> = folds
        .iter()
        .map(|fold| kernels::fold_by(width, fold.factor, fold.stride).expect("built"))
        .collect();
    let none = Specialization::none();
    let per_module_builds: Vec<(&[u32], &Specialization)> = modules
        .iter()
        .map(|words| (words.as_slice(), &none))
        .collect();

    // Warm, so the first compile of each does not land in the average.
    gpu.probe_pipelines(&per_module_builds)?;
    let per_module = repeat_result(|| gpu.probe_pipelines(&per_module_builds))?;

    // 3. One module, fourteen specializations. The same number of pipelines, one module.
    let open = kernels::fold_halves_open(width)?;
    let specializations: Vec<Specialization> = folds
        .iter()
        .map(|fold| Specialization::none().set(FOLD_HALF_SPEC_ID, fold.stride))
        .collect();
    let specialized_builds: Vec<(&[u32], &Specialization)> = specializations
        .iter()
        .map(|specialization| (open.as_slice(), specialization))
        .collect();

    gpu.probe_pipelines(&specialized_builds)?;
    let specialized = repeat_result(|| gpu.probe_pipelines(&specialized_builds))?;

    println!("\n{:>34} {:>12} {:>12}", "", "all folds", "per fold");
    row("emitting the modules", emitting, folds.len());
    row("pipelines, one module each", per_module, folds.len());
    row(
        "pipelines, one module specialized",
        specialized,
        folds.len(),
    );

    // The comparison that means something is between the two *strategies*, each paying for what it
    // actually does: fourteen modules and fourteen pipelines, against one module and fourteen
    // pipelines. Comparing the pipeline columns alone would leave out the thing specialization
    // removes.
    let one_emission = emitting.median / folds.len() as u32;
    let today = emitting.median + per_module.median;
    let deferred = one_emission + specialized.median;

    // All three feed both strategy totals, so a wandering repeat anywhere makes both of them — and
    // the percentage below, which `decisions/DR-0005` now quotes — unquotable.
    let mark = if emitting.is_steady() && per_module.is_steady() && specialized.is_steady() {
        ""
    } else {
        "!"
    };
    let saved = today.as_secs_f64() - deferred.as_secs_f64();

    println!("\n{:>34} {:>12}", "", "setup per call");
    println!(
        "{:>34} {:>12}",
        "fourteen modules, fourteen pipelines",
        format!("{}{mark}", micros(today))
    );
    println!(
        "{:>34} {:>12}",
        "one module, fourteen pipelines",
        format!("{}{mark}", micros(deferred))
    );
    println!(
        "\nSpecializing removes {:.1}%{mark} of the setup — {}. Emission is {:.1}% of what building the\n\
         pipelines costs, and a specialization constant is fixed *at* pipeline creation, so\n\
         fourteen values still need fourteen pipelines however few modules they came from.",
        saved / today.as_secs_f64() * 100.0,
        micros(Duration::from_secs_f64(saved.abs())),
        emitting.median.as_secs_f64() / per_module.median.as_secs_f64() * 100.0
    );
    println!(
        "\nCompare `cargo run --release --example reducer -p runner`, which removes the setup\n\
         entirely by keeping the pipelines. That is the larger number by a long way."
    );

    Ok(())
}

/// Time `body` `REPEATS` times, `common::SAMPLES` times over, and summarise.
///
/// The mean of one batch was what this returned, and the headline it feeds — how much of the setup
/// specializing removes — is now quoted in `decisions/DR-0005`. A number that reaches a decision
/// record has to carry its spread with it.
fn repeat(mut body: impl FnMut()) -> Result<Timing, Box<dyn std::error::Error>> {
    let batches = common::host(common::SAMPLES, || {
        for _ in 0..REPEATS {
            body();
        }
        Ok::<(), runner::Error>(())
    })?;
    Ok(per_iteration(batches))
}

/// The same, for a body that can fail.
fn repeat_result(
    mut body: impl FnMut() -> Result<(), runner::Error>,
) -> Result<Timing, Box<dyn std::error::Error>> {
    let batches = common::host(common::SAMPLES, || {
        for _ in 0..REPEATS {
            body()?;
        }
        Ok::<(), runner::Error>(())
    })?;
    Ok(per_iteration(batches))
}

/// A batch of `REPEATS` divided back down to one of them.
///
/// `common::host` reports per batch, the way `Gpu::time_repeated` does, and every figure in this
/// file means *one* emission or *one* pipeline build. Dividing here keeps that true in one place
/// rather than at each of the six sites that print one.
fn per_iteration(batches: Timing) -> Timing {
    Timing {
        best: batches.best / REPEATS,
        median: batches.median / REPEATS,
        worst: batches.worst / REPEATS,
        repeats: batches.repeats,
    }
}

/// One line of the table.
fn row(label: &str, timing: Timing, count: usize) {
    let mark = common::mark(timing);
    println!(
        "{label:>34} {:>12} {:>12}",
        format!("{}{mark}", micros(timing.median)),
        format!("{}{mark}", micros(timing.median / count as u32))
    );
}

/// Microseconds, which is the scale these land on.
fn micros(duration: Duration) -> String {
    format!("{:.1} us", duration.as_secs_f64() * 1e6)
}
