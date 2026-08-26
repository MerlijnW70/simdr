//! What "one module per parameter value" actually costs, and what deferring the value saves.
//!
//! `notes/NEXT.md` put specialization constants on the list on the grounds that `Gpu::sum` builds
//! one module per fold size, and that modules are cheap in bytes but not in *pipeline creation*.
//! That is an argument. This is the measurement, and it separates the three costs a reduction pays
//! before it dispatches anything:
//!
//! - **Emitting** a module — pure computation, no device involved.
//! - **Creating a pipeline** from it — `vkCreateShaderModule` plus `vkCreateComputePipeline`,
//!   which is where the driver compiles the shader.
//! - The same, from **one** module specialized as many different ways.
//!
//! The third is the one the argument rests on. A specialization constant is fixed when the
//! pipeline is created, so *n* values still need *n* pipelines — what it removes is *n* modules,
//! not *n* compilations. Whether that is most of the cost or a rounding error is the question.
//!
//! **Read the spread before the ratio.** The same four builds measure at 0.6 ms from this example
//! and at 21 ms from a test binary doing the identical calls, in debug and in release alike, and no
//! cause for that has been found. `decisions/DR-0005` records it as unexplained.

use runner::kernels::{self, FOLD_HALF_SPEC_ID};
use runner::{Gpu, Specialization};
use std::time::{Duration, Instant};

/// The buffer size the shape is measured at. How many folds it takes is computed, not assumed:
/// `reduction::folds` decides, and every label below is built from its length.
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
    });

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

    // 3. One module, one specialization per fold. The same number of pipelines, one module.
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
    // actually does: one module per fold and a pipeline each, against one module and a pipeline per
    // fold. Comparing the pipeline columns alone would leave out the thing specialization removes.
    let n = folds.len();
    let one_emission = emitting / folds.len() as u32;
    let today = emitting + per_module;
    let deferred = one_emission + specialized;
    let saved = today.as_secs_f64() - deferred.as_secs_f64();

    println!("\n{:>34} {:>12}", "", "setup per call");
    println!(
        "{:>34} {:>12}",
        &format!("{} modules, {} pipelines", folds.len(), folds.len()),
        micros(today)
    );
    println!(
        "{:>34} {:>12}",
        &format!("one module, {} pipelines", folds.len()),
        micros(deferred)
    );
    println!(
        "\nSpecializing removes {:.1}% of the setup — {}. Emission is {:.1}% of what building the\n\
         pipelines costs, and a specialization constant is fixed *at* pipeline creation, so\n\
         {n} values still need {n} pipelines however few modules they came from.",
        saved / today.as_secs_f64() * 100.0,
        micros(Duration::from_secs_f64(saved.abs())),
        emitting.as_secs_f64() / per_module.as_secs_f64() * 100.0
    );
    println!(
        "\nCompare `cargo run --release --example reducer -p runner`, which removes the setup\n\
         entirely by keeping the pipelines. That is the larger number by a long way."
    );

    Ok(())
}

/// Time `body` `REPEATS` times and return the mean.
fn repeat(mut body: impl FnMut()) -> Duration {
    let started = Instant::now();
    for _ in 0..REPEATS {
        body();
    }
    started.elapsed() / REPEATS
}

/// The same, for a body that can fail.
fn repeat_result(
    mut body: impl FnMut() -> Result<(), runner::Error>,
) -> Result<Duration, runner::Error> {
    let started = Instant::now();
    for _ in 0..REPEATS {
        body()?;
    }
    Ok(started.elapsed() / REPEATS)
}

/// One line of the table.
fn row(label: &str, total: Duration, count: usize) {
    println!(
        "{label:>34} {:>12} {:>12}",
        micros(total),
        micros(total / count as u32)
    );
}

/// Microseconds, which is the scale these land on.
fn micros(duration: Duration) -> String {
    format!("{:.1} us", duration.as_secs_f64() * 1e6)
}
