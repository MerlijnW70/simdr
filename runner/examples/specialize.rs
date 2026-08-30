use runner::kernels::{self, FOLD_HALF_SPEC_ID};
use runner::{Gpu, Specialization};
use std::time::{Duration, Instant};

const ELEMENTS: usize = 1 << 20;

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

    let emitting = repeat(|| {
        for fold in &folds {
            kernels::fold_by(width, fold.factor, fold.stride).expect("built");
        }
    });

    let modules: Vec<Vec<u32>> = folds
        .iter()
        .map(|fold| kernels::fold_by(width, fold.factor, fold.stride).expect("built"))
        .collect();
    let none = Specialization::none();
    let per_module_builds: Vec<(&[u32], &Specialization)> = modules
        .iter()
        .map(|words| (words.as_slice(), &none))
        .collect();

    gpu.probe_pipelines(&per_module_builds)?;
    let per_module = repeat_result(|| gpu.probe_pipelines(&per_module_builds))?;

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

fn repeat(mut body: impl FnMut()) -> Duration {
    let started = Instant::now();
    for _ in 0..REPEATS {
        body();
    }
    started.elapsed() / REPEATS
}

fn repeat_result(
    mut body: impl FnMut() -> Result<(), runner::Error>,
) -> Result<Duration, runner::Error> {
    let started = Instant::now();
    for _ in 0..REPEATS {
        body()?;
    }
    Ok(started.elapsed() / REPEATS)
}

fn row(label: &str, total: Duration, count: usize) {
    println!(
        "{label:>34} {:>12} {:>12}",
        micros(total),
        micros(total / count as u32)
    );
}

fn micros(duration: Duration) -> String {
    format!("{:.1} us", duration.as_secs_f64() * 1e6)
}
