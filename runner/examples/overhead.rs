use runner::Gpu;
use runner::kernels;
use std::time::{Duration, Instant};

const SIZES: [(usize, &str); 5] = [
    (64, "256 B"),
    (16_384, "64 KB"),
    (262_144, "1 MB"),
    (4_194_304, "16 MB"),
    (16_777_216, "64 MB"),
];

const TRIPS: u32 = 40;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}\n", gpu.limits().name);

    let empty = kernels::empty(width)?;

    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>10}",
        "buffer", "round trip", "dispatch", "overhead", "MB/s"
    );

    let mut fixed = Duration::ZERO;
    for (words, label) in SIZES {
        let input = vec![1_u32; words];

        gpu.run_u32(&empty, &input, 1)?;

        let started = Instant::now();
        for _ in 0..TRIPS {
            gpu.run_u32(&empty, &input, 1)?;
        }
        let trip = started.elapsed() / TRIPS;

        let dispatch = gpu.time(&empty, &input, 1, 1)?;
        let overhead = trip.saturating_sub(dispatch);

        if words == 64 {
            fixed = overhead;
        }
        let bytes = (words * size_of::<u32>()) as f64;

        println!(
            "{label:>8} {:>12} {:>12} {:>12} {:>10}",
            micros(trip),
            micros(dispatch),
            micros(overhead),
            format!("{:.0}", bytes / overhead.as_secs_f64() / 1e6)
        );
    }

    println!(
        "\n{:>8} {:>16} {:>18}",
        "buffer", "allocate + free", "per run (3 of them)"
    );
    for (words, label) in SIZES {
        let bytes = (words * size_of::<u32>()) as u64;

        gpu.probe_resident(bytes, 1)?;
        let started = Instant::now();
        for _ in 0..TRIPS {
            gpu.probe_resident(bytes, 1)?;
        }
        let each = started.elapsed() / TRIPS;

        println!("{label:>8} {:>16} {:>18}", micros(each), micros(each * 3));
    }

    println!("\nfixed cost, from the smallest buffer: {}", micros(fixed));
    println!(
        "That is what a persistent-resource API could remove. It is charged once per `run` call,\n\
         and `run` allocates three buffers and builds a pipeline every time."
    );

    let input = vec![1_u32; 65_536];
    gpu.time(&empty, &input, 1, 1)?;
    let batched = gpu.time(&empty, &input, 1, 1_000)? / 1_000;
    println!(
        "\nthe same empty dispatch, amortised over a thousand of them: {}",
        micros(batched)
    );
    println!(
        "The gap between that and the fixed cost above is the whole argument for reusing\n\
         buffers and pipelines rather than rebuilding them."
    );

    Ok(())
}

fn micros(duration: Duration) -> String {
    format!("{:.1} us", duration.as_secs_f64() * 1e6)
}
