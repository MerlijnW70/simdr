//! Where the buffers land when all of them are there at once.
//!
//! Three sightings of a large-working-set cliff are recorded in `notes/FINDINGS.md` and two
//! explanations have been tested and refuted — L2 capacity, and eviction under a *single*
//! allocation. What survived was a gap the code wrote into its own documentation: `probe_memory`
//! answers for one buffer, and a run holds three.
//!
//! This asks the question properly. `Gpu::run` allocates a staging buffer and two device-local
//! ones, so a working set of *n* bytes puts three *n*-byte allocations on the device together. If
//! the driver starts placing the second or third in host memory, a kernel reading it crosses the
//! bus on every access and the collapse would look exactly as it does.
//!
//! A negative result settles as much as a positive one. If all three land device-local at a size
//! where the timing has already fallen apart, that hypothesis is dead as well.

use runner::Gpu;

/// Working-set sizes in megabytes, spanning the sizes where measurements stopped being steady.
const SIZES_MB: [u64; 10] = [1, 8, 32, 48, 56, 64, 128, 256, 512, 1_024];

/// How many buffers of that size a run actually holds: staging, source, destination.
const PER_RUN: u32 = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let heap = gpu.probe_memory(1)?.largest_device_heap;
    println!("{}", gpu.limits().name);
    println!("device-local heap: {:.1} GB\n", heap as f64 / 1e9);

    println!(
        "{:>10} {:>12} {:>14} {:>14} {:>10}",
        "each", "all three", "one resident", "three resident", "of heap"
    );

    for megabytes in SIZES_MB {
        let bytes = megabytes * 1_000_000;
        let total = bytes * u64::from(PER_RUN);

        let alone = describe(gpu.probe_resident(bytes, 1));
        let together = describe(gpu.probe_resident(bytes, PER_RUN));

        println!(
            "{:>10} {:>12} {:>14} {:>14} {:>9.1}%",
            format!("{megabytes} MB"),
            format!("{} MB", total / 1_000_000),
            alone,
            together,
            total as f64 / heap as f64 * 100.0
        );
    }

    println!(
        "\n`device-local` means the driver honoured the request. `host` means it fell back, and a\n\
         kernel reading that buffer crosses the bus on every access."
    );

    Ok(())
}

/// What a probe came to, in one word.
fn describe(placement: Result<runner::Placement, runner::Error>) -> String {
    match placement {
        Ok(placement) if placement.device_local => "device-local".to_owned(),
        Ok(_) => "host".to_owned(),
        Err(_) => "refused".to_owned(),
    }
}
