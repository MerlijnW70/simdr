use runner::Gpu;

const SIZES_MB: [u64; 10] = [1, 8, 32, 48, 56, 64, 128, 256, 512, 1_024];

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

fn describe(placement: Result<runner::Placement, runner::Error>) -> String {
    match placement {
        Ok(placement) if placement.device_local => "device-local".to_owned(),
        Ok(_) => "host".to_owned(),
        Err(_) => "refused".to_owned(),
    }
}
