use runner::Gpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    println!("{}\n", gpu.limits().name);
    println!(
        "{:>5} {:>13} {:>13} {:>14} {:>12}",
        "index", "device-local", "host-visible", "host-coherent", "host-CACHED"
    );

    let types = gpu.memory_types();
    for kind in &types {
        println!(
            "{:>5} {:>13} {:>13} {:>14} {:>12}",
            kind.index, kind.device_local, kind.host_visible, kind.host_coherent, kind.host_cached
        );
    }

    let chosen = types
        .iter()
        .find(|kind| kind.host_visible && kind.host_coherent);
    let better = types
        .iter()
        .find(|kind| kind.host_visible && kind.host_coherent && kind.host_cached);

    println!("\nstaging asks for HOST_VISIBLE | HOST_COHERENT and takes the first match.");
    match (chosen, better) {
        (Some(chosen), Some(better)) if chosen.index != better.index => println!(
            "  it gets type {} (cached: {}), while type {} offers the same plus caching.\n  \
             Reading a mapping back out of uncached memory is what the transfer numbers show.",
            chosen.index, chosen.host_cached, better.index
        ),
        (Some(chosen), _) => println!(
            "  it gets type {}, cached: {} — and nothing better is offered.",
            chosen.index, chosen.host_cached
        ),
        _ => println!("  no host-visible coherent type at all, which would be unusual."),
    }

    Ok(())
}
