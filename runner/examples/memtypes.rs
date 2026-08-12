//! What memory the device offers, and which of it this crate actually asks for.
//!
//! Written to answer one question that a throughput number could not: `examples/overhead.rs`
//! measured host transfers at ~370 MB/s on a PCIe 4.0 x16 part that should manage tens of
//! gigabytes. A measurement that does not know which memory it measured is not a measurement.
//!
//! # What it found
//!
//! The staging buffer asks for `HOST_VISIBLE | HOST_COHERENT` and `buffer::memory_type` returns
//! the **first** type that satisfies the request. On this device that is index 2 — visible,
//! coherent, and *not* cached — while index 3 offers all three.
//!
//! Host-visible memory without `HOST_CACHED` is typically write-combined. Sequential writes into
//! it coalesce and go at full speed; every *read* is an uncached fetch with no prefetching and no
//! line reuse. `Buffer::read` memcpys out of exactly such a mapping on the way home from every
//! dispatch.
//!
//! Asking for the flags you want and taking the first match is the obvious thing to write and it
//! silently picks the wrong memory whenever a better type sits later in the list.

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

    // What the staging path asks for, and what it therefore gets.
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
