//! What memory the device offers, and which of it this crate actually asks for.
//!
//! Written to answer one question that a throughput number could not: `examples/overhead.rs`
//! measured host transfers at ~370 MB/s on a PCIe 4.0 x16 part that should manage tens of
//! gigabytes. A measurement that does not know which memory it measured is not a measurement.
//!
//! # What it found, and what was done about it
//!
//! The staging buffer asked for `HOST_VISIBLE | HOST_COHERENT` and `buffer::memory_type` returned
//! the **first** type that satisfied the request. On the discrete card that was index 2 — visible,
//! coherent, and *not* cached — while index 3 offered all three.
//!
//! Host-visible memory without `HOST_CACHED` is typically write-combined. Sequential writes into
//! it coalesce and go at full speed; every *read* is an uncached fetch with no prefetching and no
//! line reuse. `Buffer::read` memcpys out of exactly such a mapping on the way home from every
//! dispatch.
//!
//! # The part this example got wrong about itself
//!
//! `Buffer::staging` stopped taking the first match when that was found, and **this example went
//! on printing that it still did** until 2026-08-25. It reimplemented the old rule in its own
//! `find`, called the answer "what staging gets", and was believed — a diagnostic that argued for
//! a fix and then outlived the code it was arguing about. `simdr probe` had the right answer the
//! whole time, one command away, and the two disagreed in print on both devices in this machine.
//!
//! What `Buffer::preferring` actually does is ask for `HOST_CACHED` *as well* and keep the plain
//! flags as a second candidate — and that fallback is an **allocation** fallback, not a selection
//! one. The memory a host writes and a device reads at full speed is usually a BAR window, often
//! 256 MB against several gigabytes of plain device memory; choosing it and then failing to fit in
//! it would turn a preference into a size limit.
//!
//! So the line under the table below is what this crate *asks for*. It is not a promise about
//! which type the allocation landed in, and it does not pretend to be: `memory_type` also filters
//! by the buffer's own `memory_type_bits`, and that needs a buffer, which this example does not
//! build.

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

    // The two candidates `Buffer::preferring` builds, in its order: the wanted flags plus the
    // preferred one, then the wanted flags alone.
    let preferred = types
        .iter()
        .find(|kind| kind.host_visible && kind.host_coherent && kind.host_cached);
    let fallback = types
        .iter()
        .find(|kind| kind.host_visible && kind.host_coherent);

    println!(
        "\nstaging asks for HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, and falls back to\nHOST_VISIBLE | HOST_COHERENT only if allocating from the cached type fails."
    );
    match (preferred, fallback) {
        (Some(preferred), Some(fallback)) if preferred.index != fallback.index => println!(
            "  it asks for type {}, cached — not type {}, which is the first plain match and is\n  \
             what a first-match rule hands you. That difference was the ~370 MB/s.",
            preferred.index, fallback.index
        ),
        (Some(preferred), _) => println!(
            "  it asks for type {}, cached — which is also the first plain match, so on this\n  \
             device the preference costs and saves nothing.",
            preferred.index
        ),
        (None, Some(fallback)) => println!(
            "  no cached host-visible type is offered, so the fallback is what runs: type {}.\n  \
             Reading a mapping back out of uncached memory is what the transfer numbers show.",
            fallback.index
        ),
        (None, None) => println!("  no host-visible coherent type at all, which would be unusual."),
    }

    Ok(())
}
