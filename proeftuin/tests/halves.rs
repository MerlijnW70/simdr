//! Every `f16` bit pattern, through a device and back.

use proeftuin::halves::{EDGES, PATTERNS, Roundtrip, roundtrip};
use runner::Gpu;

#[test]
fn every_half_pattern_survives_a_load_and_a_store() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        eprintln!("SKIPPED halves: no Vulkan device");
        return Ok(());
    };

    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}", gpu.limits().name);

    let outcome = match width {
        4 => roundtrip::<4>(&gpu),
        8 => roundtrip::<8>(&gpu),
        16 => roundtrip::<16>(&gpu),
        32 => roundtrip::<32>(&gpu),
        64 => roundtrip::<64>(&gpu),
        other => {
            eprintln!("SKIPPED halves: no lane count written for a subgroup of {other}");
            return Ok(());
        }
    };

    let (sent, lost) = match outcome {
        Roundtrip::Ran { sent, lost } => (sent, lost),
        // Lost coverage rather than failure: `f16` needs `Float16` and 16-bit storage, and a device
        // without them is being honest.
        Roundtrip::Unsupported(missing) => {
            eprintln!("SKIPPED halves: device lacks {missing:?}");
            return Ok(());
        }
        Roundtrip::Refused(why) => {
            eprintln!("SKIPPED halves: refused — {why}");
            return Ok(());
        }
        // Failures, and which one says whose.
        Roundtrip::Invalid(complaint) => panic!("spirv-val rejected the identity kernel: {complaint}"),
        Roundtrip::Errored(error) => {
            panic!("the driver failed on a validated identity kernel: {error}")
        }
    };

    assert_eq!(sent, PATTERNS, "the sweep must be exhaustive to mean anything");

    // The edges, named, so a reader sees them confirmed rather than inferred from a count.
    for (name, bits) in EDGES {
        let changed = lost.iter().find(|l| l.sent == bits);
        println!(
            "  {name:<20} {bits:#06x}  {}",
            changed.map_or_else(|| String::from("unchanged"), |l| format!("became {:#06x}", l.returned))
        );
    }

    // A NaN is the one class Vulkan lets a device reshape, so it is reported rather than asserted —
    // the same distinction that made a signed-zero assertion wrong on a fourth implementation.
    let (nans, rest): (Vec<&_>, Vec<&_>) = lost.iter().partition(|l| l.is_nan);
    if !nans.is_empty() {
        println!("  {} NaN patterns came back reshaped, which Vulkan permits", nans.len());
    }

    assert!(
        rest.is_empty(),
        "{} non-NaN patterns did not survive a load and a store: {:?}",
        rest.len(),
        &rest[..rest.len().min(8)]
    );
    Ok(())
}
