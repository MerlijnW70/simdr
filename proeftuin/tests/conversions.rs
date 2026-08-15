//! The conversions, at the boundaries where their five instructions part company.

use proeftuin::conversions::{ConversionsFailed, every_target};
use runner::Gpu;

#[test]
fn every_conversion_matches_the_opcode_it_emits() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        eprintln!("SKIPPED conversions: no Vulkan device");
        return Ok(());
    };

    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}", gpu.limits().name);

    let sweeps = match width {
        4 => every_target::<4>(&gpu),
        8 => every_target::<8>(&gpu),
        16 => every_target::<16>(&gpu),
        32 => every_target::<32>(&gpu),
        64 => every_target::<64>(&gpu),
        other => {
            eprintln!("SKIPPED conversions: no lane count written for a subgroup of {other}");
            return Ok(());
        }
    };

    let mut ran = 0;
    let mut complaints = Vec::new();

    for (target, sweep) in sweeps {
        match sweep {
            Ok(conversions) => {
                ran += 1;
                for conversion in conversions {
                    if !conversion.agreed() {
                        complaints.push(format!("{conversion:?}"));
                    }
                }
            }
            // Lost coverage, printed rather than counted silently.
            Err(ConversionsFailed::Refused(why)) => {
                eprintln!("  {target}: refused — {why}");
            }
            Err(ConversionsFailed::Unsupported(missing)) => {
                eprintln!("  {target}: device lacks {missing:?}");
            }
            // Failures, and which one decides whose they are.
            Err(ConversionsFailed::Invalid(complaint)) => {
                complaints.push(format!("{target}: spirv-val rejected it — {complaint}"));
            }
            Err(ConversionsFailed::Errored(error)) => complaints.push(format!(
                "{target}: the driver failed after accepting a valid module — {error}"
            )),
        }
    }

    assert!(ran >= 4, "only {ran} of six targets swept, so this proves little");
    assert!(
        complaints.is_empty(),
        "a conversion did not match the instruction it emits:\n{}",
        complaints.join("\n")
    );
    Ok(())
}
