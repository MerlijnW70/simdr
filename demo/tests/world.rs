//! Three generated worlds, held to the number a CPU says they should be.
//!
//! **Every comparison here is exact**, which is the whole reason this directory is integers. A
//! procedural world's usual test is that it looks right; these have an answer, and the answer is
//! the same on a discrete GPU, an integrated one and a software rasteriser.
//!
//! The failure each is written against is not "the picture is ugly" but the one this repository
//! keeps finding: a mapping that pairs the wrong lanes, a shift that reads a lane that does not
//! exist, an instruction chosen by signedness. A world of 256 × 64 columns is 16 384 independent
//! answers, and one wrong lane is one wrong number in a place a picture would hide.

use demo::{Answer, PITCH, caverns, caves_reference, fractal, generate, landscape};
use demo::{heights_reference, orbits_reference};
use runner::Gpu;

/// Rows of world to generate. Enough that every workgroup runs and the grid has a second axis.
const ROWS: u32 = 64;

/// Open a device, or say why not and hand back nothing.
///
/// **A skip is printed rather than passed over.** `libtest` swallows `eprintln!` from a passing
/// test, so a run with no device prints the same summary as a run that checked everything — which
/// is the failure mode this repository has written down more than once.
fn device(label: &str) -> Option<Gpu> {
    match Gpu::open() {
        Ok(Some(gpu)) => Some(gpu),
        Ok(None) => {
            eprintln!("SKIPPED {label}: no Vulkan device");
            None
        }
        Err(error) => {
            eprintln!("SKIPPED {label}: {error}");
            None
        }
    }
}

/// Compare a generated world against its reference, element by element.
///
/// Reports the *first* disagreement with its coordinates, because the useful thing about a
/// disagreement in a 16 384-element grid is where it is: one lane, one row, or every fourth column
/// are three different bugs and they look the same in a count.
fn agrees(what: &str, world: &[u32], expected: &[u32]) {
    assert_eq!(
        world.len(),
        expected.len(),
        "{what}: the device returned {} words and the reference has {}",
        world.len(),
        expected.len()
    );

    if let Some(at) = world
        .iter()
        .zip(expected)
        .position(|(left, right)| left != right)
    {
        let (x, y) = (at as u32 % PITCH, at as u32 / PITCH);
        panic!(
            "{what}: column {x} of row {y} came back {:#010x} and the reference says {:#010x}",
            world[at], expected[at]
        );
    }
}

/// Run one generator and compare, or print why it did not run.
fn checked(
    gpu: &Gpu,
    what: &str,
    build: fn(u32, u32) -> Result<Vec<u32>, simdr::lanes::LaneError>,
    reference: fn(u32, u32) -> Vec<u32>,
) -> bool {
    match generate(gpu, build, PITCH, ROWS) {
        Answer::Ran(world) => {
            agrees(what, &world, &reference(PITCH, ROWS));
            true
        }
        // Lost coverage rather than failure, and printed: a device that does not offer what the
        // module declares is being honest, and a width with no arm is the lane API working.
        other => {
            eprintln!(
                "SKIPPED {what}: {}",
                other.why().unwrap_or_else(|| String::from("no reason given"))
            );
            false
        }
    }
}

#[test]
fn the_landscape_is_the_landscape_the_host_computes() {
    let Some(gpu) = device("landscape") else {
        return;
    };
    println!("{} — subgroup {}", gpu.limits().name, gpu.limits().subgroup_size);

    assert!(checked(&gpu, "landscape", landscape, heights_reference));
}

#[test]
fn the_caves_are_carved_where_the_host_says_they_are() {
    let Some(gpu) = device("caverns") else {
        return;
    };

    assert!(checked(&gpu, "caverns", caverns, caves_reference));
}

#[test]
fn the_fractal_escapes_on_the_iteration_the_host_escapes_on() {
    let Some(gpu) = device("fractal") else {
        return;
    };

    assert!(checked(&gpu, "fractal", fractal, orbits_reference));
}

/// The worlds are worlds rather than a flat field, and the check would pass on a flat one.
///
/// **Without this the three tests above are vacuous in a way that is easy to miss.** A kernel that
/// stored zero everywhere and a reference that computed zero everywhere would agree perfectly, and
/// so would a mixer that had collapsed to a constant. So the *reference* is asked to be varied —
/// on the host, where no device is needed — and the device is then held to it.
#[test]
fn the_references_describe_something_worth_generating() {
    let heights = heights_reference(PITCH, ROWS);
    let distinct = heights.iter().collect::<std::collections::BTreeSet<_>>().len();
    assert!(
        distinct > 64,
        "the landscape has only {distinct} distinct heights, which is a pattern rather than a world"
    );

    let caves = caves_reference(PITCH, ROWS);
    let open: u32 = caves.iter().map(|word| word.count_ones()).sum();
    let total = caves.len() as u32 * 32;
    assert!(
        open > total / 100 && open < total / 2,
        "{open} of {total} layers are open, which is either solid rock or no rock at all"
    );

    let orbits = orbits_reference(PITCH, ROWS);
    assert!(
        orbits.contains(&demo::ORBITS),
        "nothing in the window stayed bounded for every iteration, so the set is off screen"
    );
    assert!(
        orbits.iter().any(|&count| count < 4),
        "everything in the window stayed bounded, so the window is inside the set"
    );
}
