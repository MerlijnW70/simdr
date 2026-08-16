//! `cargo run --release` — three worlds, drawn, and one of them timed against the host.
//!
//! The test beside this asserts; this shows. What is worth looking at is that the pictures came out
//! of the same numbers the CPU computes, so a landscape that looks wrong here would already have
//! failed there — and that the throughput number has the round trip in it, because
//! `decisions/DR-0008` says that is what a caller actually waits for.

use demo::{Answer, PITCH, caverns, caves_reference, fractal, generate, landscape, rendered};
use demo::{STEPS, heights_reference, orbits_reference};
use runner::Gpu;
use std::time::Instant;

/// Rows drawn in each picture.
const ROWS: u32 = 24;

/// Columns drawn — narrower than [`PITCH`], because a terminal is.
const DRAWN: u32 = 110;

/// The world timed against the host, in columns and rows.
///
/// A million answers. Large enough that the device's own work is the greater part of the wall
/// clock, which at a few thousand it is not — `decisions/DR-0008` measured the fixed cost of being
/// asked at about 100 µs on this machine's discrete card.
const WIDE: u32 = 1024;
const DEEP: u32 = 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };
    let limits = gpu.limits().clone();
    println!("{} — subgroup {}\n", limits.name, limits.subgroup_size);

    draw("A landscape, seen from the side", &gpu, landscape, |world| {
        // **A cross-section rather than a map.** The kernel answers with one height per column,
        // which drawn from above is a field of shaded squares and drawn from the side is a
        // skyline — and a skyline is the thing a reader can tell is wrong.
        let mut out = String::new();
        for level in (0..ROWS).rev() {
            for column in 0..DRAWN as usize {
                let height = world.get(column).copied().unwrap_or(0) * ROWS / 256;
                out.push(match height.cmp(&level) {
                    std::cmp::Ordering::Less => ' ',
                    std::cmp::Ordering::Equal => '#',
                    std::cmp::Ordering::Greater => ':',
                });
            }
            out.push('\n');
        }
        out
    });

    draw("Caves under it, one bit a layer", &gpu, caverns, |world| {
        // A vertical slice: the row is the layer and the column is the column, so this is what a
        // miner walking east would cut through.
        let mut out = String::new();
        for layer in 0..24 {
            for column in 0..DRAWN as usize {
                let word = world.get(column).copied().unwrap_or(0);
                out.push(if word >> layer & 1 == 1 { ' ' } else { '#' });
            }
            out.push('\n');
        }
        out
    });

    draw("An escape-time fractal", &gpu, fractal, |world| {
        shaded(world, " .:-=+*#%@")
    });

    // **The renderer is a kernel too.** One invocation a pixel, marching a ray through a world
    // that is never stored — each step recomputes the terrain from the coordinate it has reached.
    println!("== The same world, raycast — {SHOT_WIDE} × {SHOT_DEEP} pixels, one a lane ==");
    match generate(&gpu, rendered, SHOT_WIDE, SHOT_DEEP) {
        Answer::Ran(picture) => print!("{}", lit(&picture)),
        other => println!(
            "  not rendered — {}",
            other.why().unwrap_or_else(|| String::from("no reason given"))
        ),
    }
    println!();

    timed(&gpu)?;
    Ok(())
}

/// Generate a world and print it, or say why it did not run.
fn draw(
    title: &str,
    gpu: &Gpu,
    build: fn(u32, u32) -> Result<Vec<u32>, simdr::lanes::LaneError>,
    render: impl Fn(&[u32]) -> String,
) {
    println!("== {title} ==");
    match generate(gpu, build, PITCH, ROWS) {
        Answer::Ran(world) => print!("{}", render(&world)),
        other => println!(
            "  not generated — {}",
            other.why().unwrap_or_else(|| String::from("no reason given"))
        ),
    }
    println!();
}

/// A world of bytes as a ramp of characters, `DRAWN` columns wide.
fn shaded(world: &[u32], ramp: &str) -> String {
    let shades: Vec<char> = ramp.chars().collect();
    let mut out = String::new();

    for row in 0..ROWS as usize {
        for column in 0..DRAWN as usize {
            let value = world
                .get(row * PITCH as usize + column)
                .copied()
                .unwrap_or(0);
            // The value's range is the caller's: a height is a byte and an escape count stops at
            // `ORBITS`, so the ramp is indexed by a fraction of whichever it is.
            let top = world.iter().copied().max().unwrap_or(1).max(1);
            let shade = (value as usize * (shades.len() - 1)) / top as usize;
            out.push(shades.get(shade).copied().unwrap_or(' '));
        }
        out.push('\n');
    }
    out
}

/// The rendered picture's size, in pixels. Wide is a multiple of the workgroup, as every grid is.
const SHOT_WIDE: u32 = 128;
const SHOT_DEEP: u32 = 36;

/// A packed pixel as a lit picture: elevation shaded, distance dimming it, zero is sky.
///
/// **Elevation and distance, because either alone is unreadable.** Depth on its own draws a horizon
/// and horizontal bands — a heightfield seen from a little above varies far more in height than in
/// range. Height on its own throws the perspective away. The product of the two is a landscape.
fn lit(picture: &[u32]) -> String {
    let shades: Vec<char> = " .:-=+*#%@".chars().collect();
    let far = (STEPS * 2) as usize;
    let mut out = String::new();

    for row in 0..SHOT_DEEP as usize {
        for column in 0..SHOT_WIDE as usize {
            let packed = picture
                .get(row * SHOT_WIDE as usize + column)
                .copied()
                .unwrap_or(0);
            if packed == 0 {
                out.push(' ');
                continue;
            }

            let depth = (packed & 0xff) as usize;
            let elevation = (packed >> 8) as usize;

            // Near ground keeps its elevation; far ground fades toward the sky it sits against.
            let lit = elevation * far.saturating_sub(depth) / far;
            let shade = (lit * (shades.len() - 1)) / 64;
            out.push(shades.get(shade.min(shades.len() - 1)).copied().unwrap_or(' '));
        }
        out.push('\n');
    }
    out
}

/// One row of the table: a name, the generator, and the host's answer for the same world.
type Timed = (
    &'static str,
    fn(u32, u32) -> Result<Vec<u32>, simdr::lanes::LaneError>,
    fn(u32, u32) -> Vec<u32>,
);

/// A million columns of each world, on the device and on the host.
///
/// **The wall clock, not the dispatch**, because `decisions/DR-0008` measured what a caller
/// actually waits for: a round trip costs about 100 µs before anything is computed, and on a buffer
/// this size the copies either side of it cost far more. A number that leaves those out is a number
/// nobody experiences.
///
/// Three workloads rather than one, because the interesting thing is not a ratio but **where the
/// ratio crosses one**. One does two octaves of noise per answer, one does thirty-two, and one runs
/// forty iterations of fixed-point arithmetic — and all three return the same four bytes. The work
/// per byte moved climbs across the table while the transfer does not move at all, which is the
/// only thing that can shift the balance.
fn timed(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    let answers = u64::from(WIDE) * u64::from(DEEP);
    println!("== {WIDE} × {DEEP} = {answers} answers, each way ==\n");
    println!(
        "  {:<12} {:>11} {:>11} {:>8}  {:>11}  agreed",
        "world", "round trip", "host", "ratio", "dispatch"
    );

    let each: [Timed; 3] = [
        ("landscape", landscape, heights_reference),
        ("caverns", caverns, caves_reference),
        ("fractal", fractal, orbits_reference),
    ];

    for (name, build, reference) in each {
        // **A discarded run first.** The first dispatch of a module pays for its pipeline and its
        // buffers, and this table is about the steady state a caller sees — timing the first one
        // reported four times the round trip for two workloads that move identical bytes.
        let _ = generate(gpu, build, WIDE, DEEP);

        let started = Instant::now();
        let world = match generate(gpu, build, WIDE, DEEP) {
            Answer::Ran(world) => world,
            other => {
                println!(
                    "  {name:<12} not generated — {}",
                    other.why().unwrap_or_else(|| String::from("no reason given"))
                );
                continue;
            }
        };
        let trip = started.elapsed();

        let started = Instant::now();
        let expected = reference(WIDE, DEEP);
        let host = started.elapsed();

        // **Checked, not just timed.** A throughput number for the wrong answer is worse than no
        // number, and this is the one place both are available at once.
        let agreed = world == expected;

        let spirv = build(gpu.limits().subgroup_size, WIDE)?;
        let empty = vec![0_u32; answers as usize];
        let grid = runner::Grid::new(WIDE / demo::WORKGROUP, DEEP);
        let alone = gpu.time_grid(&spirv, &empty, grid, 8)?;

        println!(
            "  {name:<12} {trip:>11.2?} {host:>11.2?} {:>7.1}×  {alone:>11.2?}  {}",
            host.as_secs_f64() / trip.as_secs_f64().max(f64::EPSILON),
            if agreed { "all of them" } else { "NO" }
        );
    }

    println!(
        "\n  The dispatch column is the device's own work; the round trip is what a caller waits\n  \
         for. `decisions/DR-0008` priced that gap and nothing in this directory can move it — what\n  \
         moves is the arithmetic per byte returned, which is the whole of why the ratio climbs."
    );
    println!(
        "  And {} of those bytes are uploaded and never read: `Gpu::run_grid` sizes the output from\n  \
         its input, so a generator that reads nothing still pays to send it.",
        answers * 4
    );

    Ok(())
}
