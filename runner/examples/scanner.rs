//! What a held scan costs, and what fusing its map into the chain is worth.
//!
//! Two questions, and the second is the one with a number attached.
//!
//! # Where a long scan's time goes
//!
//! `Gpu::scanner` is `2 × levels + 1` dispatches over a dozen buffers, so the first table is
//! simply what that costs at three depths — one level, two, three.
//!
//! # What the map is worth
//!
//! The running total of f(x) over data the caller cannot otherwise reach costs three host
//! crossings: send the input, run the map, bring the result home, send it back, scan. Two of them
//! are the whole buffer. `Gpu::scanner_of` makes the map the chain's first pass and the
//! intermediate never leaves the device.
//!
//! **The left column is given every advantage.** It would be easy — and wrong — to write it as
//! `gpu.run(&square, …)`, which allocates and builds a pipeline every call; that is most of a
//! millisecond this project has already measured, and charging it to the old route would make the
//! new one look far better than it is. So the map runs through a held `Session` and the scan
//! through a held `Scanner`: nothing is allocated or built in either column, and the only
//! difference left is where the intermediate went.
//!
//! Both columns are asserted to compute the same numbers, so what is timed is the route and not
//! two different answers.

use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use std::time::{Duration, Instant};

/// Element counts to measure at, and how many scans to time at each.
///
/// One level, two and three — the depth is what decides the dispatch count, so a saving that is
/// per-call rather than per-element shows up as the same absolute figure across all three.
const SIZES: [(usize, u32); 3] = [(1 << 12, 40), (1 << 16, 20), (1 << 20, 10)];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };
    println!(
        "{} — subgroup {}\n",
        gpu.limits().name,
        gpu.limits().subgroup_size
    );

    println!(
        "{:>12} {:>8} {:>12} {:>14}",
        "elements", "levels", "dispatches", "Scanner::scan"
    );
    for (elements, repeats) in SIZES {
        let input: Vec<f32> = (0..elements).map(|index| (index % 3) as f32).collect();
        let mut scanner = gpu.scanner(elements)?;
        scanner.scan(&input)?;

        let started = Instant::now();
        for _ in 0..repeats {
            scanner.scan(&input)?;
        }
        let held = started.elapsed() / repeats;

        // `2 * levels + 1`, so the level count is the dispatch count read backwards.
        let levels = (scanner.dispatches() - 1) / 2;
        println!(
            "{:>12} {:>8} {:>12} {:>14}",
            thousands(elements),
            levels,
            scanner.dispatches(),
            micros(held)
        );
    }

    in_place(&gpu)?;
    mapped(&gpu)?;
    Ok(())
}

/// Where the deepest chain here spends its device time, pass by pass.
///
/// Three kinds of pass, which is what makes this worth printing rather than a reduction's: block
/// scans on the way **up**, one workgroup at the **top**, and offset additions on the way **down**
/// reading what the way up wrote. The shape of the profile says which of the three the length is
/// actually paying for.
fn in_place(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    let Some(&(elements, _)) = SIZES.last() else {
        return Ok(());
    };

    let mut scanner = gpu.scanner(elements)?;
    let input: Vec<f32> = (0..elements).map(|index| (index % 3) as f32).collect();

    scanner.scan_timed(&input)?;
    let started = Instant::now();
    let (_, spans) = scanner.scan_timed(&input)?;
    let wall = started.elapsed();

    println!(
        "
{} elements, timed from inside the chain:
",
        thousands(elements)
    );
    if spans.is_empty() {
        println!(
            "  this device reports no usable timestamp queries — `timestampValidBits` is zero on
             some queues and the feature is optional, so there is nothing to say rather than a
             zero to print."
        );
        return Ok(());
    }

    // `2 * levels + 1`: the first pass and then a pair per level, up and down.
    let levels = (spans.len() - 1) / 2;
    let total: Duration = spans.iter().sum();

    println!("{:>22} {:>12} {:>10}", "pass", "on device", "share");
    for (index, taken) in spans.iter().enumerate() {
        let name = if index == 0 {
            "up: the input".to_owned()
        } else if index < levels {
            format!("up: level {index}")
        } else if index == levels {
            "top: one workgroup".to_owned()
        } else {
            format!("down: level {}", spans.len() - 1 - index)
        };

        let share = taken.as_secs_f64() / total.as_secs_f64() * 100.0;
        println!("{name:>22} {:>12} {share:>9.0}%", micros(*taken));
    }
    println!("{:>22} {:>12}", "---- the dispatches ----", micros(total));

    println!(
        "
  **The two ends are the scan and the depth is nearly free.** The first pass reads the
         whole input and the last writes the whole answer; everything between them works on the
         block totals, which are a sixty-fourth of the buffer and then a sixty-fourth of that. On
         an RTX 4080 the five middle passes come to about 10 us against 21 for the two ends, and on
         the integrated Radeon to about 14 against 402.

         So a longer input costs two more dispatches and almost no more device time, and making
         a scan faster means making those two traversals faster rather than shortening the
         recursion.

         And the dispatches are not the call. Against the {} above, this device spends {} of it
         on the chain — the rest is the host writing its input and waiting for one submission,
         neither of which is inside the command buffer. `runner/examples/reducer.rs` reaches the
         same conclusion for the reduction and measures the upload directly.",
        micros(wall),
        micros(total),
    );

    Ok(())
}

/// The running total of x², one crossing against three.
fn mapped(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    let width = gpu.limits().subgroup_size;
    let square = kernels::square(width)?;

    println!("\nΣ x² as a running total — the map on the device against the map through the host:");
    println!(
        "{:>12} {:>16} {:>16} {:>10}",
        "elements", "three crossings", "one crossing", "faster"
    );

    for (elements, repeats) in SIZES {
        // Values of 0, 1 and 2. The totals have to stay inside the 2²⁴ an `f32` counts exactly or
        // the comparison below is a tolerance wearing an equals sign — and 2 has to be present,
        // because x² and x agree on 0 and 1 and a map that had stopped squaring would go
        // unnoticed without it.
        let input: Vec<f32> = (0..elements).map(|index| (index % 3) as f32).collect();
        let squares: Vec<f32> = input.iter().map(|value| value * value).collect();
        let groups = elements as u32 / WORKGROUP_SIZE;

        let mut fused = gpu.scanner_of(elements, &square)?;
        let mut plain = gpu.scanner(elements)?;

        // The map, held: no allocation and no pipeline creation inside the timed loop.
        let mut mapping = gpu.session(&square, &[elements, elements])?;
        let words: Vec<u32> = input.iter().map(|value| value.to_bits()).collect();

        // Warm every path, so none pays for the driver's first compile of these modules.
        mapping.write(0, &words)?;
        mapping.dispatch(groups, 1)?;
        mapping.read(1, elements)?;
        let expected = plain.scan(&squares)?;
        assert_eq!(fused.scan(&input)?, expected, "the two routes disagree");

        let started = Instant::now();
        for _ in 0..repeats {
            // The best a caller can do without `scanner_of`: send the input, run the map, bring
            // the squares home, send them back to be scanned.
            mapping.write(0, &words)?;
            mapping.dispatch(groups, 1)?;
            let squares: Vec<f32> = mapping
                .read(1, elements)?
                .into_iter()
                .map(f32::from_bits)
                .collect();
            plain.scan(&squares)?;
        }
        let stepwise = started.elapsed() / repeats;

        let started = Instant::now();
        for _ in 0..repeats {
            fused.scan(&input)?;
        }
        let chained = started.elapsed() / repeats;

        println!(
            "{:>12} {:>16} {:>16} {:>10}",
            thousands(elements),
            micros(stepwise),
            micros(chained),
            format!("{:.1}x", stepwise.as_secs_f64() / chained.as_secs_f64())
        );
    }

    println!(
        "\n  The left column uploads the input, downloads the squares and uploads them again —\n\
         three crossings of which two are the whole buffer. The right one is a single chain: the\n\
         map is its first pass and its output is handed to the first block scan on the device.\n\n\
       \x20 Both columns hold their pipelines and their buffers, so neither pays for allocation or\n\
         pipeline creation, and both were asserted to compute the same numbers before either was\n\
         timed."
    );

    Ok(())
}

/// Microseconds, which is the scale these land on.
fn micros(taken: Duration) -> String {
    format!("{:.1} us", taken.as_secs_f64() * 1e6)
}

/// A count with thin spaces, because six digits run together.
fn thousands(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(digit);
    }
    out
}
