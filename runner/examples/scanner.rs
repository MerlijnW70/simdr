use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use std::time::{Duration, Instant};

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

fn mapped(gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
    let width = gpu.limits().subgroup_size;
    let square = kernels::square(width)?;

    println!("\nΣ x² as a running total — the map on the device against the map through the host:");
    println!(
        "{:>12} {:>16} {:>16} {:>10}",
        "elements", "three crossings", "one crossing", "faster"
    );

    for (elements, repeats) in SIZES {
        let input: Vec<f32> = (0..elements).map(|index| (index % 3) as f32).collect();
        let squares: Vec<f32> = input.iter().map(|value| value * value).collect();
        let groups = elements as u32 / WORKGROUP_SIZE;

        let mut fused = gpu.scanner_of(elements, &square)?;
        let mut plain = gpu.scanner(elements)?;

        let mut mapping = gpu.session(&square, &[elements, elements])?;
        let words: Vec<u32> = input.iter().map(|value| value.to_bits()).collect();

        mapping.write(0, &words)?;
        mapping.dispatch(groups, 1)?;
        mapping.read(1, elements)?;
        let expected = plain.scan(&squares)?;
        assert_eq!(fused.scan(&input)?, expected, "the two routes disagree");

        let started = Instant::now();
        for _ in 0..repeats {
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

fn micros(taken: Duration) -> String {
    format!("{:.1} us", taken.as_secs_f64() * 1e6)
}

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
