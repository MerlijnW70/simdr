use runner::{Gpu, Grid, kernels};
use std::time::Duration;

const SIZES: [(u32, u32); 3] = [(8, 400), (256, 200), (4_096, 50)];

const DEEP: u32 = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    let width = limits.subgroup_size;
    println!(
        "{} — subgroup {width}, so the two workgroup sizes below are {width} and {}\n",
        limits.name,
        width * DEEP
    );

    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>13} {:>13} {:>12}",
        "rows", "invocations", "flat wg=w", "flat wg=8w", "grid 1 deep", "grid 8 deep", "wide rows"
    );

    for (height, iterations) in SIZES {
        let invocations = (height * width) as usize;
        let input: Vec<u32> = (0..invocations).map(|index| index as u32).collect();

        let flat = kernels::flat_scale(width, width, 3)?;
        let flat_wide = kernels::flat_scale(width, width * DEEP, 3)?;
        let shallow = kernels::row_scale(width, width, 1, 3)?;
        let deep = kernels::row_scale(width, width, DEEP, 3)?;
        let wide = kernels::row_scale(width, width * DEEP, 1, 3)?;

        let cases = [
            (&flat, Grid::linear(height)),
            (&flat_wide, Grid::linear(height / DEEP)),
            (&shallow, Grid::new(1, height)),
            (&deep, Grid::new(1, height / DEEP)),
            (&wide, Grid::new(DEEP, height / DEEP)),
        ];

        let mut timings = Vec::with_capacity(cases.len());
        for (spirv, grid) in cases {
            if grid.x == 0 || grid.y == 0 {
                timings.push(None);
                continue;
            }
            gpu.time_grid(spirv, &input, grid, 1)?;
            timings.push(Some(
                gpu.time_grid(spirv, &input, grid, iterations)? / iterations,
            ));
        }

        println!(
            "{:>8} {:>12} {:>12} {:>12} {:>13} {:>13} {:>12}",
            height,
            thousands(invocations),
            cell(timings.first().copied().flatten()),
            cell(timings.get(1).copied().flatten()),
            cell(timings.get(2).copied().flatten()),
            cell(timings.get(3).copied().flatten()),
            cell(timings.get(4).copied().flatten()),
        );
    }

    println!(
        "\nAll five compute the same answer over the same elements. `runner/tests/plane.rs`\n\
         checks the grid ones against a host reference; `kernels::flat_scale` differs from them\n\
         in the address and in nothing else."
    );

    Ok(())
}

fn cell(duration: Option<Duration>) -> String {
    duration.map_or_else(|| "-".to_owned(), micros)
}

fn micros(duration: Duration) -> String {
    format!("{:.2} us", duration.as_secs_f64() * 1e6)
}

fn thousands(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(digit);
    }
    out
}
