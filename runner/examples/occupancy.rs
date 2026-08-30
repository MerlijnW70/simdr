use runner::{Gpu, kernels};
use simdr::lanes::LaneError;
use std::time::Duration;

type Shape<'a> = (&'a str, &'a dyn Fn(u32) -> Result<Vec<u32>, LaneError>);

const MULTIPLES: [u32; 6] = [1, 2, 4, 8, 16, 32];

const SIZES: [(u32, u32); 2] = [(1 << 14, 200), (1 << 18, 50)];

const REPEATS: u32 = 512;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    let width = limits.subgroup_size;
    let ceiling = limits.max_workgroup_invocations;
    println!(
        "{} — subgroup {width}, workgroup ceiling {ceiling} ({} subgroups)\n",
        limits.name,
        ceiling.checked_div(width).unwrap_or(0)
    );

    for (target, iterations) in SIZES {
        println!("about {} invocations:", thousands(target as usize));
        sweep(&gpu, width, ceiling, target, iterations)?;
        println!();
    }

    println!(
        "Every column runs the same total work over the same elements and differs only in how\n\
         many subgroups share a workgroup. A dash is a size past this device's ceiling, or one\n\
         that does not divide the invocation count."
    );

    Ok(())
}

fn sweep(
    gpu: &Gpu,
    width: u32,
    ceiling: u32,
    target: u32,
    iterations: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    print!("{:>12}", "subgroups");
    for multiple in MULTIPLES {
        print!("{multiple:>12}");
    }
    println!();

    let shapes: [Shape<'_>; 3] = [
        ("elementwise", &|workgroup| {
            kernels::flat_scale(width, workgroup, 3)
        }),
        ("repeated", &|workgroup| {
            kernels::sized_repeated_scale(width, workgroup, REPEATS, 3)
        }),
        ("reduction", &|workgroup| {
            kernels::sized_lane_sum(width, workgroup)
        }),
    ];

    for (name, build) in shapes {
        print!("{name:>12}");
        for multiple in MULTIPLES {
            print!(
                "{:>12}",
                one(gpu, width, ceiling, target, iterations, multiple, build)?
            );
        }
        println!();
    }

    Ok(())
}

fn one(
    gpu: &Gpu,
    width: u32,
    ceiling: u32,
    target: u32,
    iterations: u32,
    multiple: u32,
    build: &dyn Fn(u32) -> Result<Vec<u32>, LaneError>,
) -> Result<String, Box<dyn std::error::Error>> {
    let workgroup = width * multiple;
    if workgroup > ceiling {
        return Ok("-".to_owned());
    }

    let largest = width
        * MULTIPLES
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .min(ceiling / width);
    let invocations = (target / largest) * largest;
    if invocations == 0 {
        return Ok("-".to_owned());
    }

    let input: Vec<u32> = (0..invocations).collect();
    let spirv = build(workgroup)?;
    let workgroups = invocations / workgroup;

    gpu.time(&spirv, &input, workgroups, 1)?;
    let elapsed = gpu.time(&spirv, &input, workgroups, iterations)? / iterations;

    Ok(micros(elapsed))
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
