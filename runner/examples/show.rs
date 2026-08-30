use runner::Gpu;
use runner::kernels::{self, WORKGROUP_SIZE};
use simdr::lanes::{F32, U32};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = Gpu::open()? else {
        println!("no Vulkan device");
        return Ok(());
    };

    let limits = gpu.limits();
    let width = limits.subgroup_size;
    println!(
        "{} — subgroup {} (arithmetic {}, clustered {}, shuffle {})\n",
        limits.name,
        width,
        limits.subgroup_arithmetic,
        limits.subgroup_clustered,
        limits.subgroup_shuffle
    );

    if width < 32 {
        println!(
            "SKIPPED show: these kernels are built for a subgroup of at least 32 and this device \
             reports {width}. The vectors are the subject here, so picking them from the width \
             would print the same mapping three times under three names."
        );
        return Ok(());
    }

    let count = WORKGROUP_SIZE as usize;
    let input: Vec<f32> = (0..count as u32).map(|index| index as f32).collect();
    println!("in            {:?}", &input[..8]);

    println!("\nreduce_sum, one source at three widths:");
    for (label, spirv) in [
        ("Simd<f32,4> ", kernels::lane_sum::<F32, 4>(width)?),
        ("Simd<f32,8> ", kernels::lane_sum::<F32, 8>(width)?),
        ("Simd<f32,32>", kernels::lane_sum::<F32, 32>(width)?),
    ] {
        let output = gpu.run(&spirv, &input, 1)?;
        println!("  {label}  {:?}", &output[..8]);
    }

    println!("\nthe rest of the surface:");
    let scaled = gpu.run(&kernels::scale(width, 2.0)?, &input, 1)?;
    println!("  scale x2      {:?}", &scaled[..8]);

    let affine = gpu.run(&kernels::lane_affine_whole(width)?, &input, 1)?;
    println!("  x*2 + 1       {:?}", &affine[..8]);

    let largest = gpu.run(&kernels::lane_max::<F32, 8>(width)?, &input, 1)?;
    println!("  reduce_max/8  {:?}", &largest[..8]);

    let butterfly = gpu.run(&kernels::butterfly_pair_sum(width, 1)?, &input, 1)?;
    println!("  butterfly ^1  {:?}", &butterfly[..8]);

    let wider = gpu.run(&kernels::butterfly_pair_sum(width, 4)?, &input, 1)?;
    println!("  butterfly ^4  {:?}", &wider[..8]);

    let voted = gpu.run(&kernels::any_above(width, 40.0)?, &input, 1)?;
    println!("  any(x > 40)   {:?}", &voted[..8]);

    println!("\nstrip-mined, and an integer:");
    let long: Vec<f32> = (0..count as u32 * 2).map(|index| index as f32).collect();
    let strided = gpu.run(&kernels::lane_sum::<F32, 64>(width)?, &long, 1)?;
    println!("  Simd<f32,64>  {:?}", &strided[..8]);

    let integers: Vec<u32> = (0..count as u32).collect();
    let summed = gpu.run_u32(&kernels::lane_sum::<U32, 32>(width)?, &integers, 1)?;
    println!("  Simd<u32,32>  {:?}", &summed[..8]);

    Ok(())
}
