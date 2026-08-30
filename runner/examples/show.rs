use runner::Gpu;
use runner::kernels::{self, Bitwise, Comparison, WORKGROUP_SIZE};
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

    println!("\narithmetic and the six orderings, against 4:");
    let difference = gpu.run(&kernels::lane_sub(width, 1.0)?, &input, 1)?;
    println!("  x - 1         {:?}", &difference[..8]);

    let quotient = gpu.run(&kernels::lane_div(width, 2.0)?, &input, 1)?;
    println!("  x / 2         {:?}", &quotient[..8]);

    let negated = gpu.run(&kernels::lane_neg(width)?, &input, 1)?;
    println!("  -x            {:?}", &negated[..8]);

    for (label, comparison) in [
        ("x <  4", Comparison::Less),
        ("x <= 4", Comparison::LessEqual),
        ("x >  4", Comparison::Greater),
        ("x >= 4", Comparison::GreaterEqual),
        ("x == 4", Comparison::Equal),
        ("x != 4", Comparison::NotEqual),
    ] {
        let held = gpu.run(&kernels::lane_compare(width, 4.0, comparison)?, &input, 1)?;
        let flags: Vec<u32> = held[..8].iter().map(|value| *value as u32).collect();
        println!("  {label}        {flags:?}");
    }

    println!("\nthe bitwise four, in hex against 0x5:");
    let bits: Vec<u32> = (0..count as u32).collect();
    println!("  in            {}", hex(&bits[..8]));
    for (label, operation) in [
        ("x & 5", Bitwise::And),
        ("x | 5", Bitwise::Or),
        ("x ^ 5", Bitwise::Xor),
        ("!x   ", Bitwise::Not),
    ] {
        let output = gpu.run_u32(
            &kernels::lane_bitwise_with(width, 0x5, operation)?,
            &bits,
            1,
        )?;
        println!("  {label}         {}", hex(&output[..8]));
    }

    println!("\nreductions over the subgroup, and a product in clusters of four:");
    let folded: Vec<u32> = (0..count as u32)
        .map(|index| 0b1100 | (index % 3))
        .collect();
    println!("  in            {}", hex(&folded[..8]));
    for (label, spirv) in [
        ("reduce_and", kernels::lane_and_whole::<U32>(width)?),
        ("reduce_or ", kernels::lane_or_whole::<U32>(width)?),
        ("reduce_xor", kernels::lane_xor_whole::<U32>(width)?),
    ] {
        let output = gpu.run_u32(&spirv, &folded, 1)?;
        println!("  {label}    {}", hex(&output[..8]));
    }

    let product = gpu.run_u32(&kernels::lane_product::<U32, 4>(width)?, &folded, 1)?;
    println!("  product/4     {:?}", &product[..8]);

    println!("\nstrip-mined, and an integer:");
    let long: Vec<f32> = (0..count as u32 * 2).map(|index| index as f32).collect();
    let strided = gpu.run(&kernels::lane_sum::<F32, 64>(width)?, &long, 1)?;
    println!("  Simd<f32,64>  {:?}", &strided[..8]);

    let integers: Vec<u32> = (0..count as u32).collect();
    let summed = gpu.run_u32(&kernels::lane_sum::<U32, 32>(width)?, &integers, 1)?;
    println!("  Simd<u32,32>  {:?}", &summed[..8]);

    Ok(())
}

fn hex(values: &[u32]) -> String {
    let cells: Vec<String> = values
        .iter()
        .map(|value| format!("0x{value:08x}"))
        .collect();
    format!("[{}]", cells.join(", "))
}
