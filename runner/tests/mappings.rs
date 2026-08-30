mod common;

use common::{VULKAN_1_1, device, validate};
use runner::kernels::WORKGROUP_SIZE;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{I32, LaneError, U32};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dot {
    Signed,
    Unsigned,
    Mixed,
    SignedSaturating,
}

impl Dot {
    const ALL: [Self; 4] = [
        Self::Signed,
        Self::Unsigned,
        Self::Mixed,
        Self::SignedSaturating,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Signed => "OpSDot",
            Self::Unsigned => "OpUDot",
            Self::Mixed => "OpSUDot",
            Self::SignedSaturating => "OpSDotAccSat",
        }
    }
}

const ACCUMULATOR: i32 = i32::MAX - 1_000;

const WORKGROUP: u32 = WORKGROUP_SIZE;

fn layer<const LANES: u32>(kind: Dot, subgroup: u32, offset: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, WORKGROUP, 2))?;
    let weights = kernel.load::<LANES>(0)?;
    let activations = kernel.load_offset::<LANES>(0, offset)?;

    let total = {
        let mut lanes = kernel.lanes()?;

        let packed = match kind {
            Dot::Signed => {
                let products = lanes.dot_signed(weights, activations)?;
                lanes.reinterpret(products)?
            }
            Dot::Unsigned => lanes.dot_unsigned(weights, activations)?,
            Dot::Mixed => {
                let products = lanes.dot_mixed(weights, activations)?;
                lanes.reinterpret(products)?
            }
            Dot::SignedSaturating => {
                let carried = lanes.splat_bits::<I32, LANES>(ACCUMULATOR as u32)?;
                let products = lanes.dot_signed_saturating(weights, activations, carried)?;
                lanes.reinterpret(products)?
            }
        };

        lanes.reduce_sum(packed)?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

fn reference(
    kind: Dot,
    input: &[u32],
    offset: usize,
    width: u32,
    lanes: u32,
    strips: u32,
) -> Vec<u32> {
    let invocations = WORKGROUP as usize;
    let vector = (lanes.min(width) as usize).max(1);

    let mine: Vec<u32> = (0..invocations)
        .map(|invocation| {
            (0..strips as usize)
                .map(|strip| {
                    let at = invocation + strip * invocations;
                    let weights = input.get(at).copied().unwrap_or(0);
                    let activations = input.get(offset + at).copied().unwrap_or(0);
                    products(kind, weights, activations)
                })
                .fold(0_u32, u32::wrapping_add)
        })
        .collect();

    (0..invocations)
        .map(|invocation| {
            let base = invocation / vector * vector;
            mine.get(base..(base + vector).min(invocations))
                .unwrap_or_default()
                .iter()
                .fold(0_u32, |carried, &value| carried.wrapping_add(value))
        })
        .collect()
}

fn products(kind: Dot, weights: u32, activations: u32) -> u32 {
    let byte = |word: u32, index: u32| ((word >> (index * 8)) & 0xff) as u8;

    match kind {
        Dot::Unsigned => (0..4).fold(0_u32, |carried, index| {
            let product = u32::from(byte(weights, index)) * u32::from(byte(activations, index));
            carried.wrapping_add(product)
        }),
        Dot::Signed | Dot::Mixed | Dot::SignedSaturating => {
            let total = (0..4).fold(0_i32, |carried, index| {
                let weight = i32::from(byte(weights, index) as i8);
                let activation = match kind {
                    Dot::Mixed => i32::from(byte(activations, index)),
                    _ => i32::from(byte(activations, index) as i8),
                };
                carried.wrapping_add(weight.wrapping_mul(activation))
            });

            match kind {
                Dot::SignedSaturating => ACCUMULATOR.saturating_add(total) as u32,
                _ => total as u32,
            }
        }
    }
}

fn drawn(seed: u64, count: usize) -> Vec<u32> {
    let mut rng = runner::fuzz::Rng::new(seed);
    (0..count).map(|_| rng.next() as u32).collect()
}

fn agreed<const LANES: u32>(gpu: &runner::Gpu, kind: Dot, seeds: u64) -> Result<bool, String> {
    let width = gpu.limits().subgroup_size;
    let strips = (LANES / width.max(1)).max(1);
    let per_problem = WORKGROUP as usize * strips as usize;

    let spirv = match layer::<LANES>(kind, width, (seeds as usize * per_problem) as u32) {
        Ok(spirv) => spirv,
        Err(refused) => return Err(format!("refused: {refused}")),
    };

    let missing = gpu.limits().unsupported_in(&spirv);
    if !missing.is_empty() {
        return Err(format!("device lacks {missing:?}"));
    }
    validate(&spirv, &format!("dot-{}-{LANES}", kind.name()), VULKAN_1_1)
        .map_err(|complaint| format!("spirv-val rejected it: {complaint}"))?;

    let per_seed: Vec<Vec<u32>> = (0..seeds)
        .map(|seed| drawn(seed, per_problem * 2))
        .collect();

    let mut input = Vec::with_capacity(per_problem * 2 * seeds as usize);
    for words in &per_seed {
        input.extend(words.iter().take(per_problem).copied());
    }
    for words in &per_seed {
        input.extend(words.iter().skip(per_problem).copied());
    }

    let returned = gpu
        .run_u32(&spirv, &input, seeds as u32)
        .map_err(|error| format!("the driver failed on a validated module: {error}"))?;

    for (index, words) in per_seed.iter().enumerate() {
        let expected = reference(kind, words, per_problem, width, LANES, strips);
        let start = index * WORKGROUP as usize;
        let actual = returned
            .get(start..start + WORKGROUP as usize)
            .ok_or_else(|| format!("seed {index}: the device returned fewer words than asked"))?;

        if let Some(at) = actual
            .iter()
            .zip(&expected)
            .position(|(left, right)| left != right)
        {
            return Err(format!(
                "seed {index} disagreed at element {at}: device {:?}, reference {:?}",
                actual.get(at),
                expected.get(at)
            ));
        }
    }

    Ok(true)
}

macro_rules! mappings {
    ($gpu:expr, $kind:expr, $seeds:expr, $half:literal, $whole:literal, $double:literal) => {
        vec![
            ("clustered", agreed::<$half>($gpu, $kind, $seeds)),
            ("whole", agreed::<$whole>($gpu, $kind, $seeds)),
            ("strip-mined", agreed::<$double>($gpu, $kind, $seeds)),
        ]
    };
}

#[test]
fn every_packed_dot_agrees_at_every_mapping() {
    let Some(gpu) = device("dot mappings") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let seeds = 6;
    let mut executed = 0;
    let mut complaints = Vec::new();

    for kind in Dot::ALL {
        let outcomes = match width {
            4 => mappings!(&gpu, kind, seeds, 2, 4, 8),
            8 => mappings!(&gpu, kind, seeds, 4, 8, 16),
            16 => mappings!(&gpu, kind, seeds, 8, 16, 32),
            32 => mappings!(&gpu, kind, seeds, 16, 32, 64),
            64 => mappings!(&gpu, kind, seeds, 32, 64, 128),
            other => {
                eprintln!("SKIPPED dot mappings: no lane counts written for a subgroup of {other}");
                return;
            }
        };

        for (mapping, outcome) in outcomes {
            match outcome {
                Ok(_) => executed += 1,
                Err(why) if why.starts_with("refused") || why.starts_with("device lacks") => {
                    eprintln!("  {} {mapping} not run: {why}", kind.name());
                }
                Err(why) => complaints.push(format!("{} {mapping}: {why}", kind.name())),
            }
        }
    }

    assert!(
        complaints.is_empty(),
        "a packed dot product did not come back right:\n{}",
        complaints.join("\n")
    );

    if executed == 0 {
        eprintln!("SKIPPED dot mappings: this device offers no packed dot product");
        return;
    }
    assert_eq!(
        executed, 12,
        "only {executed} of twelve dot × mapping combinations ran, and the rest were neither \
         refused by name nor unsupported"
    );
}
