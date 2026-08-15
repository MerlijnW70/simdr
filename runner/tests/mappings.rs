//! The four packed dot products, at all three lane mappings.
//!
//! **Every kernel in `runner::kernels::dot` is built through `whole_subgroup!`**, which fixes
//! `LANES` to the device's width. So `OpSDot`, `OpUDot`, `OpSUDot` and `OpSDotAccSat` had only ever
//! *executed* as whole-subgroup vectors — clustered they fold inside a cluster, strip-mined they
//! fold the strips first, and the saturating one saturates **per strip**. Twelve combinations, of
//! which four had ever run.
//!
//! That matters more here than elsewhere. `OpUDot` is the instruction that shipped **invalid**:
//! emitted with a signed result type, correct on two devices for weeks, and caught the first time
//! `spirv-val` was pointed at it. Being valid says nothing about being right, and being right at
//! one mapping says nothing about the other two — the four differ only where it hides. `OpSDot` and
//! `OpUDot` agree on every byte with its top bit clear; `OpSUDot` agrees with both wherever the
//! weights happen to be positive; and `OpSDotAccSat` differs from `OpSDot` *only at the overflow*.
//!
//! # Why this file exists twice over
//!
//! A sandbox proved these twelve agree on three devices and was deleted on 2026-08-16, taking the
//! check with it — `notes/FINDINGS.md` has the account. A measurement outlives the thing that took
//! it and a check does not, so this is the check, in the suite that runs on every push.
//!
//! It builds its own modules rather than calling `kernels::dot`, and that is the point: the
//! `whole_subgroup!` macro is exactly what this is testing around.

mod common;

use common::{VULKAN_1_1, device, validate};
use runner::kernels::WORKGROUP_SIZE;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{I32, LaneError, U32};

/// Which packed dot product a module emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dot {
    /// `OpSDot` — both operands signed.
    Signed,
    /// `OpUDot` — both unsigned, and the one that shipped invalid.
    Unsigned,
    /// `OpSUDot` — signed weights, unsigned activations. What a quantised layer actually wants,
    /// and the one whose operands are easiest to swap by accident.
    Mixed,
    /// `OpSDotAccSat` — signed, with a saturating accumulator.
    SignedSaturating,
}

impl Dot {
    /// All four, so a sweep cannot quietly cover three.
    const ALL: [Self; 4] = [
        Self::Signed,
        Self::Unsigned,
        Self::Mixed,
        Self::SignedSaturating,
    ];

    /// The instruction, for a report a reader can act on.
    const fn name(self) -> &'static str {
        match self {
            Self::Signed => "OpSDot",
            Self::Unsigned => "OpUDot",
            Self::Mixed => "OpSUDot",
            Self::SignedSaturating => "OpSDotAccSat",
        }
    }
}

/// Where the saturating accumulator starts.
///
/// **Chosen by measuring, and the obvious choice was nearly useless.** Four signed byte products
/// sum to roughly ±65 000, so an accumulator far from the ceiling saturates only at the top of that
/// range — swapping the reference's `saturating_add` for a `wrapping_add`, which is exactly the
/// mistake the instruction exists to prevent, then disagreed on one seed in thirty-two for one
/// mapping and on none at all for the other two.
///
/// A thousand from the ceiling saturates whenever the products are positive, which is about half
/// the lanes, so both sides of the saturation are reached every seed. An instruction that differs
/// only at the overflow is only tested at the overflow.
const ACCUMULATOR: i32 = i32::MAX - 1_000;

/// Invocations per workgroup.
const WORKGROUP: u32 = WORKGROUP_SIZE;

/// A layer: `Σ w[j] × a[j]`, four products to a word, reduced across the vector.
///
/// Binding 0 holds every weight and then every activation, so one buffer carries both operands and
/// `Kernel::load_offset` reaches the second — the arrangement a caller with two arrays would
/// produce by concatenating them.
fn layer<const LANES: u32>(kind: Dot, subgroup: u32, offset: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, WORKGROUP, 2))?;
    let weights = kernel.load::<LANES>(0)?;
    let activations = kernel.load_offset::<LANES>(0, offset)?;

    let total = {
        let mut lanes = kernel.lanes()?;

        // **`reinterpret`, and leaving it out is a module `spirv-val` rejects.** Three of the four
        // answer with an `i32` and this kernel's buffer holds `u32`, so the store is a type
        // mismatch. When the sandbox got this wrong an RTX 4080 and an integrated Radeon each ran
        // it 192 times and agreed with the reference every time; lavapipe refused the module with
        // `ERROR_UNKNOWN` and said nothing about why. `OpUDot`'s story with the parts in the same
        // order — and `OpUDot` answers with a `u32` and needs none, which is the asymmetry the bug
        // lived in.
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

        // The reduction is what makes the mapping visible: whole-subgroup it is one instruction,
        // clustered it folds inside the cluster, and strip-mined it folds the strips first. It runs
        // on the `u32` bits in every case, so the reference below folds one way rather than four.
        lanes.reduce_sum(packed)?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// The same arithmetic on the host, written from the addressing rather than from the kernel.
///
/// **Deliberately not sharing a line with the emitter.** A reference that reuses the thing it
/// checks agrees with it about the same mistake — which is how `reduce_min` came to fold its strips
/// with a maximum and pass every hand-written test but one.
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

    // Each invocation's own total, over its strips. The dot is *per strip* — the emitter zips the
    // strips of both operands and the accumulator — so a saturating one saturates once per strip
    // rather than once per lane, and folding the strips first would hide that.
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

    // The fold across the vector the mapping describes, on the `u32` bits — which is what the
    // kernel reduces, whatever the dot answered with.
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

/// One word of each operand through one dot product, as the `u32` bits it stores as.
///
/// The byte order is the little-endian one SPIR-V's packed formats define: component zero is the
/// low byte. Getting that backwards is a wrong answer that looks plausible, and it is the only
/// thing in this reference that is a *choice* rather than arithmetic.
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
                // The one difference between `OpSDot` and `OpSUDot`, and it only shows where the
                // second operand's top bit is set: −1 or 255.
                let activation = match kind {
                    Dot::Mixed => i32::from(byte(activations, index)),
                    _ => i32::from(byte(activations, index) as i8),
                };
                carried.wrapping_add(weight.wrapping_mul(activation))
            });

            match kind {
                // Saturating, not wrapping: the whole point of the instruction, and invisible
                // unless the accumulator is near the ceiling — which is why `ACCUMULATOR` is.
                Dot::SignedSaturating => ACCUMULATOR.saturating_add(total) as u32,
                _ => total as u32,
            }
        }
    }
}

/// `count` words from `seed`, full range.
///
/// Full range because every byte of every word is an operand: a corpus of small values would leave
/// the sign of a weight and the top bit of an activation untested, and those are exactly what
/// `OpSUDot` treats differently from `OpSDot`.
fn drawn(seed: u64, count: usize) -> Vec<u32> {
    let mut rng = runner::fuzz::Rng::new(seed);
    (0..count).map(|_| rng.next() as u32).collect()
}

/// Run one (dot, mapping) pairing over every seed **in one dispatch**, and compare each.
///
/// One workgroup per seed, so the invocation's own index selects its problem. The layout is every
/// problem's weights and then every problem's activations, which is why the offset handed to
/// `layer` is the *whole batch's* and not one workgroup's — the second is right at a batch of one
/// and wrong at every larger size, which is exactly how it survived a suite once.
fn agreed<const LANES: u32>(gpu: &runner::Gpu, kind: Dot, seeds: u64) -> Result<bool, String> {
    let width = gpu.limits().subgroup_size;
    let strips = (LANES / width.max(1)).max(1);
    let per_problem = WORKGROUP as usize * strips as usize;

    let spirv = match layer::<LANES>(kind, width, (seeds as usize * per_problem) as u32) {
        Ok(spirv) => spirv,
        // A lane count with no mapping on this device is the lane API working, not a failure. The
        // caller counts these so a run made entirely of them cannot look green.
        Err(refused) => return Err(format!("refused: {refused}")),
    };

    let missing = gpu.limits().unsupported_in(&spirv);
    if !missing.is_empty() {
        return Err(format!("device lacks {missing:?}"));
    }
    // **Before the device, every time.** A driver is lenient about things the validator is not.
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

/// The three lane counts around a width, as the three mappings.
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
                // Lost coverage rather than failure, and printed rather than counted silently: a
                // device without the dot-product extension is being honest, and a skipped check
                // that looks green is worse than a red one.
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

    // **Without this the test is vacuous.** Every combination being unsupported would print tidy
    // lines and assert nothing, which is the failure this whole repository is about. A device
    // without `VK_KHR_shader_integer_dot_product` reaches none; one with it should reach twelve.
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
