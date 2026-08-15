//! Tools that put the engine under pressure, each carrying its own exact oracle.
//!
//! See `README.md` for the isolation contract and for how to delete all of this. The short version:
//! nothing in `src/`, `runner/` or `cli/` may refer to anything here, and nothing does.
//!
//! # Why a quantised layer is the first one
//!
//! A workload is only a test if something can disagree with it, and integer arithmetic is the only
//! kind here whose reference is *exact* rather than approximate. A neural-network layer over `u8`
//! activations and `i8` weights answers with an `i32`, so the CPU can compute the same number and
//! the comparison is a fact rather than a tolerance.
//!
//! It also lands on two measured gaps at once — `notes/CLAIMS.md` has both, and `README.md` here
//! restates them. The one this module attacks first is the sharper of the two:
//!
//! **The packed dot products have only ever run whole-subgroup.** `runner::kernels::dot` builds
//! every one of them through `whole_subgroup!`, which fixes `LANES` to the device's width. So
//! `OpSDot`, `OpUDot` and `OpSUDot` — the family in which `OpUDot` shipped *invalid*, correct on two
//! devices for weeks until the first `spirv-val` run against it — have never been executed as a
//! **clustered** vector or a **strip-mined** one, where they are a different instruction sequence
//! with a different fold behind them.

// The validator harness the engine's own test trees share, reached by path rather than copied —
// two copies would be two things to keep in step, and this is the layer whose absence is the whole
// lesson below.
#[path = "../../tests/common/spirv_val.rs"]
pub(crate) mod spirv_val;

pub mod conversions;

use runner::kernels::WORKGROUP_SIZE;
use runner::{Error, Gpu};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{LaneError, U32};

/// Invocations per workgroup. One workgroup keeps the addressing short enough to re-derive by hand,
/// which is the point of a reference.
pub const WORKGROUP: u32 = WORKGROUP_SIZE;

/// What one run of the layer concluded.
#[derive(Debug)]
pub struct Checked {
    /// Which mapping the vector had — the thing under test.
    pub mapping: &'static str,
    /// The vector's lane count.
    pub lanes: u32,
    /// The device's subgroup width.
    pub width: u32,
    /// How many elements each invocation held.
    pub strips: u32,
    /// The first index where the device and the reference differ, if any.
    pub disagreed_at: Option<usize>,
}

impl Checked {
    /// Whether this run agreed everywhere.
    #[must_use]
    pub const fn agreed(&self) -> bool {
        self.disagreed_at.is_none()
    }
}

/// Which packed dot product the layer uses.
///
/// **All four, because they differ in exactly the way that hides.** `OpSDot` and `OpUDot` agree on
/// every byte with its top bit clear, and `OpSUDot` agrees with both when the weights happen to be
/// positive — so a corpus of small values proves one instruction and reads as proving three. That
/// is the shape `OpUDot` was invalid in: correct on two devices for weeks, with a signed result
/// type, until the first `spirv-val` run against it.
///
/// The saturating one differs from `OpSDot` **only at the overflow**, which is the sharpest version
/// of the same trap: an accumulator nowhere near the limit makes the two indistinguishable. So the
/// accumulator here starts near `i32::MAX` on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dot {
    /// `OpSDot` — both operands signed bytes.
    Signed,
    /// `OpUDot` — both unsigned.
    Unsigned,
    /// `OpSUDot` — signed weights against unsigned activations, which is what a quantised layer is.
    Mixed,
    /// `OpSDotAccSat` — signed, accumulating with saturation.
    SignedSaturating,
}

impl Dot {
    /// The four, in a fixed order so a report reads the same twice.
    pub const ALL: [Self; 4] = [
        Self::Signed,
        Self::Unsigned,
        Self::Mixed,
        Self::SignedSaturating,
    ];

    /// How this spells in a report.
    #[must_use]
    pub const fn name(self) -> &'static str {
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
/// **Chosen by measuring, and the first choice was nearly useless.** Four signed byte products sum
/// to roughly ±65 000, so an accumulator at `i32::MAX - 40_000` only saturates on the top of that
/// range: swapping the reference's `saturating_add` for a `wrapping_add` — which is exactly the
/// mistake the instruction exists to prevent — disagreed on **one seed in thirty-two** for one
/// mapping and on none at all for the other two.
///
/// A thousand from the ceiling saturates whenever the products are positive, which is about half
/// the lanes, so both sides of the saturation are reached every seed. An instruction that differs
/// only at the overflow is only tested at the overflow.
pub const ACCUMULATOR: i32 = i32::MAX - 1_000;

/// A quantised layer: `Σ w[j] × a[j]`, four products to a word, reduced across the vector.
///
/// Binding 0 holds the weights and then the activations, so one buffer carries both operands and
/// `Kernel::load_offset` reaches the second — the same arrangement `kernels::network::clipped_dot`
/// uses, and the one a caller with two arrays would produce by concatenating them.
///
/// `OpSUDot` rather than `OpSDot`: a quantised layer's weights are signed and its activations are
/// not, which is the mixed form and the one whose operands are easiest to swap by accident.
///
/// # Errors
///
/// [`LaneError`] if `LANES` has no mapping onto this subgroup, or the module cannot be built.
pub fn layer<const LANES: u32>(
    kind: Dot,
    subgroup: u32,
    offset: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, WORKGROUP, 2))?;
    let weights = kernel.load::<LANES>(0)?;
    let activations = kernel.load_offset::<LANES>(0, offset)?;

    let total = {
        let mut lanes = kernel.lanes()?;

        // **`reinterpret`, and the first version of this had it missing.** Three of the four answer
        // with an `i32` and this kernel's buffer holds `u32`, so the store was a type mismatch —
        // invalid SPIR-V. An RTX 4080 and an integrated Radeon ran it 192 times each and agreed
        // with the reference every time; lavapipe refused the module with `ERROR_UNKNOWN` and said
        // nothing about why. That is `OpUDot`'s story with the parts in the same order.
        //
        // `OpUDot` answers with a `u32` and needs none, which is the asymmetry the bug lived in.
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
                let carried = lanes.splat_bits::<simdr::lanes::I32, LANES>(ACCUMULATOR as u32)?;
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
/// **Deliberately not sharing a line with the emitter.** A reference that reuses the thing it checks
/// is a reference that agrees with it about the same mistake — which is how `reduce_min` came to
/// fold its strips with a maximum and pass every hand-written test but one.
///
/// The addressing is `Kernel::run_start` plus the strip stride, spelled out: invocation `i` of a
/// workgroup holds elements `i`, `i + workgroup`, `i + 2·workgroup`, and the vector it belongs to is
/// `min(lanes, width)` invocations wide.
#[must_use]
pub fn reference(
    kind: Dot,
    input: &[u32],
    offset: usize,
    width: u32,
    lanes: u32,
    strips: u32,
) -> Vec<u32> {
    let invocations = WORKGROUP as usize;
    let vector = (lanes.min(width) as usize).max(1);

    // Step one: each invocation's own total, over its strips. The dot is *per strip* — the emitter
    // zips the strips of both operands and the accumulator — so a saturating one saturates once per
    // strip rather than once per lane, and folding the strips first would hide that.
    let mine: Vec<u32> = (0..invocations)
        .map(|invocation| {
            (0..strips as usize)
                .map(|strip| {
                    let at = invocation + strip * invocations;
                    let weights = input.get(at).copied().unwrap_or(0);
                    let activations = input.get(offset + at).copied().unwrap_or(0);
                    packed_products(kind, weights, activations)
                })
                .fold(0_u32, u32::wrapping_add)
        })
        .collect();

    // Step two: the fold across the vector the mapping describes, on the `u32` bits — which is what
    // the kernel reduces, whatever the dot answered with.
    (0..invocations)
        .map(|invocation| {
            let base = invocation / vector * vector;
            mine[base..(base + vector).min(invocations)]
                .iter()
                .fold(0_u32, |carried, &value| carried.wrapping_add(value))
        })
        .collect()
}

/// One word of each operand, through one of the four dot products, as the `u32` bits it stores as.
///
/// The byte order is the little-endian one SPIR-V's packed formats define: component zero is the
/// low byte. Getting that backwards is a wrong answer that looks plausible, and it is the only
/// thing in this reference that is a *choice* rather than arithmetic — so reversing it is the check
/// that this file can disagree at all.
#[must_use]
pub fn packed_products(kind: Dot, weights: u32, activations: u32) -> u32 {
    let byte = |word: u32, index: u32| ((word >> (index * 8)) & 0xff) as u8;

    match kind {
        Dot::Unsigned => (0..4).fold(0_u32, |carried, index| {
            let product = u32::from(byte(weights, index)) * u32::from(byte(activations, index));
            carried.wrapping_add(product)
        }),
        Dot::Signed | Dot::Mixed | Dot::SignedSaturating => {
            let products = (0..4).fold(0_i32, |carried, index| {
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
                Dot::SignedSaturating => ACCUMULATOR.saturating_add(products) as u32,
                _ => products as u32,
            }
        }
    }
}

/// What one attempt at a mapping came to.
///
/// **Four outcomes rather than a `Result`, because three of them are not failures and only one is
/// silence.** A lane count with no mapping is the lane API working; a device that does not offer an
/// instruction is the device being honest; a driver that accepts the module and then errors is a
/// finding of its own, and it is neither of the first two. Collapsing them loses exactly the
/// distinction a tool like this exists to draw.
#[derive(Debug)]
pub enum Outcome {
    /// The mapping refused to build the module, by name.
    Refused(LaneError),
    /// The module is legal and this device does not offer what it declares.
    Unsupported(Vec<simdr::spec::Capability>),
    /// `spirv-val` rejected the module, so it was never dispatched.
    ///
    /// **The outcome this tool did not have on its first outing, and the one that mattered.** The
    /// layer stored an `i32` into a `u32` buffer; two devices ran it 192 times each and agreed with
    /// the reference every time, and a third refused it with `ERROR_UNKNOWN`. A sandbox that
    /// dispatches without validating reproduces the exact failure the engine's own suite exists to
    /// prevent — `runner/tests/validated.rs` opens by describing it.
    Invalid(String),
    /// The device accepted the module and then failed to run it.
    ///
    /// Not a disagreement, not a refusal, and — now that the module is validated first — not a
    /// module this crate got wrong either. `notes/FINDINGS.md` records the precedent: an integrated
    /// Radeon faults inside `vkCreateComputePipelines` on a clustered scan that `spirv-val` accepts
    /// and two other implementations run correctly.
    Errored(Error),
    /// It ran, and here is whether it agreed.
    Ran(Checked),
}

/// Build the layer at `LANES`, run it, and compare against [`reference`].
pub fn check<const LANES: u32>(
    gpu: &Gpu,
    kind: Dot,
    mapping: &'static str,
    seed: u64,
) -> Outcome {
    let width = gpu.limits().subgroup_size;
    let strips = (LANES / width.max(1)).max(1);
    let per_operand = WORKGROUP as usize * strips as usize;

    let spirv = match layer::<LANES>(kind, width, per_operand as u32) {
        Ok(spirv) => spirv,
        Err(refused) => return Outcome::Refused(refused),
    };

    let missing = gpu.limits().unsupported_in(&spirv);
    if !missing.is_empty() {
        return Outcome::Unsupported(missing);
    }

    // **Before the device, every time.** A driver is lenient about things the validator is not, and
    // an invalid module here came back as 192 correct-looking answers on one device and an opaque
    // `ERROR_UNKNOWN` on another.
    if let Err(complaint) = spirv_val::validate(
        &spirv,
        &format!("proeftuin-layer-{mapping}-{LANES}"),
        spirv_val::VULKAN_1_1,
    ) {
        return Outcome::Invalid(complaint);
    }

    let input = drawn(seed, per_operand * 2);
    let actual = match gpu.run_u32(&spirv, &input, 1) {
        Ok(actual) => actual,
        Err(error) => return Outcome::Errored(error),
    };
    let expected = reference(kind, &input, per_operand, width, LANES, strips);

    Outcome::Ran(Checked {
        mapping,
        lanes: LANES,
        width,
        strips,
        disagreed_at: actual
            .iter()
            .zip(&expected)
            .position(|(left, right)| left != right),
    })
}

/// `count` words from `seed`, full range.
///
/// Full range because every byte of every word is an operand: a corpus of small values would leave
/// the sign of a weight and the top bit of an activation untested, and those are exactly what
/// `OpSUDot` treats differently from `OpSDot`.
#[must_use]
pub fn drawn(seed: u64, count: usize) -> Vec<u32> {
    let mut rng = runner::fuzz::Rng::new(seed);
    (0..count).map(|_| rng.next() as u32).collect()
}
