//! One layer of a quantised neural network, because someone had one to point this at.
//!
//! The shape is taken from a chess engine's NNUE output layer — `H:\schaak\src\nnue.rs`, a
//! `768 → 256×2 → 1` perspective network whose entire per-evaluation arithmetic is this:
//!
//! ```text
//! fn clipped_dot(a: &[i16], w: &[i8], qa: i16) -> i32 {
//!     let mut s = 0i32;
//!     for (&v, &wj) in a.iter().zip(w.iter()) {
//!         s += i32::from(v.clamp(0, qa)) * i32::from(wj);
//!     }
//!     s
//! }
//! ```
//!
//! Called twice per position over 256 elements each. Nothing here is chess-specific: it is a
//! clipped-ReLU dot product, which is the inner loop of every dense quantised layer there is.
//!
//! # Why the clamp is spelled out
//!
//! There is no elementwise max or min in [`simdr::lanes`], so `v.clamp(0, qa)` becomes two
//! comparisons and two selects. That costs four instructions per element where a CPU spends one
//! `vpminsw`, and leaving it out would have made this benchmark a comparison between a hard
//! problem and an easier one. It is in.
//!
//! # Everything is i32
//!
//! The engine stores activations as `i16` and weights as `i8` and widens both to `i32` inside the
//! loop. SPIR-V without `Int8`/`Int16` capabilities has only 32-bit integers, so the buffers hold
//! the widened values and the arithmetic is identical — at four times the memory traffic. That is
//! a real cost of this port and is not hidden: see `notes/FINDINGS.md`.

use super::shape;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{I32, LaneError};

impl Layer {
    /// The engine's own constants: activations clamped to `[0, 255]`.
    ///
    /// `QA` in `H:\schaak\src\nnue.rs:138`. Named here rather than passed so a caller comparing
    /// against that engine cannot accidentally benchmark a different network.
    pub const QA: i32 = 255;
}

/// A clipped-ReLU dot product, built for one subgroup per output.
#[derive(Debug, Clone, Copy)]
pub struct Layer;

/// `out = Σ clamp(a[j], 0, qa) × w[j]` over the `LANES` one subgroup covers.
///
/// Activations occupy the first `offset` elements of binding 0 and weights the rest, so a caller
/// with two arrays concatenates them. Every lane of a subgroup receives that subgroup's whole
/// total, which is what a reduction means here.
///
/// `LANES = 256` on a 32-wide subgroup is eight strips — one full 256-element layer per subgroup,
/// the same granularity the engine's own loop runs at.
///
/// # Errors
///
/// [`LaneError`] if `LANES` has no mapping onto this subgroup, or the module cannot be built.
pub fn clipped_dot<const LANES: u32>(
    subgroup: u32,
    offset: u32,
    qa: i32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<I32>::new(shape(subgroup))?;
    let activations = kernel.load::<LANES>(0)?;
    let weights = kernel.load_offset::<LANES>(0, offset)?;

    let total = {
        let mut lanes = kernel.lanes()?;

        let floor = lanes.splat_bits::<I32, LANES>(0)?;
        let ceiling = lanes.splat_bits::<I32, LANES>(bits(qa))?;

        // `v.clamp(0, qa)`, from the outside in. Two compares and two selects because there is no
        // elementwise min or max; a `select` computes both sides and picks, so no lane diverges.
        let above = lanes.greater_than(activations, ceiling)?;
        let capped = lanes.select(above, ceiling, activations)?;
        let positive = lanes.greater_than(capped, floor)?;
        let clipped = lanes.select(positive, capped, floor)?;

        let products = lanes.mul(clipped, weights)?;
        lanes.reduce_sum(products)?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// The same layer with the two operands in *their own buffers*.
///
/// Activations at binding 0, weights at binding 1, the result at binding 2. That is how a caller
/// actually holds them — a weight table loaded once and a per-position accumulator — and until the
/// runner could bind more than two buffers they had to be concatenated into one with the join
/// passed as an offset. The arithmetic is identical; what changes is that the caller no longer has
/// to build a copy of both arrays to hand them over.
///
/// # Errors
///
/// [`LaneError`] if `LANES` has no mapping onto this subgroup, or the module cannot be built.
pub fn clipped_dot_split<const LANES: u32>(subgroup: u32, qa: i32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<I32>::new(Shape::new(subgroup, super::WORKGROUP_SIZE, 3))?;
    let activations = kernel.load::<LANES>(0)?;
    let weights = kernel.load::<LANES>(1)?;

    let total = {
        let mut lanes = kernel.lanes()?;

        let floor = lanes.splat_bits::<I32, LANES>(0)?;
        let ceiling = lanes.splat_bits::<I32, LANES>(bits(qa))?;

        let above = lanes.greater_than(activations, ceiling)?;
        let capped = lanes.select(above, ceiling, activations)?;
        let positive = lanes.greater_than(capped, floor)?;
        let clipped = lanes.select(positive, capped, floor)?;

        let products = lanes.mul(clipped, weights)?;
        lanes.reduce_sum(products)?
    };

    kernel.store_scalar(2, total)?;
    kernel.finish()
}

/// The same dot product with the clamp removed, to price it.
///
/// Not a network layer — a network without its nonlinearity is a linear map and computes something
/// else. It exists so the benchmark can say what the four clamp instructions cost rather than
/// guessing, and it is named so nobody mistakes it for the real thing.
///
/// # Errors
///
/// As [`clipped_dot`].
pub fn unclipped_dot<const LANES: u32>(subgroup: u32, offset: u32) -> Result<Vec<u32>, LaneError> {
    super::dot_product::<I32, LANES>(subgroup, offset)
}

/// A signed value as the bits a buffer holds.
///
/// `splat_bits` takes a bit pattern for all three element types, because the standard library has
/// no numeric trait covering them. This is the `i32` spelling.
#[must_use]
pub const fn bits(value: i32) -> u32 {
    u32::from_ne_bytes(value.to_ne_bytes())
}

/// What [`clipped_dot`] should produce, on the host.
///
/// The engine's loop, transcribed. Kept beside the kernel so the two can be read against each
/// other, and so a test comparing them is comparing against something legible rather than against
/// a second copy of the same mistake.
#[must_use]
pub fn reference(activations: &[i32], weights: &[i32], qa: i32) -> i32 {
    activations
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| value.clamp(0, qa).wrapping_mul(weight))
        .fold(0_i32, i32::wrapping_add)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_clamps_at_both_ends() {
        // Below zero contributes nothing, above qa contributes qa — the whole of a clipped ReLU.
        let weights = [1, 1, 1, 1];

        assert_eq!(reference(&[-5, 0, 100, 1_000], &weights, 255), 355);
        assert_eq!(reference(&[-1, -1, -1, -1], &weights, 255), 0);
        assert_eq!(reference(&[255, 255, 255, 255], &weights, 255), 1_020);
    }

    #[test]
    fn a_negative_weight_still_subtracts_after_the_clamp() {
        // The clamp is on the activation only. A test that used positive weights throughout would
        // pass with the clamp applied to the wrong operand.
        assert_eq!(reference(&[10], &[-3], 255), -30);
        assert_eq!(reference(&[-10], &[-3], 255), 0);
    }

    #[test]
    fn the_bit_spelling_round_trips_negative_values() {
        for value in [0_i32, 1, -1, i32::MIN, i32::MAX, -255] {
            assert_eq!(bits(value) as i32, value);
        }
    }
}
