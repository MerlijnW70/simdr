//! ```text
//! fn clipped_dot(a: &[i16], w: &[i8], qa: i16) -> i32 {
//!     let mut s = 0i32;
//!     for (&v, &wj) in a.iter().zip(w.iter()) {
//!         s += i32::from(v.clamp(0, qa)) * i32::from(wj);
//!     }
//!     s
//! }
//! ```

use super::shape;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{I32, LaneError};

impl Layer {
    pub const QA: i32 = 255;
}

#[derive(Debug, Clone, Copy)]
pub struct Layer;

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

pub fn unclipped_dot<const LANES: u32>(subgroup: u32, offset: u32) -> Result<Vec<u32>, LaneError> {
    super::dot_product::<I32, LANES>(subgroup, offset)
}

#[must_use]
pub const fn bits(value: i32) -> u32 {
    u32::from_ne_bytes(value.to_ne_bytes())
}

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
        let weights = [1, 1, 1, 1];

        assert_eq!(reference(&[-5, 0, 100, 1_000], &weights, 255), 355);
        assert_eq!(reference(&[-1, -1, -1, -1], &weights, 255), 0);
        assert_eq!(reference(&[255, 255, 255, 255], &weights, 255), 1_020);
    }

    #[test]
    fn a_negative_weight_still_subtracts_after_the_clamp() {
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
