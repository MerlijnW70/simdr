pub mod control;
pub mod dot;
pub mod extended;
pub mod narrow;
pub mod network;
pub mod occupancy;
pub mod plane;
pub mod reduce;
pub mod scan;
pub mod scatter;
pub mod specialized;
pub mod unrun;

pub use control::{
    any_above, branch_in_loop, branch_only, loop_in_branch, rolled_counter_sum, rolled_doubling,
    scale_if_any_above, sum_or_max,
};
pub use dot::{
    byte_component, mixed_dot, packed_dot, repeated_packed_dot, repeated_unpacked_dot, unpacked_dot,
};
pub use extended::{clamped, fused_square, larger, magnitude, root, smaller};
pub use narrow::{narrow_add, narrow_clamp, narrow_sum, narrow_sum_whole};
pub use network::{Layer, clipped_dot, clipped_dot_split, unclipped_dot};
pub use occupancy::{sized_lane_sum, sized_repeated_scale};
pub use plane::{flat_scale, row_bias, row_index, row_scale, row_sum};
pub use reduce::{
    FOLD_HALF_SPEC_ID, butterfly_cluster_sum, butterfly_pair_sum, butterfly_tree_sum, dot_product,
    dot_product_whole, fold_by, fold_halves, fold_halves_open, lane_and, lane_and_whole, lane_max,
    lane_max_whole, lane_or, lane_or_whole, lane_product, lane_product_whole, lane_sum,
    lane_sum_whole, lane_xor, lane_xor_whole, workgroup_sum,
};
pub use scatter::{atomic_gather, claim_slots, exchange_chain, histogram, histogram_incrementing};
pub use specialized::{
    specialized_add, specialized_affine, specialized_cluster, specialized_derived,
};
pub use unrun::{
    all_above, ballot_above, broadcast, broadcast_in_cluster, centre_and_scale, equals, lane_min,
    prefix_sum, remainder, rolled_block_sum, rolled_weighted_totals, rotate_in_cluster, shift_down,
    shift_up, subgroup_agrees, subgroup_agrees_wide,
};

use simdr::kernel::{Kernel, Shape};
use simdr::lanes::LaneError;

pub const WORKGROUP_SIZE: u32 = 64;

macro_rules! whole_subgroup {
    ($subgroup:expr, $build:ident $(, $argument:expr)* $(,)?) => {
        match $subgroup {
            4 => $build::<4>($subgroup $(, $argument)*),
            8 => $build::<8>($subgroup $(, $argument)*),
            16 => $build::<16>($subgroup $(, $argument)*),
            32 => $build::<32>($subgroup $(, $argument)*),
            64 => $build::<64>($subgroup $(, $argument)*),
            width => Err(simdr::lanes::LaneError::BadWidth { width }),
        }
    };
}

macro_rules! whole_subgroup_of {
    ($element:ty, $subgroup:expr, $build:ident $(, $argument:expr)* $(,)?) => {
        match $subgroup {
            4 => $build::<$element, 4>($subgroup $(, $argument)*),
            8 => $build::<$element, 8>($subgroup $(, $argument)*),
            16 => $build::<$element, 16>($subgroup $(, $argument)*),
            32 => $build::<$element, 32>($subgroup $(, $argument)*),
            64 => $build::<$element, 64>($subgroup $(, $argument)*),
            width => Err(simdr::lanes::LaneError::BadWidth { width }),
        }
    };
}

pub(crate) use {whole_subgroup, whole_subgroup_of};

#[must_use]
pub fn shape(subgroup: u32) -> Shape {
    Shape::new(subgroup, WORKGROUP_SIZE, 2)
}

pub fn scale(subgroup: u32, factor: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, scale_at, factor)
}

fn scale_at<const LANES: u32>(subgroup: u32, factor: f32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let scaled = {
        let mut lanes = kernel.lanes()?;
        let factor = lanes.splat_bits::<F32, LANES>(factor.to_bits())?;
        lanes.mul(value, factor)?
    };
    kernel.store(1, scaled)?;
    kernel.finish()
}

/// `x - rhs`, `x / rhs` and `-x`, one operation apiece so a tour can print each
/// on its own line rather than a single number four of them agree on.
pub fn lane_sub(subgroup: u32, rhs: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_sub_at, rhs)
}

fn lane_sub_at<const LANES: u32>(subgroup: u32, rhs: f32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let difference = {
        let mut lanes = kernel.lanes()?;
        let rhs = lanes.splat_bits::<F32, LANES>(rhs.to_bits())?;
        lanes.sub(value, rhs)?
    };
    kernel.store(1, difference)?;
    kernel.finish()
}

pub fn lane_div(subgroup: u32, rhs: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_div_at, rhs)
}

fn lane_div_at<const LANES: u32>(subgroup: u32, rhs: f32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let quotient = {
        let mut lanes = kernel.lanes()?;
        let rhs = lanes.splat_bits::<F32, LANES>(rhs.to_bits())?;
        lanes.div(value, rhs)?
    };
    kernel.store(1, quotient)?;
    kernel.finish()
}

pub fn lane_neg(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_neg_at)
}

fn lane_neg_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let negated = kernel.lanes()?.neg(value)?;
    kernel.store(1, negated)?;
    kernel.finish()
}

/// Which of the six orderings [`lane_compare`] asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
}

/// `1.0` where the comparison holds and `0.0` where it does not, so the answer
/// reads as the predicate itself rather than as whatever was selected by it.
pub fn lane_compare(
    subgroup: u32,
    threshold: f32,
    comparison: Comparison,
) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_compare_at, threshold, comparison)
}

fn lane_compare_at<const LANES: u32>(
    subgroup: u32,
    threshold: f32,
    comparison: Comparison,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let held = {
        let mut lanes = kernel.lanes()?;
        let threshold = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let one = lanes.splat_bits::<F32, LANES>(1.0_f32.to_bits())?;
        let zero = lanes.splat_bits::<F32, LANES>(0.0_f32.to_bits())?;

        let predicate = match comparison {
            Comparison::Less => lanes.less_than(value, threshold)?,
            Comparison::LessEqual => lanes.less_equal(value, threshold)?,
            Comparison::Greater => lanes.greater_than(value, threshold)?,
            Comparison::GreaterEqual => lanes.greater_equal(value, threshold)?,
            Comparison::Equal => lanes.equal(value, threshold)?,
            Comparison::NotEqual => lanes.not_equal(value, threshold)?,
        };
        lanes.select(predicate, one, zero)?
    };
    kernel.store(1, held)?;
    kernel.finish()
}

/// Which of the four bitwise operations [`lane_bitwise_with`] applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bitwise {
    And,
    Or,
    Xor,
    /// The complement takes no second operand, so the mask goes unused.
    Not,
}

pub fn lane_bitwise_with(
    subgroup: u32,
    mask: u32,
    operation: Bitwise,
) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_bitwise_with_at, mask, operation)
}

fn lane_bitwise_with_at<const LANES: u32>(
    subgroup: u32,
    mask: u32,
    operation: Bitwise,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let result = {
        let mut lanes = kernel.lanes()?;
        let mask = lanes.splat_bits::<U32, LANES>(mask)?;
        match operation {
            Bitwise::And => lanes.and(value, mask)?,
            Bitwise::Or => lanes.or(value, mask)?,
            Bitwise::Xor => lanes.xor(value, mask)?,
            Bitwise::Not => lanes.not(value)?,
        }
    };
    kernel.store(1, result)?;
    kernel.finish()
}

pub fn lane_affine<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let result = {
        let mut lanes = kernel.lanes()?;
        let two = lanes.splat_bits::<F32, LANES>(2.0_f32.to_bits())?;
        let one = lanes.splat_bits::<F32, LANES>(1.0_f32.to_bits())?;
        let doubled = lanes.mul(value, two)?;
        lanes.add(doubled, one)?
    };
    kernel.store(1, result)?;
    kernel.finish()
}

/// `-((x - 1) / 2)` per element — a subtraction, a division and a negation,
/// none of which crosses a lane.
pub fn lane_arithmetic(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_arithmetic_at)
}

fn lane_arithmetic_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let result = {
        let mut lanes = kernel.lanes()?;
        let one = lanes.splat_bits::<F32, LANES>(1.0_f32.to_bits())?;
        let two = lanes.splat_bits::<F32, LANES>(2.0_f32.to_bits())?;

        let shifted = lanes.sub(value, one)?;
        let halved = lanes.div(shifted, two)?;
        lanes.neg(halved)?
    };
    kernel.store(1, result)?;
    kernel.finish()
}

/// Each of the six comparisons against `THRESHOLD` contributes its own power of
/// two, so the one number a lane writes says which of them held and no two
/// subsets share an answer.
pub const ORDERING_THRESHOLD: f32 = 4.0;

pub fn lane_ordering(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_ordering_at)
}

fn lane_ordering_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let result = {
        let mut lanes = kernel.lanes()?;
        let threshold = lanes.splat_bits::<F32, LANES>(ORDERING_THRESHOLD.to_bits())?;
        let zero = lanes.splat_bits::<F32, LANES>(0.0_f32.to_bits())?;

        let held = [
            (lanes.less_than(value, threshold)?, 1.0_f32),
            (lanes.less_equal(value, threshold)?, 2.0),
            (lanes.greater_than(value, threshold)?, 4.0),
            (lanes.greater_equal(value, threshold)?, 8.0),
            (lanes.equal(value, threshold)?, 16.0),
            (lanes.not_equal(value, threshold)?, 32.0),
        ];

        let mut total = zero;
        for (predicate, weight) in held {
            let bit = lanes.splat_bits::<F32, LANES>(weight.to_bits())?;
            let contribution = lanes.select(predicate, bit, zero)?;
            total = lanes.add(total, contribution)?;
        }
        total
    };
    kernel.store(1, result)?;
    kernel.finish()
}

/// The same division over both integer families, so a signed operand that a
/// `OpUDiv` would read as an enormous positive number is caught.
pub fn lane_divide_signed(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_divide_signed_at)
}

fn lane_divide_signed_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::I32;

    let mut kernel = Kernel::<I32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let result = {
        let mut lanes = kernel.lanes()?;
        let two = lanes.splat_bits::<I32, LANES>(2)?;
        let quotient = lanes.div(value, two)?;
        lanes.neg(quotient)?
    };
    kernel.store(1, result)?;
    kernel.finish()
}

pub fn lane_divide_unsigned(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_divide_unsigned_at)
}

fn lane_divide_unsigned_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let result = {
        let mut lanes = kernel.lanes()?;
        let two = lanes.splat_bits::<U32, LANES>(2)?;
        lanes.div(value, two)?
    };
    kernel.store(1, result)?;
    kernel.finish()
}

/// The mask each bitwise kernel works against, chosen so that no two of the
/// four operations agree on any input.
pub const BITWISE_MASK: u32 = 0b1010;

/// A mask apiece, and a weight apiece. Distinct masks keep `x | a` from being
/// `(x & a) + (x ^ a)`, which would let `and` and `xor` trade places unseen;
/// distinct weights keep any other two from doing the same.
pub const BITWISE_AND_MASK: u32 = 0b1010;
pub const BITWISE_OR_MASK: u32 = 0b0101;
pub const BITWISE_XOR_MASK: u32 = 0b1100;

pub fn lane_bitwise(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_bitwise_at)
}

fn lane_bitwise_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let result = {
        let mut lanes = kernel.lanes()?;

        let and_mask = lanes.splat_bits::<U32, LANES>(BITWISE_AND_MASK)?;
        let or_mask = lanes.splat_bits::<U32, LANES>(BITWISE_OR_MASK)?;
        let xor_mask = lanes.splat_bits::<U32, LANES>(BITWISE_XOR_MASK)?;

        let weighed = [
            (lanes.and(value, and_mask)?, 1_u32),
            (lanes.or(value, or_mask)?, 3),
            (lanes.xor(value, xor_mask)?, 5),
            (lanes.not(value)?, 7),
        ];

        let mut total = lanes.splat_bits::<U32, LANES>(0)?;
        for (term, weight) in weighed {
            let factor = lanes.splat_bits::<U32, LANES>(weight)?;
            let scaled = lanes.mul(term, factor)?;
            total = lanes.add(total, scaled)?;
        }
        total
    };
    kernel.store(1, result)?;
    kernel.finish()
}

/// The complement on a signed type is the one's complement and not a negation,
/// which `-x` and `!x` disagree about by exactly one.
pub fn lane_complement_signed(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_complement_signed_at)
}

fn lane_complement_signed_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::I32;

    let mut kernel = Kernel::<I32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let result = kernel.lanes()?.not(value)?;
    kernel.store(1, result)?;
    kernel.finish()
}

pub fn square(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, square_at)
}

fn square_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let squared = kernel.lanes()?.mul(value, value)?;
    kernel.store(1, squared)?;
    kernel.finish()
}

pub fn lane_affine_whole(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_affine)
}

pub fn empty(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    Kernel::<F32>::new(shape(subgroup))?.finish()
}
