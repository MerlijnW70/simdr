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
    dot_product_whole, fold_by, fold_halves, fold_halves_open, lane_max, lane_max_whole, lane_sum,
    lane_sum_whole, workgroup_sum,
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
