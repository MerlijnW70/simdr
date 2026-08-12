//! Kernels to run, built with the emitter.
//!
//! Almost all of this used to be sixty lines of buffer binding copied per kernel, and duplicated
//! again in `simdr`'s own tests. [`simdr::kernel::Kernel`] owns that now, so what is left here is
//! the part that differs: what each kernel computes.
//!
//! Every kernel binds two storage buffers — 0 read, 1 written — which is what
//! [`crate::Gpu::run`] expects.
//!
//! Split by what a kernel needs from the machine: the elementwise ones here, the ones that cross
//! lanes in [`reduce`], the ones whose shape is decided at runtime in [`control`], the ones that
//! reach the GLSL.std.450 set in [`extended`], and the ones addressing rows and columns in
//! [`plane`].

pub mod control;
pub mod dot;
pub mod extended;
pub mod narrow;
pub mod network;
pub mod occupancy;
pub mod plane;
pub mod reduce;
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
    FOLD_HALF_SPEC_ID, butterfly_pair_sum, butterfly_tree_sum, dot_product, dot_product_whole,
    fold_halves, fold_halves_open, lane_max, lane_max_whole, lane_sum, lane_sum_whole,
    workgroup_sum,
};
pub use scatter::{claim_slots, histogram, histogram_incrementing};
pub use specialized::{
    specialized_add, specialized_affine, specialized_cluster, specialized_derived,
};
pub use unrun::{all_above, ballot_above, broadcast, lane_min, prefix_sum, shift_down, shift_up};

use simdr::kernel::{Kernel, Shape};
use simdr::lanes::LaneError;

/// How many invocations each kernel here runs per workgroup.
///
/// **Chosen once, and measured much later.** On the three devices this runs on, 64 invocations is
/// eight subgroups, two, or one — so it is not the same quantity on any two of them.
/// `runner/examples/occupancy.rs` sweeps it, and `notes/FINDINGS.md` records what the sweep says:
/// the best size differs by device *and* by kernel shape, so there is no better constant to move
/// this to. [`occupancy`] holds the kernels that take it as an argument instead.
pub const WORKGROUP_SIZE: u32 = 64;

/// Build a kernel whose vector has to be exactly as wide as the subgroup.
///
/// Votes and shuffles have no clustered form — a vote over a 32-lane cluster of a 64-wide
/// subgroup would answer for every vector sharing that subgroup, so the lane API refuses it by
/// name. Which means these kernels cannot say `load::<32>`: on a 64-wide device that is a cluster,
/// and the kernel stops building.
///
/// `LANES` is a const generic and the width arrives at runtime, so the widths have to be listed.
/// A macro rather than the match written out twelve times: what is being repeated is the *list*,
/// and a list that appears twelve times drifts.
///
/// **This is what a second device found.** Every one of these kernels was written against a 32-wide
/// subgroup and read as though it adapted, because the width was passed in — and only the lane
/// count was wrong.
///
/// **And a third device made the list itself the limit.** It held 32 and 64, which is every width
/// real hardware here reports — and lavapipe reports **8**, so every kernel below refused to build
/// with `BadWidth` on a device that was perfectly capable of running them. The list now covers 4,
/// 8, 16, 32 and 64: every power of two a Vulkan implementation is known to report.
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

/// The same, for a builder that is also generic over its element type.
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

/// The shape every kernel here is built to, for a device of `subgroup` lanes.
#[must_use]
pub fn shape(subgroup: u32) -> Shape {
    Shape::new(subgroup, WORKGROUP_SIZE, 2)
}

/// `out[i] = in[i] * factor`.
///
/// The control: no lane talks to any other, so a wrong answer here is a wrong *harness* rather
/// than a wrong subgroup mapping. Run it first.
///
/// **The lane count is the device's width, not 32.** It was 32, which is one element per
/// invocation on a 32- or 64-wide subgroup and *eight* on a four-wide one — so on a narrow device
/// this kernel silently read and wrote eight times the buffer every caller hands it. On lavapipe at
/// four lanes that is an access violation; at eight it was undefined behaviour that happened to
/// return zeros. An elementwise kernel has no reason to strip-mine, so it does not.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
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

/// `out[i] = in[i] * 2 + 1`, elementwise only.
///
/// Two operations where the reduction kernels have one, to show that elementwise work stays one
/// instruction per strip and never touches a subgroup capability.
///
/// `LANES` is open here because the point is to watch what strip-mining does to an elementwise
/// kernel. A caller that just wants one element per invocation wants [`lane_affine_whole`] — see
/// [`scale`] for what a hard-coded 32 does on a four-wide device.
///
/// # Errors
///
/// As [`scale`].
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

/// `out[i] = in[i] × in[i]`.
///
/// The map half of a map-reduce, and the reason [`crate::Gpu::reducer_of`] exists: Σ x² is the
/// squared L2 norm, and computing it the obvious way sends the input to the device, brings the
/// squares home, and sends them straight back to be summed. As one chain it is a single crossing.
///
/// # Errors
///
/// As [`scale`].
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

/// [`lane_affine`] over a vector as wide as the device's subgroup — one element per invocation.
///
/// What every caller of it actually wanted. The generic form was called with a literal 32, which
/// strip-mines on anything narrower and reads past the end of a buffer sized for one element each.
///
/// # Errors
///
/// As [`scale`].
pub fn lane_affine_whole(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, lane_affine)
}

/// An empty kernel, for measuring what a dispatch costs before any work is added.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
pub fn empty(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    Kernel::<F32>::new(shape(subgroup))?.finish()
}
