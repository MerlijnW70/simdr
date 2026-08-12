//! Kernels whose shape is decided at runtime: votes, branches and loops.
//!
//! Every condition here comes from a vote, so the whole subgroup takes the same edge. That is
//! `decisions/DR-0003`, and it is what makes a reduction inside a branch mean anything.

use super::{shape, whole_subgroup};
use simdr::kernel::Kernel;
use simdr::lanes::LaneError;
use simdr::spec::SelectionControl;

/// `out[i] = 1.0` if any element of the subgroup exceeds `threshold`, else `0.0`.
///
/// A `Mask` reduced to an answer: compare, vote, select. Every lane writes the same value, which
/// is what a vote means.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn any_above_at<const LANES: u32>(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let answer = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let over = lanes.greater_than(value, limit)?;
        let verdict = lanes.any(over)?;

        // The vote is one boolean for the whole subgroup; widen it back to a value every lane can
        // store. `splat` a pair of constants and pick between them.
        let yes = lanes.splat_bits::<F32, LANES>(1.0_f32.to_bits())?;
        let no = lanes.splat_bits::<F32, LANES>(0.0_f32.to_bits())?;
        let element = lanes.type_of::<F32>()?;
        lanes.module().select(element, verdict, yes.id(), no.id())?
    };
    kernel.store_scalar(1, answer)?;
    kernel.finish()
}

/// `out[i] = in[i] * 10` when any element of the subgroup exceeds `threshold`, else `in[i]`.
///
/// A uniform branch with the store *inside* it, opened and closed by hand. Kept in that form
/// deliberately: `store` needs the kernel rather than the lane builder, so this is what the block
/// structure looks like without a helper, and it is worth seeing once.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn scale_if_any_above_at<const LANES: u32>(
    subgroup: u32,
    threshold: f32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;

    // The unconditional part: everyone writes their own value first.
    kernel.store(1, value)?;

    let over = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let above = lanes.greater_than(value, limit)?;
        lanes.any_uniform(above)?
    };

    // And then the subgroups that qualified overwrite it.
    let scaled = {
        let mut lanes = kernel.lanes()?;
        let ten = lanes.splat_bits::<F32, LANES>(10.0_f32.to_bits())?;
        lanes.mul(value, ten)?
    };

    let then_block = kernel.module().alloc_id()?;
    let merge_block = kernel.module().alloc_id()?;
    kernel
        .module()
        .selection_merge(merge_block, SelectionControl::None)?;
    kernel
        .module()
        .branch_conditional(over.id(), then_block, merge_block)?;
    kernel.module().label_at(then_block)?;
    kernel.store(1, scaled)?;
    kernel.module().branch(merge_block)?;
    kernel.module().label_at(merge_block)?;

    kernel.finish()
}

/// The same shape, but with the body built by [`simdr::lanes::Lanes::if_uniform`].
///
/// Computes inside the branch and writes nothing there, so the observable difference is only in
/// the instruction stream — which is the point: it exercises the helper's own block structure.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn branch_only_at<const LANES: u32>(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    kernel.store(1, value)?;

    {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let above = lanes.greater_than(value, limit)?;
        let over = lanes.any_uniform(above)?;

        lanes.if_uniform(over, |lanes| {
            let two = lanes.splat_bits::<F32, LANES>(2.0_f32.to_bits())?;
            lanes.mul(value, two)?;
            Ok(())
        })?;
    }

    kernel.finish()
}

/// `out[i] = sum(in)` when any element of the subgroup exceeds `threshold`, else `max(in)`.
///
/// The kernel that could not be written before [`simdr::lanes::Lanes::choose_uniform`]: both arms
/// end in a subgroup reduction, and exactly one of them runs. A `select` would have computed both
/// and thrown one away; a bare `if_uniform` could not have carried either out.
///
/// It is also the end-to-end test that the `OpPhi` names the right predecessors — get that wrong
/// and the driver reads a value from an edge that never carried it, which no amount of validation
/// catches.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn sum_or_max_at<const LANES: u32>(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let answer = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let above = lanes.greater_than(value, limit)?;
        let over = lanes.any_uniform(above)?;

        lanes.choose_uniform(
            over,
            element,
            |lanes| lanes.reduce_sum(value),
            |lanes| lanes.reduce_max(value),
        )?
    };

    kernel.store_scalar(1, answer)?;
    kernel.finish()
}

/// `out[i] = in[i] + (0 + 1 + … + times - 1)`, accumulated in a rolled loop.
///
/// What the loop counter is for. The body is built once and reads the counter phi, so the sum it
/// produces is only right if that phi is the same value the continue block steps — a body handed
/// a copy, or a fresh zero, would return `in[i]` and look plausible.
///
/// Integers rather than floats so the expected answer is exact by construction rather than by
/// staying under 2^24.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn rolled_counter_sum_at<const LANES: u32>(
    subgroup: u32,
    times: u32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes()?;
        lanes.repeat_rolled(times, element, value.id(), |lanes, carried, iteration| {
            let held = lanes.from_lane_value::<U32, LANES>(carried)?;
            let step = lanes.from_lane_value::<U32, LANES>(iteration)?;
            Ok(lanes.add(held, step)?.id())
        })?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// A branch inside a loop: each trip doubles, or adds one, according to a vote.
///
/// The nesting `repeat_rolled` was never asked for. Its body is built once and now opens blocks of
/// its own, so the loop's own bookkeeping — the copy into the phi's promised name, then the branch
/// to the continue target — happens in the *selection's merge block* rather than in the body block
/// the loop opened. That is correct and it is not obvious, which is why it is run rather than
/// reasoned about.
///
/// The vote is on the loaded value and does not change between trips, so every iteration takes the
/// same arm and the answer is predictable: `in[i] × 2^times` or `in[i] + times`.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn branch_in_loop_at<const LANES: u32>(
    subgroup: u32,
    times: u32,
    threshold: f32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let above = lanes.greater_than(value, limit)?;
        let over = lanes.any_uniform(above)?;
        let one = lanes.splat_bits::<F32, LANES>(1.0_f32.to_bits())?;

        lanes.repeat_rolled(times, element, value.id(), |lanes, carried, _| {
            let held = lanes.from_lane_value::<F32, LANES>(carried)?;
            lanes.choose_uniform(
                over,
                element,
                |lanes| Ok(lanes.add(held, held)?.id()),
                |lanes| Ok(lanes.add(held, one)?.id()),
            )
        })?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// A loop inside a branch: one arm runs a rolled loop, the other returns the value untouched.
///
/// The other nesting, and the one `Module::current_block` exists for. The taken arm finishes in
/// the *loop's* merge block, so the selection's `OpPhi` must name that block and not the one the
/// arm opened. Naming the wrong one is a dominance failure the validator catches — and if it ever
/// did not, the driver would read a value from an edge that never carried it.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn loop_in_branch_at<const LANES: u32>(
    subgroup: u32,
    times: u32,
    threshold: f32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<F32, LANES>(threshold.to_bits())?;
        let above = lanes.greater_than(value, limit)?;
        let over = lanes.any_uniform(above)?;

        lanes.choose_uniform(
            over,
            element,
            |lanes| {
                lanes.repeat_rolled(times, element, value.id(), |lanes, carried, _| {
                    let held = lanes.from_lane_value::<F32, LANES>(carried)?;
                    Ok(lanes.add(held, held)?.id())
                })
            },
            |_| Ok(value.id()),
        )?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// `out[i] = in[i] * 2^times`, as a rolled loop rather than an unrolled one.
///
/// The counterpart to [`super::butterfly_tree_sum`]: same threading question, but through a real
/// four-block loop with two phis instead of a straight line.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn rolled_doubling_at<const LANES: u32>(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let total = {
        let mut lanes = kernel.lanes()?;
        lanes.repeat_rolled(times, element, value.id(), |lanes, carried, _| {
            let held = lanes.from_lane_value::<F32, LANES>(carried)?;
            Ok(lanes.add(held, held)?.id())
        })?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// `any_above_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn any_above(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, any_above_at, threshold)
}

/// `scale_if_any_above_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn scale_if_any_above(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, scale_if_any_above_at, threshold)
}

/// `branch_only_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn branch_only(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, branch_only_at, threshold)
}

/// `sum_or_max_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn sum_or_max(subgroup: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, sum_or_max_at, threshold)
}

/// `rolled_counter_sum_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn rolled_counter_sum(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, rolled_counter_sum_at, times)
}

/// `branch_in_loop_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn branch_in_loop(subgroup: u32, times: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, branch_in_loop_at, times, threshold)
}

/// `loop_in_branch_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn loop_in_branch(subgroup: u32, times: u32, threshold: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, loop_in_branch_at, times, threshold)
}

/// `rolled_doubling_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn rolled_doubling(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, rolled_doubling_at, times)
}
