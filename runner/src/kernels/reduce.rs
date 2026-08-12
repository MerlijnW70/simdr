//! Kernels where lanes talk to each other.
//!
//! Two ways of doing the same arithmetic: the subgroup instruction, and the butterfly tree that
//! is what the hardware does underneath. They have to agree exactly, and that agreement is a
//! stronger test than either alone — a wrong mapping would have to be wrong twice, identically.

use super::{shape, whole_subgroup, whole_subgroup_of};
use simdr::kernel::Kernel;
use simdr::lanes::{Element, LaneError};

/// `out[i] = Simd::<T, LANES>::reduce_sum(in[…])`.
///
/// One piece of source for every `LANES` and every `T`: nothing here names a reduction shape, a
/// cluster size, or an opcode. The kernel derives all of them — DR-0002 in four lines.
///
/// # Errors
///
/// [`LaneError`] if the lane count cannot sit on this subgroup, or the module cannot be built.
pub fn lane_sum<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let total = kernel.lanes()?.reduce_sum(value)?;
    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// `out[i] = Simd::<T, LANES>::reduce_max(in[…])`.
///
/// # Errors
///
/// As [`lane_sum`].
pub fn lane_max<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let largest = kernel.lanes()?.reduce_max(value)?;
    kernel.store_scalar(1, largest)?;
    kernel.finish()
}

/// `out[i] = in[i] + in[i ^ mask]` — a butterfly exchange with the lane `mask` away.
///
/// Was written against [`simdr::module`] by hand until shuffles reached the lane API; it is four
/// lines now, and takes the distance as a parameter because it can.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn butterfly_pair_sum_at<const LANES: u32>(
    subgroup: u32,
    mask: u32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let paired = {
        let mut lanes = kernel.lanes()?;
        let neighbour = lanes.butterfly(value, mask)?;
        lanes.add(value, neighbour)?
    };
    kernel.store(1, paired)?;
    kernel.finish()
}

/// `out[i] = in[i] + in[i + half]` — one halving step of a full-buffer reduction.
///
/// The pass that does the bulk of the work in [`crate::reduction`]. No lane talks to any other and
/// there is no branch: the dispatch is sized to exactly `half` invocations, so `i + half` is in
/// range by construction rather than by a bounds test that would diverge.
///
/// `half` is a build-time constant because the offset is, which means one module per step. They
/// are a few hundred words each and building them is cheaper than the dispatch that runs them.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn fold_halves_at<const LANES: u32>(subgroup: u32, half: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let near = kernel.load::<LANES>(0)?;
    let far = kernel.load_offset::<LANES>(0, half)?;
    let folded = kernel.lanes()?.add(near, far)?;
    kernel.store(1, folded)?;
    kernel.finish()
}

/// `out[i] = Σ in[j] × in[j + offset]` over the `LANES` this invocation's subgroup covers.
///
/// A dot product, which is the inner loop of every dense neural-network layer and of a great deal
/// else. Both operands live in binding 0: the first `offset` elements are one vector, the rest are
/// the other, so a caller with two arrays concatenates them and passes the join.
///
/// Elementwise multiply then one reduction — the two things this crate does — so the whole layer
/// is four lines and names no opcode. `LANES` picks how many products each subgroup folds: 32 is
/// one per lane, 128 strip-mines four.
///
/// # Errors
///
/// [`LaneError`] if `LANES` has no mapping onto this subgroup, or the module cannot be built.
pub fn dot_product<T: Element, const LANES: u32>(
    subgroup: u32,
    offset: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let left = kernel.load::<LANES>(0)?;
    let right = kernel.load_offset::<LANES>(0, offset)?;
    let total = {
        let mut lanes = kernel.lanes()?;
        let products = lanes.mul(left, right)?;
        lanes.reduce_sum(products)?
    };
    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// `out[i] = Σ in[…]` over the whole **workgroup**, not just the subgroup.
///
/// The kernel that could not be written before workgroup shared memory arrived. A subgroup
/// reduction stops at 32 lanes; this combines every subgroup of the workgroup and gives all 64
/// invocations the same total.
///
/// ```text
///   total = reduce_sum(value)     each lane holds its own subgroup's total
///   shared[local] = total         every invocation writes a different slot
///   barrier                       reached by all of them, so it is well defined
///   answer = shared[0] + shared[w] + …    constant indices, one per subgroup
/// ```
///
/// The final reads are at build-time constant indices, so every invocation runs the identical
/// instructions and no lane diverges — nothing `decisions/DR-0003` refuses. Every invocation
/// redundantly computes the same last few adds, which is cheaper than any way of avoiding it.
///
/// # Errors
///
/// [`LaneError::BadShape`] if the workgroup is not a whole number of subgroups, otherwise as
/// [`lane_sum`].
fn workgroup_sum_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let workgroup = super::WORKGROUP_SIZE;
    if subgroup == 0 || !workgroup.is_multiple_of(subgroup) {
        return Err(LaneError::BadShape {
            workgroup,
            buffers: subgroup,
        });
    }
    let subgroups = workgroup / subgroup;

    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let mine = kernel.lanes()?.reduce_sum(value)?;

    let shared = kernel.shared(workgroup)?;
    let slot = kernel.local_index();
    kernel.store_shared(shared, slot, mine)?;
    kernel.barrier()?;

    // One slot per subgroup, at 0, w, 2w, … — every one a constant, so this is straight-line code
    // identical in every invocation.
    let mut total = kernel.load_shared(shared, 0)?;
    for index in 1..subgroups {
        let next = kernel.load_shared(shared, index * subgroup)?;
        let element = kernel.element();
        total = kernel.module().binary(T::ADD, element, total, next)?;
    }

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// A subgroup sum built by hand out of butterflies, rather than by `reduce_sum`.
///
/// `log2(width)` butterfly steps, each pairing lanes twice as far apart as the last, so every
/// lane ends holding the total. It is what `OpGroupNonUniformFAdd` does internally — writing it
/// out is the test that [`simdr::lanes::Lanes::repeat`] threads a value correctly, because the
/// answer has to match the built-in reduction exactly.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn butterfly_tree_sum_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let steps = subgroup.trailing_zeros();
    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;

    let total = {
        let mut lanes = kernel.lanes()?;
        lanes.repeat(steps, value.id(), |lanes, carried, step| {
            let held = lanes.from_lane_value::<F32, LANES>(carried)?;
            let partner = lanes.butterfly(held, 1 << step)?;
            Ok(lanes.add(held, partner)?.id())
        })?
    };

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// [`lane_sum`] over a vector as wide as this device's subgroup.
///
/// The distinction the second device made necessary. `lane_sum::<T, 32>` is a *cluster* on a
/// 64-wide subgroup and reduces 32 lanes; this reduces the subgroup, whatever the subgroup is.
/// Tests that mean "the subgroup total" want this one, and tests that mean "a 32-lane vector"
/// want the other — and until there was a device where they differ, one name served for both.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn lane_sum_whole<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, lane_sum)
}

/// [`lane_max`] over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// As [`lane_sum_whole`].
pub fn lane_max_whole<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, lane_max)
}

/// `butterfly_pair_sum_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn butterfly_pair_sum(subgroup: u32, mask: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, butterfly_pair_sum_at, mask)
}

/// `fold_halves_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn fold_halves(subgroup: u32, half: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, fold_halves_at, half)
}

/// `butterfly_tree_sum_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn butterfly_tree_sum(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, butterfly_tree_sum_at)
}

/// `workgroup_sum_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn workgroup_sum<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, workgroup_sum_at)
}

#[cfg(test)]
mod tests {
    use super::{workgroup_sum, workgroup_sum_at};
    use simdr::lanes::{F32, LaneError};

    /// A width the workgroup is not a whole number of.
    ///
    /// The guard existed and nothing reached it: every caller passes a real device's width, so the
    /// refusal was dead code that read as a safeguard. A mutation run replacing the condition with
    /// `false` left the whole suite green.
    ///
    /// Asked of the builder directly rather than through the public wrapper, because the wrapper
    /// now refuses anything that is not 32 or 64 *first* — it has to, since the lane count is a
    /// const generic and those are the two it instantiates. Going through it would test the
    /// dispatcher and leave this guard unreached all over again.
    #[test]
    fn a_subgroup_the_workgroup_does_not_divide_by_is_refused() {
        // 64 invocations do not split into whole subgroups of 24 or 48.
        for width in [24_u32, 48, 63] {
            assert!(
                matches!(
                    workgroup_sum_at::<F32, 32>(width),
                    Err(LaneError::BadShape { .. })
                ),
                "a subgroup of {width} was accepted"
            );
        }
    }

    #[test]
    fn a_subgroup_of_zero_is_refused_before_it_divides() {
        // The other half of the same condition, and the one that would divide by zero if the
        // guard were reordered.
        assert!(matches!(
            workgroup_sum_at::<F32, 32>(0),
            Err(LaneError::BadShape { .. })
        ));
    }

    #[test]
    fn a_width_that_is_no_devices_subgroup_is_refused_by_the_dispatcher() {
        // What the wrapper adds. The lane count has to be instantiated at build time, so only the
        // widths listed in `whole_subgroup!` can be built at all — and a caller passing 24 gets
        // told that rather than getting a kernel for some other width.
        for width in [0_u32, 24, 16, 128] {
            assert!(
                matches!(workgroup_sum::<F32>(width), Err(LaneError::BadWidth { .. })),
                "a subgroup of {width} was accepted"
            );
        }
    }

    #[test]
    fn the_widths_a_device_reports_are_accepted() {
        // Both real subgroup widths divide 64, so the guard must not be refusing those.
        for width in [32_u32, 64] {
            assert!(
                workgroup_sum::<F32>(width).is_ok(),
                "a subgroup of {width} was refused"
            );
        }
    }
}
