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
    fold_by_at::<LANES>(subgroup, 2, half)
}

/// `out[i] = Σ in[i + k × stride]` for `k` in `0..factor` — a fold by more than two.
///
/// A chain of these is a quarter as long as a chain of halvings: `log₁₆` passes instead of `log₂`,
/// or five dispatches instead of fifteen over 2²⁰ elements. `super::super::reduction::MAX_FOLD`
/// records what that is worth once measured, which is **less than the arithmetic suggests** —
/// about 8% at 2²⁰ and nothing at 8 192.
///
/// `stride` is three numbers at once and they are the same number: the distance between the
/// elements one invocation adds, how many invocations there are, and how many elements the pass
/// leaves behind.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
pub fn fold_by(subgroup: u32, factor: u32, stride: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, fold_by_at, factor, stride)
}

fn fold_by_at<const LANES: u32>(
    subgroup: u32,
    factor: u32,
    stride: u32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;

    // The first outside the fold, so there is no "nothing yet" case to default. A factor of one is
    // a copy, which is what it should be rather than an error: the plan never asks for one, and an
    // arm nothing can reach is an equivalent mutant waiting to be reported.
    let mut total = kernel.load::<LANES>(0)?;
    for step in 1..factor {
        let next = kernel.load_offset::<LANES>(0, step.saturating_mul(stride))?;
        total = kernel.lanes()?.add(total, next)?;
    }

    kernel.store(1, total)?;
    kernel.finish()
}

/// The same fold, with the offset left open until the pipeline is created.
///
/// One module for every step of a reduction instead of one per step. Whether that is worth
/// anything is a measurement rather than an argument — `runner/examples/specialize.rs` makes it,
/// and `decisions/DR-0005` records what it said.
///
/// The cost inside the kernel is one `OpIAdd` per strip: a constant offset folds into the address
/// arithmetic for free and a value cannot.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
/// The lane count comes from the width, exactly as [`fold_halves`] does. It was a literal 32 —
/// one element per invocation at 32 and 64 lanes, and eight strips at four, which reads eight
/// times the buffer the caller sized. The two are a *pair*, and only one of them had been fixed.
pub fn fold_halves_open(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, fold_halves_open_at)
}

fn fold_halves_open_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    // The kernel's own index type, not one built here: the offset is added to an address, so it
    // has to be whatever the addresses are. This used to be `type_int(32, false)` and the `false`
    // was not load-bearing — `OpIAdd` is sign-agnostic — which is a decision written down twice
    // where only one copy could ever matter.
    let element = kernel.index_type();
    let half = kernel
        .module()
        .spec_constant(element, 0, FOLD_HALF_SPEC_ID)?;

    let near = kernel.load::<LANES>(0)?;
    let far = kernel.load_offset_by::<LANES>(0, half)?;
    let folded = kernel.lanes()?.add(near, far)?;
    kernel.store(1, folded)?;
    kernel.finish()
}

/// The `SpecId` [`fold_halves_open`] leaves its offset under.
pub const FOLD_HALF_SPEC_ID: u32 = 0;

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

/// The same tree, inside each cluster of a vector narrower than the subgroup.
///
/// **The kernel the butterfly's refusal used to make unwritable.** A `Simd<f32, 8>` on a 32-wide
/// device is four vectors sharing the subgroup, and `Lanes::butterfly` refused every one of them —
/// so the mapping that exists to run several small vectors at once could be reduced by the
/// hardware and not swizzled at all. A mask below the cluster's width cannot leave it, which is
/// arithmetic rather than a special case, and this is what it buys: `log2(cluster)` steps, four
/// independent trees, one instruction each.
///
/// It is also the stronger of the two comparisons the file's header describes. `lane_sum` over the
/// same vector emits a single `ClusteredReduce`; this emits a tree of shuffles and adds. Nothing is
/// shared between them but the answer.
///
/// # Errors
///
/// [`LaneError::NoMapping`] if `cluster` is not a power of two that divides the subgroup, otherwise
/// if the module cannot be built.
pub fn butterfly_cluster_sum(subgroup: u32, cluster: u32) -> Result<Vec<u32>, LaneError> {
    match cluster {
        1 => butterfly_cluster_sum_at::<1>(subgroup),
        2 => butterfly_cluster_sum_at::<2>(subgroup),
        4 => butterfly_cluster_sum_at::<4>(subgroup),
        8 => butterfly_cluster_sum_at::<8>(subgroup),
        16 => butterfly_cluster_sum_at::<16>(subgroup),
        32 => butterfly_cluster_sum_at::<32>(subgroup),
        lanes => Err(LaneError::NoMapping {
            lanes,
            width: subgroup,
        }),
    }
}

/// The builder for [`butterfly_cluster_sum`], at a known cluster width.
///
/// The step count is the *vector's* `log2` and not the subgroup's — which is the whole difference
/// from [`butterfly_tree_sum_at`], and the mistake that would look like a working kernel returning
/// the subgroup's total in every lane.
fn butterfly_cluster_sum_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    if LANES > subgroup {
        return Err(LaneError::NoMapping {
            lanes: LANES,
            width: subgroup,
        });
    }

    let steps = LANES.trailing_zeros();
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

/// [`dot_product`] over a vector as wide as this device's subgroup.
///
/// The `offset` is the join between the two operands in binding 0, and is a property of the
/// caller's buffer rather than of the mapping — so it stays a parameter while the lane count
/// follows the device.
///
/// # Errors
///
/// As [`lane_sum_whole`].
pub fn dot_product_whole<T: Element>(subgroup: u32, offset: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, dot_product, offset)
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
    use super::{butterfly_cluster_sum, workgroup_sum, workgroup_sum_at};
    use simdr::lanes::{F32, LaneError};

    /// A cluster exactly the subgroup's width, and one wider than it.
    ///
    /// **The third time this file has met the same shape**, and the note above records the first:
    /// a guard nothing reaches reads exactly like one that works. Here the mutation gate found two
    /// at once — `LANES > subgroup` weakened to `>=`, which refuses a cluster that is simply a
    /// whole-subgroup vector, and the condition removed altogether, which lets a wider one through.
    ///
    /// Neither consumer could see either. `runner/tests/validated.rs` sweeps clusters 2, 4 and 8
    /// across every width and *reports* a build refusal rather than failing on it — by design, since
    /// a cluster wider than the subgroup is a real refusal there — so `>=` reads as that. And
    /// `runner/tests/loops.rs` skips the equal case by name before it builds anything.
    ///
    /// The wider case is the sharper half: without this guard the call is still refused, but by the
    /// *butterfly's* own bound, as a mask reaching outside its subgroup. That is a true statement
    /// about a different thing, and every consumer here would print it as "not built" either way.
    #[test]
    fn a_cluster_the_subgroups_width_builds_and_a_wider_one_is_refused_by_name() {
        assert!(
            butterfly_cluster_sum(32, 32).is_ok(),
            "a cluster exactly the subgroup's width is a whole-subgroup vector, not a missing \
             mapping"
        );

        assert_eq!(
            butterfly_cluster_sum(16, 32).err(),
            Some(LaneError::NoMapping {
                lanes: 32,
                width: 16
            }),
            "a cluster wider than the subgroup is refused here and names both numbers"
        );
    }

    /// A width the workgroup is not a whole number of.
    ///
    /// The guard existed and nothing reached it: every caller passes a real device's width, so the
    /// refusal was dead code that read as a safeguard. A mutation run replacing the condition with
    /// `false` left the whole suite green.
    ///
    /// Asked of the builder directly rather than through the public wrapper, because the wrapper
    /// refuses any width it has no lane count for *first* — it has to, since the lane count is a
    /// const generic and it can only instantiate the widths `whole_subgroup_of!` lists, which are
    /// 4, 8, 16, 32 and 64. Going through it would test the dispatcher and leave this guard
    /// unreached all over again.
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
        // Not 16: that was here until lavapipe reported 8 and the dispatcher had to learn every
        // power of two a Vulkan implementation is known to report. What is left is the widths no
        // implementation reports at all.
        for width in [0_u32, 24, 48, 128] {
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
