//! The lane operations that had unit tests and had never run.
//!
//! A coverage sweep on 2026-08-11 found five of them — `prefix_sum`, `ballot`, `shift_down`,
//! `broadcast`, `all_uniform` — with unit tests only, plus `reduce_min` with a fuzz case and no
//! kernel. That is the weakest evidence this project has: a unit test decodes the module and
//! agrees that the emitter emitted what the test expected, which says nothing whatever about what
//! the hardware then does with it.
//!
//! It is also not hypothetical. `reduce_min` folded its strips with a maximum for weeks under
//! seven passing unit tests, and only a differential run against a CPU reference noticed.
//!
//! Everything here exists to be executed and compared.
//!
//! # It stopped being only the lane API, and the reason is the same one
//!
//! A second sweep on 2026-08-25 asked the question of the whole tree rather than of `src/lanes/`,
//! and found the shape one layer down and one layer up. `f_sub`, `f_div`, `f_negate`, `i_sub` and
//! `u_div` — added a week earlier for an activation and for the arithmetic that says which of a
//! batch a lane is working on — had **one consumer between them**, a test in the emitter that
//! hands one module to `spirv-val`. `Kernel::repeat_rolled` and `repeat_rolled_many` had their own
//! unit tests and `tests/control_flow.rs`, and nothing that ran.
//!
//! Both passed `tests/integrity.rs`'s check that every public operation is named outside its own
//! file, because a test is a consumer. That is the check doing what it says; it is also exactly the
//! gap this module is named after, so the answer was to widen the module rather than the check.
//!
//! The validator is a weaker witness here than anywhere else in this tree, and `decisions/DR-0001`
//! says why: an opcode number read wrong assembles into a *different well-formed instruction*.
//! `spirv-val` accepts it. Only an answer compared against a host reference does not.

use super::{shape, whole_subgroup, whole_subgroup_of};
use simdr::kernel::Kernel;
use simdr::lanes::{Element, LaneError};

/// `out[i] = Σ in[first..=i]` within the subgroup — an *inclusive* scan.
///
/// Inclusive is the whole question. An exclusive scan is the same instruction with a different
/// `GroupOperation`, the two differ by exactly one element, and every unit test that counted
/// opcodes would pass for either. Only running it says which one came out.
///
/// Whole-subgroup vectors only, which is this kernel's choice rather than the lane API's limit:
/// the other two mappings scan as well and are the business of `kernels::scan`, which runs them
/// against a CPU reference. Here `LANES` equals the width so that what is under test is the one
/// instruction.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn prefix_sum_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let running = kernel.lanes()?.prefix_sum(value)?;
    kernel.store(1, running)?;
    kernel.finish()
}

/// `out[i] = in[source]` — one lane's value, delivered to all of them.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn broadcast_at<T: Element, const LANES: u32>(
    subgroup: u32,
    source: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let shared = kernel.lanes()?.broadcast(value, source)?;
    kernel.store(1, shared)?;
    kernel.finish()
}

/// `out[i] = in[i + delta]` within the subgroup, where that lane exists.
///
/// The lanes it does not exist for are **undefined** by the specification, so the test that runs
/// this checks only the lanes that have a source and reports what the rest happened to hold rather
/// than asserting it. Observing an undefined value is fine; pinning one is how a test starts
/// depending on a driver.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn shift_down_at<T: Element, const LANES: u32>(
    subgroup: u32,
    delta: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let shifted = kernel.lanes()?.shift_down(value, delta)?;
    kernel.store(1, shifted)?;
    kernel.finish()
}

/// The same upward: `out[i] = in[i - delta]`.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn shift_up_at<T: Element, const LANES: u32>(
    subgroup: u32,
    delta: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let shifted = kernel.lanes()?.shift_up(value, delta)?;
    kernel.store(1, shifted)?;
    kernel.finish()
}

/// `out[i] = Simd::<T, LANES>::reduce_min(in[…])`.
///
/// The operation whose strip fold was wrong, and which had no kernel at all until it was.
///
/// # Errors
///
/// [`LaneError`] if the lane count has no mapping onto this subgroup.
pub fn lane_min<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let smallest = kernel.lanes()?.reduce_min(value)?;
    kernel.store_scalar(1, smallest)?;
    kernel.finish()
}

/// `out[i] = 1` where *every* element of the subgroup exceeds `threshold`, else `0`.
///
/// The other vote. `any` has been run since the first week; `all` reaches a different instruction
/// and had never been dispatched.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn all_above_at<const LANES: u32>(subgroup: u32, threshold: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let answer = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<U32, LANES>(threshold)?;
        let above = lanes.greater_than(value, limit)?;
        let verdict = lanes.all(above)?;

        let yes = lanes.splat_bits::<U32, LANES>(1)?;
        let no = lanes.splat_bits::<U32, LANES>(0)?;
        let element = lanes.type_of::<U32>()?;
        lanes.module().select(element, verdict, yes.id(), no.id())?
    };
    kernel.store_scalar(1, answer)?;
    kernel.finish()
}

/// `out[i] = 1` when every lane of the subgroup holds the same input value, else `0`.
///
/// The third vote, and the one that asks about a **value**. `all_above` compares against a
/// constant every lane already knows; this compares the lanes against each other, which no
/// predicate can express — the value a lane would compare against is the one it is trying to
/// learn.
///
/// A subgroup that agrees is the case a kernel takes a fast path for, and `decisions/DR-0003` will
/// only branch on a `Uniform` — so the answer goes through `Lanes::all_equal_uniform` and drives a
/// real branch here rather than a select. That makes the module say what the operation is for, and
/// it is why the answer is written inside the branch: a kernel that took both paths would prove
/// the vote's value and not its uniformity.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn subgroup_agrees_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;

    let zero = kernel.module().constant_u32(0)?;
    let one = kernel.module().constant_u32(1)?;
    let slot = kernel.local_index();
    let out = kernel.element_pointer_to(1, slot)?;

    // Zero first, so the slot a divergent subgroup never reaches says so rather than holding
    // whatever the buffer came with.
    kernel.module().store(out, zero)?;

    let agreed = kernel.lanes()?.all_equal_uniform(value)?;
    kernel
        .lanes()?
        .if_uniform(agreed, |lanes| Ok(lanes.module().store(out, one)?))?;

    kernel.finish()
}

/// The same vote over a vector **wider** than the subgroup — the strip-mined form.
///
/// `LANES` elements per lane, and two questions rather than one: every lane holds the same strip 0,
/// *and* in every lane the other strips equal strip 0. The second is what an elementwise equality
/// is for, and without it this mapping was refused by name.
///
/// **The input that separates the two implementations** is one where every strip is internally
/// uniform and the strips differ from each other — all lanes hold 1 in strip 0 and 2 in strip 1.
/// A folded vote says `true` for that, and it is false.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
pub fn subgroup_agrees_wide<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;

    let zero = kernel.module().constant_u32(0)?;
    let one = kernel.module().constant_u32(1)?;
    let slot = kernel.local_index();
    let out = kernel.element_pointer_to(1, slot)?;
    kernel.module().store(out, zero)?;

    let agreed = kernel.lanes()?.all_equal_uniform(value)?;
    kernel
        .lanes()?
        .if_uniform(agreed, |lanes| Ok(lanes.module().store(out, one)?))?;

    kernel.finish()
}

/// `out[i] = in[i - delta]` within the vector, **wrapping** — the rotate.
///
/// The operation a cluster's edge was waiting for. `shift_up` leaves the bottom `delta` lanes
/// undefined and refuses a clustered vector outright; a rotate has no edge, so it is defined
/// everywhere and allowed everywhere a vector is one strip.
///
/// `cluster` picks the vector's width, so this runs the whole-subgroup form and the clustered one
/// from the same source — which is the point: they are one instruction sequence with a different
/// `size` in the masks.
///
/// # Errors
///
/// [`LaneError::NoMapping`] if `cluster` is not a power of two that divides the subgroup, otherwise
/// if the module cannot be built.
pub fn rotate_in_cluster(subgroup: u32, cluster: u32, delta: u32) -> Result<Vec<u32>, LaneError> {
    fn build<const LANES: u32>(subgroup: u32, delta: u32) -> Result<Vec<u32>, LaneError> {
        use simdr::lanes::U32;

        let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
        let value = kernel.load::<LANES>(0)?;
        let rotated = kernel.lanes()?.rotate_up(value, delta)?;
        kernel.store(1, rotated)?;
        kernel.finish()
    }

    match cluster {
        2 => build::<2>(subgroup, delta),
        4 => build::<4>(subgroup, delta),
        8 => build::<8>(subgroup, delta),
        16 => build::<16>(subgroup, delta),
        32 => build::<32>(subgroup, delta),
        64 => build::<64>(subgroup, delta),
        lanes => Err(LaneError::NoMapping {
            lanes,
            width: subgroup,
        }),
    }
}

/// `out[i] = 1` where `in[i]` equals `wanted`, else `0` — the elementwise comparison.
///
/// `Lanes::equal` is the comparison a `Simd` layer is asked for first and this library had none.
/// Run rather than counted, because an opcode-counting test passes just as well for
/// `OpINotEqual`.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn equals_at<const LANES: u32>(subgroup: u32, wanted: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let flag = {
        let mut lanes = kernel.lanes()?;
        let target = lanes.splat_bits::<U32, LANES>(wanted)?;
        let same = lanes.equal(value, target)?;
        let yes = lanes.splat_bits::<U32, LANES>(1)?;
        let no = lanes.splat_bits::<U32, LANES>(0)?;
        lanes.select(same, yes, no)?
    };
    kernel.store(1, flag)?;
    kernel.finish()
}

/// `equals_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is not one the dispatcher lists.
pub fn equals(subgroup: u32, wanted: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, equals_at, wanted)
}

/// `subgroup_agrees_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is not one the dispatcher lists.
pub fn subgroup_agrees(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, subgroup_agrees_at)
}

/// `out[i] = ` the low 32 bits of the ballot of `in[i] > threshold`.
///
/// One bit per lane, set where that lane's predicate held. Every lane of the subgroup receives the
/// same mask, which is what makes it checkable: the host knows which lanes qualify and can build
/// the same word.
///
/// A ballot is a `uvec4` in SPIR-V and only the first component is read here — a 32-wide subgroup
/// fits in it, and a 64-wide one would need two.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn ballot_above_at<const LANES: u32>(subgroup: u32, threshold: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let low = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.splat_bits::<U32, LANES>(threshold)?;
        let above = lanes.greater_than(value, limit)?;
        let mask = lanes.ballot(above)?;

        let uint = lanes.type_of::<U32>()?;
        lanes.module().composite_extract(uint, mask, &[0])?
    };
    kernel.store_scalar(1, low)?;
    kernel.finish()
}

/// `all_above_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn all_above(subgroup: u32, threshold: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, all_above_at, threshold)
}

/// `ballot_above_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn ballot_above(subgroup: u32, threshold: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, ballot_above_at, threshold)
}

/// `prefix_sum_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn prefix_sum<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, prefix_sum_at)
}

/// `broadcast_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn broadcast<T: Element>(subgroup: u32, source: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, broadcast_at, source)
}

/// The same, over a vector *narrower* than the subgroup — one broadcast per cluster.
///
/// **The lane read differs per invocation, which is what makes this worth running.** `source` is a
/// position in the vector, so on a 32-wide device a `Simd<f32, 8>` broadcasting lane 3 has four
/// vectors each reading their own lane 3 — subgroup lanes 3, 11, 19 and 27. A version that took
/// `source` as a subgroup lane would put one value in all thirty-two, and would agree with this
/// one for the first cluster.
///
/// # Errors
///
/// [`LaneError::NoMapping`] if `cluster` is not a power of two that divides the subgroup,
/// [`LaneError::LaneOutOfRange`] if `source` is outside the vector, otherwise if the module cannot
/// be built.
///
/// **It said `NoSuchForm` until somebody called it with a `source` outside the vector.**
/// `Lanes::broadcast` holds every mapping to `Lanes::within_group`, whose refusal names the
/// operand and the width it exceeded; `NoSuchForm` is what a *shape* with no lowering gets, and
/// this operation has one at every mapping. Nothing could have caught the difference: matching
/// the variant against this body finds neither, because the body delegates and the refusal is
/// two layers down. Counting the sites that *produce* each variant and reading them against the
/// docs that claim it is what found it, and this was the only one of seven that had no producer.
pub fn broadcast_in_cluster<T: Element>(
    subgroup: u32,
    cluster: u32,
    source: u32,
) -> Result<Vec<u32>, LaneError> {
    match cluster {
        1 => broadcast_at::<T, 1>(subgroup, source),
        2 => broadcast_at::<T, 2>(subgroup, source),
        4 => broadcast_at::<T, 4>(subgroup, source),
        8 => broadcast_at::<T, 8>(subgroup, source),
        16 => broadcast_at::<T, 16>(subgroup, source),
        32 => broadcast_at::<T, 32>(subgroup, source),
        lanes => Err(LaneError::NoMapping {
            lanes,
            width: subgroup,
        }),
    }
}

/// `shift_down_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn shift_down<T: Element>(subgroup: u32, delta: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, shift_down_at, delta)
}

/// `shift_up_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is neither 32 nor 64.
pub fn shift_up<T: Element>(subgroup: u32, delta: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, shift_up_at, delta)
}

/// `out[i] = -((in[i] - centre) / scale)` — the three instructions an activation needs.
///
/// `f_sub`, `f_div` and `f_negate` arrived together on 2026-08-18 because a layer that centres its
/// input, scales it and flips the sign needs all three, and for a week afterwards the only thing
/// that named any of them was `tests/instructions.rs`. That is one module handed to `spirv-val`:
/// evidence that the words are legal, and none at all that `OpFNegate` negates. A wrong opcode
/// number here assembles into a *different well-formed instruction* — which is the failure
/// `decisions/DR-0001` is about, and the validator cannot see it.
///
/// **Exact by construction rather than by tolerance.** `centre` is an integer and `scale` a power
/// of two, so every step is representable: the subtraction of small integers is exact, a division
/// by a power of two moves the exponent and nothing else, and a negation flips one bit. The host
/// computes the same expression and the two are compared for equality — no epsilon, which is the
/// bar `notes/NEXT.md` sets when it refuses to fuzz `sqrt` and `exp` for want of one.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn centre_and_scale_at<const LANES: u32>(
    subgroup: u32,
    centre: f32,
    scale: f32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::F32;

    let mut kernel = Kernel::<F32>::new(shape(subgroup))?;
    let element = kernel.element();
    let value = kernel.load::<LANES>(0)?.id();

    let middle = F32::constant_from_bits(kernel.module(), centre.to_bits())?;
    let divisor = F32::constant_from_bits(kernel.module(), scale.to_bits())?;

    let centred = kernel.module().f_sub(element, value, middle)?;
    let scaled = kernel.module().f_div(element, centred, divisor)?;
    let flipped = kernel.module().f_negate(element, scaled)?;

    kernel.store_scalar(1, flipped)?;
    kernel.finish()
}

/// `out[i] = in[i] % divisor`, spelled as the divide, multiply and subtract it actually is.
///
/// SPIR-V has `OpUMod`, and this crate does not emit it. `u_div` and `i_sub` were added for the
/// arithmetic that says *which of a batch* a lane is working on, and a remainder written out of
/// them is the shape that arithmetic takes: `x - (x / d) * d`.
///
/// **`divisor` is deliberately not a power of two.** Seven makes `OpUDiv` a real division rather
/// than a shift the driver folds, so what runs is the instruction under test.
///
/// The identity holds exactly in integers for every input, which is what makes the host reference
/// a `%` rather than an approximation of one.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn remainder_at<const LANES: u32>(subgroup: u32, divisor: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let element = kernel.element();
    let value = kernel.load::<LANES>(0)?.id();

    let by = kernel.module().constant_u32(divisor)?;
    let whole = kernel.module().u_div(element, value, by)?;
    let back = kernel.module().i_mul(element, whole, by)?;
    let left = kernel.module().i_sub(element, value, back)?;

    kernel.store_scalar(1, left)?;
    kernel.finish()
}

/// `out[i] = Σ over `times` blocks of `in[block * 64 + i]` — a rolled loop that **reaches a buffer**.
///
/// The kernel `decisions/DR-0010` was written for, and the one that could not be built before it.
/// `Lanes::repeat_rolled` hands its body a `Lanes`, which holds a module and a width and no
/// bindings — so a rolled body could compute and could not *fetch*, and every kernel here unrolled
/// its strips instead. `Kernel::repeat_rolled` hands the body the kernel, and this is the shape
/// that difference exists for: one body, `times` trips, a different block of the buffer each time.
///
/// **What makes it a test rather than a demonstration.** The body is built once, so the offset it
/// loads from is `counter * 64` where `counter` is the loop's own phi. A phi naming the wrong
/// predecessor satisfies `spirv-val` and then reads from an edge that never carried it — so a loop
/// that re-read block zero every trip, or stepped its counter in the header rather than the
/// continue block, emits a valid module and returns `times * block[0]`. Only running it says which
/// happened.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn rolled_block_sum_at<const LANES: u32>(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let element = kernel.element();
    let uint = kernel.index_type();
    let span = kernel.module().constant_u32(super::WORKGROUP_SIZE)?;
    let nought = kernel.module().constant_u32(0)?;

    let total = kernel.repeat_rolled(times, element, nought, |kernel, carried, counter| {
        let offset = kernel.module().i_mul(uint, counter, span)?;
        let block = kernel.load_offset_by::<LANES>(0, offset)?.id();
        Ok(kernel.module().i_add(element, carried, block)?)
    })?;

    kernel.store_scalar(1, total)?;
    kernel.finish()
}

/// `out[i] = Σ block * in[block * 64 + i]`, from two running totals in **one** pass over the input.
///
/// [`Kernel::repeat_rolled_many`]'s own documentation names this workload: *"a weighted sum over
/// sixteen vectors reads its input sixteen times, which is a bandwidth problem rather than an
/// arithmetic one"*. Carrying one value forces one loop per total; carrying two reads each block
/// once and updates both.
///
/// The loop keeps a plain sum and a sum weighted by `block + 1`, and the answer stored is the
/// difference — which reduces to a sum weighted by `block`, and is why the reference is one line.
///
/// **Both phis are load-bearing, and that is the point of subtracting them.** Storing the weighted
/// total alone would pass with the plain one wired to anything at all; a wrong value on either back
/// edge moves the difference. The emitter's own test for this shape asserts *two* `OpPhi`
/// instructions and cannot say what either of them carries.
///
/// # Errors
///
/// [`LaneError::BadCarry`] if the body is handed a number of values it was not built for, which
/// cannot happen here and is stated rather than unwrapped. [`LaneError`] otherwise.
fn rolled_weighted_totals_at<const LANES: u32>(
    subgroup: u32,
    times: u32,
) -> Result<Vec<u32>, LaneError> {
    use simdr::lanes::U32;

    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let element = kernel.element();
    let uint = kernel.index_type();
    let span = kernel.module().constant_u32(super::WORKGROUP_SIZE)?;
    let one = kernel.module().constant_u32(1)?;
    let nought = kernel.module().constant_u32(0)?;

    let totals = kernel.repeat_rolled_many(
        times,
        element,
        &[nought, nought],
        |kernel, carried, counter| {
            let [plain, weighted] = *carried else {
                return Err(LaneError::BadCarry {
                    given: carried.len(),
                    wanted: 2,
                });
            };

            let offset = kernel.module().i_mul(uint, counter, span)?;
            let block = kernel.load_offset_by::<LANES>(0, offset)?.id();

            // `counter + 1` rather than `counter`, so the first trip contributes to the weighted
            // total as well. A weight of zero on trip zero would hide a body that never ran.
            let weight = kernel.module().i_add(uint, counter, one)?;
            let scaled = kernel.module().i_mul(element, block, weight)?;

            Ok(vec![
                kernel.module().i_add(element, plain, block)?,
                kernel.module().i_add(element, weighted, scaled)?,
            ])
        },
    )?;

    let [plain, weighted] = totals[..] else {
        return Err(LaneError::BadCarry {
            given: totals.len(),
            wanted: 2,
        });
    };
    let answer = kernel.module().i_sub(element, weighted, plain)?;

    kernel.store_scalar(1, answer)?;
    kernel.finish()
}

/// `centre_and_scale_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is not one this crate knows.
pub fn centre_and_scale(subgroup: u32, centre: f32, scale: f32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, centre_and_scale_at, centre, scale)
}

/// `remainder_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is not one this crate knows.
pub fn remainder(subgroup: u32, divisor: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, remainder_at, divisor)
}

/// `rolled_block_sum_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is not one this crate knows.
pub fn rolled_block_sum(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, rolled_block_sum_at, times)
}

/// `rolled_weighted_totals_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built, or the width is not one this crate knows.
pub fn rolled_weighted_totals(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, rolled_weighted_totals_at, times)
}
