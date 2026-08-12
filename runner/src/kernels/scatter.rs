//! Kernels whose output slot depends on the data.
//!
//! Everything else in `kernels/` writes to an address derived from the invocation index, so the
//! answer is a `map` and no two invocations collide. These are the other kind: a histogram's bin
//! comes from the value being counted, and two invocations counting the same value must both be
//! counted.
//!
//! # The index is clamped, and that is not a detail
//!
//! An out-of-range index into a storage buffer is undefined behaviour, not an error — so the bin
//! is held inside the buffer with `Lanes::min` before it is used. That costs one instruction per
//! element and is the price of letting the data choose an address.

use super::shape;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{LaneError, U32};

/// Count how many inputs fall in each of `bins` bins, one bin per distinct value.
///
/// `out[in[i] % bins] += 1`, atomically. Written as a clamp of a masked value rather than a
/// modulus because the masking is what keeps the index in range and the clamp is what makes that
/// true even if the mask is wrong.
///
/// Binding 1 must hold at least `bins` elements and start at zero — this adds to whatever is
/// there, which is what an atomic counter does.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
pub fn histogram(subgroup: u32, workgroup: u32, bins: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, workgroup, 2))?;
    let value = kernel.load::<32>(0)?;

    let bin = {
        let mut lanes = kernel.lanes()?;
        // `min(value, bins - 1)` rather than a modulus: it is one instruction, and it makes the
        // index in-range by construction rather than by an argument about the input.
        let ceiling = lanes.splat_bits::<U32, 32>(bins.saturating_sub(1))?;
        lanes.min(value, ceiling)?
    };

    let one = kernel.module().constant_u32(1)?;
    kernel.atomic_add_at(1, bin.id(), one)?;
    kernel.finish()
}

/// The same, counting with `OpAtomicIIncrement` instead of an add of one.
///
/// A different instruction for the same arithmetic, which is why it is worth having both: they
/// must agree, and only one of them takes a value operand.
///
/// # Errors
///
/// As [`histogram`].
pub fn histogram_incrementing(
    subgroup: u32,
    workgroup: u32,
    bins: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, workgroup, 2))?;
    let value = kernel.load::<32>(0)?;

    let bin = {
        let mut lanes = kernel.lanes()?;
        let ceiling = lanes.splat_bits::<U32, 32>(bins.saturating_sub(1))?;
        lanes.min(value, ceiling)?
    };

    kernel.atomic_increment_at(1, bin.id())?;
    kernel.finish()
}

/// Every invocation claims a consecutive slot from one counter, and writes its own index there.
///
/// The allocator shape, and the one that shows what an atomic *returns*: `OpAtomicIAdd` yields the
/// value the slot held before, so no two invocations get the same answer and the answers cover
/// `0..n` exactly. The output is therefore a permutation of the invocation indices — which is a
/// stronger statement than a histogram's, because a lost or duplicated claim shows up as a
/// repeated or missing entry rather than as an off-by-one in a total.
///
/// Slot 0 of binding 1 is the counter; the claims are written from slot 1 onwards.
///
/// # Errors
///
/// As [`histogram`].
pub fn claim_slots(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;

    // Unsigned, and it matters less than it looks: `OpIAdd` is sign-agnostic and these values are
    // small, so a signed type here would compute the same answer. What it would *not* do is reuse
    // the `u32` the kernel already interned — it would declare a second 32-bit integer type, and
    // the module would carry two types that mean the same thing with an index of one and a
    // constant of the other. The test below is what says so.
    let uint = kernel.module().type_int(32, false)?;
    let counter = kernel.module().constant_u32(0)?;
    let one = kernel.module().constant_u32(1)?;

    // The slot this invocation was given, which is different in every one of them.
    let claimed = kernel.atomic_add_at(1, counter, one)?;

    // Written one further along, so the counter itself is not overwritten by a claim.
    let slot = kernel.module().i_add(uint, claimed, one)?;
    let local = kernel.local_index();
    let pointer = kernel.element_pointer_to(1, slot)?;
    kernel.module().store(pointer, local)?;

    kernel.finish()
}

#[cfg(test)]
mod tests {
    use super::claim_slots;
    use simdr::decode;
    use simdr::module::op;

    /// The allocator's index arithmetic reuses the kernel's own `u32`.
    ///
    /// A mutation run flipped the signedness of that declaration and the whole suite stayed green,
    /// which is correct as far as the *answer* goes: `OpIAdd` does not care, and neither does an
    /// access chain's index. What changes is the module — a second `OpTypeInt 32` appears, and the
    /// kernel is then carrying two types that mean the same thing.
    ///
    /// One 32-bit integer type is the invariant worth stating, and it is not one any dispatch
    /// could have told us.
    #[test]
    fn the_allocator_declares_one_32_bit_integer_type_and_not_two() {
        let words = claim_slots(32).expect("built");

        let integers: Vec<Vec<u32>> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::TYPE_INT)
            .map(|instruction| instruction.operands().to_vec())
            .collect();

        assert_eq!(
            integers.len(),
            1,
            "two integer types of the same width: {integers:?}"
        );
        // Result id, width, signedness.
        assert_eq!(
            integers.first().and_then(|operands| operands.get(1)),
            Some(&32)
        );
        assert_eq!(
            integers.first().and_then(|operands| operands.get(2)),
            Some(&0),
            "unsigned, which is what the kernel's own index arithmetic uses"
        );
    }
}
