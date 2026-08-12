//! Kernels using the packed integer dot product.
//!
//! Each of these has a twin that computes the same answer out of ordinary instructions, and the
//! pair is the point: `OpSDot` is one instruction where the twin is four shifts, four
//! sign-extensions, four multiplies and three adds, and they have to agree exactly.
//!
//! The buffer holds packed words either way. A `Simd<u32, N>` is one `u32` per lane as it always
//! is — `decisions/DR-0004` is not bent here — and what differs between the two kernels is only
//! how the *instruction* reads those bits.

use super::{shape, whole_subgroup};
use simdr::kernel::Kernel;
use simdr::lanes::{I32, LaneError, U32, Vector};

/// `out[i] = Σ signed_bytes(in[i]) × signed_bytes(in[i])`, in one instruction per lane.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn packed_dot_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let packed = kernel.load::<LANES>(0)?;
    let totals = {
        let mut lanes = kernel.lanes()?;
        let products = lanes.dot_signed(packed, packed)?;
        lanes.reinterpret(products)?
    };
    kernel.store(1, totals)?;
    kernel.finish()
}

/// The same answer, built out of shifts and multiplies instead.
///
/// The twin. Every byte is extracted, sign-extended by shifting up and back down, squared and
/// added — which is what `OpSDot` replaces, and what it has to agree with.
///
/// # Errors
///
/// As `packed_dot_at`.
fn unpacked_dot_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let packed = kernel.load::<LANES>(0)?;

    let totals = {
        let mut lanes = kernel.lanes()?;

        // The first component outside the fold, so there is no "no components yet" case. An
        // `Option` here would have been an arm nothing could reach — and an unreachable arm is an
        // equivalent mutant waiting to be reported, which `notes/FINDINGS.md` has four of already.
        let mut total = square_of_byte::<LANES>(&mut lanes, packed, 0)?;
        for byte in 1_u32..4 {
            let squared = square_of_byte::<LANES>(&mut lanes, packed, byte)?;
            total = lanes.add(total, squared)?;
        }

        lanes.reinterpret(total)?
    };

    kernel.store(1, totals)?;
    kernel.finish()
}

/// One packed byte, sign-extended into a whole `i32`.
///
/// Shift it to the top of the word and arithmetic-shift it back: the 24 bits above become copies
/// of its sign. A mask would extract the same bits and read every negative byte as a number
/// between 128 and 255 — which is the mistake `OpSDot` exists partly to make unwriteable.
///
/// **The byte index is the part worth testing.** `24 - byte × 8` is the only place the four
/// positions are told apart, and every kernel below sums the squares of all four — which is
/// symmetric, so a *permutation* of the positions computes the same answer. A mutation of the
/// minus to a plus gives shift counts of 32, 40 and 48; SPIR-V leaves a shift at or past the
/// operand's width undefined, this device masks it to five bits, and the result was byte 0, 3, 2, 1
/// squared and summed — the same number, from the wrong bytes. It survived because nothing here
/// could see a single position on its own. [`byte_component`] is what can.
fn signed_byte<const LANES: u32>(
    lanes: &mut simdr::lanes::Lanes<'_>,
    packed: Vector<U32, LANES>,
    byte: u32,
) -> Result<Vector<I32, LANES>, LaneError> {
    let up = lanes.splat_bits::<U32, LANES>(24 - byte * 8)?;
    let down = lanes.splat_bits::<U32, LANES>(24)?;

    let raised = lanes.shift_left(packed, up)?;
    let signed = lanes.reinterpret_unsigned(raised)?;
    lanes.shift_right_arithmetic(signed, down)
}

/// The same byte, squared.
fn square_of_byte<const LANES: u32>(
    lanes: &mut simdr::lanes::Lanes<'_>,
    packed: Vector<U32, LANES>,
    byte: u32,
) -> Result<Vector<I32, LANES>, LaneError> {
    let component = signed_byte::<LANES>(lanes, packed, byte)?;
    lanes.mul(component, component)
}

/// `out[i] = signed_bytes(in[i])[byte]` — one packed byte on its own, sign and all.
///
/// The discriminator the summing kernels cannot be. Every other kernel here folds all four
/// positions together with an operation that does not care which is which; this one names a
/// position and writes what it found, so a shift amount that lands on the wrong byte is a wrong
/// answer rather than the same answer by a different route.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
pub fn byte_component(subgroup: u32, byte: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, byte_component_at, byte)
}

fn byte_component_at<const LANES: u32>(subgroup: u32, byte: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let packed = kernel.load::<LANES>(0)?;
    let component = {
        let mut lanes = kernel.lanes()?;
        let signed = signed_byte::<LANES>(&mut lanes, packed, byte)?;
        lanes.reinterpret(signed)?
    };
    kernel.store(1, component)?;
    kernel.finish()
}

/// `packed_dot_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// As `packed_dot_at`.
pub fn packed_dot(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, packed_dot_at)
}

/// `unpacked_dot_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// As `packed_dot_at`.
pub fn unpacked_dot(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, unpacked_dot_at)
}

/// `times` dot products accumulated into one running total, saturating.
///
/// The arithmetic-bound shape. `packed_dot_at` does one dot product per element loaded, so on a
/// device with bandwidth to spare it measures the load rather than the instruction; this does
/// `times` of them per load, so the arithmetic is what is left.
///
/// The operand is salted with the iteration number — `packed + i` — because otherwise every
/// iteration is the same expression and a driver is free to compute it once. The salt costs one
/// add per iteration and `repeated_unpacked_at` pays exactly the same one.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
fn repeated_packed_at<const LANES: u32>(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let packed = kernel.load::<LANES>(0)?;

    let totals = {
        let mut lanes = kernel.lanes()?;
        let mut total = lanes.splat_bits::<I32, LANES>(0)?;

        for step in 0..times {
            let salt = lanes.splat_bits::<U32, LANES>(step)?;
            let operand = lanes.add(packed, salt)?;
            total = lanes.dot_signed_saturating(operand, operand, total)?;
        }

        lanes.reinterpret(total)?
    };

    kernel.store(1, totals)?;
    kernel.finish()
}

/// The same, with every dot product written out as shifts and multiplies.
///
/// # Errors
///
/// As `repeated_packed_at`.
fn repeated_unpacked_at<const LANES: u32>(
    subgroup: u32,
    times: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let packed = kernel.load::<LANES>(0)?;

    let totals = {
        let mut lanes = kernel.lanes()?;
        let mut total = lanes.splat_bits::<I32, LANES>(0)?;

        for step in 0..times {
            let salt = lanes.splat_bits::<U32, LANES>(step)?;
            let operand = lanes.add(packed, salt)?;

            let mut sum = square_of_byte::<LANES>(&mut lanes, operand, 0)?;
            for byte in 1_u32..4 {
                let squared = square_of_byte::<LANES>(&mut lanes, operand, byte)?;
                sum = lanes.add(sum, squared)?;
            }
            // Wrapping rather than saturating, which is the one way this differs from the twin —
            // there is no written-out saturating add here, and at these magnitudes neither
            // overflows. The example says so rather than leaving it implied.
            total = lanes.add(total, sum)?;
        }

        lanes.reinterpret(total)?
    };

    kernel.store(1, totals)?;
    kernel.finish()
}

/// `repeated_packed_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// As `repeated_packed_at`.
pub fn repeated_packed_dot(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, repeated_packed_at, times)
}

/// `repeated_unpacked_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// As `repeated_packed_at`.
pub fn repeated_unpacked_dot(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, repeated_unpacked_at, times)
}

/// `out[i] = Σ signed(a) × unsigned(b)` over two halves of one buffer.
///
/// The mixed form, which is what a quantised layer's weights and activations usually are. Both
/// operands live in binding 0 — the first half signed, the second unsigned — so a caller with two
/// arrays concatenates them, as [`super::dot_product`] already expects.
///
/// # Errors
///
/// As `packed_dot_at`.
fn mixed_dot_at<const LANES: u32>(subgroup: u32, offset: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let signed = kernel.load::<LANES>(0)?;
    let unsigned = kernel.load_offset::<LANES>(0, offset)?;

    let totals = {
        let mut lanes = kernel.lanes()?;
        let products = lanes.dot_mixed(signed, unsigned)?;
        lanes.reinterpret(products)?
    };
    kernel.store(1, totals)?;
    kernel.finish()
}

/// `mixed_dot_at` over a vector as wide as this device's subgroup.
///
/// # Errors
///
/// As `packed_dot_at`.
pub fn mixed_dot(subgroup: u32, offset: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, mixed_dot_at, offset)
}
