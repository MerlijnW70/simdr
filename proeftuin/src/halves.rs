//! Every `f16` bit pattern there is, through a device and back.
//!
//! `src/half.rs` opens by saying *"every one of the 65 536 half bit patterns is round-tripped
//! through [`to_f32`] and back"* — on the **CPU**. Nothing checks that a device agrees, and the one
//! layer that might have cannot: the differential fuzzer's `Domain::Half` has a ceiling of **8** and
//! refuses any round whose arithmetic leaves ±2048, because that is what lets its comparison be
//! exact. So denormals, infinities, NaNs, negative zero and every rounding boundary are outside it
//! **by construction** rather than by oversight.
//!
//! That leaves the 16-bit storage path exercised at values below 8 and nowhere else, on a type whose
//! whole difficulty is at the edges.
//!
//! # What is guaranteed, and what is only observed
//!
//! This distinction is the design, and the project has been caught by it before: a test once
//! asserted that a sum of sixty-four negative zeros keeps its sign, which IEEE 754 says and **Vulkan
//! does not require** — it is `shaderSignedZeroInfNanPreserveFloat32`, binding only a module that
//! declares the matching execution mode. Two GPUs and a local lavapipe preserved it; Ubuntu's Mesa
//! folded it to `+0.0`, and a shared CI runner turned out to be a fourth implementation.
//!
//! So:
//!
//! * **Asserted** — a half that is *loaded and stored* comes back with the same bits. No arithmetic
//!   happens, so no rounding mode, denormal flush or NaN quieting is licensed to touch it. This is
//!   the exhaustive half.
//! * **Reported** — what arithmetic does at the edges. Vulkan permits a device to flush denormals
//!   and to reshape NaNs unless the matching feature is enabled and declared, so a difference there
//!   is news about the device rather than a defect. Printing it is the honest thing; asserting it
//!   would be the mistake above, again.

use crate::batch::{self, Answer, Batch};
use runner::Gpu;
use runner::kernels::WORKGROUP_SIZE;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{F16, LaneError};

/// How many halves there are.
pub const PATTERNS: usize = 1 << 16;

/// A kernel that loads a half and stores it, and does nothing else.
///
/// The identity on purpose. Every instruction between the load and the store would be a licence for
/// the device to change the value, and the claim under test is that the *storage path* — an `f16`
/// buffer, `Int16` storage access, a load and a store — is bit-exact for every pattern.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
pub fn identity<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<F16>::new(Shape::new(subgroup, WORKGROUP_SIZE, 2))?;
    let value = kernel.load::<LANES>(0)?;
    kernel.store(1, value)?;
    kernel.finish()
}

/// A pattern that did not survive the round trip.
#[derive(Debug, Clone, Copy)]
pub struct Lost {
    /// The bits that went in.
    pub sent: u16,
    /// The bits that came back.
    pub returned: u16,
    /// What `simdr::half` says the pattern means, for a reader.
    pub as_f32: f32,
    /// Whether it is a NaN, which is the one class a device is allowed to reshape.
    pub is_nan: bool,
}

/// What the round trip found, when it ran.
///
/// The four ways it can *not* run live in [`Answer`] now, which is where the other two tools' four
/// live too. They were three enums with the same arms and different names — a fifth reason would
/// have had to be added three times, and adding it to two of the three reads exactly like adding it
/// to all.
#[derive(Debug)]
pub struct Roundtrip {
    /// How many patterns were sent.
    pub sent: usize,
    /// The ones that changed, NaNs included — the caller decides what to do about those.
    pub lost: Vec<Lost>,
}

/// Send all 65 536 patterns through the device and see which come back.
///
/// **The one tool here that was already a batch**, and by accident rather than by design: one
/// dispatch, 1 024 workgroups, 64 patterns each. Saying so through [`Batch`] is the point — the
/// shape the other two had to be rebuilt into is the shape this one was already in, and a shared
/// name is what makes that visible rather than a coincidence in three files.
///
/// Exhaustive is cheap here and a sample would be the wrong shape: the interesting patterns are a
/// few hundred out of 65 536 and a sweep would find them by luck.
pub fn roundtrip<const LANES: u32>(gpu: &Gpu) -> Answer<Roundtrip> {
    let width = gpu.limits().subgroup_size;
    let batch = Batch::of(PATTERNS / WORKGROUP_SIZE as usize, WORKGROUP_SIZE as usize);

    // Every pattern in order, so a failure names a bit pattern rather than an index into a shuffle.
    let sent: Vec<u16> = (0..PATTERNS).map(|bits| bits as u16).collect();

    batch::run(
        gpu,
        "proeftuin-half-identity",
        identity::<LANES>(width),
        &sent,
        batch.workgroups(),
    )
    .map(|returned| Roundtrip {
        sent: sent.len(),
        lost: sent
            .iter()
            .zip(&returned)
            .filter(|(before, after)| before != after)
            .map(|(&before, &after)| Lost {
                sent: before,
                returned: after,
                as_f32: simdr::half::to_f32(before),
                // The exponent all ones with a non-zero mantissa. A device may quiet a signalling
                // NaN or reshape its payload, and Vulkan does not forbid it — so these are counted
                // apart.
                is_nan: (before & 0x7c00) == 0x7c00 && (before & 0x03ff) != 0,
            })
            .collect(),
    })
}

/// The patterns worth naming in a report, whatever the round trip said about them.
///
/// Not a test corpus — the round trip is exhaustive — but the list a reader wants to see confirmed,
/// because these are the ones the fuzzer's ceiling of 8 puts out of reach.
pub const EDGES: [(&str, u16); 10] = [
    ("+0", 0x0000),
    ("-0", 0x8000),
    ("smallest denormal", 0x0001),
    ("largest denormal", 0x03ff),
    ("smallest normal", 0x0400),
    ("one", 0x3c00),
    ("largest finite", 0x7bff),
    ("+inf", 0x7c00),
    ("-inf", 0xfc00),
    ("a NaN", 0x7e00),
];
