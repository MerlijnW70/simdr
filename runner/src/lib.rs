//! Running `simdr`'s SPIR-V on a real device.
//!
//! Everything in the emitter proves modules are *valid*. Validity is not correctness: a kernel
//! can satisfy `spirv-val` down to the last rule and still compute the wrong number. This crate
//! is what closes that gap — it hands a module to a driver, dispatches it, and reads the answer
//! back so it can be compared against a CPU reference.
//!
//! # Why this is a separate crate
//!
//! Vulkan is FFI, and FFI is `unsafe`. The emitter forbids `unsafe` outright and takes no
//! dependencies; both hold because emitting a binary format needs neither. Running one needs
//! both. So the boundary is drawn here, the arrow points `runner -> simdr`, and nothing in the
//! emitter can ever reach back.
//!
//! # Why `ash` and not `wgpu`
//!
//! `wgpu` would be a fraction of the code, but it runs SPIR-V through naga, which re-parses and
//! re-emits it. That would make this a test of naga's reading of our module. The whole point is
//! to ask the *driver*, so `ash` hands the words to `vkCreateShaderModule` untouched.

// What the emitter forbids outright, this crate cannot: Vulkan is FFI and FFI is `unsafe`. The
// rest of the discipline still applies and was being followed by habit rather than by anything
// checking — so it is checked now.
//
// `unsafe_op_in_unsafe_fn` is on by default in edition 2024 and is the one that matters most here:
// an `unsafe fn` body is not automatically an unsafe block, so every FFI call names itself and
// carries its own `SAFETY` note. That convention is why the audit of those notes was possible at
// all, and why it found the one that had stopped being true.
// `undocumented_unsafe_blocks` is the half of that convention which was not being checked. Every
// `unsafe fn` here carried a `# Safety` clause and most of the blocks inside them carried nothing,
// which is the wrong way round: the clause is what a *caller* owes, and the note is what the
// callee's own reasoning was. There were 79 of the latter missing.
//
// The temptation was to collapse each function's blocks into one and write one note. That would
// have deleted the granularity this crate says above is what made an audit of these possible at
// all — and that audit found a note which had stopped being true. So the notes were written
// instead, one per block, and where two calls share an argument they now share a block as well.
#![deny(missing_docs)]
#![deny(clippy::unwrap_used, clippy::panic)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod buffer;
mod device;
mod dispatch;

pub mod fuzz;
pub mod kernels;
pub mod reduction;
pub mod scan;
pub mod timing;

use ash::vk;
use std::fmt;

pub use device::{Limits, Narrow};
pub use dispatch::{Grid, MemoryType, Pass, Placement, Session, Specialization};
pub use reduction::{BadLength, Reducer, Reduction};
pub use scan::Scanner;
pub use timing::Timing;

/// Something that stopped a kernel running.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The Vulkan loader is not present, or refused to load.
    NoLoader(ash::LoadingError),
    /// A Vulkan call failed.
    Vulkan(vk::Result),
    /// The loader started but reported no physical device with a compute queue.
    NoComputeDevice,
    /// No memory type with the properties a buffer needed was available.
    NoHostVisibleMemory,
    /// The host tried to read or write a buffer that lives only on the device.
    ///
    /// A caller bug rather than a device limitation: device-local buffers move through a staging
    /// copy, and reaching for one directly means that copy was skipped.
    NotMappable,
    /// The driver accepted the pipeline creation call but returned no pipeline.
    NoPipeline,
    /// More words than the buffer holds.
    ///
    /// A caller bug, and one that used to be undefined behaviour rather than an error: the host
    /// copies memcpy through a mapping sized when the buffer was allocated, and the length was
    /// *assumed* because for a while only one caller existed and it always allocated exactly what
    /// it wrote. `Session` takes a slice from outside and does not.
    TooLarge {
        /// How many words were asked for.
        words: usize,
        /// How many the buffer holds.
        capacity: usize,
    },
    /// A dispatch would touch more of a buffer than the buffer holds.
    ///
    /// Apart from [`Error::TooLarge`], which is a *host copy* that would not fit — a failed memcpy,
    /// caught before anything happens. This is a **kernel** reading or writing past the end of a
    /// binding, which is undefined behaviour: an access violation on one device here and plausible
    /// wrong numbers on another. `dispatch::extent` reads the module's own workgroup size, element
    /// stride and address arithmetic and refuses the submission rather than making it.
    Overrun {
        /// Which binding the kernel would run off the end of.
        ///
        /// `None` when the module's address arithmetic could not be read and the floor of one
        /// element per invocation was used instead: there is no binding to name there.
        binding: Option<u32>,
        /// How many words of it the dispatch would touch.
        needed: usize,
        /// How many it holds.
        held: usize,
    },
    /// A buffer was not a shape [`Gpu::sum`] can fold.
    BadLength(BadLength),
    /// A pass could not be built.
    ///
    /// An emitter failure rather than a device one, which is worth keeping apart: nothing was
    /// submitted, and the fix is in the kernel rather than on the machine.
    Emit(simdr::lanes::LaneError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLoader(error) => write!(f, "the Vulkan loader would not load: {error}"),
            Self::Vulkan(result) => write!(f, "a Vulkan call failed: {result:?}"),
            Self::NoComputeDevice => f.write_str("no physical device offers a compute queue"),
            Self::NoHostVisibleMemory => f.write_str("no memory type offers what a buffer needed"),
            Self::NotMappable => {
                f.write_str("that buffer lives on the device and the host cannot map it")
            }
            Self::NoPipeline => f.write_str("the driver returned no compute pipeline"),
            Self::TooLarge { words, capacity } => {
                write!(f, "{words} words asked for and the buffer holds {capacity}")
            }
            Self::Overrun {
                binding: Some(binding),
                needed,
                held,
            } => write!(
                f,
                "this dispatch touches {needed} words of binding {binding} and it holds {held}"
            ),
            Self::Overrun {
                binding: None,
                needed,
                held,
            } => write!(
                f,
                "this dispatch touches at least {needed} words and the buffers hold {held}"
            ),
            Self::BadLength(BadLength::NotAPowerOfTwo(length)) => write!(
                f,
                "{length} elements cannot be halved down: every fold needs a power of two"
            ),
            Self::BadLength(BadLength::TooSmall { length, minimum }) => write!(
                f,
                "{length} elements is below the {minimum} the final workgroup needs"
            ),
            Self::Emit(error) => write!(f, "a pass could not be built: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoLoader(error) => Some(error),
            Self::Emit(error) => Some(error),
            _ => None,
        }
    }
}

impl From<vk::Result> for Error {
    fn from(result: vk::Result) -> Self {
        Self::Vulkan(result)
    }
}

/// An open compute device.
///
/// Opening one is expensive and running kernels on it is not, so a caller keeps this alive across
/// however many dispatches it needs.
pub struct Gpu {
    // Field order is drop order, and Vulkan cares: the device must go before the instance, and
    // the instance before the entry that loaded it.
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    limits: Limits,
    instance: ash::Instance,
    _entry: ash::Entry,
}

impl Gpu {
    /// What the device reports about itself — its name, and how wide a subgroup is.
    #[must_use]
    pub const fn limits(&self) -> &Limits {
        &self.limits
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: both handles were created by this struct and nothing else holds them; every
        // per-run object is destroyed before `run` returns, so the device is idle by now.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
