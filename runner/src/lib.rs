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

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    NoLoader(ash::LoadingError),
    Vulkan(vk::Result),
    NoComputeDevice,
    NoSuchDevice {
        wanted: String,
        present: Vec<String>,
    },
    NoHostVisibleMemory,
    NotMappable,
    NoPipeline,
    TooLarge {
        words: usize,
        capacity: usize,
    },
    Overrun {
        binding: Option<u32>,
        needed: usize,
        held: usize,
    },
    BadLength(BadLength),
    Emit(simdr::lanes::LaneError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLoader(error) => write!(f, "the Vulkan loader would not load: {error}"),
            Self::Vulkan(result) => write!(f, "a Vulkan call failed: {result:?}"),
            Self::NoComputeDevice => f.write_str("no physical device offers a compute queue"),
            Self::NoSuchDevice { wanted, present } => write!(
                f,
                "no device here is called {wanted:?} — SIMDR_DEVICE matches a substring of {present:?}"
            ),
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
                "this dispatch reaches {needed} words into binding {binding} and it holds {held}"
            ),
            Self::Overrun {
                binding: None,
                needed,
                held,
            } => write!(
                f,
                "this dispatch reaches at least {needed} words in and the buffers hold {held}"
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

pub struct Gpu {
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    limits: Limits,
    instance: ash::Instance,
    _entry: ash::Entry,
}

impl Gpu {
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
