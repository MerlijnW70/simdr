//! ```
//! use simdr::kernel::{Kernel, Shape};
//! use simdr::lanes::F32;
//!
//! # fn main() -> Result<(), simdr::lanes::LaneError> {
//! let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2))?;
//! let value = kernel.load::<32>(0)?;
//! let total = kernel.lanes()?.reduce_sum(value)?;
//! kernel.store_scalar(1, total)?;
//!
//! let spirv: Vec<u32> = kernel.finish()?;
//! assert_eq!(spirv.first(), Some(&0x0723_0203), "the SPIR-V magic number");
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

pub mod decode;
pub mod encode;
pub mod half;
pub mod kernel;
pub mod lanes;
pub mod module;
pub mod spec;
