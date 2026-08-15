//! SIMD on the GPU: portable-SIMD semantics lowered onto SPIR-V subgroup operations.
//!
//! A `Simd<T, N>` is a vector of `N` lanes, and a GPU subgroup is a vector unit of 32 or 64
//! lanes. This crate emits the SPIR-V that makes those the same thing.
//!
//! # What this crate is, and is not
//!
//! It **emits** SPIR-V: words in, words out, no I/O and no device. That is what lets the
//! dependency table stay empty, and it is the reason the boundary is drawn here rather than a
//! layer higher. Running a module needs a Vulkan loader, which needs FFI, which needs `unsafe` —
//! so running lives in a separate workspace member that may depend on whatever it likes. The
//! arrow points that way and never back.
//!
//! # Where to start
//!
//! [`kernel::Kernel`] is the front door: it builds the buffer interface every compute shader needs
//! and hands out a [`lanes::Lanes`] for the arithmetic. [`module`] is underneath, for anything the
//! lane layer does not cover, and [`spec`] holds Khronos' numbers — every one of them read out of
//! the grammar or the assembler rather than from memory, which is `decisions/DR-0001`.
//!
//! ```
//! use simdr::kernel::{Kernel, Shape};
//! use simdr::lanes::F32;
//!
//! # fn main() -> Result<(), simdr::lanes::LaneError> {
//! // 32-wide subgroup, 64 invocations, two buffers. `simdr probe` reports the width.
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
//!
//! # What it cannot express
//!
//! No matrices and no cooperative matrices. A packed `i8` *mapping*, where four elements share a
//! lane, is deliberately absent — `decisions/DR-0004` says why, and why the packed integer dot
//! product does not amount to one. Dispatch has two axes and not three, which is
//! `decisions/DR-0006`. There is no compare-and-exchange among the atomics, and no `f64`.
//!
//! `notes/NEXT.md` says which of those is worth building and what each one blocks.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
// No input may cause a panic — every failure a caller can provoke is a `Result`. These four are
// the constructs that turn bad input into an abort, so they are denied rather than discouraged.
// Integer overflow is the fifth way to panic and is *not* denied wholesale: `clippy::
// arithmetic_side_effects` would fire on loop counters that provably cannot overflow and cost more
// in readability than it buys. Where a length taken from a caller drives arithmetic, the checked
// form is used explicitly and says why.
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
