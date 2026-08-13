//! A prefix sum over more elements than one workgroup can reach.
//!
//! [`crate::kernels::scan`] has the kernels; this composes them. A scan longer than
//! [`crate::kernels::WORKGROUP_SIZE`] elements is three steps — scan each block, scan the block
//! totals, add each block what it owes — and the middle step is itself a scan, so past 64 blocks
//! it is the same three steps again one level up. [`plan`] decides how many levels that is.
//!
//! # Why it is an object rather than a function
//!
//! The same trade [`crate::Reducer`] makes. Seven pipelines and a dozen buffers is a great deal to
//! build for one call and nothing to keep for a caller that scans repeatedly, so the pipelines and
//! the buffers are made once, sized for a length, and held.

mod held;
mod plan;

pub use held::Scanner;
