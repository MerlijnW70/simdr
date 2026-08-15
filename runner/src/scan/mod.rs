//! A prefix sum over more elements than one workgroup can reach.
//!
//! [`crate::kernels::scan`] has the kernels; this composes them. A scan longer than
//! [`crate::kernels::WORKGROUP_SIZE`] elements is three steps — scan each block, scan the block
//! totals, add each block what it owes — and the middle step is itself a scan, so past 64 blocks
//! it is the same three steps again one level up. `scan::plan` decides how many levels that is.
//!
//! # Why it is an object rather than a function
//!
//! The same trade [`crate::Reducer`] makes. Seven pipelines and a dozen buffers is a great deal to
//! build for one call and nothing to keep for a caller that scans repeatedly, so the pipelines and
//! the buffers are made once, sized for a length, and held.

mod held;
mod passes;
mod plan;

pub use held::Scanner;

use crate::{Error, Gpu};

impl Gpu {
    /// The inclusive prefix sum of `input`, building everything it needs and throwing it away.
    ///
    /// The one-shot form, and it is here for symmetry with [`Gpu::sum`] rather than for speed: it
    /// allocates a dozen buffers and builds `2 × levels + 1` pipelines on every call, all of which
    /// [`Gpu::scanner`] does once and keeps. `runner/examples/scanner.rs` measures a held scan over
    /// 2²⁰ at about a millisecond, and the setup this repeats is the larger part of that.
    ///
    /// What it is good for is a test, an example, or scanning something once — the cases where
    /// building a `Scanner` to use it a single time is ceremony rather than economy.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scanner`], and [`Error::TooLarge`] can never come from here: the scanner is built
    /// for exactly the length it is then given.
    pub fn scan(&self, input: &[f32]) -> Result<Vec<f32>, Error> {
        self.scanner(input.len())?.scan(input)
    }
}
