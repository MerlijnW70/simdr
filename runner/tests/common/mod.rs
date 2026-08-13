//! Shared harness for the execution tests.
//!
//! Not a test binary of its own — Cargo compiles `tests/*.rs` as separate binaries and leaves
//! directories alone, so each of those declares `mod common;`. The allow below follows from that:
//! a helper only one of them uses is dead code in the other.

#![allow(
    dead_code,
    unused_imports,
    reason = "each test binary compiles this file and uses a different subset of it — and a               re-export nobody in *this* binary names is an unused import rather than dead code,               which is a second lint saying the same thing about the same arrangement"
)]

// The validator, from the emitter's test tree rather than copied into this one. Two copies would
// be two things to keep in step, and the one covering these kernels is the one that did not exist
// until now — every kernel in `runner::kernels` reached a driver without ever meeting `spirv-val`.
// The path is relative to *this file's* directory, so it climbs out of `runner/tests/common/`
// three times to reach the workspace root and not twice.
#[path = "../../../tests/common/spirv_val.rs"]
mod spirv_val;
pub use spirv_val::{VULKAN_1_1, expect_valid, validate, validator};

use runner::Gpu;

/// Open a device, or report why the test is not running and hand back `None`.
///
/// A machine without a GPU is a normal state for a suite to find. It reports loudly rather than
/// passing quietly, because a skipped correctness test that looks green is worse than a red one.
pub fn device(label: &str) -> Option<Gpu> {
    match Gpu::open() {
        Ok(Some(gpu)) => Some(gpu),
        Ok(None) => {
            eprintln!("SKIPPED {label}: no Vulkan device");
            None
        }
        Err(error) => {
            eprintln!("SKIPPED {label}: could not open a device — {error}");
            None
        }
    }
}

/// A deterministic input that makes a wrong answer obvious.
///
/// Every element distinct, and small enough that sums stay exact in `f32`: a reduction over a few
/// hundred of these stays well inside the 24 bits a float carries, so comparing exactly is
/// legitimate rather than lucky — and it holds whatever order the hardware reduces in, which
/// matters because floating-point addition is not associative.
pub fn ramp(count: usize) -> Vec<f32> {
    (0..count).map(|index| index as f32).collect()
}

/// What a reduction over consecutive groups of `group` elements should give each lane.
pub fn grouped_sums(count: usize, group: usize) -> Vec<f32> {
    (0..count)
        .map(|lane| {
            let first = lane / group * group;
            (first..(first + group).min(count))
                .map(|other| other as f32)
                .sum()
        })
        .collect()
}
