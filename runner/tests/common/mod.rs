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
///
/// **A device asked for by name and not here is not that state, and fails the test.** Setting
/// `SIMDR_DEVICE` asserts a device exists; when the assertion is wrong, every test in the suite
/// skips, and a skip is invisible — `libtest` swallows `eprintln!` from a passing test, so the run
/// prints the same summary it would have printed after running all of it. `SIMDR_DEVICE=llvmpipe`
/// here, where the two devices are called something else, was 157 skips and an exit code of zero.
///
/// So that one is a panic. It is the only outcome a two-device sweep can act on: the whole point of
/// the sweep is that both widths ran, and a typo in the variable that chooses them must not be able
/// to report that they did.
pub fn device(label: &str) -> Option<Gpu> {
    match Gpu::open() {
        Ok(Some(gpu)) => Some(gpu),
        Ok(None) => {
            eprintln!("SKIPPED {label}: no Vulkan device");
            None
        }
        Err(error @ runner::Error::NoSuchDevice { .. }) => {
            panic!("SIMDR_DEVICE names a device that is not here — {error}")
        }
        Err(error) => {
            eprintln!("SKIPPED {label}: could not open a device — {error}");
            None
        }
    }
}

/// Whether this device offers everything `modules` declare, reporting by name when it does not.
///
/// **The gate that cannot name the wrong feature.** Every skip in this suite used to pick a feature
/// bit by hand, and picking is where it goes wrong: three kernels using votes were gated on the
/// *ballot*, which is a different capability and a different bit. That was right on all three
/// implementations here — no device offers one without the other — and would have been wrong on the
/// first that did.
///
/// It is not only a past mistake. Five gates in this suite named `subgroup_arithmetic` alone for
/// kernels that also reach `any_uniform`, and a vote is `GroupNonUniformVote`: on a device with
/// arithmetic and no vote they would have run and failed at pipeline creation rather than skipping.
///
/// `Limits::unsupported_in` reads the requirement out of the module's own `OpCapability`
/// instructions, so a kernel that starts needing something new brings its own gate with it and no
/// caller has to remember. Pass every module the test dispatches: a test that gates on one and runs
/// another is the same class of mistake one level up.
///
/// # What it cannot see
///
/// A permission that leaves no trace in the module. `shaderSubgroupExtendedTypes` is the one this
/// crate meets — a device may accept `OpGroupNonUniformIAdd` on a 32-bit integer and refuse it on an
/// 8-bit one with nothing in the SPIR-V to say so — so the narrow-type tests still ask [`Narrow`] by
/// hand, and say why where they do.
///
/// [`Narrow`]: runner::Narrow
pub fn runnable(gpu: &Gpu, label: &str, modules: &[&[u32]]) -> bool {
    for spirv in modules {
        let missing = gpu.limits().unsupported_in(spirv);
        if !missing.is_empty() {
            eprintln!("SKIPPED {label}: this device does not offer {missing:?}");
            return false;
        }
    }
    true
}

/// How many elements a kernel built for `lanes` touches, on a device of `width` lanes.
///
/// **A buffer of one workgroup is wrong for most kernels here, and was wrong for eleven tests.**
/// A vector of 32 lanes is one element per invocation only on a 32-wide subgroup; narrower, it
/// strip-mines — four elements each at 8 lanes, eight at 4. Sizing to `WORKGROUP_SIZE` then hands
/// the kernel an eighth of what it reads.
///
/// Nothing caught that for months. Every test doing it passed on the two GPUs in this machine,
/// which report 32 and 64, and read off the end of its input on lavapipe at 4, 8 and 16 — right in
/// the first sixty-four elements and undefined after them. `dispatch::extent` learnt to recover the
/// strip count from a module's own address arithmetic and refused all eleven the first time it ran.
///
/// `lanes` is the count the *kernel* was built for, not the device's. Passing the device's width
/// gives one element per invocation, which is what a whole-subgroup kernel does at every width.
pub fn elements(width: u32, lanes: u32) -> usize {
    let strips = (lanes / width.max(1)).max(1) as usize;
    runner::kernels::WORKGROUP_SIZE as usize * strips
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
