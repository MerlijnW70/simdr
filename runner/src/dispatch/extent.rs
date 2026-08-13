//! How many invocations a dispatch launches, read from the module rather than assumed.
//!
//! A dispatch is a workgroup count; a kernel is a workgroup *size*. Multiply them and you have the
//! number of invocations that will run, and every kernel this project emits writes at least one
//! *element* per invocation. So a dispatch whose invocations need more of the buffer than it holds
//! is writing past the end of it — undefined behaviour, which on the devices here has shown up as
//! an access violation on one and plausible wrong numbers on another.
//!
//! # Why the module is decoded instead of the caller being asked
//!
//! Both numbers this needs are already in the SPIR-V: the workgroup size as the `LocalSize`
//! execution mode, and the size of an element as the buffer's `ArrayStride`. Asking the caller for
//! either would let the two disagree, and the caller's copy is the one that would be wrong — it is
//! a number they typed, while the module's is the number the kernel was built with.
//!
//! This is the same rule `decisions/DR-0001` states for opcodes, applied to a shape: read it from
//! the artefact, never from memory. It also happens to be the only version of this check that
//! works. **Element, not word**: `i8` puts four elements in a word and `f16` two, so a dispatch of
//! 128 invocations over a 32-word buffer is exactly full rather than four times over. Comparing
//! invocations against words instead — which this did for one run — refused two fuzzer suites that
//! were doing nothing wrong.
//!
//! # What the check is and is not
//!
//! **Necessary, not sufficient.** One word per invocation is the floor: a strip-mined kernel reads
//! and writes `LANES` of them, and nothing in the module says how many strips a given load walks.
//! So this catches dispatching too many workgroups and does not catch a lane mapping that reads
//! eight times its buffer — which is a real bug this project has had, and is the reason the limit
//! of the check is written down here rather than left to be discovered.

use super::Grid;
use simdr::decode;
use simdr::module::op;
use simdr::spec::{Decoration, ExecutionMode};

/// The workgroup size declared by `spirv`, as `x * y * z`.
///
/// `None` if the module declares no `LocalSize` at all, which is not a shape this can reason
/// about — a caller gets no check rather than a wrong one.
pub(crate) fn workgroup_size(spirv: &[u32]) -> Option<u64> {
    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::EXECUTION_MODE)
        .find_map(|instruction| {
            // entry point, mode, then the three literals the mode takes.
            let operands = instruction.operands();
            let (mode, sizes) = match operands {
                [_entry, mode, x, y, z] => (*mode, [*x, *y, *z]),
                _ => return None,
            };
            if mode != ExecutionMode::LocalSize.word() {
                return None;
            }

            Some(u64::from(sizes[0]) * u64::from(sizes[1]) * u64::from(sizes[2]))
        })
}

/// How many invocations `grid` launches of a kernel whose workgroup holds `workgroup` of them.
pub(crate) const fn invocations(grid: Grid, workgroup: u64) -> u64 {
    (grid.x as u64) * (grid.y as u64) * workgroup
}

/// How many bytes one element of this kernel's buffers takes, from the module's `ArrayStride`.
///
/// **This is the difference between a check and a false alarm.** A buffer is words to Vulkan and
/// elements to a kernel, and for `i8` or `f16` those are not the same count: 32 words hold 128
/// bytes, and 128 invocations each writing one of them is exactly full rather than four times
/// over. The first version of this check compared invocations against *words* and refused two
/// fuzzer suites that were doing nothing wrong.
///
/// Read from the module for the same reason the workgroup size is: the emitter decorates the
/// buffer with the stride its element type requires, so the artefact already knows.
pub(crate) fn element_bytes(spirv: &[u32]) -> Option<u64> {
    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::DECORATE)
        .find_map(|instruction| match instruction.operands() {
            [_target, decoration, stride] if *decoration == Decoration::ArrayStride.word() => {
                Some(u64::from(*stride))
            }
            _ => None,
        })
}

/// Whether a buffer of `words` can hold one element for every invocation the dispatch launches.
///
/// `true` when the module declares no workgroup size or no stride: there is nothing to check
/// against, and refusing on an unknown would turn "this runner cannot tell" into "your module is
/// wrong".
pub(crate) fn fits(spirv: &[u32], grid: Grid, words: usize) -> bool {
    let (Some(workgroup), Some(stride)) = (workgroup_size(spirv), element_bytes(spirv)) else {
        return true;
    };

    // A stride of zero needs no guard of its own, and the guard that was here could never fire.
    // Elements of no size take no room: the product below is zero, zero fits every buffer, and
    // that is the same answer an explicit `if stride == 0 { return true }` gave. The mutation gate
    // found it by flipping the condition to `false` and killing nothing — the second unfalsifiable
    // branch this project has written and the second to be deleted rather than tested.
    let bytes = (words as u64).saturating_mul(size_of::<u32>() as u64);
    invocations(grid, workgroup).saturating_mul(stride) <= bytes
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used)]

    use super::{element_bytes, fits, invocations, workgroup_size};
    use crate::dispatch::Grid;
    use crate::kernels;
    use simdr::lanes::I8;
    use simdr::module::op;
    use simdr::spec::ExecutionMode;

    #[test]
    fn the_workgroup_size_is_read_out_of_the_module_the_emitter_built() {
        // Not a constant repeated here: `kernels::WORKGROUP_SIZE` is what the kernel was built
        // for, and this proves the module says the same thing.
        let spirv = kernels::empty(32).expect("built");
        assert_eq!(
            workgroup_size(&spirv),
            Some(u64::from(kernels::WORKGROUP_SIZE))
        );
    }

    #[test]
    fn a_module_with_no_execution_mode_at_all_reports_nothing() {
        // The header alone. Nothing to decode, so nothing is claimed.
        assert_eq!(workgroup_size(&[]), None);
        assert_eq!(workgroup_size(&[0x0723_0203, 0x0001_0300, 0, 1, 0]), None);
    }

    /// A module body of one hand-built `OpExecutionMode`, for shapes the emitter does not produce.
    ///
    /// Written out by hand because that is the point: every kernel here declares `LocalSize` and
    /// nothing else with five operands, so a module that does can only come from somewhere else —
    /// and "somewhere else" is exactly who this decoder has to be right about.
    fn module_with_execution_mode(mode: u32, sizes: [u32; 3]) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 1, 0];
        // Word count in the high half, opcode in the low: one for the instruction word, one for
        // the entry point, one for the mode, three for the literals.
        words.push((6 << 16) | u32::from(op::EXECUTION_MODE));
        words.push(1);
        words.push(mode);
        words.extend_from_slice(&sizes);
        words
    }

    #[test]
    fn an_execution_mode_that_is_not_local_size_declares_no_workgroup() {
        // `LocalSizeHint` takes the same three literals and means something else entirely — it is
        // advice to a driver, not the shape of a dispatch. Reading its operands as a workgroup
        // size would invent a number out of an unrelated instruction.
        let hint = module_with_execution_mode(ExecutionMode::LocalSize.word() + 1, [2, 3, 4]);
        assert_eq!(workgroup_size(&hint), None);

        let real = module_with_execution_mode(ExecutionMode::LocalSize.word(), [2, 3, 4]);
        assert_eq!(workgroup_size(&real), Some(24));
    }

    #[test]
    fn the_three_axes_of_a_workgroup_are_multiplied_and_not_divided() {
        // Every kernel here declares `LocalSize x 1 1`, where a product and a quotient agree:
        // `x * 1 * 1` and `x / 1 / 1` are both `x`. So the one shape that tells them apart has to
        // be built by hand, and without it the operator is untested on real modules.
        let module = module_with_execution_mode(ExecutionMode::LocalSize.word(), [2, 3, 4]);

        assert_eq!(workgroup_size(&module), Some(24), "2 * 3 * 4");
        assert_ne!(workgroup_size(&module), Some(0), "2 / 3 / 4");
    }

    #[test]
    fn invocations_multiply_both_axes_by_the_workgroup() {
        assert_eq!(invocations(Grid::linear(1), 64), 64);
        assert_eq!(invocations(Grid::linear(16), 64), 1024);
        assert_eq!(invocations(Grid::new(4, 4), 64), 1024);
    }

    #[test]
    fn the_product_is_computed_wide_enough_not_to_wrap() {
        // 2^20 workgroups of 2^16 invocations is 2^36, which does not fit in the `u32` both
        // factors are. Computed narrowly it is *zero*, and a dispatch of zero invocations fits
        // every buffer — so the wrap would not merely misreport, it would report the safe answer.
        assert_eq!(invocations(Grid::linear(1 << 20), 1 << 16), 1 << 36);
        assert_eq!(
            (1_u32 << 20).wrapping_mul(1 << 16),
            0,
            "the narrow product this avoids"
        );
    }

    #[test]
    fn a_dispatch_that_matches_its_buffer_fits_and_one_word_more_does_not() {
        let spirv = kernels::empty(32).expect("built");
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert!(fits(&spirv, Grid::linear(1), workgroup));
        assert!(fits(&spirv, Grid::linear(1), workgroup + 1));
        assert!(
            !fits(&spirv, Grid::linear(1), workgroup - 1),
            "one word short of a workgroup is one invocation with nowhere to write"
        );
        assert!(!fits(&spirv, Grid::linear(2), workgroup));
    }

    #[test]
    fn an_undecodable_module_is_let_through_rather_than_refused() {
        // "This runner cannot tell" must not be reported as "your module is wrong".
        assert!(fits(&[], Grid::linear(1 << 20), 1));
    }

    #[test]
    fn the_stride_is_the_one_the_element_type_needs() {
        // Four bytes to a word, so these are 1, 2 and 4 elements per word respectively.
        assert_eq!(element_bytes(&kernels::empty(32).expect("built")), Some(4));

        let bytes = kernels::narrow::narrow_add::<I8, 32>(32, 1).expect("built");
        assert_eq!(element_bytes(&bytes), Some(1));
    }

    #[test]
    fn a_byte_kernel_fills_a_word_with_four_invocations_rather_than_one() {
        // The false alarm this check had for one run. 128 invocations of a byte kernel need 128
        // bytes, which is 32 words — not 128 of them.
        let spirv = kernels::narrow::narrow_add::<I8, 32>(32, 1).expect("built");
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert!(
            fits(&spirv, Grid::linear(2), workgroup / 2),
            "128 byte-writing invocations fit in 32 words"
        );
        assert!(
            !fits(&spirv, Grid::linear(2), workgroup / 2 - 1),
            "and do not fit in 31"
        );
    }

    #[test]
    fn a_word_kernel_and_a_byte_kernel_disagree_by_exactly_four() {
        // The same dispatch against the same buffer, differing only in element size. If these
        // ever agree, the stride is not reaching the comparison.
        let words = kernels::empty(32).expect("built");
        let bytes = kernels::narrow::narrow_add::<I8, 32>(32, 1).expect("built");
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert!(!fits(&words, Grid::linear(4), workgroup));
        assert!(fits(&bytes, Grid::linear(4), workgroup));
    }
}
