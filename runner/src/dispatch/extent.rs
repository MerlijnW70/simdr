//! How many invocations a dispatch launches, read from the module rather than assumed.
//!
//! A dispatch is a workgroup count; a kernel is a workgroup *size*. Multiply them and you have the
//! number of invocations that will run, and every kernel this project emits writes at least one
//! *element* per invocation. So a dispatch whose invocations need more of a buffer than it holds is
//! writing past the end of it — undefined behaviour, which on the devices here has shown up as an
//! access violation on one and plausible wrong numbers on another.
//!
//! # Why the module is decoded instead of the caller being asked
//!
//! Every number this needs is already in the SPIR-V: the workgroup size as the `LocalSize`
//! execution mode, the size of an element as the buffer's `ArrayStride`, and how many elements each
//! invocation touches as the address arithmetic itself. Asking the caller for any of them would let
//! the two disagree, and the caller's copy is the one that would be wrong — it is a number they
//! typed, while the module's is the number the kernel was built with.
//!
//! This is the same rule `decisions/DR-0001` states for opcodes, applied to a shape: read it from
//! the artefact, never from memory. It also happens to be the only version of this check that
//! works. **Element, not word**: `i8` puts four elements in a word and `f16` two, so a dispatch of
//! 128 invocations over a 32-word buffer is exactly full rather than four times over. Comparing
//! invocations against words instead — which this did for one run — refused two fuzzer suites that
//! were doing nothing wrong.
//!
//! # It is a check on every dispatch, and it was a check on one
//!
//! [`Bounds`] is decoded once from a module and asked once per dispatch, which is what lets the
//! held paths carry it: a [`crate::Session`] or a [`crate::Reducer`] builds its pipelines long
//! before it knows how many workgroups anyone will ask for.
//!
//! That shape is the fix for the hole this file had. `Gpu::run` was checked and **nothing else
//! was** — not `run_bound`, not `Session::dispatch`, not `run_chain`, not the held reducer or
//! scanner. The layer that caught eleven tests reading past their inputs covered one of the six
//! ways this crate dispatches, while `README.md` listed it as a layer of the stack.
//!
//! # What the check is and is not
//!
//! **Necessary, not sufficient.** Four numbers go into it and all four are read from the module:
//! how many invocations the dispatch launches, how many *elements* each one touches of a given
//! binding, how many elements past the end of the run it reaches, and how many bytes an element
//! takes.
//!
//! The third of those was outside it until now. `Kernel::load_offset` reads `in[i + half]`, and a
//! buffer exactly as long as the run is one this said a dispatch fit while the kernel read `half`
//! elements past the end of it. The direction was safe — under-counting refuses less than it might
//! and never more — but the hole was real, and closing it needed nothing declared: the emitter
//! folds `strip × workgroup + offset` into one constant, and the strip term is a number this
//! already knows. See [`addressing`].
//!
//! Two things stay outside. A grid kernel's `row × pitch` is not read at all, and
//! `Kernel::load_offset_by`'s offset is a *specialization* constant — a number chosen after the
//! module was built, with no literal in it to find. Both under-count, which is the direction this
//! check must always take when it cannot see.

mod addressing;

use super::Grid;
use simdr::decode;
use simdr::module::op;
use simdr::spec::{Decoration, ExecutionMode};
use std::collections::BTreeMap;

/// What a module needs of the buffers it is dispatched over, decoded once.
///
/// Held rather than recomputed because the two halves of the question arrive at different times: a
/// module is known when a pipeline is built and a workgroup count when it is dispatched, and a
/// [`crate::Session`] can be a long way between the two.
#[derive(Debug, Clone)]
pub(crate) struct Bounds {
    /// Invocations per workgroup, from `LocalSize`. `None` when the module declares none, which is
    /// not a shape this can reason about.
    workgroup: Option<u64>,
    /// Bytes per element, from the buffer's `ArrayStride`.
    stride: Option<u64>,
    /// What each binding's addressing asks of it. A binding whose address does not vary per
    /// invocation is absent — see [`addressing`].
    needs: BTreeMap<u32, addressing::Needs>,
}

/// A buffer a dispatch would touch past the end of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Overrun {
    /// Which binding, when the module's addressing named one.
    ///
    /// `None` when it could not be read and the floor below was used instead — there is no binding
    /// to name there, and naming one anyway would be a guess dressed as a fact.
    pub(crate) binding: Option<u32>,
    /// How many words the dispatch would touch of it.
    pub(crate) needed: usize,
    /// How many it holds.
    pub(crate) held: usize,
}

impl From<Overrun> for crate::Error {
    fn from(overrun: Overrun) -> Self {
        Self::Overrun {
            binding: overrun.binding,
            needed: overrun.needed,
            held: overrun.held,
        }
    }
}

impl Bounds {
    /// Read what `spirv` needs.
    pub(crate) fn of(spirv: &[u32]) -> Self {
        let workgroup = workgroup_size(spirv);
        Self {
            workgroup,
            stride: element_bytes(spirv),
            needs: addressing::needs(spirv, workgroup.unwrap_or(0)),
        }
    }

    /// The binding a dispatch of `grid` would overrun, given each binding's size in words.
    ///
    /// `words` is one entry per binding, in binding order. A binding with no entry, or one whose
    /// addressing this could not read, is **not checked**: with a size per binding the buffers are
    /// deliberately different sizes — `Gpu::run_bound`'s whole reason for existing is a weight
    /// table beside a one-word answer — so a binding this cannot read is one it must not guess
    /// about. [`Bounds::fits`] takes the other view, and says why.
    pub(crate) fn overrun(&self, grid: Grid, words: &[usize]) -> Option<Overrun> {
        let (Some(workgroup), Some(stride)) = (self.workgroup, self.stride) else {
            return None;
        };
        let invocations = invocations(grid, workgroup);

        self.needs
            .iter()
            .filter_map(|(&binding, needs)| {
                let held = *words.get(binding as usize)?;
                let needed = words_for(elements_of(invocations, *needs), stride);
                (needed > held).then_some(Overrun {
                    binding: Some(binding),
                    needed,
                    held,
                })
            })
            .next()
    }

    /// The same, for buffers that are **all** `words` long.
    ///
    /// The shape [`crate::Gpu::run`] and [`crate::Gpu::run_chain`] have: one length, every buffer
    /// allocated to it. That is what makes the floor below safe here and not in [`Bounds::overrun`]
    /// — with one size for every buffer, "at least one element per invocation" cannot ask more of a
    /// small binding than the caller already gave every binding.
    ///
    /// `None` when the module declares no workgroup size or no stride: there is nothing to check
    /// against, and refusing on an unknown would turn "this runner cannot tell" into "your module
    /// is wrong".
    pub(crate) fn overrun_uniform(&self, grid: Grid, words: usize) -> Option<Overrun> {
        let (Some(workgroup), Some(stride)) = (self.workgroup, self.stride) else {
            return None;
        };
        let invocations = invocations(grid, workgroup);

        // The hungriest binding, or the floor for a module whose addressing this could not read at
        // all. Such a module touches an element per invocation as far as anyone here knows, and
        // treating it as touching *none* would make every dispatch fit every buffer.
        //
        // By elements *needed* rather than by elements per invocation: a binding read one at a time
        // with a constant offset past the run can want more than one read four at a time without
        // it, and comparing the multipliers alone would pick the wrong one to report.
        let (binding, elements) = self
            .needs
            .iter()
            .map(|(&binding, needs)| (Some(binding), elements_of(invocations, *needs)))
            .max_by_key(|&(_, elements)| elements)
            .unwrap_or((None, invocations));

        let needed = words_for(elements, stride);
        (needed > words).then_some(Overrun {
            binding,
            needed,
            held: words,
        })
    }

    /// Whether a dispatch of `grid` fits buffers that are all `words` long.
    ///
    /// The reading of [`Bounds::overrun_uniform`] the tests below want. Every caller reports which
    /// binding and by how much, so this is the tests' spelling rather than a second question.
    #[cfg(test)]
    fn fits(&self, grid: Grid, words: usize) -> bool {
        self.overrun_uniform(grid, words).is_none()
    }

    /// How many elements the widest binding gives each invocation, or one for a module this could
    /// not read.
    ///
    /// What the tests below assert against, and what [`Bounds::fits`] compares with. Kept as its
    /// own name because "the strip count the emitter used" is the thing being recovered, and a
    /// reader looking for it should not have to know it lives in a map.
    #[cfg(test)]
    pub(crate) fn elements_per_invocation(&self) -> u64 {
        self.needs
            .values()
            .map(|needs| needs.per_invocation)
            .max()
            .unwrap_or(1)
    }

    /// The largest constant any binding is read past the end of its run by.
    #[cfg(test)]
    pub(crate) fn offset(&self) -> u64 {
        self.needs
            .values()
            .map(|needs| needs.offset)
            .max()
            .unwrap_or(0)
    }
}

/// The highest element index a dispatch of `invocations` reaches in this binding, as a count.
///
/// `invocations × strips` is the run itself, and the offset sits past the end of it: the last
/// invocation of the last strip lands at `invocations × strips - 1 + offset`, so the buffer needs
/// one more than that. A binding nobody offsets into has an offset of zero and this is the product
/// alone, which is the number this file compared before there was an offset to add.
fn elements_of(invocations: u64, needs: addressing::Needs) -> u64 {
    invocations
        .saturating_mul(needs.per_invocation)
        .saturating_add(needs.offset)
}

/// How many words `elements` of `stride` bytes occupy, rounded up.
///
/// **Rounded up, and that is not pedantry**: four `i8` share a word, so 129 byte-writing
/// invocations need 33 words rather than 32, and the last one has nowhere to write if this
/// divides instead.
fn words_for(elements: u64, stride: u64) -> usize {
    let bytes = elements
        .saturating_mul(stride)
        .saturating_add(size_of::<u32>() as u64 - 1);

    usize::try_from(bytes / size_of::<u32>() as u64).unwrap_or(usize::MAX)
}

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

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used)]

    use super::{Bounds, Overrun, element_bytes, invocations, workgroup_size};
    use crate::dispatch::Grid;
    use crate::kernels;
    use simdr::lanes::{F32, I8};
    use simdr::module::op;
    use simdr::spec::ExecutionMode;

    /// How many elements each invocation touches, as the tests below ask it.
    fn per_invocation(spirv: &[u32]) -> u64 {
        Bounds::of(spirv).elements_per_invocation()
    }

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
        let bounds = Bounds::of(&kernels::empty(32).expect("built"));
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert!(bounds.fits(Grid::linear(1), workgroup));
        assert!(bounds.fits(Grid::linear(1), workgroup + 1));
        assert!(
            !bounds.fits(Grid::linear(1), workgroup - 1),
            "one word short of a workgroup is one invocation with nowhere to write"
        );
        assert!(!bounds.fits(Grid::linear(2), workgroup));
    }

    #[test]
    fn the_strip_count_is_read_back_out_of_the_module() {
        // The emitter was told `LANES`; nothing wrote the strip count down. It is recovered from
        // the address arithmetic, so these assertions are against what the *mapping* decides:
        // 32 lanes on a 32-wide subgroup is one element per invocation, 128 is four, and on a
        // 64-wide subgroup the same 128 is two.
        for (width, lanes, strips) in [(32_u32, 32_u32, 1_u64), (32, 128, 4), (64, 128, 2)] {
            let spirv = match lanes {
                32 => kernels::reduce::lane_sum::<F32, 32>(width),
                _ => kernels::reduce::lane_sum::<F32, 128>(width),
            }
            .expect("built");

            assert_eq!(
                per_invocation(&spirv),
                strips,
                "{lanes} lanes on a {width}-wide subgroup"
            );
        }
    }

    #[test]
    fn a_kernel_that_touches_one_element_per_invocation_reports_one() {
        assert_eq!(per_invocation(&kernels::empty(32).expect("built")), 1);
        assert_eq!(per_invocation(&kernels::scale(32, 2.0).expect("built")), 1);
    }

    #[test]
    fn a_module_with_no_workgroup_arithmetic_reports_one_rather_than_nothing() {
        // Under-counting is the safe direction: a floor that cannot read a module still refuses
        // the dispatches it can prove wrong, and lets the rest through.
        assert_eq!(per_invocation(&[]), 1);
    }

    #[test]
    fn a_strip_mined_kernel_needs_more_buffer_than_its_invocation_count() {
        // **The check the first version could not make.** Four strips at 32 lanes: one workgroup
        // of 64 invocations touches 256 elements, so 64 words is not enough however many
        // invocations there are.
        let bounds = Bounds::of(&kernels::reduce::lane_sum::<F32, 128>(32).expect("built"));
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert!(bounds.fits(Grid::linear(1), workgroup * 4));
        assert!(
            !bounds.fits(Grid::linear(1), workgroup),
            "one element per invocation is what this kernel does not do"
        );
        assert!(!bounds.fits(Grid::linear(1), workgroup * 4 - 1));
    }

    #[test]
    fn the_strip_count_multiplies_the_requirement_and_does_not_replace_it() {
        // Both factors, together. A version that used the strip count *instead of* the invocation
        // count would accept a dispatch of any width.
        let bounds = Bounds::of(&kernels::reduce::lane_sum::<F32, 128>(32).expect("built"));
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert!(bounds.fits(Grid::linear(2), workgroup * 8));
        assert!(
            !bounds.fits(Grid::linear(2), workgroup * 4),
            "twice the workgroups needs twice the buffer"
        );
    }

    #[test]
    fn the_width_four_bug_this_check_was_written_for_is_now_refused() {
        // **The regression this exists to prevent, stated as itself.** A kernel with a hard-coded
        // 32 lanes is one element per invocation on a 32-wide subgroup and *eight* on a four-wide
        // one. `kernels::scale` was written that way, and on lavapipe at four lanes it read and
        // wrote eight times the buffer every caller handed it: undefined behaviour returning zeros
        // at width 8 for a day before it became an access violation at 4.
        let spirv = kernels::lane_affine::<32>(4).expect("built");
        let bounds = Bounds::of(&spirv);
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert_eq!(
            per_invocation(&spirv),
            8,
            "32 lanes on a four-wide subgroup is eight elements each"
        );
        assert!(
            !bounds.fits(Grid::linear(1), workgroup),
            "a buffer of one element per invocation is an eighth of what this reads"
        );
        assert!(bounds.fits(Grid::linear(1), workgroup * 8));
    }

    #[test]
    fn a_constant_offset_past_the_run_is_read_out_of_the_folded_address() {
        // `kernels::network::clipped_dot` puts activations in the first `offset` elements of
        // binding 0 and weights after them, so it is the kernel this check was blind to: everything
        // it reads past `invocations × strips` is the offset, and the offset is a literal in the
        // module's own address arithmetic.
        //
        // 512 rather than a round number on purpose — it is `WORKGROUP_SIZE × 8`, the same shape as
        // a strip term, so a walk that subtracted the wrong number of strips would land near it
        // rather than obviously away from it.
        let offset = kernels::WORKGROUP_SIZE * 8;
        let spirv = kernels::network::clipped_dot::<256>(32, offset, 255).expect("built");
        let bounds = Bounds::of(&spirv);

        assert_eq!(
            per_invocation(&spirv),
            8,
            "256 lanes on a 32-wide subgroup is eight strips"
        );
        assert_eq!(
            bounds.offset(),
            u64::from(offset),
            "the strip term is subtracted back off, leaving what the caller asked for"
        );
    }

    #[test]
    fn a_kernel_reading_past_its_run_needs_a_buffer_past_its_run() {
        // The hole this closed, as the numbers a caller sees. One workgroup of 64 invocations at
        // eight strips is a run of 512 elements, and the kernel reads 512 more past it — so a
        // buffer of exactly the run is half of what it touches, and this used to say it fit.
        let offset = kernels::WORKGROUP_SIZE * 8;
        let bounds =
            Bounds::of(&kernels::network::clipped_dot::<256>(32, offset, 255).expect("built"));
        let run = kernels::WORKGROUP_SIZE as usize * 8;

        assert!(
            !bounds.fits(Grid::linear(1), run),
            "the run alone leaves nothing for the half this kernel reads past it"
        );
        assert!(!bounds.fits(Grid::linear(1), run + offset as usize - 1));
        assert!(
            bounds.fits(Grid::linear(1), run + offset as usize),
            "the run and the offset is exactly what it touches"
        );
    }

    #[test]
    fn a_kernel_that_offsets_into_nothing_reports_no_offset() {
        // The other direction, and the one every other kernel here takes: `load` is `load_offset`
        // with a zero, the emitter folds the zero away rather than emitting an add, and this must
        // report nothing rather than inventing a term out of the strip stride.
        assert_eq!(Bounds::of(&kernels::empty(32).expect("built")).offset(), 0);
        assert_eq!(
            Bounds::of(&kernels::reduce::lane_sum::<F32, 128>(32).expect("built")).offset(),
            0,
            "four strips of address arithmetic, and not one element past the run"
        );
    }

    #[test]
    fn an_undecodable_module_is_let_through_rather_than_refused() {
        // "This runner cannot tell" must not be reported as "your module is wrong".
        assert!(Bounds::of(&[]).fits(Grid::linear(1 << 20), 1));
        assert_eq!(
            Bounds::of(&[]).overrun(Grid::linear(1 << 20), &[1, 1]),
            None
        );
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
        let bounds = Bounds::of(&kernels::narrow::narrow_add::<I8, 32>(32, 1).expect("built"));
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert!(
            bounds.fits(Grid::linear(2), workgroup / 2),
            "128 byte-writing invocations fit in 32 words"
        );
        assert!(
            !bounds.fits(Grid::linear(2), workgroup / 2 - 1),
            "and do not fit in 31"
        );
    }

    #[test]
    fn a_word_kernel_and_a_byte_kernel_disagree_by_exactly_four() {
        // The same dispatch against the same buffer, differing only in element size. If these
        // ever agree, the stride is not reaching the comparison.
        let words = Bounds::of(&kernels::empty(32).expect("built"));
        let bytes = Bounds::of(&kernels::narrow::narrow_add::<I8, 32>(32, 1).expect("built"));
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert!(!words.fits(Grid::linear(4), workgroup));
        assert!(bytes.fits(Grid::linear(4), workgroup));
    }

    #[test]
    fn a_partly_filled_word_still_needs_the_whole_word() {
        // Rounding up rather than down. 129 byte-writing invocations need 33 words; a version that
        // divided would accept 32 and leave the last invocation writing past the end.
        let bounds = Bounds::of(&kernels::narrow::narrow_add::<I8, 32>(32, 1).expect("built"));
        let three = Grid::linear(3);
        let invocations = 3 * kernels::WORKGROUP_SIZE as usize;

        assert_eq!(invocations, 192, "three workgroups of 64");
        assert!(bounds.fits(three, 48), "192 bytes is exactly 48 words");
        assert!(!bounds.fits(three, 47));
    }

    #[test]
    fn each_binding_is_measured_against_its_own_size() {
        // **What the module-wide answer could not say.** A reduction reads four strips from binding
        // 0 and writes one scalar per invocation to binding 1, so the two need 256 elements and 64.
        // Taking the largest and applying it to both — which is what a single number has to do —
        // would refuse an output buffer that is exactly the right size.
        let bounds = Bounds::of(&kernels::reduce::lane_sum::<F32, 128>(32).expect("built"));
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert_eq!(
            bounds.overrun(Grid::linear(1), &[workgroup * 4, workgroup]),
            None
        );
        assert_eq!(
            bounds.overrun(Grid::linear(1), &[workgroup, workgroup]),
            Some(Overrun {
                binding: Some(0),
                needed: workgroup * 4,
                held: workgroup,
            }),
            "the input is four strips and only the input is"
        );
        assert_eq!(
            bounds.overrun(Grid::linear(1), &[workgroup * 4, workgroup - 1]),
            Some(Overrun {
                binding: Some(1),
                needed: workgroup,
                held: workgroup - 1,
            }),
            "and the output is still checked, against its own size"
        );
    }

    #[test]
    fn a_binding_addressed_by_workgroup_rather_than_by_invocation_is_left_alone() {
        // `scan_blocks` writes one total per *workgroup* to binding 2, at `workgroup_index`. There
        // is no invocation count to multiply there, and a check that multiplied anyway would demand
        // 64 words for a buffer that legitimately holds one per block.
        let spirv = kernels::scan::scan_blocks::<F32>(32).expect("built");
        let bounds = Bounds::of(&spirv);
        let workgroup = kernels::WORKGROUP_SIZE as usize;

        assert_eq!(
            bounds.overrun(Grid::linear(4), &[workgroup * 4, workgroup * 4, 4]),
            None,
            "four blocks, four totals"
        );
        assert_eq!(
            bounds.overrun(Grid::linear(4), &[workgroup, workgroup * 4, 4]),
            Some(Overrun {
                binding: Some(0),
                needed: workgroup * 4,
                held: workgroup,
            }),
            "and the per-invocation bindings are checked as before"
        );
    }

    #[test]
    fn a_binding_with_no_size_given_is_not_checked() {
        // `overrun` takes what the caller has. A short list is a caller who bound fewer buffers
        // than the module names, which is a pipeline failure with a better message than this one.
        let bounds = Bounds::of(&kernels::reduce::lane_sum::<F32, 128>(32).expect("built"));

        assert_eq!(bounds.overrun(Grid::linear(1), &[]), None);
        assert_eq!(
            bounds.overrun(Grid::linear(1), &[kernels::WORKGROUP_SIZE as usize * 4]),
            None,
            "binding 1 has no entry, so it is not judged"
        );
    }
}
