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
//! **Necessary, not sufficient.** Five numbers go into it and all five are read from the module:
//! the dispatch's shape, how many *elements* each invocation touches of a given binding, how many
//! elements past the end of the run it reaches, how far apart that buffer's rows are, and how many
//! bytes an element takes.
//!
//! **The middle two were outside it, and both under-counted.** `Kernel::load_offset` reads
//! `in[i + half]`, so a buffer exactly as long as the run was one this said a dispatch fit while
//! the kernel read `half` elements past the end. And a grid's rows are `pitch` elements apart
//! whether or not the dispatch covers a row, so a kernel reading a narrow slab of a wide matrix
//! reached its last row `(rows - 1) × pitch` elements in while this counted only the columns
//! dispatched — 800 elements measured as 128, in the shape `plane`'s own header describes.
//!
//! Neither needed anything declared, for the reason the strip count did not: the numbers are in the
//! module. The emitter folds `strip × workgroup + offset` into one constant, and it multiplies the
//! row by the pitch. See [`addressing`].
//!
//! **Nothing stays outside any more.** `Kernel::load_offset_by`'s offset is a *specialization*
//! constant — a number chosen after the module was built, with no literal in it to find — and this
//! counted zero for it and said so. That was the wrong direction: everywhere else a term it cannot
//! read makes the check *weaker*, and here it made the check **permissive**, because an address
//! this under-counts is a dispatch this lets through.
//!
//! The number is not unknowable; it is known somewhere else. A pipeline is created *with* its
//! specialization, and every caller that bounds a dispatch has that value in scope at the moment it
//! asks. So [`Bounds::of`] takes one and [`addressing`] resolves each constant to the value the
//! pipeline will carry — the caller's, or the module's own default where the caller sets nothing.

mod addressing;

use super::{Grid, Specialization};
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
    /// The three `LocalSize` axes. `None` when the module declares none, which is not a shape this
    /// can reason about.
    ///
    /// Kept as three rather than as their product because for a grid they are two different things:
    /// x is the columns a workgroup covers and y is its rows, and only x appears in the address
    /// arithmetic. The product is the invocation count and is taken from these where it is wanted.
    local: Option<[u64; 3]>,
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
    /// How many words the buffer must hold for this dispatch to stay inside it.
    ///
    /// The **extent** rather than the count touched, and the two stopped being the same when the
    /// pitch came in: a kernel reading a narrow slab of a wide matrix touches only its own columns
    /// and needs every word up to the last of them, because the rows it skips are inside the
    /// buffer. For a linear binding they still agree.
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
    /// Read what `spirv` needs, when built with `chosen`.
    ///
    /// **The specialization is an argument because forgetting it was the failure this had.** An
    /// address that adds a specialization constant reaches further than the module's own literals
    /// say, by exactly the amount the pipeline was created with — so the two have to be read
    /// together or the answer is a guess in the permissive direction. A caller that specializes
    /// nothing passes `Specialization::none()`, which resolves every constant to the default the
    /// module declared, which is what the driver will do with it.
    pub(crate) fn of(spirv: &[u32], chosen: &Specialization) -> Self {
        let local = local_size(spirv);
        Self {
            local,
            stride: element_bytes(spirv),
            // The **x** axis, not the product. `Kernel::run_start` emits `group.x × (workgroup ×
            // strips)` where `workgroup` is `Shape::workgroup`, which is `LocalSize`'s x alone — so
            // dividing that constant by the product would recover the strip count of a grid kernel
            // as `strips / rows`, and the two agreed only because every kernel with more than one
            // strip has a y of one.
            needs: addressing::needs(spirv, local.map_or(0, |sizes| sizes[0]), chosen),
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
        let (Some(local), Some(stride)) = (self.local, self.stride) else {
            return None;
        };

        self.needs
            .iter()
            .filter_map(|(&binding, needs)| {
                let held = *words.get(binding as usize)?;
                let needed = words_for(elements_of(grid, local, *needs), stride);
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
        let (Some(local), Some(stride)) = (self.local, self.stride) else {
            return None;
        };

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
            .map(|(&binding, needs)| (Some(binding), elements_of(grid, local, *needs)))
            .max_by_key(|&(_, elements)| elements)
            .unwrap_or((None, invocations(grid, local)));

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

/// How many elements of this binding a dispatch of `grid` reaches, counting from its first.
///
/// # A linear buffer
///
/// `invocations × strips`, and the offset sits past the end of that: the last invocation of the
/// last strip lands at `invocations × strips - 1 + offset`, so the buffer needs one more than that.
/// A binding nobody offsets into has an offset of zero and this is the product alone, which is the
/// number this file compared before there was an offset to add.
///
/// # A plane
///
/// **Not that product, and the difference is unbounded.** A grid's rows are `pitch` elements apart
/// whether or not the dispatch covers a row, so the last row *starts* at `(rows - 1) × pitch` and
/// the columns are what it reaches from there:
///
/// ```text
/// (grid.y × local.y - 1) × pitch  +  grid.x × local.x × strips  +  offset
/// ```
///
/// Where the dispatch covers a whole row those agree exactly — `pitch = grid.x × local.x × strips`
/// makes this `grid.y × local.y × pitch`, which is the invocation product — and that is every grid
/// kernel in this crate, which is why the product served. `plane`'s own header describes the shape
/// where they do not: a matrix 4096 elements to the row, read 64 columns at a time. There the
/// product is 64 rows × 64 columns and the kernel's last row begins 258 048 elements in.
fn elements_of(grid: Grid, local: [u64; 3], needs: addressing::Needs) -> u64 {
    let columns = u64::from(grid.x)
        .saturating_mul(local[0])
        .saturating_mul(needs.per_invocation);

    let reached = match needs.pitch {
        Some(pitch) => u64::from(grid.y)
            .saturating_mul(local[1])
            .saturating_sub(1)
            .saturating_mul(pitch)
            .saturating_add(columns),
        None => columns
            .saturating_mul(u64::from(grid.y))
            .saturating_mul(local[1])
            .saturating_mul(local[2]),
    };

    reached.saturating_add(needs.offset)
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

/// The three axes of the workgroup `spirv` declares, as `LocalSize` spells them.
///
/// **Separately, because for a grid they mean different things.** `kernel::binding` emits
/// `LocalSize workgroup rows 1`, so x is how many *columns* a workgroup covers and y is how many
/// *rows* — and the address arithmetic uses only x. A linear kernel has y and z of one and the two
/// readings agree, which is why the product served for as long as there were only linear kernels
/// with more than one strip.
///
/// `None` if the module declares no `LocalSize` at all, which is not a shape this can reason
/// about — a caller gets no check rather than a wrong one.
pub(crate) fn local_size(spirv: &[u32]) -> Option<[u64; 3]> {
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

            Some(sizes.map(u64::from))
        })
}

/// How many invocations a dispatch of `grid` launches of a workgroup shaped `local`.
///
/// Every axis multiplied, which is the whole of it: `LocalSize`'s three are invocations per
/// workgroup and the grid's two are workgroups. It is a **product** and not a quotient, and the
/// only shape that can tell those apart is one with a y or z above 1 — so the tests build one by
/// hand rather than waiting for a kernel to have it.
pub(crate) const fn invocations(grid: Grid, local: [u64; 3]) -> u64 {
    (grid.x as u64) * (grid.y as u64) * local[0] * local[1] * local[2]
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
mod tests;
