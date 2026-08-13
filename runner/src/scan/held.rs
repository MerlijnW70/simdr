//! The scan that owns its pipelines and its buffers, and runs them in one submission.
//!
//! Excused from the mutation gate as FFI, so the arithmetic it runs on lives in [`super::plan`]
//! where the gate can reach it. What is left here is allocation, descriptor sets, and the order
//! the dispatches go in.
//!
//! # The order
//!
//! ```text
//!   up      scan each block of the input, keeping every block's total          1 dispatch
//!           scan each block of those totals, keeping their totals              per level below the top
//!   top     one workgroup scans what is left, exclusively                      1 dispatch
//!   down    add each block of a level the offset its level above computed      per level
//! ```
//!
//! Every pass reads what an earlier one wrote, and `Gpu::replay` puts a barrier between them, so
//! the whole thing is one command buffer and one fence however deep it goes.

use super::plan::{self, Level};
use crate::buffer::Buffer;
use crate::dispatch::{Pipeline, Staged, deliver_floats};
use crate::kernels::{self, WORKGROUP_SIZE};
use crate::{Error, Gpu};
use simdr::lanes::F32;

/// A prefix sum over a fixed number of elements, with its pipelines already built.
pub struct Scanner<'gpu> {
    gpu: &'gpu Gpu,
    /// How many elements this was built for. A different count needs a different `Scanner`.
    elements: usize,
    /// The host's way in and out.
    ///
    /// `Option` because [`Buffer::destroy`] consumes the buffer and `Drop` has only `&mut self`.
    staging: Option<Buffer>,
    /// Every device buffer, destroyed in reverse order of creation.
    ///
    /// Held as one list rather than named fields because how many there are depends on how many
    /// levels the length needs, and a field per level is not a thing Rust has.
    buffers: Vec<Buffer>,
    /// Which buffer the answer ends up in.
    answer: usize,
    /// One per dispatch, in order.
    pipelines: Vec<Pipeline>,
    /// How many workgroups each dispatch runs, in the same order.
    workgroups: Vec<u32>,
}

/// Where each of a level's buffers sits in [`Scanner::buffers`].
struct Slots {
    /// The block totals this level holds.
    totals: usize,
    /// This level scanned within its own blocks, and `None` at the top, where the scan of the
    /// level *is* the offsets and no second buffer is needed.
    scanned: Option<usize>,
    /// What each block of the level below owes the blocks before it.
    offsets: usize,
}

impl Gpu {
    /// Build every pipeline a scan over `elements` needs, and hold them.
    ///
    /// `elements` must be a whole number of [`WORKGROUP_SIZE`] and at least one of them.
    ///
    /// # Errors
    ///
    /// [`Error::BadLength`] if `elements` is not a shape this can scan, [`Error::Emit`] if a
    /// kernel cannot be built, otherwise as [`Gpu::run`].
    pub fn scanner<'gpu>(&'gpu self, elements: usize) -> Result<Scanner<'gpu>, Error> {
        let levels = plan::levels(elements)?;
        let width = self.limits().subgroup_size;

        // Every module first, so a kernel that will not build fails before anything is allocated.
        let blocks = kernels::scan::scan_blocks::<F32>(width).map_err(Error::Emit)?;
        let blocks_exclusive =
            kernels::scan::scan_blocks_exclusive::<F32>(width).map_err(Error::Emit)?;
        let top = kernels::scan::scan_workgroup_exclusive::<F32>(width).map_err(Error::Emit)?;
        let add = kernels::scan::add_offsets::<F32>(width).map_err(Error::Emit)?;

        let words = size_of::<f32>() as u64;
        let bytes = (elements.max(1) as u64) * words;

        // SAFETY: everything allocated here is owned by the `Scanner` and destroyed in its `Drop`,
        // which cannot run while a dispatch is in flight — every dispatch waits on a fence before
        // returning. `Held::fail` releases whatever was allocated before an early return.
        unsafe {
            let staging = Buffer::staging(self, bytes)?;
            let mut held = Held {
                gpu: self,
                staging,
                buffers: Vec::new(),
                pipelines: Vec::new(),
                workgroups: Vec::new(),
            };

            // The input is the one buffer the host writes, so it is the one worth asking to be
            // reachable from both sides — see `Buffer::shared`.
            let input = match held.shared(bytes) {
                Ok(index) => index,
                Err(error) => return Err(held.fail(error)),
            };
            let scanned = match held.local(bytes) {
                Ok(index) => index,
                Err(error) => return Err(held.fail(error)),
            };
            let output = match held.local(bytes) {
                Ok(index) => index,
                Err(error) => return Err(held.fail(error)),
            };

            let mut slots = Vec::with_capacity(levels.len());
            for (depth, level) in levels.iter().enumerate() {
                let at_top = depth + 1 == levels.len();
                match held.level(level, at_top, words) {
                    Ok(found) => slots.push(found),
                    Err(error) => return Err(held.fail(error)),
                }
            }

            if let Err(error) = held.record(
                &levels,
                &slots,
                input,
                scanned,
                output,
                &Modules {
                    blocks: &blocks,
                    blocks_exclusive: &blocks_exclusive,
                    top: &top,
                    add: &add,
                },
            ) {
                return Err(held.fail(error));
            }

            // **Two derivations of the same number, made to agree.** The plan says how many
            // dispatches a scan of this depth takes; the loop above emits them one at a time. If
            // those ever disagree the scanner is running a different algorithm from the one that
            // was planned, and the difference would show up as a wrong answer at some depth rather
            // than as a failure here.
            if held.pipelines.len() != plan::dispatches(levels.len()) {
                return Err(held.fail(Error::NoPipeline));
            }

            Ok(held.into_scanner(elements, output))
        }
    }
}

/// The four modules a scan runs, so the recording below takes one argument rather than four.
struct Modules<'a> {
    blocks: &'a [u32],
    blocks_exclusive: &'a [u32],
    top: &'a [u32],
    add: &'a [u32],
}

/// A `Scanner` under construction, with the release path for a failure part way through.
struct Held<'gpu> {
    gpu: &'gpu Gpu,
    staging: Buffer,
    buffers: Vec<Buffer>,
    pipelines: Vec<Pipeline>,
    workgroups: Vec<u32>,
}

impl<'gpu> Held<'gpu> {
    /// Allocate a device-local buffer and return where it landed.
    ///
    /// # Safety
    ///
    /// As [`Buffer::device_local`].
    unsafe fn local(&mut self, bytes: u64) -> Result<usize, Error> {
        // SAFETY: `Buffer::device_local` asks for a live device and a caller who will destroy
        // what comes back. The device outlives this builder, and everything pushed here is
        // released by either `Held::fail` or `Scanner::drop`.
        let buffer = unsafe { Buffer::device_local(self.gpu, bytes) }?;
        self.buffers.push(buffer);
        Ok(self.buffers.len() - 1)
    }

    /// The same, in memory the host can write where the device offers it.
    ///
    /// # Safety
    ///
    /// As [`Buffer::shared`].
    unsafe fn shared(&mut self, bytes: u64) -> Result<usize, Error> {
        // SAFETY: as `local` — the same contract, for a buffer that also asks to be host-writable.
        let buffer = unsafe { Buffer::shared(self.gpu, bytes) }?;
        self.buffers.push(buffer);
        Ok(self.buffers.len() - 1)
    }

    /// One level's buffers, zeroed.
    ///
    /// # Safety
    ///
    /// As [`Buffer::device_local`].
    unsafe fn level(&mut self, level: &Level, at_top: bool, words: u64) -> Result<Slots, Error> {
        let bytes = (level.capacity as u64) * words;

        // SAFETY: `zeroed` asks what this function's own contract asks, and each of the three
        // calls allocates a separate buffer this builder then owns.
        let totals = unsafe { self.zeroed(bytes) }?;
        // At the top there is no block structure left to scan and re-offset: one workgroup scans
        // the level straight into the offsets, so the intermediate buffer would never be read.
        let scanned = if at_top {
            None
        } else {
            // SAFETY: as above.
            Some(unsafe { self.zeroed(bytes) }?)
        };
        // SAFETY: as above.
        let offsets = unsafe { self.zeroed(bytes) }?;

        Ok(Slots {
            totals,
            scanned,
            offsets,
        })
    }

    /// A device-local buffer whose whole contents have been written once, with zeros.
    ///
    /// **The padding is the reason.** A level of four totals still gets a buffer of sixty-four,
    /// because a workgroup is sixty-four invocations, and the tail would otherwise be memory
    /// nobody wrote. `notes/FINDINGS.md` records three tests that assumed such memory reads as
    /// zero — true on both GPUs here and false on lavapipe.
    ///
    /// # Safety
    ///
    /// As [`Buffer::device_local`].
    unsafe fn zeroed(&mut self, bytes: u64) -> Result<usize, Error> {
        // SAFETY: as this function's own contract.
        let index = unsafe { self.local(bytes) }?;
        let Some(buffer) = self.buffers.get(index) else {
            return Err(Error::NoPipeline);
        };

        let zeros = vec![0_u32; (bytes / size_of::<u32>() as u64) as usize];
        // SAFETY: the staging buffer is this builder's own and is at least as large as any level —
        // it was sized for the whole input, which every level is a fraction of. Nothing is in
        // flight: no pipeline has been recorded yet.
        unsafe {
            self.staging.write(self.gpu, &zeros)?;
            self.gpu.copy(&self.staging, buffer, bytes)?;
        }
        Ok(index)
    }

    /// Build a pipeline over `bound` and remember how many workgroups it runs.
    ///
    /// # Safety
    ///
    /// Every index must name a buffer this holds.
    unsafe fn pass(
        &mut self,
        spirv: &[u32],
        bound: &[(usize, u64)],
        workgroups: u32,
    ) -> Result<(), Error> {
        let mut buffers = Vec::with_capacity(bound.len());
        for &(index, bytes) in bound {
            let Some(buffer) = self.buffers.get(index) else {
                return Err(Error::NoPipeline);
            };
            buffers.push((buffer, bytes));
        }

        // SAFETY: every buffer named is one this builder allocated and still owns, and none is in
        // use — nothing has been submitted yet.
        let pipeline =
            unsafe { Pipeline::new(self.gpu, spirv, &buffers, &crate::Specialization::none()) }?;
        self.pipelines.push(pipeline);
        self.workgroups.push(workgroups);
        Ok(())
    }

    /// Release everything allocated so far and hand the error back.
    fn fail(self, error: Error) -> Error {
        // SAFETY: nothing was ever submitted, so no pipeline or buffer is in flight. Pipelines go
        // first: a descriptor set naming a destroyed buffer would be a dangling reference.
        unsafe {
            for pipeline in self.pipelines {
                pipeline.destroy(self.gpu);
            }
            for buffer in self.buffers {
                buffer.destroy(self.gpu);
            }
            self.staging.destroy(self.gpu);
        }
        error
    }

    /// Hand the finished pieces to a `Scanner`.
    fn into_scanner(self, elements: usize, answer: usize) -> Scanner<'gpu> {
        Scanner {
            gpu: self.gpu,
            elements,
            staging: Some(self.staging),
            buffers: self.buffers,
            answer,
            pipelines: self.pipelines,
            workgroups: self.workgroups,
        }
    }
}

impl Held<'_> {
    /// Record every dispatch, in order.
    ///
    /// # Safety
    ///
    /// As [`Held::pass`].
    unsafe fn record(
        &mut self,
        levels: &[Level],
        slots: &[Slots],
        input: usize,
        scanned: usize,
        output: usize,
        modules: &Modules<'_>,
    ) -> Result<(), Error> {
        let words = size_of::<f32>() as u64;
        let elements = self.buffers.get(input).map_or(0, Buffer::capacity) as u64 * words;

        let (Some(first), Some(first_slots)) = (levels.first(), slots.first()) else {
            return Err(Error::NoPipeline);
        };
        let level_bytes = |level: &Level| (level.capacity as u64) * words;

        // Up, from the input: every block scanned inclusively, every block's total kept.
        //
        // SAFETY: every index names a buffer this builder allocated above and still owns, which
        // is what `pass` asks. Nothing has been submitted, so none of them is in use.
        unsafe {
            self.pass(
                modules.blocks,
                &[
                    (input, elements),
                    (scanned, elements),
                    (first_slots.totals, level_bytes(first)),
                ],
                (self.buffers.get(input).map_or(0, Buffer::capacity) / WORKGROUP_SIZE as usize)
                    as u32,
            )?;
        }

        // Up, through the levels below the top: the same, but **exclusively**, because what a
        // block owes is the total of the blocks before it and not including it.
        for depth in 0..levels.len().saturating_sub(1) {
            let (Some(level), Some(here), Some(above)) =
                (levels.get(depth), slots.get(depth), slots.get(depth + 1))
            else {
                return Err(Error::NoPipeline);
            };
            let (Some(upper), Some(scanned_here)) = (levels.get(depth + 1), here.scanned) else {
                return Err(Error::NoPipeline);
            };

            // SAFETY: as the first pass — the indices come from `slots`, which holds only
            // buffers allocated by this builder.
            unsafe {
                self.pass(
                    modules.blocks_exclusive,
                    &[
                        (here.totals, level_bytes(level)),
                        (scanned_here, level_bytes(level)),
                        (above.totals, level_bytes(upper)),
                    ],
                    level.workgroups,
                )?;
            }
        }

        // The top: one workgroup, scanning what is left straight into the offsets it produces.
        let (Some(last), Some(last_slots)) = (levels.last(), slots.last()) else {
            return Err(Error::NoPipeline);
        };
        // SAFETY: as above.
        unsafe {
            self.pass(
                modules.top,
                &[
                    (last_slots.totals, level_bytes(last)),
                    (last_slots.offsets, level_bytes(last)),
                ],
                1,
            )?;
        }

        // Down: each level takes the offsets from the level above and pays its own blocks.
        for depth in (0..levels.len().saturating_sub(1)).rev() {
            let (Some(level), Some(here), Some(above)) =
                (levels.get(depth), slots.get(depth), slots.get(depth + 1))
            else {
                return Err(Error::NoPipeline);
            };
            let (Some(upper), Some(scanned_here)) = (levels.get(depth + 1), here.scanned) else {
                return Err(Error::NoPipeline);
            };

            // SAFETY: as above.
            unsafe {
                self.pass(
                    modules.add,
                    &[
                        (scanned_here, level_bytes(level)),
                        (above.offsets, level_bytes(upper)),
                        (here.offsets, level_bytes(level)),
                    ],
                    level.workgroups,
                )?;
            }
        }

        // And the input's own blocks, which is the answer.
        //
        // SAFETY: as above.
        unsafe {
            self.pass(
                modules.add,
                &[
                    (scanned, elements),
                    (first_slots.offsets, level_bytes(first)),
                    (output, elements),
                ],
                (self.buffers.get(input).map_or(0, Buffer::capacity) / WORKGROUP_SIZE as usize)
                    as u32,
            )?;
        }

        Ok(())
    }
}

impl Scanner<'_> {
    /// How many elements this was built for.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// How many dispatches one call runs.
    #[must_use]
    pub fn dispatches(&self) -> usize {
        self.pipelines.len()
    }

    /// The inclusive prefix sum of `input`, reusing the pipelines and the buffers.
    ///
    /// `input.len()` must equal [`Scanner::elements`]. A shorter slice would leave the tail of the
    /// buffer holding the previous call's data and scan that too, which is a wrong answer rather
    /// than an error — so it is refused.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if `input` is not the length this was built for, otherwise as
    /// [`Gpu::run`].
    pub fn scan(&mut self, input: &[f32]) -> Result<Vec<f32>, Error> {
        if input.len() != self.elements {
            return Err(Error::TooLarge {
                words: input.len(),
                capacity: self.elements,
            });
        }

        let (Some(staging), Some(source), Some(answer)) = (
            self.staging.as_ref(),
            self.buffers.first(),
            self.buffers.get(self.answer),
        ) else {
            return Err(Error::NoPipeline);
        };
        let bytes = (self.elements.max(1) * size_of::<f32>()) as u64;

        // SAFETY: every buffer and pipeline here is owned by `self` and outlives the call, and the
        // submission waits on a fence before returning.
        let output = unsafe {
            let upload = deliver_floats(self.gpu, input, staging, source)?;

            self.gpu.replay(
                &self.pipelines,
                &self.workgroups,
                upload,
                Some(Staged {
                    from: answer,
                    to: staging,
                    bytes,
                }),
            )?;
            staging.read(self.gpu, self.elements)?
        };

        Ok(output.into_iter().map(f32::from_bits).collect())
    }
}

impl Drop for Scanner<'_> {
    fn drop(&mut self) {
        // SAFETY: every object here was created by `Gpu::scanner` and nothing else holds it. The
        // device is idle with respect to them: `scan` waits on a fence before returning. Pipelines
        // go first — a descriptor set naming a destroyed buffer would be a dangling reference for
        // as long as it existed.
        unsafe {
            for pipeline in std::mem::take(&mut self.pipelines) {
                pipeline.destroy(self.gpu);
            }
            for buffer in std::mem::take(&mut self.buffers) {
                buffer.destroy(self.gpu);
            }
            if let Some(staging) = self.staging.take() {
                staging.destroy(self.gpu);
            }
        }
    }
}
