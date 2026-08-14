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
use std::time::Duration;

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
        self.build_scanner(elements, None)
    }

    /// The same, with one elementwise pass of `map` run over the input first.
    ///
    /// **What removes a crossing of the bus.** The running total of f(x) over data the caller
    /// cannot reach otherwise costs three host crossings: send the input, run `f`, bring the
    /// result home, send it back, scan. Two of those are the whole buffer.
    ///
    /// Here `map` is the first pass of the same chain — its output never leaves the device, and
    /// the first block scan reads it where the input would have been. The same trade
    /// [`Gpu::reducer_of`] makes, which `runner/examples/reducer.rs` measures at 3.7× for a
    /// reduction over 2²⁰.
    ///
    /// `map` must be a two-binding kernel built for [`WORKGROUP_SIZE`] invocations, reading
    /// binding 0 and writing binding 1, and it must write **every** element the first scan reads.
    /// `elements / WORKGROUP_SIZE` workgroups of it are dispatched, worked out here rather than
    /// taken as an argument so the count cannot disagree with the length the levels were built
    /// for.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scanner`].
    pub fn scanner_of<'gpu>(
        &'gpu self,
        elements: usize,
        map: &[u32],
    ) -> Result<Scanner<'gpu>, Error> {
        self.build_scanner(elements, Some(map))
    }

    /// The construction both of them share.
    fn build_scanner<'gpu>(
        &'gpu self,
        elements: usize,
        map: Option<&[u32]>,
    ) -> Result<Scanner<'gpu>, Error> {
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
            // Where the map writes, and where the first block scan then reads. Only allocated
            // when there is a map: a scanner without one would hold a buffer nothing touches.
            let mapped = match map {
                None => None,
                Some(_) => match held.local(bytes) {
                    Ok(index) => Some(index),
                    Err(error) => return Err(held.fail(error)),
                },
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
                Ends {
                    input,
                    mapped,
                    scanned,
                    output,
                },
                &Modules {
                    blocks: &blocks,
                    blocks_exclusive: &blocks_exclusive,
                    top: &top,
                    add: &add,
                    map,
                },
            ) {
                return Err(held.fail(error));
            }

            // **Two derivations of the same number, made to agree.** The plan says how many
            // dispatches a scan of this depth takes; the loop above emits them one at a time. If
            // those ever disagree the scanner is running a different algorithm from the one that
            // was planned, and the difference would show up as a wrong answer at some depth rather
            // than as a failure here.
            if held.pipelines.len() != plan::dispatches(levels.len(), map.is_some()) {
                return Err(held.fail(Error::NoPipeline));
            }

            Ok(held.into_scanner(elements, output))
        }
    }
}

/// The modules a scan runs, so the recording below takes one argument rather than five.
struct Modules<'a> {
    blocks: &'a [u32],
    blocks_exclusive: &'a [u32],
    top: &'a [u32],
    add: &'a [u32],
    /// One elementwise pass over the input first, and `None` when there is none.
    map: Option<&'a [u32]>,
}

/// The buffers at the ends of the chain — the input's own, rather than a level's.
///
/// Grouped for the same reason `Modules` is: `record` was growing an argument per buffer, and a
/// caller passing `scanned` where `output` belongs would build a scanner that scans its own answer.
#[derive(Clone, Copy)]
struct Ends {
    /// What the host writes.
    input: usize,
    /// What the map writes and the first scan reads, when there is a map.
    mapped: Option<usize>,
    /// The input's blocks, scanned from their own starts.
    scanned: usize,
    /// Where the answer lands.
    output: usize,
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
        ends: Ends,
        modules: &Modules<'_>,
    ) -> Result<(), Error> {
        let Ends {
            input,
            mapped,
            scanned,
            output,
        } = ends;

        let words = size_of::<f32>() as u64;
        let elements = self.buffers.get(input).map_or(0, Buffer::capacity) as u64 * words;
        let workgroups =
            (self.buffers.get(input).map_or(0, Buffer::capacity) / WORKGROUP_SIZE as usize) as u32;

        // The map, when there is one: elementwise over the whole input, writing where the first
        // block scan will read. Its output never crosses the bus, which is the whole point.
        let first_read = match (modules.map, mapped) {
            (Some(map), Some(mapped)) => {
                // SAFETY: as the passes below — both indices name buffers this builder allocated.
                unsafe {
                    self.pass(map, &[(input, elements), (mapped, elements)], workgroups)?;
                }
                mapped
            }
            // A map with nowhere to write, or a buffer with no map, is a construction bug rather
            // than a caller's mistake — the two are decided together in `build_scanner`.
            (Some(_), None) | (None, Some(_)) => return Err(Error::NoPipeline),
            (None, None) => input,
        };

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
                    (first_read, elements),
                    (scanned, elements),
                    (first_slots.totals, level_bytes(first)),
                ],
                workgroups,
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
                workgroups,
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
        self.run(input).map(|(answer, _)| answer)
    }

    /// The same scan, reporting how long each pass took on the device's own clock.
    ///
    /// **The deepest chain here, and the only one where a per-pass profile can say something a
    /// reduction's could not.** A reduction's passes shrink by a fixed factor and all do the same
    /// kind of work; a scan's do three different kinds — block scans on the way up, one workgroup
    /// at the top, offset additions on the way down — and the way down reads buffers the way up
    /// wrote.
    ///
    /// One timestamp per dispatch, written into the chain's own command buffer, so each pass is
    /// measured beside the passes it actually runs beside. `runner/examples/reducer.rs` records
    /// what that correction was worth for the reduction: a probe had the step cost five times too
    /// high.
    ///
    /// The vector is empty on a device with no usable timestamp queries, which is a thing to
    /// report rather than a zero to print.
    ///
    /// # Errors
    ///
    /// As [`Scanner::scan`].
    pub fn scan_timed(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<Duration>), Error> {
        self.run(input)
    }

    /// Both of the above: the answer, and where the time went.
    fn run(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<Duration>), Error> {
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
        let (output, spans) = unsafe {
            let upload = deliver_floats(self.gpu, input, staging, source)?;

            let spans = self.gpu.replay_timed(
                &self.pipelines,
                &self.workgroups,
                upload,
                Some(Staged {
                    from: answer,
                    to: staging,
                    bytes,
                }),
            )?;
            (staging.read(self.gpu, self.elements)?, spans)
        };

        Ok((output.into_iter().map(f32::from_bits).collect(), spans))
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
