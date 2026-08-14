//! A reduction that keeps its pipelines and its buffers between calls.
//!
//! [`crate::Gpu::sum`] builds a pipeline per fold on every call and throws them all away. That is
//! the right shape for a test and the wrong one for anything that reduces more than once:
//! `runner/examples/reducer.rs` measures the same reduction at about **1000 µs** rebuilt and
//! **200 µs** held over 8 192 elements — 5.0× — and **2.2×** over 2²⁰, where the setup is a
//! smaller share of a larger call.
//!
//! That last clause used to read "because by then the arithmetic is most of the time", and the
//! arithmetic is not. Broken down, a held reduction over 2²⁰ is mostly **the host writing its
//! input** — around 70% of it once everything else had been taken out — with the chained
//! dispatches and the single submission accounting for most of the rest. `notes/FINDINGS.md` has
//! the table, and `runner/examples/reducer.rs` prints a fresh one on whatever device it is run on.
//!
//! Getting there took four passes, and the reduction over 2²⁰ went from ~1930 µs to ~280 µs on an
//! RTX 4080. Only the last of them touched the arithmetic. The others removed a download of the
//! whole buffer to read one word out of it, a `Vec<u32>` built to reinterpret bits that were
//! already the right bits, two of three submissions, and — this pass — the staging copy, by
//! writing the input into memory the device could already read.
//!
//! This is the same trade [`crate::Session`] makes for one pipeline, applied to a chain of them.
//!
//! # Why it is built for a length
//!
//! How many folds a reduction needs depends on how many elements it is reducing, so the pipelines
//! and the buffers are both sized when the [`Reducer`] is made. A different length needs a
//! different one. That is a real limit and it is stated in the type rather than hidden behind a
//! resize that would quietly rebuild everything the object exists to keep.
//!
//! # What owns what, and why together
//!
//! A pipeline holds a descriptor set, and a descriptor set points at particular buffers. Caching
//! pipelines apart from the buffers they were built against would be a use-after-free written in
//! safe-looking code — the buffers would drop, the descriptors would still name them, and the next
//! dispatch would read freed memory. So one type owns both, and its `Drop` releases them in order.

use super::Reduction;
use crate::buffer::Buffer;
use crate::dispatch::{Ends, Pipeline, Staged, answer_in_destination, deliver_floats};
use crate::{Error, Gpu};
use std::time::Duration;

/// How much of the answer buffer comes home: one `f32`.
///
/// Every invocation of the final workgroup holds the whole total, so slot zero is the answer and
/// the rest of the buffer is the last fold's leftovers. Reading it all was 37% of a reduction over
/// 2²⁰ elements — see `notes/FINDINGS.md`.
const ANSWER_BYTES: u64 = size_of::<f32>() as u64;

/// A reduction over a fixed number of elements, with its pipelines already built.
pub struct Reducer<'gpu> {
    gpu: &'gpu Gpu,
    /// How many elements this was built for. A different count needs a different `Reducer`.
    elements: usize,
    /// The host's way in and out.
    ///
    /// `Option` because [`Buffer::destroy`] consumes the buffer and `Drop` has only `&mut self`.
    /// `Some` for the whole life of the reducer and `None` only while dropping.
    staging: Option<Buffer>,
    source: Option<Buffer>,
    destination: Option<Buffer>,
    /// One per pass, in order: the folds, then the workgroup reduction that finishes.
    pipelines: Vec<Pipeline>,
    /// How many workgroups each pass dispatches, in the same order.
    ///
    /// Kept beside the pipelines rather than recomputed, because `folds` is arithmetic on the
    /// element count and recomputing it in two places is how the two come to disagree.
    workgroups: Vec<u32>,
}

impl Gpu {
    /// Build every pipeline a reduction over `elements` needs, and hold them.
    ///
    /// `elements` must be a power of two and at least `2 × WORKGROUP_SIZE`, exactly as
    /// [`Gpu::sum`] requires — a reducer that accepted a length it could not then reduce would
    /// move the failure from here to the first call.
    ///
    /// # Errors
    ///
    /// [`Error::BadLength`] if `elements` is not a shape this can fold, [`Error::Emit`] if a pass
    /// cannot be built, otherwise as [`Gpu::run`].
    pub fn reducer<'gpu>(&'gpu self, elements: usize) -> Result<Reducer<'gpu>, Error> {
        self.build_reducer(elements, None)
    }

    /// The same, with one elementwise pass of `map` run over the input first.
    ///
    /// **This is what removes an upload and a download.** Σ f(x) over a device the caller cannot
    /// reach otherwise costs three host crossings: send the input, run `f`, bring the result home,
    /// send it back, reduce. Two of those are 4 MB each at 2²⁰ elements, and
    /// `runner/examples/reducer.rs` prices a 4 MB upload at ~190 µs and a download at ~710 µs on
    /// an RTX 4080 — the download being the more expensive direction because the memory a host
    /// reads back is uncached.
    ///
    /// Here `map` is simply the first pass of the same chain. Its output never leaves the device —
    /// the ping-pong hands it straight to the first fold — so the intermediate crossing does not
    /// happen rather than happening faster.
    ///
    /// `map` must be a two-binding kernel built for [`crate::kernels::WORKGROUP_SIZE`]
    /// invocations, reading
    /// binding 0 and writing binding 1, and it must write **every** element the first fold reads —
    /// which for an elementwise kernel over the whole input it does. `elements / WORKGROUP_SIZE`
    /// workgroups of it are dispatched, computed here rather than taken as an argument so that the
    /// count cannot disagree with the length the folds were built for.
    ///
    /// # Errors
    ///
    /// As [`Gpu::reducer`].
    pub fn reducer_of<'gpu>(
        &'gpu self,
        elements: usize,
        map: &[u32],
    ) -> Result<Reducer<'gpu>, Error> {
        self.build_reducer(elements, Some(map))
    }

    /// The construction both of them share.
    ///
    /// What to run is decided in [`super::plan`], which has no `unsafe` in it and is therefore
    /// inside the mutation gate; this half owns the Vulkan objects and is not.
    fn build_reducer<'gpu>(
        &'gpu self,
        elements: usize,
        map: Option<&[u32]>,
    ) -> Result<Reducer<'gpu>, Error> {
        // Every module first, so a build failure happens before anything is allocated.
        let stages = super::plan::stages(self.limits().subgroup_size, elements, map)?;

        let bytes = (elements.max(1) * size_of::<u32>()) as u64;
        let workgroups: Vec<u32> = stages.iter().map(|stage| stage.workgroups).collect();

        // SAFETY: everything allocated here is owned by the `Reducer` and destroyed in its `Drop`,
        // which cannot run while a dispatch is in flight — every dispatch waits on a fence before
        // returning. The early returns below release what was allocated before them.
        unsafe {
            let staging = Buffer::staging(self, bytes)?;
            // The source is the one buffer the host writes, so it is the one worth asking to be
            // reachable from both sides — see `Buffer::shared`. The destination is written only by
            // the device and read only by the device, and asking for a BAR window to hold it would
            // spend a scarce heap on a buffer no host ever touches.
            let source = match Buffer::shared(self, bytes) {
                Ok(buffer) => buffer,
                Err(error) => {
                    staging.destroy(self);
                    return Err(error);
                }
            };
            let destination = match Buffer::device_local(self, bytes) {
                Ok(buffer) => buffer,
                Err(error) => {
                    source.destroy(self);
                    staging.destroy(self);
                    return Err(error);
                }
            };

            let mut pipelines = Vec::with_capacity(stages.len());
            for (index, stage) in stages.iter().enumerate() {
                // The pair alternates: pass 0 reads the source and writes the destination, pass 1
                // the other way round. It is the descriptor set that decides, so the modules are
                // untouched — `fold_halves` has no idea which buffer it is reading.
                let (read, written) = Ends::of(index).order(&source, &destination);
                match Pipeline::new(
                    self,
                    &stage.words,
                    &[(read, bytes), (written, bytes)],
                    &crate::Specialization::none(),
                ) {
                    Ok(pipeline) => pipelines.push(pipeline),
                    Err(error) => {
                        for pipeline in pipelines {
                            pipeline.destroy(self);
                        }
                        destination.destroy(self);
                        source.destroy(self);
                        staging.destroy(self);
                        return Err(error);
                    }
                }
            }

            Ok(Reducer {
                gpu: self,
                elements,
                staging: Some(staging),
                source: Some(source),
                destination: Some(destination),
                pipelines,
                workgroups,
            })
        }
    }
}

impl Reducer<'_> {
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

    /// The same sum, reporting how long each pass took on the device's own clock.
    ///
    /// **This is what a breakdown should be made of.** `runner/examples/reducer.rs` used to time
    /// each row on its own — a chain of empty kernels for the dispatch cost, a bare submit-and-wait
    /// for the submission, a `Session` write for the upload — and the rows came to 123% of the call
    /// they described, because each probe paid fixed costs the real call pays once between them
    /// all.
    ///
    /// These are timestamps written *into the chain's own command buffer*, between the dispatches
    /// that actually run. Pass `i` is measured beside the passes it runs beside.
    ///
    /// The vector is empty on a device with no usable timestamp queries, which is a thing to
    /// report rather than a zero to print.
    ///
    /// # Errors
    ///
    /// As [`Reducer::sum`].
    pub fn sum_timed(&mut self, input: &[f32]) -> Result<(Reduction, Vec<Duration>), Error> {
        self.run(input)
    }

    /// Sum every element of `input`, reusing the pipelines and the buffers.
    ///
    /// `input.len()` must equal [`Reducer::elements`]. A shorter slice would leave the tail of the
    /// buffer holding whatever the last call put there, and the answer would be that call's data
    /// added to this one's — which is a wrong number rather than an error, so it is refused.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if `input` is not the length this was built for, otherwise as
    /// [`Gpu::run`].
    pub fn sum(&mut self, input: &[f32]) -> Result<Reduction, Error> {
        self.run(input).map(|(reduction, _)| reduction)
    }

    /// Both of the above: the answer, and where the time went.
    fn run(&mut self, input: &[f32]) -> Result<(Reduction, Vec<Duration>), Error> {
        if input.len() != self.elements {
            return Err(Error::TooLarge {
                words: input.len(),
                capacity: self.elements,
            });
        }

        let (Some(staging), Some(source), Some(destination)) = (
            self.staging.as_ref(),
            self.source.as_ref(),
            self.destination.as_ref(),
        ) else {
            return Err(Error::NoPipeline);
        };

        // SAFETY: every buffer and pipeline here is owned by `self` and outlives the call, and the
        // submission waits on a fence before returning.
        let (output, spans) = unsafe {
            // Straight from the caller's slice, and — where the device has memory both sides can
            // reach — straight into the buffer the first pass reads. `deliver_floats` picks, and
            // returns the copy that is left to record, which on such a device is none.
            //
            // The slice half came first: this used to build a `Vec<u32>` of the whole input to
            // reinterpret bits that were already the right bits — four megabytes allocated and
            // copied per call, **52%** of it by measurement.
            let upload = deliver_floats(self.gpu, input, staging, source)?;

            // The buffers alternate, so which one holds the answer depends on how many passes ran.
            // Reading the wrong one returns the *second to last* fold — a plausible number, and
            // roughly twice the right one.
            let answer = if answer_in_destination(self.pipelines.len()) {
                destination
            } else {
                source
            };

            // **One submission, not three.** The upload and the answer used to be `Gpu::copy`
            // calls either side of this, and a `copy` is a whole command buffer, submission and
            // fence — about 65 µs each against a 540 µs call. Recorded inside the chain they cost
            // a barrier apiece.
            //
            // The answer is **one word**, not the buffer: a reduction produces a single number and
            // this used to bring whole megabytes home to call `.first()` on them.
            let spans = self.gpu.replay_timed(
                &self.pipelines,
                &self.workgroups,
                upload,
                Some(Staged {
                    from: answer,
                    to: staging,
                    bytes: ANSWER_BYTES,
                }),
            )?;
            (staging.read(self.gpu, 1)?, spans)
        };

        let total = output
            .first()
            .copied()
            .map(f32::from_bits)
            .ok_or(Error::NoPipeline)?;

        Ok((
            Reduction {
                total,
                dispatches: self.pipelines.len(),
                host_combined: 1,
            },
            spans,
        ))
    }
}

impl Drop for Reducer<'_> {
    fn drop(&mut self) {
        // SAFETY: every object here was created by `Gpu::reducer` and nothing else holds it. The
        // device is idle with respect to them: `sum` waits on a fence before returning, so nothing
        // can still be in flight. Pipelines go first — a descriptor set naming a destroyed buffer
        // would be a dangling reference for as long as it existed.
        unsafe {
            for pipeline in std::mem::take(&mut self.pipelines) {
                pipeline.destroy(self.gpu);
            }
            if let Some(buffer) = self.destination.take() {
                buffer.destroy(self.gpu);
            }
            if let Some(buffer) = self.source.take() {
                buffer.destroy(self.gpu);
            }
            if let Some(staging) = self.staging.take() {
                staging.destroy(self.gpu);
            }
        }
    }
}
