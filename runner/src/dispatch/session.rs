//! Buffers and a pipeline that outlive one dispatch.
//!
//! [`crate::Gpu::run`] allocates three buffers, builds a pipeline, submits three times and throws
//! all of it away. That is the right shape for a test — nothing leaks between cases, and every run
//! starts from the same state — and it is the wrong shape for anything that asks more than once.
//!
//! # What it costs, measured
//!
//! `examples/overhead.rs` timed an *empty* kernel over a 256-byte buffer at ~875 us a round trip,
//! against 0.8 us for the same dispatch amortised over a thousand of them. Allocating and freeing
//! one buffer costs ~310 us on this device whatever its size, and a run does three.
//!
//! So better than 99% of a small call is setup. A [`Session`] pays it once.
//!
//! # What it does not do
//!
//! It does not make the *dispatch* faster, and it does not remove the host copies — a caller that
//! writes the same data every time is still crossing the bus every time. It removes allocation and
//! pipeline creation, which is what the measurement said was there to remove.

use super::pipeline::Pipeline;
use crate::buffer::Buffer;
use crate::{Error, Gpu};
use std::time::Duration;

/// A kernel with its buffers, ready to be dispatched repeatedly.
///
/// Bindings are `0..n-1` in the order the sizes were given, matching [`Gpu::run_bound`]. Which of
/// them a kernel reads and which it writes is the kernel's business; this holds the memory and
/// says nothing about it.
pub struct Session<'gpu> {
    gpu: &'gpu Gpu,
    /// What the module needs of each binding, read once when the session was built.
    ///
    /// Kept rather than recomputed because the two halves of the question arrive apart: the module
    /// is known here and the workgroup count only at [`Session::dispatch`]. Decoding it per
    /// dispatch would also make the check cost something on the path whose entire purpose is that
    /// nothing costs anything after setup.
    bounds: super::extent::Bounds,
    /// The host's way in and out. One, reused, sized to the largest binding.
    ///
    /// `Option` because [`Buffer::destroy`] consumes the buffer and `Drop` has only `&mut self`.
    /// It is `Some` for the whole life of the session and `None` only while dropping.
    staging: Option<Buffer>,
    /// Device-local, one per binding.
    ///
    /// Each knows its own size, so nothing here keeps a parallel list of lengths — one that could
    /// drift from the buffers it describes.
    buffers: Vec<Buffer>,
    /// Built once. Creating one costs far more than running one.
    pipeline: Option<Pipeline>,
}

impl Gpu {
    /// Hold `spirv` and one device-local buffer per entry of `sizes`, in words.
    ///
    /// Everything expensive happens here: three allocations and a pipeline for a two-binding
    /// kernel, and none of it again until the session is dropped.
    ///
    /// # Errors
    ///
    /// [`Error::NoPipeline`] if `sizes` is empty, otherwise as [`Gpu::run`].
    pub fn session<'gpu>(
        &'gpu self,
        spirv: &[u32],
        sizes: &[usize],
    ) -> Result<Session<'gpu>, Error> {
        if sizes.is_empty() {
            return Err(Error::NoPipeline);
        }

        let bytes: Vec<u64> = sizes
            .iter()
            .map(|&words| (words.max(1) * size_of::<u32>()) as u64)
            .collect();
        let staging_bytes = bytes.iter().copied().max().unwrap_or(4);

        // SAFETY: everything allocated here is owned by the `Session` and destroyed in its `Drop`,
        // which cannot run while a dispatch is in flight because every dispatch waits on a fence
        // before returning.
        unsafe {
            let staging = Buffer::staging(self, staging_bytes)?;

            let mut buffers = Vec::with_capacity(bytes.len());
            for &size in &bytes {
                // Host-writable where the device offers it, so that `Session::write` can put the
                // caller's words in the binding itself. `Buffer::shared` falls back to plain
                // device-local per buffer, which matters here more than anywhere else: a session
                // can hold several large bindings, and the memory both sides can reach is often a
                // small window. The ones that fit it get it and the rest stage as before.
                match Buffer::shared(self, size) {
                    Ok(buffer) => buffers.push(buffer),
                    Err(error) => {
                        for buffer in buffers {
                            buffer.destroy(self);
                        }
                        staging.destroy(self);
                        return Err(error);
                    }
                }
            }

            let bound: Vec<(&Buffer, u64)> = buffers
                .iter()
                .zip(&bytes)
                .map(|(buffer, &size)| (buffer, size))
                .collect();

            match Pipeline::new(self, spirv, &bound, &super::Specialization::none()) {
                Ok(pipeline) => Ok(Session {
                    gpu: self,
                    bounds: super::extent::Bounds::of(spirv),
                    staging: Some(staging),
                    buffers,
                    pipeline: Some(pipeline),
                }),
                Err(error) => {
                    for buffer in buffers {
                        buffer.destroy(self);
                    }
                    staging.destroy(self);
                    Err(error)
                }
            }
        }
    }
}

impl Session<'_> {
    /// How many bindings this holds.
    #[must_use]
    pub fn bindings(&self) -> usize {
        self.buffers.len()
    }

    /// Copy `words` into binding `index`.
    ///
    /// `&mut self`, so two writes cannot overlap — which is what a `&self` receiver would let a
    /// caller believe. That is the reason for the receiver, and it holds either way; what varies
    /// underneath is only the route. Where the binding is host-writable the words go into it
    /// directly and no submission happens at all; otherwise they go through the shared staging
    /// buffer and one copy. `Buffer::shared` says which devices offer which, and
    /// `runner/examples/reducer.rs` measures the difference at about 30% of a 4 MB reduction.
    ///
    /// # Errors
    ///
    /// [`Error::NoPipeline`] if `index` names no binding, [`Error::TooLarge`] if `words` is longer
    /// than that binding holds, otherwise as [`Gpu::run`].
    pub fn write(&mut self, index: usize, words: &[u32]) -> Result<(), Error> {
        let (Some(target), Some(staging)) = (self.buffers.get(index), self.staging.as_ref()) else {
            return Err(Error::NoPipeline);
        };

        // Writing nothing writes nothing. `fitting` floors at one word because a zero-byte
        // `vkCmdCopyBuffer` is not allowed, and copying that one word would put whatever the
        // staging buffer last held into the caller's binding — a side effect from a call that
        // asked for none.
        if words.is_empty() {
            return Ok(());
        }

        // Against the *binding's* capacity, not the staging buffer's. Staging is sized to the
        // largest binding, so clamping to it — which this did — would silently write a prefix
        // into a smaller binding and report success. A short write is a wrong answer arriving
        // later, and this crate refuses rather than truncates everywhere else.
        let bytes = fitting(words.len(), target.capacity())?;

        // SAFETY: both buffers are this session's and no dispatch is in flight — every one of them
        // waits on its fence before returning.
        unsafe {
            match super::deliver(self.gpu, words, staging, target)? {
                // Staged: the copy is a submission of its own, as it has always been.
                Some(_) => self.gpu.copy(staging, target, bytes),
                // Written into the binding itself. No copy, and therefore no submission and no
                // fence to wait on — the words are simply there. A host write to coherent memory
                // is made visible to the device by the next queue submission, which is the
                // dispatch this write was preparing for.
                None => Ok(()),
            }
        }
    }

    /// Read `count` words back from binding `index`.
    ///
    /// # Errors
    ///
    /// As [`Session::write`].
    pub fn read(&mut self, index: usize, count: usize) -> Result<Vec<u32>, Error> {
        let (Some(source), Some(staging)) = (self.buffers.get(index), self.staging.as_ref()) else {
            return Err(Error::NoPipeline);
        };
        let bytes = fitting(count, source.capacity())?;

        // SAFETY: as above.
        unsafe {
            self.gpu.copy(source, staging, bytes)?;
            staging.read(self.gpu, count)
        }
    }

    /// Dispatch `workgroups` groups, `iterations` times, and report what the device's clock said.
    ///
    /// No allocation and no pipeline creation: that is the whole point of the type.
    ///
    /// # Errors
    ///
    /// [`Error::NoPipeline`] if the session has been torn down, otherwise as [`Gpu::run`].
    pub fn dispatch(&mut self, workgroups: u32, iterations: u32) -> Result<Duration, Error> {
        self.dispatch_grid(super::Grid::linear(workgroups), iterations)
    }

    /// The same, over both axes.
    ///
    /// # Errors
    ///
    /// As [`Session::dispatch`].
    pub fn dispatch_grid(&mut self, grid: super::Grid, iterations: u32) -> Result<Duration, Error> {
        // **Checked per dispatch, because that is when the count arrives.** A session's buffers are
        // fixed at construction and its workgroup count is not, so this is the one path where the
        // caller can ask for a dispatch too large for buffers that were the right size a moment
        // ago. It had no check of any kind until this was written.
        let held: Vec<usize> = self.buffers.iter().map(Buffer::capacity).collect();
        if let Some(overrun) = self.bounds.overrun(grid, &held) {
            return Err(overrun.into());
        }

        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(Error::NoPipeline);
        };

        // SAFETY: the pipeline and its buffers are alive for as long as `self` is, and this waits
        // on a fence before returning.
        unsafe { self.gpu.dispatch(pipeline, grid, iterations.max(1)) }
    }
}

/// How many bytes `words` occupies, once it is known to fit in `capacity` words.
///
/// A minimum of one byte: a zero-length copy is not an error and `vkCmdCopyBuffer` will not take a
/// size of zero.
fn fitting(words: usize, capacity: usize) -> Result<u64, Error> {
    if words > capacity {
        return Err(Error::TooLarge { words, capacity });
    }
    Ok((words.max(1) * size_of::<u32>()) as u64)
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        // SAFETY: every object here was created by `Gpu::session` and nothing else holds it. The
        // device is idle with respect to them: `dispatch`, `write` and `read` each wait on a fence
        // before returning, so nothing can still be in flight.
        unsafe {
            if let Some(pipeline) = self.pipeline.take() {
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
