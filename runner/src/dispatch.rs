//! Building a pipeline out of a SPIR-V module and running it.
//!
//! # What is timed, and what is not
//!
//! A run is three submissions: upload, dispatch, download. Only the middle one is timed, and the
//! kernel's buffers are device-local, so the number [`Gpu::time`] reports is the kernel reading
//! VRAM rather than the host's copies crossing the bus. Getting that wrong is what made two
//! earlier benchmarks meaningless — see `notes/FINDINGS.md`.
//!
//! # What is where
//!
//! This file is the staging machinery: allocate three buffers, copy in, dispatch, copy out, tear
//! down. `run` is the surface over it — one call per way a caller might spell its data. The rest
//! are the pieces each of those needs: `pipeline` and `specialization` to build one, `grid` to say
//! how many workgroups on how many axes, `step` to say what a chain hands each pass, `submit` to
//! record and wait, `session` and `chain` to keep things alive across calls, and `extent` to refuse
//! a dispatch that would run off the end of a binding.
//!
//! **Every one of them goes through `extent`.** It guarded this file's `execute` and nothing else
//! for four days, which is a sixth of the ways this crate dispatches; `extent`'s own header has
//! what that cost and what closing it needed.

mod bindings;
mod chain;
mod extent;
mod grid;
mod pipeline;
mod placement;
mod run;
mod session;
mod specialization;
mod step;
mod submit;
mod timestamps;
mod upload;

pub use grid::Grid;
pub use placement::{MemoryType, Placement};
pub use session::Session;
pub use specialization::Specialization;
pub use step::Pass;

pub(crate) use chain::Staged;
pub(crate) use extent::Bounds;
pub(crate) use pipeline::Pipeline;
pub(crate) use step::{Ends, answer_in_destination};
pub(crate) use upload::{deliver, deliver_floats};

use crate::buffer::Buffer;
use crate::{Error, Gpu};
use ash::vk;
use std::time::Duration;

impl Gpu {
    /// Upload, dispatch, download — returning the output and what the dispatch alone took.
    fn execute(
        &self,
        spirv: &[u32],
        input: &[u32],
        grid: Grid,
        iterations: u32,
        specialization: &Specialization,
    ) -> Result<(Vec<u32>, Duration), Error> {
        let count = input.len();
        let bytes = (count.max(1) * size_of::<u32>()) as u64;

        // **The output buffer is exactly as long as the input, so the dispatch has to fit in it.**
        // That equal-length rule is what makes this call a one-argument one, and it is also the
        // trap in it: nothing about `workgroups` is checked against `input.len()`, so a caller who
        // dispatches twice what their buffer holds gets a kernel writing off the end of it. That
        // is undefined behaviour — an access violation on one device here and plausible wrong
        // numbers on another — rather than an error.
        //
        // `extent::Bounds` reads the workgroup size out of the module and refuses instead. It is a
        // floor rather than a proof: see `dispatch::extent` for what it cannot catch.
        // **With the specialization this dispatch will actually use.** `Kernel::load_offset_by`
        // reaches past its run by a number chosen here rather than written into the module, and a
        // bound that read the module alone counted zero for it — permissively, which is the one
        // direction a refusal must never take.
        if let Some(overrun) =
            extent::Bounds::of(spirv, specialization).overrun_uniform(grid, count)
        {
            return Err(overrun.into());
        }

        // SAFETY: every object below is created here and destroyed before returning, and each is
        // used only between a submission and the fence that completes it.
        unsafe {
            let staging = Buffer::staging(self, bytes)?;
            let source = Buffer::device_local(self, bytes)?;
            let destination = Buffer::device_local(self, bytes)?;

            // The host's only way in. Forgetting this once made every computing kernel return
            // whatever the device memory happened to hold, and the empty-kernel test still
            // passed — which is why that one is not the floor it looks like.
            staging.write(self, input)?;

            let outcome = self.staged_run(
                spirv,
                &staging,
                &source,
                &destination,
                bytes,
                grid,
                count,
                iterations.max(1),
                specialization,
            );

            staging.destroy(self);
            source.destroy(self);
            destination.destroy(self);
            outcome
        }
    }

    /// The three submissions, with everything torn down afterwards.
    ///
    /// # Safety
    ///
    /// The buffers must be live and the device idle with respect to them.
    #[expect(
        clippy::too_many_arguments,
        reason = "every one is a distinct thing the run needs, and bundling them into a struct \
                  only moved the list somewhere else last time"
    )]
    unsafe fn staged_run(
        &self,
        spirv: &[u32],
        staging: &Buffer,
        source: &Buffer,
        destination: &Buffer,
        bytes: u64,
        grid: Grid,
        count: usize,
        iterations: u32,
        specialization: &Specialization,
    ) -> Result<(Vec<u32>, Duration), Error> {
        // SAFETY: this function's own contract says the buffers are live and the device idle with
        // respect to them, which is what `Pipeline::new` needs of the ones it points descriptors
        // at. The pipeline is destroyed at the end of this function.
        let pipeline = unsafe {
            Pipeline::new(
                self,
                spirv,
                &[(source, bytes), (destination, bytes)],
                specialization,
            )
        }?;

        // Upload: the host's words are already in `staging`; copy them where the kernel can see
        // them. Untimed, because a benchmark of PCIe is not what anyone asked for.
        // SAFETY: both buffers are the caller's, alive for this call, and nothing is using them —
        // `copy` waits on its own fence before returning, so the dispatch below cannot overlap it.
        unsafe { self.copy(staging, source, bytes) }?;

        // SAFETY: the pipeline was built above and outlives the submission, which `dispatch` waits
        // for before returning.
        let elapsed = unsafe { self.dispatch(&pipeline, grid, iterations) }?;

        // SAFETY: as the upload copy. The dispatch has completed — it waited on its fence — so the
        // destination holds the kernel's output rather than a partial write.
        unsafe { self.copy(destination, staging, bytes) }?;
        // SAFETY: the copy above waited on its fence, so the staging buffer holds the whole
        // result, which is exactly what `Buffer::read` asks of its caller.
        let output = unsafe { staging.read(self, count) }?;

        // SAFETY: every submission that used this pipeline waited on a fence before returning, so
        // none is in flight.
        unsafe { pipeline.destroy(self) };
        Ok((output, elapsed))
    }

    /// Record `iterations` dispatches and time the submission.
    ///
    /// # Safety
    ///
    /// The pipeline must be live.
    unsafe fn dispatch(
        &self,
        pipeline: &Pipeline,
        grid: Grid,
        iterations: u32,
    ) -> Result<Duration, Error> {
        let (x, y) = grid.counts();
        // SAFETY: `record_and_wait` asks that whatever the closure names outlive the submission.
        // It names only `pipeline`, which this function's contract says is live, and the call
        // waits for the submission before returning.
        unsafe {
            self.record_and_wait(0, |device, command, _| {
                device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.handle(),
                );
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout(),
                    0,
                    &[pipeline.descriptors()],
                    &[],
                );

                for iteration in 0..iterations {
                    if iteration > 0 {
                        // Keep the dispatches from overlapping, so the elapsed time is the sum of
                        // their own rather than a measure of the scheduler's appetite.
                        let barrier = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(vk::AccessFlags::SHADER_READ)];
                        device.cmd_pipeline_barrier(
                            command,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &barrier,
                            &[],
                            &[],
                        );
                    }
                    device.cmd_dispatch(command, x, y, 1);
                }
                Ok(())
            })
            .map(|recorded| recorded.whole)
        }
    }
}
