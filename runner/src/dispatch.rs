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
//! down. [`run`] is the surface over it — one call per way a caller might spell its data. The rest
//! are the pieces each of those needs: [`pipeline`] and [`specialization`] to build one, [`grid`]
//! to say how many workgroups on how many axes, [`submit`] to record and wait, [`session`] and
//! [`chain`] to keep things alive across calls.

mod bindings;
mod chain;
mod grid;
mod pipeline;
mod placement;
mod run;
mod session;
mod specialization;
mod submit;
mod timestamps;

pub use chain::Pass;
pub use grid::Grid;
pub use placement::{MemoryType, Placement};
pub use session::Session;
pub use specialization::Specialization;

pub(crate) use pipeline::Pipeline;

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
        unsafe { self.copy(staging, source, bytes) }?;

        let elapsed = unsafe { self.dispatch(&pipeline, grid, iterations) }?;

        unsafe { self.copy(destination, staging, bytes) }?;
        let output = unsafe { staging.read(self, count) }?;

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
        unsafe {
            self.record_and_wait(|device, command| {
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
        }
    }
}
