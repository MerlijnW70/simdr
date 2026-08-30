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
