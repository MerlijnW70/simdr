use super::pipeline::Pipeline;
use super::step::{Ends, Pass, answer_in_destination};
use crate::buffer::Buffer;
use crate::{Error, Gpu};
use ash::vk;
use std::time::Duration;

impl Gpu {
    pub fn run_chain(&self, passes: &[Pass<'_>], input: &[u32]) -> Result<Vec<u32>, Error> {
        self.run_chain_head(passes, input, input.len())
    }

    pub fn run_chain_head(
        &self,
        passes: &[Pass<'_>],
        input: &[u32],
        head: usize,
    ) -> Result<Vec<u32>, Error> {
        if passes.is_empty() {
            return Err(Error::NoPipeline);
        }

        for pass in passes {
            let grid = super::Grid::linear(pass.workgroups);
            if let Some(overrun) =
                super::extent::Bounds::of(pass.spirv, &super::Specialization::none())
                    .overrun_uniform(grid, input.len())
            {
                return Err(overrun.into());
            }
        }

        let count = head.min(input.len()).max(1);
        let bytes = (input.len().max(1) * size_of::<u32>()) as u64;

        // SAFETY: every object below is created here and destroyed before returning, and each is
        // used only between a submission and the fence that completes it.
        unsafe {
            let staging = Buffer::staging(self, bytes)?;
            let source = Buffer::device_local(self, bytes)?;
            let destination = Buffer::device_local(self, bytes)?;

            let upload = super::deliver(self, input, &staging, &source)?;
            let outcome = self.chained_run(
                passes,
                Workspace {
                    staging: &staging,
                    source: &source,
                    destination: &destination,
                    bytes,
                },
                count,
                upload,
            );

            staging.destroy(self);
            source.destroy(self);
            destination.destroy(self);
            outcome
        }
    }

    unsafe fn chained_run(
        &self,
        passes: &[Pass<'_>],
        buffers: Workspace<'_>,
        count: usize,
        upload: Option<Staged<'_>>,
    ) -> Result<Vec<u32>, Error> {
        let Workspace {
            staging,
            source,
            destination,
            bytes,
        } = buffers;

        let mut pipelines = Vec::with_capacity(passes.len());
        for (index, pass) in passes.iter().enumerate() {
            let (read, written) = Ends::of(index).order(source, destination);
            // SAFETY: this function's contract says the buffers are live and the device idle with
            // respect to them, which is what pointing descriptors at them requires. Every pipeline
            // built here is destroyed before returning, on both the error and the success path.
            match unsafe {
                Pipeline::new(
                    self,
                    pass.spirv,
                    &[(read, bytes), (written, bytes)],
                    &super::Specialization::none(),
                )
            } {
                Ok(pipeline) => pipelines.push(pipeline),
                Err(error) => {
                    for pipeline in pipelines {
                        // SAFETY: nothing was submitted — the failure happened while building, so
                        // none of these has ever been dispatched.
                        unsafe { pipeline.destroy(self) };
                    }
                    return Err(error);
                }
            }
        }

        let groups: Vec<u32> = passes.iter().map(|pass| pass.workgroups).collect();
        let answer = if answer_in_destination(passes.len()) {
            destination
        } else {
            source
        };
        let home = (count.max(1) * size_of::<u32>()) as u64;

        // SAFETY: every pipeline in `pipelines` was built above and is still live, the buffers are
        // the caller's and outlive the call, and `replay` waits on its fence before returning.
        let recorded = unsafe {
            self.replay(
                &pipelines,
                &groups,
                upload,
                Some(Staged {
                    from: answer,
                    to: staging,
                    bytes: home.min(bytes),
                }),
            )
        };

        // SAFETY: `replay` waited on the submission that wrote it, so the answer is in staging
        // rather than still in flight — which is exactly what `Buffer::read` asks.
        let output = recorded.and_then(|()| unsafe { staging.read(self, count) });

        for pipeline in pipelines {
            // SAFETY: the one submission that used them has completed, as above.
            unsafe { pipeline.destroy(self) };
        }
        output
    }
}

#[derive(Clone, Copy)]
struct Workspace<'a> {
    staging: &'a Buffer,
    source: &'a Buffer,
    destination: &'a Buffer,
    bytes: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct Staged<'a> {
    pub(crate) from: &'a Buffer,
    pub(crate) to: &'a Buffer,
    pub(crate) bytes: u64,
}

impl Gpu {
    pub(crate) unsafe fn replay(
        &self,
        pipelines: &[Pipeline],
        workgroups: &[u32],
        before: Option<Staged<'_>>,
        after: Option<Staged<'_>>,
    ) -> Result<(), Error> {
        // SAFETY: forwarded unchanged; this function's contract is `replay_timed`'s.
        unsafe { self.replay_timed(pipelines, workgroups, before, after) }.map(|_| ())
    }

    pub(crate) unsafe fn replay_timed(
        &self,
        pipelines: &[Pipeline],
        workgroups: &[u32],
        before: Option<Staged<'_>>,
        after: Option<Staged<'_>>,
    ) -> Result<Vec<Duration>, Error> {
        let marks = pipelines.len() as u32;
        // SAFETY: `record_and_wait` asks that whatever the closure names outlive the submission.
        // It names the pipelines and the buffers in `before`/`after`, all of which are the
        // caller's and live for this call by its own contract, and the wait happens inside.
        let recorded = unsafe {
            self.record_and_wait(marks, |device, command, clock| {
                if let Some(upload) = before {
                    let region = [vk::BufferCopy::default().size(upload.bytes)];
                    device.cmd_copy_buffer(command, upload.from.handle, upload.to.handle, &region);
                    across(device, command, TRANSFER_TO_SHADER);
                }

                for (index, (pipeline, groups)) in pipelines.iter().zip(workgroups).enumerate() {
                    if index > 0 {
                        barrier(device, command);
                    }

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
                    device.cmd_dispatch(command, (*groups).max(1), 1, 1);

                    if let Some(clock) = clock {
                        // SAFETY: covered by the block this closure runs inside — the command
                        // buffer is recording, and the index is below the count `marks` promised
                        // the pool.
                        clock.mark(self, command, index as u32);
                    }
                }

                if let Some(download) = after {
                    across(device, command, SHADER_TO_TRANSFER);
                    let region = [vk::BufferCopy::default().size(download.bytes)];
                    device.cmd_copy_buffer(
                        command,
                        download.from.handle,
                        download.to.handle,
                        &region,
                    );
                }
                Ok(())
            })
        };

        recorded.map(|recorded| recorded.spans)
    }
}

type Stages = (
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
    vk::AccessFlags,
    vk::AccessFlags,
);

const TRANSFER_TO_SHADER: Stages = (
    vk::PipelineStageFlags::TRANSFER,
    vk::PipelineStageFlags::COMPUTE_SHADER,
    vk::AccessFlags::TRANSFER_WRITE,
    vk::AccessFlags::SHADER_READ,
);

const SHADER_TO_TRANSFER: Stages = (
    vk::PipelineStageFlags::COMPUTE_SHADER,
    vk::PipelineStageFlags::TRANSFER,
    vk::AccessFlags::SHADER_WRITE,
    vk::AccessFlags::TRANSFER_READ,
);

fn across(device: &ash::Device, command: vk::CommandBuffer, stages: Stages) {
    let (from, to, written, read) = stages;
    let memory = [vk::MemoryBarrier::default()
        .src_access_mask(written)
        .dst_access_mask(read)];

    // SAFETY: the command buffer is recording, and a memory barrier names no resource that could
    // have been destroyed.
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            from,
            to,
            vk::DependencyFlags::empty(),
            &memory,
            &[],
            &[],
        );
    }
}

fn barrier(device: &ash::Device, command: vk::CommandBuffer) {
    let memory = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];

    // SAFETY: the command buffer is recording, and a memory barrier names no resource that could
    // have been destroyed.
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &memory,
            &[],
            &[],
        );
    }
}
