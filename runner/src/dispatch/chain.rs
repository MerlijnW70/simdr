//! Several dispatches in a row, with the data staying on the device between them.
//!
//! A reduction that starts wider than one workgroup cannot finish in one dispatch: there is no
//! barrier across a dispatch in Vulkan, so the only way one workgroup reads another's output is a
//! second dispatch. Chaining them through the host would work and would measure the bus, which is
//! the mistake `notes/FINDINGS.md` records twice already. So the whole chain is one command
//! buffer, one submission, and the words never leave device memory until the end.
//!
//! # Feeding each pass its predecessor's output
//!
//! Every kernel this crate emits binds buffer 0 read and buffer 1 written, and those bindings are
//! baked into the module. Rather than build a second descriptor set and alternate, each pass is
//! followed by a device-to-device copy back into the source.
//!
//! **The copy is as long as the next pass will read, and no longer.** It used to be the whole
//! buffer every time, which for a shrinking reduction is mostly copying elements nobody will look
//! at again: fourteen copies of 4 MB where the fourteen passes between them read 4 MB in total.
//! `runner/examples/reducer.rs` measured that at ~20% of a held reduction over 2²⁰ elements — not
//! the majority `notes/NEXT.md` predicted, and not nothing either.
//!
//! [`Pass::new`] still copies everything, because a caller who has not said how much its pass
//! writes has not given anyone the right to guess. [`Pass::writing`] is where the saving is, and it
//! is what [`crate::Reducer`] and [`crate::Gpu::sum`] use — they know, because the fold sizes are
//! what they were built from.
//!
//! A ping-pong across two descriptor sets would remove the copy entirely and is still the right
//! shape for a chain built to be fast. This one is built to be correct.

use super::pipeline::Pipeline;
use super::step::{Pass, Step};
use crate::buffer::Buffer;
use crate::{Error, Gpu};
use ash::vk;

impl Gpu {
    /// Run every pass in order over one pair of device-local buffers, and read the result back.
    ///
    /// Each pass reads what the one before it wrote. The buffers are both `input.len()` words
    /// wide throughout — a shrinking chain simply stops reading the tail — so the caller sizes
    /// once, for the first pass.
    ///
    /// The returned vector is the whole output buffer, not just the part the last pass touched.
    /// Which prefix is meaningful is the caller's arithmetic, and it is the caller who knows it.
    ///
    /// # Errors
    ///
    /// [`Error::NoPipeline`] if `passes` is empty, otherwise as [`Gpu::run`].
    pub fn run_chain(&self, passes: &[Pass<'_>], input: &[u32]) -> Result<Vec<u32>, Error> {
        if passes.is_empty() {
            return Err(Error::NoPipeline);
        }

        let count = input.len();
        let bytes = (count.max(1) * size_of::<u32>()) as u64;

        // SAFETY: every object below is created here and destroyed before returning, and each is
        // used only between a submission and the fence that completes it.
        unsafe {
            let staging = Buffer::staging(self, bytes)?;
            let source = Buffer::device_local(self, bytes)?;
            let destination = Buffer::device_local(self, bytes)?;

            staging.write(self, input)?;
            let outcome = self.chained_run(passes, &staging, &source, &destination, bytes, count);

            staging.destroy(self);
            source.destroy(self);
            destination.destroy(self);
            outcome
        }
    }

    /// Upload, run the chain, download — with everything torn down afterwards.
    ///
    /// # Safety
    ///
    /// The buffers must be live and the device idle with respect to them.
    unsafe fn chained_run(
        &self,
        passes: &[Pass<'_>],
        staging: &Buffer,
        source: &Buffer,
        destination: &Buffer,
        bytes: u64,
        count: usize,
    ) -> Result<Vec<u32>, Error> {
        let mut pipelines = Vec::with_capacity(passes.len());
        for pass in passes {
            // Built before recording rather than inside it: a failure here must not leave a
            // half-recorded command buffer, and every pipeline has to outlive the submission.
            match unsafe {
                Pipeline::new(
                    self,
                    pass.spirv,
                    &[(source, bytes), (destination, bytes)],
                    &super::Specialization::none(),
                )
            } {
                Ok(pipeline) => pipelines.push(pipeline),
                Err(error) => {
                    for pipeline in pipelines {
                        unsafe { pipeline.destroy(self) };
                    }
                    return Err(error);
                }
            }
        }

        unsafe { self.copy(staging, source, bytes) }?;

        let steps = Step::plan(passes, bytes);
        let recorded = unsafe { self.replay(&pipelines, &steps, source, destination) };

        let output = recorded.and_then(|()| {
            unsafe { self.copy(destination, staging, bytes) }?;
            unsafe { staging.read(self, count) }
        });

        for pipeline in pipelines {
            unsafe { pipeline.destroy(self) };
        }
        output
    }
}

impl Gpu {
    /// Record every pipeline in order, copying `destination` back into `source` between them, and
    /// wait for the whole thing.
    ///
    /// The half of a chain that does not care where the pipelines came from. [`Gpu::run_chain`]
    /// builds them and throws them away; [`crate::Reducer`] keeps them. Both record the same
    /// sequence, and it lives here so there is one of it.
    ///
    /// `steps` is one entry per pipeline, in the same order, each saying how many workgroups to
    /// dispatch and how many bytes to hand it from the pass before.
    ///
    /// # Safety
    ///
    /// The pipelines and both buffers must be live, and the pipelines' descriptor sets must name
    /// these buffers — a set built against different ones would read freed memory. Every
    /// `copy_bytes` must be within both buffers.
    pub(crate) unsafe fn replay(
        &self,
        pipelines: &[Pipeline],
        steps: &[Step],
        source: &Buffer,
        destination: &Buffer,
    ) -> Result<(), Error> {
        // The duration `record_and_wait` reports is the host's view of the whole submission, which
        // is not what a chain's caller wants — `Gpu::sum` reports dispatch *counts* and leaves
        // timing to the examples. Discarded here rather than threaded out to nobody.
        let recorded = unsafe {
            self.record_and_wait(|device, command| {
                for (pipeline, step) in pipelines.iter().zip(steps) {
                    if step.copy_bytes > 0 {
                        // The previous pass wrote `destination`; this one reads `source`. Both the
                        // copy and the dispatch after it wait on what came before.
                        barrier(device, command, TRANSFER_AFTER_SHADER);
                        let region = [vk::BufferCopy::default().size(step.copy_bytes)];
                        device.cmd_copy_buffer(command, destination.handle, source.handle, &region);
                        barrier(device, command, SHADER_AFTER_TRANSFER);
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
                    device.cmd_dispatch(command, step.workgroups.max(1), 1, 1);
                }
                Ok(())
            })
        };

        recorded.map(|_| ())
    }
}

/// Which stages a barrier separates, and what access it makes visible.
type Stages = (
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
    vk::AccessFlags,
    vk::AccessFlags,
);

/// The copy reads what the dispatch wrote.
const TRANSFER_AFTER_SHADER: Stages = (
    vk::PipelineStageFlags::COMPUTE_SHADER,
    vk::PipelineStageFlags::TRANSFER,
    vk::AccessFlags::SHADER_WRITE,
    vk::AccessFlags::TRANSFER_READ,
);

/// The next dispatch reads what the copy wrote.
const SHADER_AFTER_TRANSFER: Stages = (
    vk::PipelineStageFlags::TRANSFER,
    vk::PipelineStageFlags::COMPUTE_SHADER,
    vk::AccessFlags::TRANSFER_WRITE,
    vk::AccessFlags::SHADER_READ,
);

/// Record one memory barrier.
fn barrier(device: &ash::Device, command: vk::CommandBuffer, stages: Stages) {
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
