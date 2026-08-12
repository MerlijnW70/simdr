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
//! followed by a device-to-device copy of the whole buffer back into the source.
//!
//! That copy is real work — for a shrinking reduction it is mostly copying elements the next pass
//! will not read. A ping-pong across two descriptor sets would avoid it, and is the right shape
//! for a chain that is being *timed*. This one is built to be *correct*, and the copy is the
//! honest, obvious version of correct.

use super::pipeline::Pipeline;
use crate::buffer::Buffer;
use crate::{Error, Gpu};
use ash::vk;

/// One dispatch of a chain.
#[derive(Debug, Clone, Copy)]
pub struct Pass<'words> {
    /// The module to run.
    pub spirv: &'words [u32],
    /// How many workgroups of it.
    pub workgroups: u32,
}

impl<'words> Pass<'words> {
    /// A pass running `workgroups` groups of `spirv`.
    #[must_use]
    pub const fn new(spirv: &'words [u32], workgroups: u32) -> Self {
        Self { spirv, workgroups }
    }
}

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

        let recorded = unsafe {
            self.record_and_wait(|device, command| {
                for (index, (pass, pipeline)) in passes.iter().zip(&pipelines).enumerate() {
                    if index > 0 {
                        // The previous pass wrote `destination`; this one reads `source`. Both the
                        // copy and the dispatch after it wait on what came before.
                        barrier(device, command, TRANSFER_AFTER_SHADER);
                        let region = [vk::BufferCopy::default().size(bytes)];
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
                    device.cmd_dispatch(command, pass.workgroups.max(1), 1, 1);
                }
                Ok(())
            })
        };

        let output = recorded.and_then(|_| {
            unsafe { self.copy(destination, staging, bytes) }?;
            unsafe { staging.read(self, count) }
        });

        for pipeline in pipelines {
            unsafe { pipeline.destroy(self) };
        }
        output
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
