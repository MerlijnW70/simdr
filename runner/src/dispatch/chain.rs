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
//! Two device buffers, alternating: pass 0 reads A and writes B, pass 1 reads B and writes A. Only
//! the descriptor set changes — every kernel this crate emits binds buffer 0 read and buffer 1
//! written, and the module never learns which is which. [`super::step`] has the arithmetic and the
//! consequence, which is that the answer ends up in a different buffer depending on how many
//! passes ran.
//!
//! This file is the recording: barriers, dispatches, and the submission around them. It replaced a
//! device-to-device copy of B back into A after every pass — 22% of a held reduction over 2^20
//! elements, two thirds of which was the pair of pipeline barriers the copy needed rather than the
//! copy itself. What is left between passes is one barrier, because a pass still has to wait for
//! the one before it, and that one barrier turned out to cost nearly what the pair did.
//!
//! So this is shorter code rather than faster code, except on a device short of bandwidth — 5.5%
//! on the integrated Radeon and nothing measurable on the other two. `super::step` has the
//! numbers.

use super::pipeline::Pipeline;
use super::step::{Ends, Pass, answer_in_destination};
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
        self.run_chain_head(passes, input, input.len())
    }

    /// The same, bringing only the first `head` words home.
    ///
    /// A chain that ends in one number — a reduction — has no use for the rest of the buffer, and
    /// copying 4 MB back to read four bytes was **37%** of a held reduction over 2²⁰ elements. The
    /// dispatches are identical; only the download shrinks.
    ///
    /// `head` is clamped to the buffer, so asking for more than there is returns what there is.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run_chain`].
    pub fn run_chain_head(
        &self,
        passes: &[Pass<'_>],
        input: &[u32],
        head: usize,
    ) -> Result<Vec<u32>, Error> {
        if passes.is_empty() {
            return Err(Error::NoPipeline);
        }

        let count = head.min(input.len()).max(1);
        let bytes = (input.len().max(1) * size_of::<u32>()) as u64;

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
        for (index, pass) in passes.iter().enumerate() {
            // Built before recording rather than inside it: a failure here must not leave a
            // half-recorded command buffer, and every pipeline has to outlive the submission.
            //
            // The buffer pair alternates, which is the whole ping-pong: this pipeline's descriptor
            // set is what decides which end the pass reads.
            let (read, written) = Ends::of(index).order(source, destination);
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
                        unsafe { pipeline.destroy(self) };
                    }
                    return Err(error);
                }
            }
        }

        unsafe { self.copy(staging, source, bytes) }?;

        let groups: Vec<u32> = passes.iter().map(|pass| pass.workgroups).collect();
        let recorded = unsafe { self.replay(&pipelines, &groups) };

        let output = recorded.and_then(|()| {
            let answer = if answer_in_destination(passes.len()) {
                destination
            } else {
                source
            };
            // Only what the caller asked for. `count` is words and the copy is bytes, and the
            // difference between those two is exactly the mistake that made this read megabytes.
            let home = (count.max(1) * size_of::<u32>()) as u64;
            unsafe { self.copy(answer, staging, home.min(bytes)) }?;
            unsafe { staging.read(self, count) }
        });

        for pipeline in pipelines {
            unsafe { pipeline.destroy(self) };
        }
        output
    }
}

impl Gpu {
    /// Record every pipeline in order with a barrier between them, and wait for the whole thing.
    ///
    /// The half of a chain that does not care where the pipelines came from. [`Gpu::run_chain`]
    /// builds them and throws them away; [`crate::Reducer`] keeps them. Both record the same
    /// sequence, and it lives here so there is one of it.
    ///
    /// `workgroups` is one entry per pipeline, in the same order. Nothing here knows which buffers
    /// a pipeline is bound to — that was decided when it was built, and this only has to order the
    /// dispatches against each other.
    ///
    /// # Safety
    ///
    /// The pipelines must be live, and their descriptor sets must name buffers that are — a set
    /// built against freed ones would read freed memory.
    pub(crate) unsafe fn replay(
        &self,
        pipelines: &[Pipeline],
        workgroups: &[u32],
    ) -> Result<(), Error> {
        // The duration `record_and_wait` reports is the host's view of the whole submission, which
        // is not what a chain's caller wants — `Gpu::sum` reports dispatch *counts* and leaves
        // timing to the examples. Discarded here rather than threaded out to nobody.
        let recorded = unsafe {
            self.record_and_wait(|device, command| {
                for (index, (pipeline, groups)) in pipelines.iter().zip(workgroups).enumerate() {
                    if index > 0 {
                        // This pass reads what the one before it wrote, and writes what the one
                        // before *that* read. The first is a memory dependency and needs the access
                        // masks; the second is an execution dependency and needs only the barrier
                        // to exist, which it does.
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
                }
                Ok(())
            })
        };

        recorded.map(|_| ())
    }
}

/// Record the one barrier a chained pass needs.
///
/// `SHADER_WRITE` becoming visible to `SHADER_READ` is the read-after-write: this pass reads the
/// buffer the last one wrote. `SHADER_WRITE` in the destination mask as well is the
/// write-after-read: this pass writes the buffer the pass before it *read*, and while an execution
/// dependency alone is enough for that, naming it is what stops the next reader of this code from
/// removing it.
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
