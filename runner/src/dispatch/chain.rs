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
            // **Not `Buffer::shared`, and that was measured rather than assumed.** This path
            // allocates its buffers on every call, and allocating out of the memory both sides can
            // reach costs more than the upload it saves: `Gpu::sum` over 2²⁰ went from ~2153 µs to
            // ~3492 µs on an RTX 4080 when this asked for shared memory — 62% slower. The tell is
            // the 8 192-element case, which has 32 KB to upload and no transfer worth saving, and
            // still lost 22%. So the cost is in the allocation, not the writing.
            //
            // `Reducer` and `Session` do ask for it, because they allocate once and upload many
            // times. Same memory, opposite answer, and the difference is only how often the buffer
            // is made. `notes/FINDINGS.md` has both columns.
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

    /// Upload, run the chain, download — with everything torn down afterwards.
    ///
    /// # Safety
    ///
    /// The buffers must be live and the device idle with respect to them.
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

        let groups: Vec<u32> = passes.iter().map(|pass| pass.workgroups).collect();
        let answer = if answer_in_destination(passes.len()) {
            destination
        } else {
            source
        };
        // Only what the caller asked for. `count` is words and the copy is bytes, and the
        // difference between those two is exactly the mistake that made this read megabytes.
        let home = (count.max(1) * size_of::<u32>()) as u64;

        // One submission for the upload, the chain and the answer together. These were three, and
        // on a device with memory both sides can reach the upload is not even one of them —
        // `deliver` has already put the input into `source` and handed back `None`.
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

        let output = recorded.and_then(|()| unsafe { staging.read(self, count) });

        for pipeline in pipelines {
            unsafe { pipeline.destroy(self) };
        }
        output
    }
}

/// The three buffers a chain runs over, and how wide they are.
///
/// Grouped because they are one decision, not three: the pair alternates and the staging buffer
/// serves both ends, so a caller that had `source` right and `destination` wrong would have a
/// working program that returned the second-to-last pass. Passing them together also keeps them
/// the same width, which every dispatch below assumes.
#[derive(Clone, Copy)]
struct Workspace<'a> {
    /// The host's way in and out.
    staging: &'a Buffer,
    /// Read by pass 0, and written by every odd pass after it.
    source: &'a Buffer,
    /// Written by pass 0, and read by every odd pass after it.
    destination: &'a Buffer,
    /// How wide all three are, in bytes.
    bytes: u64,
}

/// One buffer-to-buffer copy, recorded inside a submission rather than being one.
#[derive(Clone, Copy)]
pub(crate) struct Staged<'a> {
    /// Where the bytes come from.
    pub(crate) from: &'a Buffer,
    /// Where they go.
    pub(crate) to: &'a Buffer,
    /// How many.
    pub(crate) bytes: u64,
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
    /// # The copies belong in here
    ///
    /// `before` moves the host's data into the buffer the first pass reads, and `after` brings the
    /// answer back out. Both used to be separate [`Gpu::copy`] calls around this one, and a `copy`
    /// is a whole `record_and_wait` — its own command buffer, its own submission, its own fence.
    ///
    /// `runner/examples/reducer.rs` measured a bare submit-and-wait at **~65 µs**, so a reduction
    /// that submitted three times spent about 195 µs of a 540 µs call doing nothing but starting
    /// and stopping. Recorded here they cost two barriers instead.
    ///
    /// # Safety
    ///
    /// The pipelines must be live, and their descriptor sets must name buffers that are — a set
    /// built against freed ones would read freed memory. Every buffer named by `before` and
    /// `after` must be live and large enough for its `bytes`.
    pub(crate) unsafe fn replay(
        &self,
        pipelines: &[Pipeline],
        workgroups: &[u32],
        before: Option<Staged<'_>>,
        after: Option<Staged<'_>>,
    ) -> Result<(), Error> {
        // The duration `record_and_wait` reports is the host's view of the whole submission, which
        // is not what a chain's caller wants — `Gpu::sum` reports dispatch *counts* and leaves
        // timing to the examples. Discarded here rather than threaded out to nobody.
        let recorded = unsafe {
            self.record_and_wait(|device, command| {
                if let Some(upload) = before {
                    // Nothing has run yet, so there is nothing to wait for — only to make visible.
                    // The barrier after it is what the first dispatch needs.
                    let region = [vk::BufferCopy::default().size(upload.bytes)];
                    device.cmd_copy_buffer(command, upload.from.handle, upload.to.handle, &region);
                    across(device, command, TRANSFER_TO_SHADER);
                }

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

        recorded.map(|_| ())
    }
}

/// Which stages a barrier separates, and what it makes visible.
type Stages = (
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
    vk::AccessFlags,
    vk::AccessFlags,
);

/// The first dispatch reads what the upload copy wrote.
const TRANSFER_TO_SHADER: Stages = (
    vk::PipelineStageFlags::TRANSFER,
    vk::PipelineStageFlags::COMPUTE_SHADER,
    vk::AccessFlags::TRANSFER_WRITE,
    vk::AccessFlags::SHADER_READ,
);

/// The download copy reads what the last dispatch wrote.
const SHADER_TO_TRANSFER: Stages = (
    vk::PipelineStageFlags::COMPUTE_SHADER,
    vk::PipelineStageFlags::TRANSFER,
    vk::AccessFlags::SHADER_WRITE,
    vk::AccessFlags::TRANSFER_READ,
);

/// Record a barrier between two differing stages.
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
