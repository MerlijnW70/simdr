//! Building a pipeline out of a SPIR-V module and running it.
//!
//! # What is timed, and what is not
//!
//! A run is three submissions: upload, dispatch, download. Only the middle one is timed, and the
//! kernel's buffers are device-local, so the number [`Gpu::time`] reports is the kernel reading
//! VRAM rather than the host's copies crossing the bus. Getting that wrong is what made two
//! earlier benchmarks meaningless — see `notes/FINDINGS.md`.

mod bindings;
mod chain;
mod pipeline;
mod placement;
mod session;
mod specialization;
mod submit;
mod timestamps;

pub use chain::Pass;
pub use placement::{MemoryType, Placement};
pub use session::Session;
pub use specialization::Specialization;

pub(crate) use pipeline::Pipeline;

use crate::buffer::Buffer;
use crate::timing::Timing;
use crate::{Error, Gpu};
use ash::vk;
use std::time::Duration;

impl Gpu {
    /// Run `spirv` over `input`, and return what it wrote to its output binding.
    ///
    /// The module must be a compute shader named `main` with two `StorageBuffer` bindings in
    /// descriptor set 0 — binding 0 read, binding 1 written — which is the shape every kernel in
    /// `simdr` emits. `workgroups` is the dispatch's x dimension.
    ///
    /// The words go to `vkCreateShaderModule` exactly as the emitter produced them. Nothing here
    /// inspects or rewrites them, which is the point: any disagreement that shows up is between
    /// our module and the driver, with no third opinion in between.
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`] if any call fails, [`Error::NoPipeline`] if the driver returns none.
    pub fn run(&self, spirv: &[u32], input: &[f32], workgroups: u32) -> Result<Vec<f32>, Error> {
        let words: Vec<u32> = input.iter().map(|value| value.to_bits()).collect();
        let output = self.run_words(spirv, &words, workgroups)?;
        Ok(output.into_iter().map(f32::from_bits).collect())
    }

    /// The same, for a kernel whose buffers hold 32-bit integers.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_u32(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
    ) -> Result<Vec<u32>, Error> {
        self.run_words(spirv, input, workgroups)
    }

    /// The same, for a kernel whose buffers hold bytes — `i8` or `u8`.
    ///
    /// The buffer is still words underneath, because that is what a Vulkan allocation is; four
    /// elements share each one. Element `i` lands at byte offset `i`, which is what the kernel's
    /// `ArrayStride` of 1 says it should, so the packing is little-endian and not a choice.
    ///
    /// A length that is not a multiple of four is padded up. The padding is written and read back
    /// and then dropped, so a caller sees exactly the elements it passed.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_bytes(
        &self,
        spirv: &[u32],
        input: &[u8],
        workgroups: u32,
    ) -> Result<Vec<u8>, Error> {
        let words: Vec<u32> = input
            .chunks(4)
            .map(|chunk| {
                let mut word = [0_u8; 4];
                // `chunks` yields at most four, so this is a copy with the tail left zero rather
                // than a case that can fail.
                if let Some(slot) = word.get_mut(..chunk.len()) {
                    slot.copy_from_slice(chunk);
                }
                u32::from_le_bytes(word)
            })
            .collect();

        let output = self.run_words(spirv, &words, workgroups)?;
        let mut bytes: Vec<u8> = output.iter().flat_map(|word| word.to_le_bytes()).collect();
        bytes.truncate(input.len());
        Ok(bytes)
    }

    /// The same, for a kernel whose buffers hold 16-bit elements — `i16`, `u16` or `f16`.
    ///
    /// A half's bits rather than a number: [`simdr::half::from_f32`] is what turns a float into
    /// one, and this layer has no way to tell the three 16-bit types apart.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_halves(
        &self,
        spirv: &[u32],
        input: &[u16],
        workgroups: u32,
    ) -> Result<Vec<u16>, Error> {
        let words: Vec<u32> = input
            .chunks(2)
            .map(|chunk| match chunk {
                [low, high] => u32::from(*low) | (u32::from(*high) << 16),
                [low] => u32::from(*low),
                _ => 0,
            })
            .collect();

        let output = self.run_words(spirv, &words, workgroups)?;
        let mut halves: Vec<u16> = output
            .iter()
            .flat_map(|word| [*word as u16, (word >> 16) as u16])
            .collect();
        halves.truncate(input.len());
        Ok(halves)
    }

    /// Run over raw words, which is what the buffers actually hold.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_words(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
    ) -> Result<Vec<u32>, Error> {
        self.execute(spirv, input, workgroups, 1, &Specialization::none())
            .map(|out| out.0)
    }

    /// The same, with some of the module's specialization constants given values.
    ///
    /// The point of the whole mechanism: one `spirv` here, several `specialization`s, and a
    /// different pipeline each time without the module being rebuilt. What can and cannot be
    /// deferred this way is `decisions/DR-0005`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_specialized(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
        specialization: &Specialization,
    ) -> Result<Vec<u32>, Error> {
        self.execute(spirv, input, workgroups, 1, specialization)
            .map(|out| out.0)
    }

    /// Time `iterations` dispatches of a specialized pipeline.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn time_specialized(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
        iterations: u32,
        specialization: &Specialization,
    ) -> Result<Duration, Error> {
        self.execute(spirv, input, workgroups, iterations, specialization)
            .map(|out| out.1)
    }

    /// Time `iterations` back-to-back dispatches of `spirv`.
    ///
    /// Only the dispatch submission is timed: allocation, pipeline creation and both host copies
    /// happen outside it, and the kernel's buffers are device-local. A memory barrier separates
    /// the iterations, so this is the sum of the kernels' own times rather than a measure of how
    /// well the scheduler overlaps them.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn time(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
        iterations: u32,
    ) -> Result<Duration, Error> {
        self.execute(
            spirv,
            input,
            workgroups,
            iterations,
            &Specialization::none(),
        )
        .map(|out| out.1)
    }

    /// Time the same dispatch `repeats` times over, and report the spread.
    ///
    /// One number per measurement reads like a result whether or not it is one — the sweep that
    /// refuted this project's cache-capacity story looked authoritative in its first run and moved
    /// its cliff in the second. [`Timing`] is what says so on the spot.
    ///
    /// Each repeat is a fresh submission of `iterations` dispatches over buffers that stay
    /// allocated, so the repeats differ only in what the machine was doing at the time — which is
    /// exactly the thing being measured.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`], or [`Error::NoPipeline`] if `repeats` is zero and there is nothing to
    /// summarise.
    pub fn time_repeated(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
        iterations: u32,
        repeats: u32,
    ) -> Result<Timing, Error> {
        let mut samples = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats.max(1) {
            samples.push(self.time(spirv, input, workgroups, iterations)?);
        }
        Timing::of(&samples).ok_or(Error::NoPipeline)
    }

    /// Upload, dispatch, download — returning the output and what the dispatch alone took.
    fn execute(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
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
                workgroups,
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
        workgroups: u32,
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

        let elapsed = unsafe { self.dispatch(&pipeline, workgroups, iterations) }?;

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
        workgroups: u32,
        iterations: u32,
    ) -> Result<Duration, Error> {
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
                    device.cmd_dispatch(command, workgroups, 1, 1);
                }
                Ok(())
            })
        }
    }
}
