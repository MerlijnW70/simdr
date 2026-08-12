//! Dispatching over as many buffers as the kernel binds.
//!
//! [`crate::Gpu::run`] binds exactly two — 0 read, 1 written — which is what every kernel in
//! `kernels/` needed until one did not. The NNUE layer has weights and activations, two arrays
//! from two places, and had to be handed over concatenated into one buffer with the join passed
//! as an offset. That works and it is not the shape of the problem.
//!
//! The emitter never had this limit: `simdr::kernel::Shape` has always taken a buffer count. This
//! is the runner catching up.
//!
//! # The convention
//!
//! Bindings `0..n-1` are read and binding `n-1` is written, which extends the two-buffer rule
//! rather than replacing it. Each buffer is sized to its own contents, so a kernel reading a large
//! weight table and writing one value per invocation does not have to allocate the larger of the
//! two twice.

use super::pipeline::Pipeline;
use crate::buffer::Buffer;
use crate::{Error, Gpu};

impl Gpu {
    /// Run `spirv` with one buffer per entry of `inputs`, plus an output of `output_len` words.
    ///
    /// The inputs are bound at 0, 1, … in order and the output last. Every buffer is device-local
    /// and reached through one staging buffer, so nothing here measures the bus except the
    /// upload it has to do anyway.
    ///
    /// # Errors
    ///
    /// [`Error::NoPipeline`] if `inputs` is empty or `output_len` is zero, otherwise as
    /// [`Gpu::run`].
    pub fn run_bound(
        &self,
        spirv: &[u32],
        inputs: &[&[u32]],
        output_len: usize,
        workgroups: u32,
    ) -> Result<Vec<u32>, Error> {
        if inputs.is_empty() || output_len == 0 {
            return Err(Error::NoPipeline);
        }

        let sizes: Vec<u64> = inputs
            .iter()
            .map(|words| (words.len().max(1) * size_of::<u32>()) as u64)
            .chain(std::iter::once((output_len * size_of::<u32>()) as u64))
            .collect();
        let staging_bytes = sizes.iter().copied().max().unwrap_or(4);

        // SAFETY: every object below is created here and destroyed before returning, and each is
        // used only between a submission and the fence that completes it.
        unsafe {
            let staging = Buffer::staging(self, staging_bytes)?;
            let mut devices = Vec::with_capacity(sizes.len());
            for &bytes in &sizes {
                match Buffer::device_local(self, bytes) {
                    Ok(buffer) => devices.push(buffer),
                    Err(error) => {
                        for buffer in devices {
                            buffer.destroy(self);
                        }
                        staging.destroy(self);
                        return Err(error);
                    }
                }
            }

            let outcome = self.bound_run(
                spirv, &staging, &devices, &sizes, inputs, output_len, workgroups,
            );

            for buffer in devices {
                buffer.destroy(self);
            }
            staging.destroy(self);
            outcome
        }
    }

    /// Upload each input, dispatch, read the last buffer back.
    ///
    /// # Safety
    ///
    /// The buffers must be live and the device idle with respect to them.
    #[expect(
        clippy::too_many_arguments,
        reason = "every one is a distinct thing the run needs; bundling them into a struct only \
                  moved the list somewhere else last time this was tried"
    )]
    unsafe fn bound_run(
        &self,
        spirv: &[u32],
        staging: &Buffer,
        devices: &[Buffer],
        sizes: &[u64],
        inputs: &[&[u32]],
        output_len: usize,
        workgroups: u32,
    ) -> Result<Vec<u32>, Error> {
        // One staging buffer, reused: each input goes through it in turn. Sequential rather than
        // concurrent, and untimed, because a benchmark of PCIe is not what anyone asked for.
        for (index, words) in inputs.iter().enumerate() {
            let Some(target) = devices.get(index) else {
                return Err(Error::NoPipeline);
            };
            let bytes = (words.len().max(1) * size_of::<u32>()) as u64;
            unsafe {
                staging.write(self, words)?;
                self.copy(staging, target, bytes)?;
            }
        }

        let bound: Vec<(&Buffer, u64)> = devices
            .iter()
            .zip(sizes)
            .map(|(buffer, &bytes)| (buffer, bytes))
            .collect();
        let pipeline =
            unsafe { Pipeline::new(self, spirv, &bound, &super::Specialization::none()) }?;

        let dispatched = unsafe { self.dispatch(&pipeline, workgroups, 1) };
        unsafe { pipeline.destroy(self) };
        dispatched?;

        let Some(output) = devices.last() else {
            return Err(Error::NoPipeline);
        };
        let bytes = (output_len * size_of::<u32>()) as u64;
        unsafe {
            self.copy(output, staging, bytes)?;
            staging.read(self, output_len)
        }
    }
}
