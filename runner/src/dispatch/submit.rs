//! Recording a command buffer, submitting it, and waiting — with a clock attached.
//!
//! Everything that runs on the device goes through [`Gpu::record_and_wait`], which is why the
//! timing lives here rather than beside any one caller: a copy and a dispatch are measured the
//! same way, and neither gets to pick a more flattering clock.

use super::timestamps::Timestamps;
use crate::buffer::Buffer;
use crate::{Error, Gpu};
use ash::vk;
use std::time::{Duration, Instant};

impl Gpu {
    /// Copy `bytes` from one buffer to another, and wait for it.
    ///
    /// # Safety
    ///
    /// Both buffers must be live and nothing may be using them.
    pub(crate) unsafe fn copy(&self, from: &Buffer, to: &Buffer, bytes: u64) -> Result<(), Error> {
        // SAFETY: `record_and_wait` requires whatever the closure names to outlive the submission,
        // and it waits before returning — so the caller's `from` and `to`, which this function's
        // own contract says are live and unused, are live for the whole of it.
        unsafe {
            self.record_and_wait(|device, command| {
                let region = [vk::BufferCopy::default().size(bytes)];
                device.cmd_copy_buffer(command, from.handle, to.handle, &region);
                Ok(())
            })
        }
        .map(|_| ())
    }

    /// Record a one-shot command buffer, submit it, and wait — returning how long that took.
    ///
    /// **The device's own clock when it has one**, and the host's only as a fallback. An `Instant`
    /// around a submit-and-wait measures the driver's scheduling and the operating system's along
    /// with the work; at a few microseconds of kernel that is mostly not the kernel, and it is
    /// what made this project's sweep disagree with itself twenty-fold.
    ///
    /// # Safety
    ///
    /// Whatever `record` refers to must outlive the submission.
    pub(super) unsafe fn record_and_wait<F>(&self, record: F) -> Result<Duration, Error>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer) -> Result<(), Error>,
    {
        let device = self.device();
        // SAFETY: the device outlives this call — it is owned by the `Gpu` this is a method on,
        // and `Gpu::drop` cannot run while a `&self` borrow of it exists.
        let timestamps = unsafe { Timestamps::new(self) }?;

        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family());
        // SAFETY: as above for the device, and the queue family index is the one this `Gpu`
        // recorded when it opened that device rather than a number chosen here.
        let pool = unsafe { device.create_command_pool(&pool_info, None) }?;

        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool was created immediately above, is not in use by any other thread — it
        // has not been shared — and outlives the buffers allocated from it, which are freed with
        // it at the end of this function.
        let commands = unsafe { device.allocate_command_buffers(&allocate) }?;
        let Some(&command) = commands.first() else {
            // SAFETY: the pool is the one created above and nothing was allocated from it that
            // could still be recording — the allocation returned nothing.
            unsafe { device.destroy_command_pool(pool, None) };
            return Err(Error::NoPipeline);
        };

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `command` came from the pool above and has never been begun, so it is in the
        // initial state this call requires.
        unsafe { device.begin_command_buffer(command, &begin) }?;
        if let Some(timestamps) = timestamps.as_ref() {
            // SAFETY: the command buffer is between `begin` and `end`, which is the only state a
            // timestamp write may be recorded in, and the query pool is `timestamps`' own.
            unsafe { timestamps.begin(self, command) };
        }
        record(device, command)?;
        if let Some(timestamps) = timestamps.as_ref() {
            // SAFETY: as the opening write — still recording, same pool, and the caller's `record`
            // cannot have ended the buffer because it is handed one that is already open.
            unsafe { timestamps.end(self, command) };
        }
        // SAFETY: the buffer was begun above and every command since has been recorded into it, so
        // it is in the recording state this ends.
        unsafe { device.end_command_buffer(command) }?;

        // SAFETY: the device outlives this call, as above.
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let buffers = [command];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);

        let started = Instant::now();
        // SAFETY: the command buffer is recorded and ended, is submitted exactly once — it was
        // allocated in this function and never handed anywhere else — and the fence is fresh and
        // unsignalled. Everything the recorded commands refer to outlives the wait below, which is
        // what this function's own contract asks of its caller.
        unsafe {
            device.queue_submit(self.queue(), &[submit], fence)?;
            // No timeout: a hung dispatch should hang visibly rather than return a wrong answer
            // that looks like a measurement.
            device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        let host = started.elapsed();

        // The device's answer wins where there is one. `None` means the query pool existed but
        // gave nothing usable, which is a fallback rather than a failure.
        let elapsed = match timestamps {
            Some(timestamps) => {
                // SAFETY: the fence above has been waited on, so the writes these queries record
                // have completed and the results are available rather than in flight.
                let measured = unsafe { timestamps.read(self) }?;
                // SAFETY: the same wait means no submission still refers to the query pool, and
                // nothing reads `timestamps` after this — `measured` is already a plain `Duration`.
                unsafe { timestamps.destroy(self) };
                measured.unwrap_or(host)
            }
            None => host,
        };

        // SAFETY: the fence is signalled and has been waited on, so the command buffer is no
        // longer executing and both it and the pool that owns it are free to destroy. Destroying
        // the pool frees the buffer allocated from it, which is why that is not done separately.
        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
        Ok(elapsed)
    }
}
