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
    pub(super) unsafe fn copy(&self, from: &Buffer, to: &Buffer, bytes: u64) -> Result<(), Error> {
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
        let timestamps = unsafe { Timestamps::new(self) }?;

        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family());
        let pool = unsafe { device.create_command_pool(&pool_info, None) }?;

        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let commands = unsafe { device.allocate_command_buffers(&allocate) }?;
        let Some(&command) = commands.first() else {
            unsafe { device.destroy_command_pool(pool, None) };
            return Err(Error::NoPipeline);
        };

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(command, &begin) }?;
        if let Some(timestamps) = timestamps.as_ref() {
            unsafe { timestamps.begin(self, command) };
        }
        record(device, command)?;
        if let Some(timestamps) = timestamps.as_ref() {
            unsafe { timestamps.end(self, command) };
        }
        unsafe { device.end_command_buffer(command) }?;

        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let buffers = [command];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);

        let started = Instant::now();
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
                let measured = unsafe { timestamps.read(self) }?;
                unsafe { timestamps.destroy(self) };
                measured.unwrap_or(host)
            }
            None => host,
        };

        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
        Ok(elapsed)
    }
}
