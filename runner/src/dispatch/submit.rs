use super::timestamps::Timestamps;
use crate::buffer::Buffer;
use crate::{Error, Gpu};
use ash::vk;
use std::time::{Duration, Instant};

pub(super) struct Recorded {
    pub(super) whole: Duration,
    pub(super) spans: Vec<Duration>,
}

struct Submission<'gpu> {
    gpu: &'gpu Gpu,
    pool: vk::CommandPool,
    fence: vk::Fence,
    timestamps: Option<Timestamps>,
    in_flight: bool,
}

impl Drop for Submission<'_> {
    fn drop(&mut self) {
        if self.in_flight {
            return;
        }

        let device = self.gpu.device();
        // SAFETY: nothing is executing — either the wait succeeded or the work was never
        // submitted, which is exactly what `in_flight` records. The pool goes last: destroying it
        // frees the command buffer allocated from it, which is why that is not released separately.
        unsafe {
            if let Some(timestamps) = self.timestamps.take() {
                timestamps.destroy(self.gpu);
            }
            if self.fence != vk::Fence::null() {
                device.destroy_fence(self.fence, None);
            }
            device.destroy_command_pool(self.pool, None);
        }
    }
}

impl Gpu {
    pub(crate) unsafe fn copy(&self, from: &Buffer, to: &Buffer, bytes: u64) -> Result<(), Error> {
        // SAFETY: `record_and_wait` requires whatever the closure names to outlive the submission,
        // and it waits before returning — so the caller's `from` and `to`, which this function's
        // own contract says are live and unused, are live for the whole of it.
        unsafe {
            self.record_and_wait(0, |device, command, _| {
                let region = [vk::BufferCopy::default().size(bytes)];
                device.cmd_copy_buffer(command, from.handle, to.handle, &region);
                Ok(())
            })
        }
        .map(|_| ())
    }

    pub(super) unsafe fn record_and_wait<F>(&self, marks: u32, record: F) -> Result<Recorded, Error>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer, Option<&Timestamps>) -> Result<(), Error>,
    {
        let device = self.device();
        // SAFETY: the device outlives this call — it is owned by the `Gpu` this is a method on,
        // and `Gpu::drop` cannot run while a `&self` borrow of it exists.
        let timestamps = unsafe { Timestamps::new(self, marks) }?;

        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family());
        // SAFETY: as above for the device, and the queue family index is the one this `Gpu`
        // recorded when it opened that device rather than a number chosen here.
        let pool = unsafe { device.create_command_pool(&pool_info, None) }?;

        let mut submission = Submission {
            gpu: self,
            pool,
            fence: vk::Fence::null(),
            timestamps,
            in_flight: false,
        };

        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool was created immediately above, is not in use by any other thread — it
        // has not been shared — and outlives the buffers allocated from it, which are freed with
        // it when the guard drops.
        let commands = unsafe { device.allocate_command_buffers(&allocate) }?;
        let Some(&command) = commands.first() else {
            return Err(Error::NoPipeline);
        };

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `command` came from the pool above and has never been begun, so it is in the
        // initial state this call requires.
        unsafe { device.begin_command_buffer(command, &begin) }?;
        if let Some(timestamps) = submission.timestamps.as_ref() {
            // SAFETY: the command buffer is between `begin` and `end`, which is the only state a
            // timestamp write may be recorded in, and the query pool is `timestamps`' own.
            unsafe { timestamps.begin(self, command) };
        }
        record(device, command, submission.timestamps.as_ref())?;
        if let Some(timestamps) = submission.timestamps.as_ref() {
            // SAFETY: as the opening write — still recording, same pool, and the caller's `record`
            // cannot have ended the buffer because it is handed one that is already open.
            unsafe { timestamps.end(self, command) };
        }
        // SAFETY: the buffer was begun above and every command since has been recorded into it, so
        // it is in the recording state this ends.
        unsafe { device.end_command_buffer(command) }?;

        // SAFETY: the device outlives this call, as above.
        submission.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let buffers = [command];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);

        let started = Instant::now();
        submission.in_flight = true;
        // SAFETY: the command buffer is recorded and ended, is submitted exactly once — it was
        // allocated in this function and never handed anywhere else — and the fence is fresh and
        // unsignalled. Everything the recorded commands refer to outlives the wait below, which is
        // what this function's own contract asks of its caller.
        unsafe {
            device.queue_submit(self.queue(), &[submit], submission.fence)?;
            device.wait_for_fences(&[submission.fence], true, u64::MAX)?;
        }
        submission.in_flight = false;
        let host = started.elapsed();

        let (elapsed, spans) = match submission.timestamps.as_ref() {
            Some(timestamps) => {
                // SAFETY: the fence above has been waited on, so the writes these queries record
                // have completed and the results are available rather than in flight.
                let measured = unsafe { timestamps.read(self) }?;
                // SAFETY: the same wait, and the marks were written into the same pool.
                let spans = unsafe { timestamps.spans(self) }?;
                (measured.unwrap_or(host), spans)
            }
            None => (host, Vec::new()),
        };

        Ok(Recorded {
            whole: elapsed,
            spans,
        })
    }
}
