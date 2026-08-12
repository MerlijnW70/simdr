//! Asking the device how long it took, instead of asking the clock on this side of the bus.
//!
//! `Instant` around a submit-and-wait measures everything between the two calls: the driver's
//! scheduling, the operating system's, whatever else wanted the GPU. At a few microseconds of
//! actual work that is mostly not the kernel — and it is why this project's sweep disagreed with
//! itself by twenty times and had no way to say so.
//!
//! A timestamp query is written *into the command stream*, so the two readings bracket the work
//! on the device's own clock. The difference is in ticks; `timestampPeriod` converts them to
//! nanoseconds.
//!
//! Not every queue supports them. `timestampValidBits` is zero on some transfer queues, and the
//! whole feature is optional — so this reports `None` rather than pretending, and the caller
//! falls back to the host clock knowing that is what it has.

use crate::{Error, Gpu};
use ash::vk;
use std::time::Duration;

/// A pair of timestamp queries bracketing one submission.
pub(super) struct Timestamps {
    pool: vk::QueryPool,
    /// Nanoseconds per tick, from the device's limits.
    period: f32,
}

impl Timestamps {
    /// Create a query pool, or `None` if this device cannot answer.
    ///
    /// # Safety
    ///
    /// [`Timestamps::destroy`] must run before the device goes away.
    pub(super) unsafe fn new(gpu: &Gpu) -> Result<Option<Self>, Error> {
        let period = gpu.limits().timestamp_period_ns;
        if period <= 0.0 {
            return Ok(None);
        }

        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);
        let pool = unsafe { gpu.device().create_query_pool(&info, None) }?;

        Ok(Some(Self { pool, period }))
    }

    /// Reset the pool and write the opening timestamp.
    ///
    /// The reset has to be recorded too: a query pool's contents are undefined until it is, and a
    /// second submission would otherwise read the first one's answer.
    ///
    /// # Safety
    ///
    /// `command` must be a command buffer in the recording state.
    pub(super) unsafe fn begin(&self, gpu: &Gpu, command: vk::CommandBuffer) {
        let device = gpu.device();
        unsafe {
            device.cmd_reset_query_pool(command, self.pool, 0, 2);
            device.cmd_write_timestamp(command, vk::PipelineStageFlags::TOP_OF_PIPE, self.pool, 0);
        }
    }

    /// Write the closing timestamp.
    ///
    /// # Safety
    ///
    /// As [`Timestamps::begin`], and after it.
    pub(super) unsafe fn end(&self, gpu: &Gpu, command: vk::CommandBuffer) {
        unsafe {
            gpu.device().cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                1,
            );
        }
    }

    /// Read the elapsed device time, once the submission has completed.
    ///
    /// # Safety
    ///
    /// The fence for the submission that wrote these must have been waited on.
    pub(super) unsafe fn read(&self, gpu: &Gpu) -> Result<Option<Duration>, Error> {
        let mut ticks = [0_u64; 2];
        unsafe {
            gpu.device().get_query_pool_results(
                self.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }?;

        // A device may report the two in either order if the pipeline stages allowed reordering;
        // a negative interval is not a measurement, so it becomes `None` rather than a wrap.
        let Some(elapsed) = ticks
            .get(1)
            .and_then(|end| ticks.first().and_then(|start| end.checked_sub(*start)))
        else {
            return Ok(None);
        };

        let nanos = elapsed as f64 * f64::from(self.period);
        Ok(Some(Duration::from_nanos(nanos as u64)))
    }

    /// Release the pool.
    ///
    /// # Safety
    ///
    /// No submission using it may still be in flight.
    pub(super) unsafe fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device().destroy_query_pool(self.pool, None) };
    }
}
