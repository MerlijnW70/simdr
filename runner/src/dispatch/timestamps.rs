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

/// Timestamp queries bracketing one submission, and optionally marking points inside it.
///
/// Query 0 opens and query 1 closes; anything after them is a **mark** a caller wrote part way
/// through. That is what turns a breakdown into the call rather than a set of probes resembling it:
/// a mark after each pass of a chain is measured *in company*, paying whatever the pass beside it
/// makes it pay, where a probe timed on its own pays its own fixed costs and nothing else's.
pub(super) struct Timestamps {
    pool: vk::QueryPool,
    /// Nanoseconds per tick, from the device's limits.
    period: f32,
    /// How many marks this pool has room for, past the opening and closing pair.
    marks: u32,
}

impl Timestamps {
    /// Create a query pool, or `None` if this device cannot answer.
    ///
    /// # Safety
    ///
    /// [`Timestamps::destroy`] must run before the device goes away.
    pub(super) unsafe fn new(gpu: &Gpu, marks: u32) -> Result<Option<Self>, Error> {
        let period = gpu.limits().timestamp_period_ns;
        if period <= 0.0 {
            return Ok(None);
        }

        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2 + marks);
        // SAFETY: the device outlives this call, and this function's own contract puts the
        // matching `destroy` before it goes away. The pool is sized for the opening and closing
        // pair plus however many marks the caller says it will write, and `mark` refuses any index
        // past that rather than writing outside the pool.
        let pool = unsafe { gpu.device().create_query_pool(&info, None) }?;

        Ok(Some(Self {
            pool,
            period,
            marks,
        }))
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
        // SAFETY: `command` is in the recording state, which this function's contract requires,
        // and the pool is this object's own — created with exactly the two queries indexed here.
        // The reset is recorded rather than assumed: a pool's contents are undefined until it is.
        unsafe {
            device.cmd_reset_query_pool(command, self.pool, 0, 2 + self.marks);
            device.cmd_write_timestamp(command, vk::PipelineStageFlags::TOP_OF_PIPE, self.pool, 0);
        }
    }

    /// Write the closing timestamp.
    ///
    /// # Safety
    ///
    /// As [`Timestamps::begin`], and after it.
    pub(super) unsafe fn end(&self, gpu: &Gpu, command: vk::CommandBuffer) {
        // SAFETY: as `begin`, and query 1 is the second of the two the pool was created with —
        // reset by the `begin` this function's contract says came first.
        unsafe {
            gpu.device().cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                1,
            );
        }
    }

    /// Write mark `index`, which a caller records part way through its own command buffer.
    ///
    /// **`BOTTOM_OF_PIPE`, so the mark lands after the work before it has finished** rather than
    /// after it was merely issued. A `TOP_OF_PIPE` mark between two dispatches would say when the
    /// second was *submitted*, which on a device that overlaps them is not when the first ended.
    ///
    /// Out-of-range indices are dropped rather than written: the pool was sized for a promise the
    /// caller made, and writing past it is undefined where doing nothing costs one measurement.
    ///
    /// # Safety
    ///
    /// As [`Timestamps::begin`].
    pub(super) unsafe fn mark(&self, gpu: &Gpu, command: vk::CommandBuffer, index: u32) {
        if index >= self.marks {
            return;
        }
        // SAFETY: `command` is recording, and the query index is inside the pool because the
        // check above is what makes it so.
        unsafe {
            gpu.device().cmd_write_timestamp(
                command,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                2 + index,
            );
        }
    }

    /// How long each marked span took, in order, measured from the opening timestamp.
    ///
    /// Span `i` runs from mark `i - 1` to mark `i`, and span 0 from the opening timestamp. Empty
    /// if the device gave nothing usable, which is the same fallback [`Timestamps::read`] takes.
    ///
    /// # Safety
    ///
    /// As [`Timestamps::read`].
    pub(super) unsafe fn spans(&self, gpu: &Gpu) -> Result<Vec<Duration>, Error> {
        if self.marks == 0 {
            return Ok(Vec::new());
        }

        let mut ticks = vec![0_u64; (2 + self.marks) as usize];
        // SAFETY: the submission has been waited on, so every query is available, and the vector
        // is exactly as long as the pool.
        unsafe {
            gpu.device().get_query_pool_results(
                self.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }?;

        let mut spans = Vec::with_capacity(self.marks as usize);
        let mut previous = ticks.first().copied().unwrap_or_default();
        for index in 0..self.marks as usize {
            let Some(at) = ticks.get(2 + index).copied() else {
                break;
            };
            // A device may reorder these; a negative interval is not a measurement, so it becomes
            // zero rather than an enormous number wrapped round.
            let elapsed = at.saturating_sub(previous);
            previous = at;
            spans.push(Duration::from_nanos(
                (elapsed as f64 * f64::from(self.period)) as u64,
            ));
        }
        Ok(spans)
    }

    /// Read the elapsed device time, once the submission has completed.
    ///
    /// # Safety
    ///
    /// The fence for the submission that wrote these must have been waited on.
    pub(super) unsafe fn read(&self, gpu: &Gpu) -> Result<Option<Duration>, Error> {
        let mut ticks = [0_u64; 2];
        // SAFETY: the submission that wrote these has been waited on, which this function's
        // contract requires, so both queries are available. `ticks` is two `u64`s for the two
        // queries the pool holds, which is what `TYPE_64` says they are.
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
        // SAFETY: `self` is taken by value so nothing else names this pool, and the caller's
        // contract says no submission using it is still in flight.
        unsafe { gpu.device().destroy_query_pool(self.pool, None) };
    }
}
