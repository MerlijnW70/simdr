use crate::{Error, Gpu};
use ash::vk;
use std::time::Duration;

pub(super) struct Timestamps {
    pool: vk::QueryPool,
    period: f32,
    marks: u32,
}

impl Timestamps {
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
            let elapsed = at.saturating_sub(previous);
            previous = at;
            spans.push(Duration::from_nanos(
                (elapsed as f64 * f64::from(self.period)) as u64,
            ));
        }
        Ok(spans)
    }

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

        let Some(elapsed) = ticks
            .get(1)
            .and_then(|end| ticks.first().and_then(|start| end.checked_sub(*start)))
        else {
            return Ok(None);
        };

        let nanos = elapsed as f64 * f64::from(self.period);
        Ok(Some(Duration::from_nanos(nanos as u64)))
    }

    pub(super) unsafe fn destroy(self, gpu: &Gpu) {
        // SAFETY: `self` is taken by value so nothing else names this pool, and the caller's
        // contract says no submission using it is still in flight.
        unsafe { gpu.device().destroy_query_pool(self.pool, None) };
    }
}
