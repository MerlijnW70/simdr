use super::pipeline::Pipeline;
use crate::buffer::Buffer;
use crate::{Error, Gpu};
use std::time::Duration;

pub struct Session<'gpu> {
    gpu: &'gpu Gpu,
    bounds: super::extent::Bounds,
    held: Vec<usize>,
    staging: Option<Buffer>,
    buffers: Vec<Buffer>,
    pipeline: Option<Pipeline>,
}

impl Gpu {
    pub fn session<'gpu>(
        &'gpu self,
        spirv: &[u32],
        sizes: &[usize],
    ) -> Result<Session<'gpu>, Error> {
        if sizes.is_empty() {
            return Err(Error::NoPipeline);
        }

        let bytes: Vec<u64> = sizes
            .iter()
            .map(|&words| (words.max(1) * size_of::<u32>()) as u64)
            .collect();
        let staging_bytes = bytes.iter().copied().max().unwrap_or(4);

        // SAFETY: everything allocated here is owned by the `Session` and destroyed in its `Drop`,
        // which cannot run while a dispatch is in flight because every dispatch waits on a fence
        // before returning.
        unsafe {
            let staging = Buffer::staging(self, staging_bytes)?;

            let mut buffers = Vec::with_capacity(bytes.len());
            for &size in &bytes {
                match Buffer::shared(self, size) {
                    Ok(buffer) => buffers.push(buffer),
                    Err(error) => {
                        for buffer in buffers {
                            buffer.destroy(self);
                        }
                        staging.destroy(self);
                        return Err(error);
                    }
                }
            }

            let bound: Vec<(&Buffer, u64)> = buffers
                .iter()
                .zip(&bytes)
                .map(|(buffer, &size)| (buffer, size))
                .collect();

            match Pipeline::new(self, spirv, &bound, &super::Specialization::none()) {
                Ok(pipeline) => Ok(Session {
                    gpu: self,
                    bounds: super::extent::Bounds::of(spirv, &super::Specialization::none()),
                    held: buffers.iter().map(Buffer::capacity).collect(),
                    staging: Some(staging),
                    buffers,
                    pipeline: Some(pipeline),
                }),
                Err(error) => {
                    for buffer in buffers {
                        buffer.destroy(self);
                    }
                    staging.destroy(self);
                    Err(error)
                }
            }
        }
    }
}

impl Session<'_> {
    #[must_use]
    pub fn bindings(&self) -> usize {
        self.buffers.len()
    }

    pub fn write(&mut self, index: usize, words: &[u32]) -> Result<(), Error> {
        let (Some(target), Some(staging)) = (self.buffers.get(index), self.staging.as_ref()) else {
            return Err(Error::NoPipeline);
        };

        if words.is_empty() {
            return Ok(());
        }

        let bytes = fitting(words.len(), target.capacity())?;

        // SAFETY: both buffers are this session's and no dispatch is in flight — every one of them
        // waits on its fence before returning.
        unsafe {
            match super::deliver(self.gpu, words, staging, target)? {
                Some(_) => self.gpu.copy(staging, target, bytes),
                None => Ok(()),
            }
        }
    }

    pub fn read(&mut self, index: usize, count: usize) -> Result<Vec<u32>, Error> {
        let (Some(source), Some(staging)) = (self.buffers.get(index), self.staging.as_ref()) else {
            return Err(Error::NoPipeline);
        };
        let bytes = fitting(count, source.capacity())?;

        // SAFETY: as above.
        unsafe {
            self.gpu.copy(source, staging, bytes)?;
            staging.read(self.gpu, count)
        }
    }

    pub fn dispatch(&mut self, workgroups: u32, iterations: u32) -> Result<Duration, Error> {
        self.dispatch_grid(super::Grid::linear(workgroups), iterations)
    }

    pub fn dispatch_grid(&mut self, grid: super::Grid, iterations: u32) -> Result<Duration, Error> {
        if let Some(overrun) = self.bounds.overrun(grid, &self.held) {
            return Err(overrun.into());
        }

        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(Error::NoPipeline);
        };

        // SAFETY: the pipeline and its buffers are alive for as long as `self` is, and this waits
        // on a fence before returning.
        unsafe { self.gpu.dispatch(pipeline, grid, iterations.max(1)) }
    }
}

fn fitting(words: usize, capacity: usize) -> Result<u64, Error> {
    if words > capacity {
        return Err(Error::TooLarge { words, capacity });
    }
    Ok((words.max(1) * size_of::<u32>()) as u64)
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        // SAFETY: every object here was created by `Gpu::session` and nothing else holds it. The
        // device is idle with respect to them: `dispatch`, `write` and `read` each wait on a fence
        // before returning, so nothing can still be in flight.
        unsafe {
            if let Some(pipeline) = self.pipeline.take() {
                pipeline.destroy(self.gpu);
            }
            for buffer in std::mem::take(&mut self.buffers) {
                buffer.destroy(self.gpu);
            }
            if let Some(staging) = self.staging.take() {
                staging.destroy(self.gpu);
            }
        }
    }
}
