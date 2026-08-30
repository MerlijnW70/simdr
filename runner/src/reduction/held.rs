use super::Reduction;
use crate::buffer::Buffer;
use crate::dispatch::{Ends, Pipeline, Staged, answer_in_destination, deliver_floats};
use crate::{Error, Gpu};
use std::time::Duration;

const ANSWER_BYTES: u64 = size_of::<f32>() as u64;

pub struct Reducer<'gpu> {
    gpu: &'gpu Gpu,
    elements: usize,
    staging: Option<Buffer>,
    source: Option<Buffer>,
    destination: Option<Buffer>,
    pipelines: Vec<Pipeline>,
    workgroups: Vec<u32>,
}

impl Gpu {
    pub fn reducer<'gpu>(&'gpu self, elements: usize) -> Result<Reducer<'gpu>, Error> {
        self.build_reducer(elements, None)
    }

    pub fn reducer_of<'gpu>(
        &'gpu self,
        elements: usize,
        map: &[u32],
    ) -> Result<Reducer<'gpu>, Error> {
        self.build_reducer(elements, Some(map))
    }

    fn build_reducer<'gpu>(
        &'gpu self,
        elements: usize,
        map: Option<&[u32]>,
    ) -> Result<Reducer<'gpu>, Error> {
        let stages = super::plan::stages(self.limits().subgroup_size, elements, map)?;

        let bytes = (elements.max(1) * size_of::<u32>()) as u64;
        let workgroups: Vec<u32> = stages.iter().map(|stage| stage.workgroups).collect();

        // SAFETY: everything allocated here is owned by the `Reducer` and destroyed in its `Drop`,
        // which cannot run while a dispatch is in flight — every dispatch waits on a fence before
        // returning. The early returns below release what was allocated before them.
        unsafe {
            let staging = Buffer::staging(self, bytes)?;
            let source = match Buffer::shared(self, bytes) {
                Ok(buffer) => buffer,
                Err(error) => {
                    staging.destroy(self);
                    return Err(error);
                }
            };
            let destination = match Buffer::device_local(self, bytes) {
                Ok(buffer) => buffer,
                Err(error) => {
                    source.destroy(self);
                    staging.destroy(self);
                    return Err(error);
                }
            };

            let mut pipelines: Vec<Pipeline> = Vec::with_capacity(stages.len());
            for (index, stage) in stages.iter().enumerate() {
                let (read, written) = Ends::of(index).order(&source, &destination);

                if let Some(overrun) = crate::dispatch::Bounds::of(
                    &stage.words,
                    &crate::dispatch::Specialization::none(),
                )
                .overrun_uniform(crate::Grid::linear(stage.workgroups), elements)
                {
                    for pipeline in pipelines {
                        pipeline.destroy(self);
                    }
                    destination.destroy(self);
                    source.destroy(self);
                    staging.destroy(self);
                    return Err(overrun.into());
                }

                match Pipeline::new(
                    self,
                    &stage.words,
                    &[(read, bytes), (written, bytes)],
                    &crate::Specialization::none(),
                ) {
                    Ok(pipeline) => pipelines.push(pipeline),
                    Err(error) => {
                        for pipeline in pipelines {
                            pipeline.destroy(self);
                        }
                        destination.destroy(self);
                        source.destroy(self);
                        staging.destroy(self);
                        return Err(error);
                    }
                }
            }

            Ok(Reducer {
                gpu: self,
                elements,
                staging: Some(staging),
                source: Some(source),
                destination: Some(destination),
                pipelines,
                workgroups,
            })
        }
    }
}

impl Reducer<'_> {
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    #[must_use]
    pub fn dispatches(&self) -> usize {
        self.pipelines.len()
    }

    pub fn sum_timed(&mut self, input: &[f32]) -> Result<(Reduction, Vec<Duration>), Error> {
        self.run(input)
    }

    pub fn sum(&mut self, input: &[f32]) -> Result<Reduction, Error> {
        self.run(input).map(|(reduction, _)| reduction)
    }

    fn run(&mut self, input: &[f32]) -> Result<(Reduction, Vec<Duration>), Error> {
        if input.len() != self.elements {
            return Err(Error::TooLarge {
                words: input.len(),
                capacity: self.elements,
            });
        }

        let (Some(staging), Some(source), Some(destination)) = (
            self.staging.as_ref(),
            self.source.as_ref(),
            self.destination.as_ref(),
        ) else {
            return Err(Error::NoPipeline);
        };

        // SAFETY: every buffer and pipeline here is owned by `self` and outlives the call, and the
        // submission waits on a fence before returning.
        let (output, spans) = unsafe {
            let upload = deliver_floats(self.gpu, input, staging, source)?;

            let answer = if answer_in_destination(self.pipelines.len()) {
                destination
            } else {
                source
            };

            let spans = self.gpu.replay_timed(
                &self.pipelines,
                &self.workgroups,
                upload,
                Some(Staged {
                    from: answer,
                    to: staging,
                    bytes: ANSWER_BYTES,
                }),
            )?;
            (staging.read(self.gpu, 1)?, spans)
        };

        let total = output
            .first()
            .copied()
            .map(f32::from_bits)
            .ok_or(Error::NoPipeline)?;

        Ok((
            Reduction {
                total,
                dispatches: self.pipelines.len(),
                host_combined: 1,
            },
            spans,
        ))
    }
}

impl Drop for Reducer<'_> {
    fn drop(&mut self) {
        // SAFETY: every object here was created by `Gpu::reducer` and nothing else holds it. The
        // device is idle with respect to them: `sum` waits on a fence before returning, so nothing
        // can still be in flight. Pipelines go first — a descriptor set naming a destroyed buffer
        // would be a dangling reference for as long as it existed.
        unsafe {
            for pipeline in std::mem::take(&mut self.pipelines) {
                pipeline.destroy(self.gpu);
            }
            if let Some(buffer) = self.destination.take() {
                buffer.destroy(self.gpu);
            }
            if let Some(buffer) = self.source.take() {
                buffer.destroy(self.gpu);
            }
            if let Some(staging) = self.staging.take() {
                staging.destroy(self.gpu);
            }
        }
    }
}
