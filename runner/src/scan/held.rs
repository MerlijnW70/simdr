use super::passes::{self, Ends, Modules, Slots};
use super::plan::{self, Level};
use crate::buffer::Buffer;
use crate::dispatch::{Pipeline, Staged, deliver_floats};
use crate::kernels;
use crate::{Error, Gpu};
use simdr::lanes::F32;
use std::time::Duration;

pub struct Scanner<'gpu> {
    gpu: &'gpu Gpu,
    elements: usize,
    staging: Option<Buffer>,
    buffers: Vec<Buffer>,
    answer: usize,
    pipelines: Vec<Pipeline>,
    workgroups: Vec<u32>,
}

impl Gpu {
    pub fn scanner<'gpu>(&'gpu self, elements: usize) -> Result<Scanner<'gpu>, Error> {
        self.build_scanner(elements, None)
    }

    pub fn scanner_of<'gpu>(
        &'gpu self,
        elements: usize,
        map: &[u32],
    ) -> Result<Scanner<'gpu>, Error> {
        self.build_scanner(elements, Some(map))
    }

    fn build_scanner<'gpu>(
        &'gpu self,
        elements: usize,
        map: Option<&[u32]>,
    ) -> Result<Scanner<'gpu>, Error> {
        let levels = plan::levels(elements)?;
        let width = self.limits().subgroup_size;

        let blocks = kernels::scan::scan_blocks::<F32>(width).map_err(Error::Emit)?;
        let blocks_exclusive =
            kernels::scan::scan_blocks_exclusive::<F32>(width).map_err(Error::Emit)?;
        let top = kernels::scan::scan_workgroup_exclusive::<F32>(width).map_err(Error::Emit)?;
        let add = kernels::scan::add_offsets::<F32>(width).map_err(Error::Emit)?;

        let words = size_of::<f32>() as u64;
        let bytes = (elements.max(1) as u64) * words;

        // SAFETY: everything allocated here is owned by the `Scanner` and destroyed in its `Drop`,
        // which cannot run while a dispatch is in flight — every dispatch waits on a fence before
        // returning. `Held::fail` releases whatever was allocated before an early return.
        unsafe {
            let staging = Buffer::staging(self, bytes)?;
            let mut held = Held {
                gpu: self,
                staging,
                buffers: Vec::new(),
                pipelines: Vec::new(),
                workgroups: Vec::new(),
            };

            let input = match held.shared(bytes) {
                Ok(index) => index,
                Err(error) => return Err(held.fail(error)),
            };
            let mapped = match map {
                None => None,
                Some(_) => match held.local(bytes) {
                    Ok(index) => Some(index),
                    Err(error) => return Err(held.fail(error)),
                },
            };
            let scanned = match held.local(bytes) {
                Ok(index) => index,
                Err(error) => return Err(held.fail(error)),
            };
            let output = match held.local(bytes) {
                Ok(index) => index,
                Err(error) => return Err(held.fail(error)),
            };

            let mut slots = Vec::with_capacity(levels.len());
            for (depth, level) in levels.iter().enumerate() {
                let at_top = depth + 1 == levels.len();
                match held.level(level, at_top, words) {
                    Ok(found) => slots.push(found),
                    Err(error) => return Err(held.fail(error)),
                }
            }

            if let Err(error) = held.record(
                &levels,
                &slots,
                Ends {
                    input,
                    mapped,
                    scanned,
                    output,
                },
                &Modules {
                    blocks: &blocks,
                    blocks_exclusive: &blocks_exclusive,
                    top: &top,
                    add: &add,
                    map,
                },
            ) {
                return Err(held.fail(error));
            }

            if held.pipelines.len() != plan::dispatches(levels.len(), map.is_some()) {
                return Err(held.fail(Error::NoPipeline));
            }

            Ok(held.into_scanner(elements, output))
        }
    }
}

struct Held<'gpu> {
    gpu: &'gpu Gpu,
    staging: Buffer,
    buffers: Vec<Buffer>,
    pipelines: Vec<Pipeline>,
    workgroups: Vec<u32>,
}

impl<'gpu> Held<'gpu> {
    unsafe fn local(&mut self, bytes: u64) -> Result<usize, Error> {
        // SAFETY: `Buffer::device_local` asks for a live device and a caller who will destroy
        // what comes back. The device outlives this builder, and everything pushed here is
        // released by either `Held::fail` or `Scanner::drop`.
        let buffer = unsafe { Buffer::device_local(self.gpu, bytes) }?;
        self.buffers.push(buffer);
        Ok(self.buffers.len() - 1)
    }

    unsafe fn shared(&mut self, bytes: u64) -> Result<usize, Error> {
        // SAFETY: as `local` — the same contract, for a buffer that also asks to be host-writable.
        let buffer = unsafe { Buffer::shared(self.gpu, bytes) }?;
        self.buffers.push(buffer);
        Ok(self.buffers.len() - 1)
    }

    unsafe fn level(&mut self, level: &Level, at_top: bool, words: u64) -> Result<Slots, Error> {
        let bytes = (level.capacity as u64) * words;

        // SAFETY: `zeroed` asks what this function's own contract asks, and each of the three
        // calls allocates a separate buffer this builder then owns.
        let totals = unsafe { self.zeroed(bytes) }?;
        let scanned = if at_top {
            None
        } else {
            // SAFETY: as above.
            Some(unsafe { self.zeroed(bytes) }?)
        };
        // SAFETY: as above.
        let offsets = unsafe { self.zeroed(bytes) }?;

        Ok(Slots {
            totals,
            scanned,
            offsets,
        })
    }

    unsafe fn zeroed(&mut self, bytes: u64) -> Result<usize, Error> {
        // SAFETY: as this function's own contract.
        let index = unsafe { self.local(bytes) }?;
        let Some(buffer) = self.buffers.get(index) else {
            return Err(Error::NoPipeline);
        };

        let zeros = vec![0_u32; (bytes / size_of::<u32>() as u64) as usize];
        // SAFETY: the staging buffer is this builder's own and is at least as large as any level —
        // it was sized for the whole input, which every level is a fraction of. Nothing is in
        // flight: no pipeline has been recorded yet.
        unsafe {
            self.staging.write(self.gpu, &zeros)?;
            self.gpu.copy(&self.staging, buffer, bytes)?;
        }
        Ok(index)
    }

    unsafe fn pass(
        &mut self,
        spirv: &[u32],
        bound: &[(usize, u64)],
        workgroups: u32,
    ) -> Result<(), Error> {
        let mut buffers = Vec::with_capacity(bound.len());
        for &(index, bytes) in bound {
            let Some(buffer) = self.buffers.get(index) else {
                return Err(Error::NoPipeline);
            };
            buffers.push((buffer, bytes));
        }

        let held: Vec<usize> = buffers
            .iter()
            .map(|&(_, bytes)| (bytes / size_of::<u32>() as u64) as usize)
            .collect();
        if let Some(overrun) =
            crate::dispatch::Bounds::of(spirv, &crate::dispatch::Specialization::none())
                .overrun(crate::dispatch::Grid::linear(workgroups), &held)
        {
            return Err(overrun.into());
        }

        // SAFETY: every buffer named is one this builder allocated and still owns, and none is in
        // use — nothing has been submitted yet.
        let pipeline =
            unsafe { Pipeline::new(self.gpu, spirv, &buffers, &crate::Specialization::none()) }?;
        self.pipelines.push(pipeline);
        self.workgroups.push(workgroups);
        Ok(())
    }

    fn fail(self, error: Error) -> Error {
        // SAFETY: nothing was ever submitted, so no pipeline or buffer is in flight. Pipelines go
        // first: a descriptor set naming a destroyed buffer would be a dangling reference.
        unsafe {
            for pipeline in self.pipelines {
                pipeline.destroy(self.gpu);
            }
            for buffer in self.buffers {
                buffer.destroy(self.gpu);
            }
            self.staging.destroy(self.gpu);
        }
        error
    }

    fn into_scanner(self, elements: usize, answer: usize) -> Scanner<'gpu> {
        Scanner {
            gpu: self.gpu,
            elements,
            staging: Some(self.staging),
            buffers: self.buffers,
            answer,
            pipelines: self.pipelines,
            workgroups: self.workgroups,
        }
    }
}

impl Held<'_> {
    unsafe fn record(
        &mut self,
        levels: &[Level],
        slots: &[Slots],
        ends: Ends,
        modules: &Modules<'_>,
    ) -> Result<(), Error> {
        let elements = self.buffers.get(ends.input).map_or(0, Buffer::capacity);

        for pass in passes::passes(levels, slots, ends, modules, elements)? {
            // SAFETY: every slot in a pass came from `ends` or `slots`, both of which hold only
            // indices this builder allocated above and still owns. Nothing has been submitted, so
            // none of them is in use.
            unsafe {
                self.pass(pass.module, &pass.bound, pass.workgroups)?;
            }
        }

        Ok(())
    }
}

impl Scanner<'_> {
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    #[must_use]
    pub fn dispatches(&self) -> usize {
        self.pipelines.len()
    }

    pub fn scan(&mut self, input: &[f32]) -> Result<Vec<f32>, Error> {
        self.run(input).map(|(answer, _)| answer)
    }

    pub fn scan_timed(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<Duration>), Error> {
        self.run(input)
    }

    fn run(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<Duration>), Error> {
        if input.len() != self.elements {
            return Err(Error::TooLarge {
                words: input.len(),
                capacity: self.elements,
            });
        }

        let (Some(staging), Some(source), Some(answer)) = (
            self.staging.as_ref(),
            self.buffers.first(),
            self.buffers.get(self.answer),
        ) else {
            return Err(Error::NoPipeline);
        };
        let bytes = (self.elements.max(1) * size_of::<f32>()) as u64;

        // SAFETY: every buffer and pipeline here is owned by `self` and outlives the call, and the
        // submission waits on a fence before returning.
        let (output, spans) = unsafe {
            let upload = deliver_floats(self.gpu, input, staging, source)?;

            let spans = self.gpu.replay_timed(
                &self.pipelines,
                &self.workgroups,
                upload,
                Some(Staged {
                    from: answer,
                    to: staging,
                    bytes,
                }),
            )?;
            (staging.read(self.gpu, self.elements)?, spans)
        };

        Ok((output.into_iter().map(f32::from_bits).collect(), spans))
    }
}

impl Drop for Scanner<'_> {
    fn drop(&mut self) {
        // SAFETY: every object here was created by `Gpu::scanner` and nothing else holds it. The
        // device is idle with respect to them: `scan` waits on a fence before returning. Pipelines
        // go first — a descriptor set naming a destroyed buffer would be a dangling reference for
        // as long as it existed.
        unsafe {
            for pipeline in std::mem::take(&mut self.pipelines) {
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
