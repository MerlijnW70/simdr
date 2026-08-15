//! The calls a caller makes: run a module over a buffer, or time one.
//!
//! Every one of these is a thin shell around [`Gpu::execute`] in [`super`] — what differs between
//! them is how the caller's data is spelled on the way in and out. `f32` and `u32` are the buffer's
//! words as they stand; `i8` and `f16` are four and two elements to a word, packed here because
//! the kernel's `ArrayStride` says so and nothing below this layer knows the element type.
//!
//! Split from [`super`] because that file is the staging machinery — three buffers, three
//! submissions, a fence — and this is the surface over it. Both were one file until the surface
//! grew a conversion per element width.

use super::{Grid, Specialization};
use crate::timing::Timing;
use crate::{Error, Gpu};
use std::time::Duration;

impl Gpu {
    /// Run `spirv` over `input`, and return what it wrote to its output binding.
    ///
    /// The module must be a compute shader named `main` with two `StorageBuffer` bindings in
    /// descriptor set 0 — binding 0 read, binding 1 written — which is the shape every kernel in
    /// `simdr` emits. `workgroups` is the dispatch's x dimension; [`Gpu::run_grid`] takes both.
    ///
    /// The words go to `vkCreateShaderModule` exactly as the emitter produced them. Nothing here
    /// rewrites them, which is the point: any disagreement that shows up is between our module and
    /// the driver, with no third opinion in between. They are *read* — see below.
    ///
    /// # Both buffers are `input.len()` long
    ///
    /// The output is allocated to the same length as the input and returned whole, so a kernel
    /// that writes fewer elements than it read leaves the rest of the returned vector holding
    /// whatever the device's memory held — which is zeros on two drivers here and is not on
    /// lavapipe. Which prefix is meaningful is the caller's arithmetic. [`Gpu::run_bound`] sizes
    /// the output separately, and takes several inputs while it is there.
    ///
    /// That equal length is what makes this a one-argument call, and it is also the trap in it: a
    /// dispatch big enough to need more elements than the buffer holds writes off the end of it.
    /// So `workgroups` is checked against the buffer before anything is allocated — the workgroup
    /// size and the element size are read out of `spirv` rather than taken on trust. See
    /// `dispatch::extent` for what that check is and, more importantly, what it is not.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if the dispatch needs more elements than `input` holds,
    /// [`Error::Vulkan`] if any call fails, [`Error::NoPipeline`] if the driver returns none.
    pub fn run(&self, spirv: &[u32], input: &[f32], workgroups: u32) -> Result<Vec<f32>, Error> {
        let words: Vec<u32> = input.iter().map(|value| value.to_bits()).collect();
        let output = self.run_words(spirv, &words, workgroups)?;
        Ok(output.into_iter().map(f32::from_bits).collect())
    }

    /// The same, for a kernel whose buffers hold 32-bit integers.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_u32(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
    ) -> Result<Vec<u32>, Error> {
        self.run_words(spirv, input, workgroups)
    }

    /// The same, for a kernel whose buffers hold bytes — `i8` or `u8`.
    ///
    /// The buffer is still words underneath, because that is what a Vulkan allocation is; four
    /// elements share each one. Element `i` lands at byte offset `i`, which is what the kernel's
    /// `ArrayStride` of 1 says it should, so the packing is little-endian and not a choice.
    ///
    /// A length that is not a multiple of four is padded up. The padding is written and read back
    /// and then dropped, so a caller sees exactly the elements it passed.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_bytes(
        &self,
        spirv: &[u32],
        input: &[u8],
        workgroups: u32,
    ) -> Result<Vec<u8>, Error> {
        let words: Vec<u32> = input
            .chunks(4)
            .map(|chunk| {
                let mut word = [0_u8; 4];
                // `chunks` yields at most four, so this is a copy with the tail left zero rather
                // than a case that can fail.
                if let Some(slot) = word.get_mut(..chunk.len()) {
                    slot.copy_from_slice(chunk);
                }
                u32::from_le_bytes(word)
            })
            .collect();

        let output = self.run_words(spirv, &words, workgroups)?;
        let mut bytes: Vec<u8> = output.iter().flat_map(|word| word.to_le_bytes()).collect();
        bytes.truncate(input.len());
        Ok(bytes)
    }

    /// The same, for a kernel whose buffers hold 16-bit elements — `i16`, `u16` or `f16`.
    ///
    /// A half's bits rather than a number: [`simdr::half::from_f32`] is what turns a float into
    /// one, and this layer has no way to tell the three 16-bit types apart.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_halves(
        &self,
        spirv: &[u32],
        input: &[u16],
        workgroups: u32,
    ) -> Result<Vec<u16>, Error> {
        let words: Vec<u32> = input
            .chunks(2)
            .map(|chunk| match chunk {
                [low, high] => u32::from(*low) | (u32::from(*high) << 16),
                [low] => u32::from(*low),
                _ => 0,
            })
            .collect();

        let output = self.run_words(spirv, &words, workgroups)?;
        let mut halves: Vec<u16> = output
            .iter()
            .flat_map(|word| [*word as u16, (word >> 16) as u16])
            .collect();
        halves.truncate(input.len());
        Ok(halves)
    }

    /// Run over raw words, which is what the buffers actually hold.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_words(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
    ) -> Result<Vec<u32>, Error> {
        self.run_grid(spirv, input, Grid::linear(workgroups))
    }

    /// Run over raw words, dispatching on both axes.
    ///
    /// What a kernel built from [`simdr::kernel::Shape::grid`] needs: its rows come from the
    /// dispatch's y, from a workgroup several invocations deep, or from both. Everything else here
    /// is [`Grid::linear`] of the count it was given.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_grid(&self, spirv: &[u32], input: &[u32], grid: Grid) -> Result<Vec<u32>, Error> {
        self.execute(spirv, input, grid, 1, &Specialization::none())
            .map(|out| out.0)
    }

    /// The same, with some of the module's specialization constants given values.
    ///
    /// The point of the whole mechanism: one `spirv` here, several `specialization`s, and a
    /// different pipeline each time without the module being rebuilt. What can and cannot be
    /// deferred this way is `decisions/DR-0005`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn run_specialized(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
        specialization: &Specialization,
    ) -> Result<Vec<u32>, Error> {
        self.execute(spirv, input, Grid::linear(workgroups), 1, specialization)
            .map(|out| out.0)
    }

    /// Time `iterations` dispatches of a specialized pipeline.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn time_specialized(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
        iterations: u32,
        specialization: &Specialization,
    ) -> Result<Duration, Error> {
        self.execute(
            spirv,
            input,
            Grid::linear(workgroups),
            iterations,
            specialization,
        )
        .map(|out| out.1)
    }

    /// Time `iterations` back-to-back dispatches of `spirv`.
    ///
    /// Only the dispatch submission is timed: allocation, pipeline creation and both host copies
    /// happen outside it, and the kernel's buffers are device-local. A memory barrier separates
    /// the iterations, so this is the sum of the kernels' own times rather than a measure of how
    /// well the scheduler overlaps them.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn time(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
        iterations: u32,
    ) -> Result<Duration, Error> {
        self.time_grid(spirv, input, Grid::linear(workgroups), iterations)
    }

    /// Time `iterations` back-to-back dispatches over both axes.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`].
    pub fn time_grid(
        &self,
        spirv: &[u32],
        input: &[u32],
        grid: Grid,
        iterations: u32,
    ) -> Result<Duration, Error> {
        self.execute(spirv, input, grid, iterations, &Specialization::none())
            .map(|out| out.1)
    }

    /// Time the same dispatch `repeats` times over, and report the spread.
    ///
    /// One number per measurement reads like a result whether or not it is one — the sweep that
    /// refuted this project's cache-capacity story looked authoritative in its first run and moved
    /// its cliff in the second. [`Timing`] is what says so on the spot.
    ///
    /// Each repeat is a fresh submission of `iterations` dispatches over buffers that stay
    /// allocated, so the repeats differ only in what the machine was doing at the time — which is
    /// exactly the thing being measured.
    ///
    /// # Errors
    ///
    /// As [`Gpu::run`], or [`Error::NoPipeline`] if `repeats` is zero and there is nothing to
    /// summarise.
    pub fn time_repeated(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
        iterations: u32,
        repeats: u32,
    ) -> Result<Timing, Error> {
        let mut samples = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats.max(1) {
            samples.push(self.time(spirv, input, workgroups, iterations)?);
        }
        Timing::of(&samples).ok_or(Error::NoPipeline)
    }
}
