use super::{Grid, Specialization};
use crate::timing::Timing;
use crate::{Error, Gpu};
use std::time::Duration;

impl Gpu {
    pub fn run(&self, spirv: &[u32], input: &[f32], workgroups: u32) -> Result<Vec<f32>, Error> {
        let words: Vec<u32> = input.iter().map(|value| value.to_bits()).collect();
        let output = self.run_words(spirv, &words, workgroups)?;
        Ok(output.into_iter().map(f32::from_bits).collect())
    }

    pub fn run_u32(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
    ) -> Result<Vec<u32>, Error> {
        self.run_words(spirv, input, workgroups)
    }

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

    fn run_words(&self, spirv: &[u32], input: &[u32], workgroups: u32) -> Result<Vec<u32>, Error> {
        self.run_grid(spirv, input, Grid::linear(workgroups))
    }

    pub fn run_grid(&self, spirv: &[u32], input: &[u32], grid: Grid) -> Result<Vec<u32>, Error> {
        self.execute(spirv, input, grid, 1, &Specialization::none())
            .map(|out| out.0)
    }

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

    pub fn time(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
        iterations: u32,
    ) -> Result<Duration, Error> {
        self.time_grid(spirv, input, Grid::linear(workgroups), iterations)
    }

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

    pub fn time_repeated(
        &self,
        spirv: &[u32],
        input: &[u32],
        workgroups: u32,
        iterations: u32,
        repeats: u32,
    ) -> Result<Timing, Error> {
        let repeats = repeats.max(1);
        let mut samples = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats {
            samples.push(self.time(spirv, input, workgroups, iterations)?);
        }
        Timing::of(&samples).ok_or(Error::NoPipeline)
    }
}
