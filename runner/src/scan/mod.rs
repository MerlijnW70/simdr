mod held;
mod passes;
mod plan;

pub use held::Scanner;

use crate::{Error, Gpu};

impl Gpu {
    pub fn scan(&self, input: &[f32]) -> Result<Vec<f32>, Error> {
        self.scanner(input.len())?.scan(input)
    }
}
