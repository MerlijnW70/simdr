mod addressing;

use super::{Grid, Specialization};
use simdr::decode;
use simdr::module::op;
use simdr::spec::{Decoration, ExecutionMode};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub(crate) struct Bounds {
    local: Option<[u64; 3]>,
    stride: Option<u64>,
    needs: BTreeMap<u32, addressing::Needs>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Overrun {
    pub(crate) binding: Option<u32>,
    pub(crate) needed: usize,
    pub(crate) held: usize,
}

impl From<Overrun> for crate::Error {
    fn from(overrun: Overrun) -> Self {
        Self::Overrun {
            binding: overrun.binding,
            needed: overrun.needed,
            held: overrun.held,
        }
    }
}

impl Bounds {
    pub(crate) fn of(spirv: &[u32], chosen: &Specialization) -> Self {
        let local = local_size(spirv);
        Self {
            local,
            stride: element_bytes(spirv),
            needs: addressing::needs(spirv, local.map_or(0, |sizes| sizes[0]), chosen),
        }
    }

    pub(crate) fn overrun(&self, grid: Grid, words: &[usize]) -> Option<Overrun> {
        let (Some(local), Some(stride)) = (self.local, self.stride) else {
            return None;
        };

        self.needs
            .iter()
            .filter_map(|(&binding, needs)| {
                let held = *words.get(binding as usize)?;
                let needed = words_for(elements_of(grid, local, *needs), stride);
                (needed > held).then_some(Overrun {
                    binding: Some(binding),
                    needed,
                    held,
                })
            })
            .next()
    }

    pub(crate) fn overrun_uniform(&self, grid: Grid, words: usize) -> Option<Overrun> {
        let (Some(local), Some(stride)) = (self.local, self.stride) else {
            return None;
        };

        let (binding, elements) = self
            .needs
            .iter()
            .map(|(&binding, needs)| (Some(binding), elements_of(grid, local, *needs)))
            .max_by_key(|&(_, elements)| elements)
            .unwrap_or((None, invocations(grid, local)));

        let needed = words_for(elements, stride);
        (needed > words).then_some(Overrun {
            binding,
            needed,
            held: words,
        })
    }

    #[cfg(test)]
    fn fits(&self, grid: Grid, words: usize) -> bool {
        self.overrun_uniform(grid, words).is_none()
    }

    #[cfg(test)]
    pub(crate) fn elements_per_invocation(&self) -> u64 {
        self.needs
            .values()
            .map(|needs| needs.per_invocation)
            .max()
            .unwrap_or(1)
    }

    #[cfg(test)]
    pub(crate) fn offset(&self) -> u64 {
        self.needs
            .values()
            .map(|needs| needs.offset)
            .max()
            .unwrap_or(0)
    }
}

/// ```text
/// (grid.y × local.y - 1) × pitch  +  grid.x × local.x × strips  +  offset
/// ```
fn elements_of(grid: Grid, local: [u64; 3], needs: addressing::Needs) -> u64 {
    let columns = u64::from(grid.x)
        .saturating_mul(local[0])
        .saturating_mul(needs.per_invocation);

    let reached = match needs.pitch {
        Some(pitch) => u64::from(grid.y)
            .saturating_mul(local[1])
            .saturating_sub(1)
            .saturating_mul(pitch)
            .saturating_add(columns),
        None => columns
            .saturating_mul(u64::from(grid.y))
            .saturating_mul(local[1])
            .saturating_mul(local[2]),
    };

    reached.saturating_add(needs.offset)
}

fn words_for(elements: u64, stride: u64) -> usize {
    let bytes = elements
        .saturating_mul(stride)
        .saturating_add(size_of::<u32>() as u64 - 1);

    usize::try_from(bytes / size_of::<u32>() as u64).unwrap_or(usize::MAX)
}

pub(crate) fn local_size(spirv: &[u32]) -> Option<[u64; 3]> {
    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::EXECUTION_MODE)
        .find_map(|instruction| {
            let operands = instruction.operands();
            let (mode, sizes) = match operands {
                [_entry, mode, x, y, z] => (*mode, [*x, *y, *z]),
                _ => return None,
            };
            if mode != ExecutionMode::LocalSize.word() {
                return None;
            }

            Some(sizes.map(u64::from))
        })
}

pub(crate) const fn invocations(grid: Grid, local: [u64; 3]) -> u64 {
    (grid.x as u64) * (grid.y as u64) * local[0] * local[1] * local[2]
}

pub(crate) fn element_bytes(spirv: &[u32]) -> Option<u64> {
    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::DECORATE)
        .find_map(|instruction| match instruction.operands() {
            [_target, decoration, stride] if *decoration == Decoration::ArrayStride.word() => {
                Some(u64::from(*stride))
            }
            _ => None,
        })
}

#[cfg(test)]
mod tests;
