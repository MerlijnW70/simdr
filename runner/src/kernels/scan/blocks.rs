//! ```text
//!   scan_blocks             each block scanned, and each block's total kept
//!   scan_blocks_exclusive   the same over the totals, exclusively — what each block OWES
//!   add_offsets             paid
//! ```

use super::{Scan, scanned_at};
use crate::kernels::{WORKGROUP_SIZE, whole_subgroup_of};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, LaneError};

pub fn scan_blocks<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, scan_blocks_at)
}

pub fn scan_blocks_exclusive<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, scan_blocks_exclusive_at)
}

fn scan_blocks_exclusive_at<T: Element, const LANES: u32>(
    subgroup: u32,
) -> Result<Vec<u32>, LaneError> {
    blocks_at::<T, LANES>(subgroup, Scan::Exclusive)
}

pub fn add_offsets<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, add_offsets_at)
}

fn add_offsets_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(Shape::new(subgroup, WORKGROUP_SIZE, 3))?;

    let value = kernel.load::<LANES>(0)?;
    let block = kernel.workgroup_index();
    let offset = kernel.load_at(1, block)?;

    let element = kernel.element();
    let raised = kernel
        .module()
        .binary(T::ADD, element, value.id(), offset)?;

    kernel.store_scalar(2, raised)?;
    kernel.finish()
}

fn scan_blocks_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    blocks_at::<T, LANES>(subgroup, Scan::Inclusive)
}

fn blocks_at<T: Element, const LANES: u32>(
    subgroup: u32,
    kind: Scan,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(Shape::new(subgroup, WORKGROUP_SIZE, 3))?;
    let scanned = scanned_at::<T, LANES>(&mut kernel, kind, Some(2))?;

    kernel.store_scalar(1, scanned)?;
    kernel.finish()
}
