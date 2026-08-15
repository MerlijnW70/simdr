//! The kernels a scan longer than one workgroup needs.
//!
//! [`super`] scans a workgroup: 64 elements, and the answer is the answer. Everything here exists
//! because a longer input has to be cut into blocks, and a block scanned on its own is short by the
//! total of every block before it.
//!
//! ```text
//!   scan_blocks             each block scanned, and each block's total kept
//!   scan_blocks_exclusive   the same over the totals, exclusively — what each block OWES
//!   add_offsets             paid
//! ```
//!
//! Split from [`super`] because the two are different jobs at different scales, and the file held
//! both at 639 lines. What they share is `super::scanned_at`, which is the scan itself and is
//! written once.
//!
//! `crate::scan::Scanner` composes these; `runner/tests/scan.rs` also composes them by hand, which
//! is how the arithmetic was checked before the object existed.

use super::{Scan, scanned_at};
use crate::kernels::{WORKGROUP_SIZE, whole_subgroup_of};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, LaneError};

/// `out[i] = in[0] + … + in[i]` within each block, **and** each block's total to binding 2.
///
/// The same scan as [`super::scan_workgroup`] with one instruction more: the last thing every invocation
/// holds is its subgroup's running total plus the offset of the subgroups before it, so the *last*
/// subgroup's lanes are holding the block's whole total. That value goes to
/// [`simdr::kernel::Kernel::workgroup_index`] of binding 2.
///
/// **This is what a scan longer than one workgroup needs.** With block totals in a buffer of their
/// own, scanning *those* and adding the result back to each block turns 64 elements into 64 × 64,
/// and again for as many levels as the length needs. Nothing here does that yet; this is the pass
/// that makes it possible, and it is useful on its own to anyone who wants per-block sums.
///
/// # Every invocation writes the block total, and they all write the same one
///
/// The whole workgroup runs the store, so binding 2's slot is written 64 times. That is the case
/// [`simdr::kernel::Kernel::store_at`] documents as its own: identical values to one address, where
/// the order they land in cannot change the answer. Electing one lane to write instead would need a
/// branch that some invocations do not take, which `decisions/DR-0003` refuses.
///
/// # Errors
///
/// As [`super::scan_workgroup`].
pub fn scan_blocks<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, scan_blocks_at)
}

/// The **exclusive** scan within each block, and each block's total to binding 2.
///
/// What the levels above the first need. A long scan works by scanning each block, scanning the
/// block totals, and adding each block its offset — and the offset a block owes is the total of
/// every block *before* it, which is an exclusive scan. Running the upper levels inclusively and
/// shifting by one afterwards would need either a read at `block - 1`, which underflows at block
/// zero, or a subtraction that in floating point does not give back the number it took away.
///
/// The block totals are the same either way: a total is a total whichever scan reported the
/// running sums, and it comes from a reduce rather than from the scan — see `scanned_at`.
///
/// # Errors
///
/// As [`super::scan_workgroup`].
pub fn scan_blocks_exclusive<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, scan_blocks_exclusive_at)
}

/// The builder for [`scan_blocks_exclusive`].
fn scan_blocks_exclusive_at<T: Element, const LANES: u32>(
    subgroup: u32,
) -> Result<Vec<u32>, LaneError> {
    blocks_at::<T, LANES>(subgroup, Scan::Exclusive)
}

/// `out[i] = in[i] + offsets[the workgroup i belongs to]`.
///
/// The second half of a long scan. Every block has been scanned from its own start, so each is
/// short by the total of the blocks before it; that number is one element of binding 1, read once
/// per invocation at [`simdr::kernel::Kernel::workgroup_index`], and added.
///
/// **One value per workgroup, read by all 64 of its invocations.** Concurrent reads need no
/// ordering, which is what makes this a plain load and the whole pass branch-free.
///
/// # Errors
///
/// As [`super::scan_workgroup`].
pub fn add_offsets<T: Element>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup_of!(T, subgroup, add_offsets_at)
}

/// The builder for [`add_offsets`].
fn add_offsets_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    // Binding 0 the scanned values, 1 the per-block offsets, 2 the output. Three buffers rather
    // than adding in place, because a kernel that reads and writes one binding is a kernel whose
    // correctness depends on no other workgroup having reached it yet.
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

/// The builder for [`scan_blocks`].
fn scan_blocks_at<T: Element, const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    blocks_at::<T, LANES>(subgroup, Scan::Inclusive)
}

/// Both block-scanning kernels, which differ only in which scan they run.
fn blocks_at<T: Element, const LANES: u32>(
    subgroup: u32,
    kind: Scan,
) -> Result<Vec<u32>, LaneError> {
    // Three buffers rather than two: the block totals need one of their own, and naming it here is
    // what tells `scanned_at` to compute one at all.
    let mut kernel = Kernel::<T>::new(Shape::new(subgroup, WORKGROUP_SIZE, 3))?;
    let scanned = scanned_at::<T, LANES>(&mut kernel, kind, Some(2))?;

    kernel.store_scalar(1, scanned)?;
    kernel.finish()
}
