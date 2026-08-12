//! How many workgroups to dispatch, on how many axes.
//!
//! `vkCmdDispatch` has always taken three counts and this crate has always passed `(x, 1, 1)`. A
//! kernel built from [`simdr::kernel::Shape::grid`] addresses `(row, column)`, and the rows have
//! to come from somewhere: either from a workgroup that is several invocations deep, or from the
//! dispatch's y, or from both. This is the second of the three.
//!
//! # Why a type rather than two more arguments
//!
//! Because `dispatch(pipeline, 64, 8, 1)` reads the same whether the 8 is a y count or an
//! iteration count, and this crate already passes an iteration count next to a workgroup count.
//! One of those mistakes returns a plausible number rather than an error.
//!
//! # There is no z
//!
//! `vkCmdDispatch` has one and nothing here has ever needed it — the emitter computes no z
//! address, so a z count above 1 would run every workgroup again over the same elements.
//! `decisions/DR-0006` records that, and adding it later is one field and one argument.

/// How many workgroups a dispatch runs, along x and along y.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    /// Workgroups along x, which is the axis a linear kernel's address runs along.
    pub x: u32,
    /// Workgroups along y, which is the axis a grid kernel's rows run along.
    pub y: u32,
}

impl Grid {
    /// `x` workgroups on one axis — what every dispatch in this crate was before there was a
    /// second one.
    #[must_use]
    pub const fn linear(x: u32) -> Self {
        Self { x, y: 1 }
    }

    /// `x` workgroups across and `y` down.
    #[must_use]
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    /// How many workgroups this is in all.
    ///
    /// `u64` because two `u32`s multiply past one: a device's limit is per axis, and the product
    /// of two legal counts is not bounded by anything a `u32` holds.
    #[must_use]
    pub const fn workgroups(self) -> u64 {
        (self.x as u64) * (self.y as u64)
    }

    /// The counts to hand `vkCmdDispatch`, with each axis floored at one.
    ///
    /// A count of zero dispatches nothing, which is legal and is almost never what a caller with a
    /// zero in hand meant. Flooring here rather than refusing keeps it consistent with
    /// [`crate::Gpu::run`], which has always run at least one workgroup.
    pub(super) const fn counts(self) -> (u32, u32) {
        (
            if self.x == 0 { 1 } else { self.x },
            if self.y == 0 { 1 } else { self.y },
        )
    }
}

impl From<u32> for Grid {
    fn from(x: u32) -> Self {
        Self::linear(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_axis_is_two_with_a_y_of_one() {
        assert_eq!(Grid::linear(64), Grid::new(64, 1));
        assert_eq!(Grid::from(64), Grid::linear(64));
    }

    #[test]
    fn the_total_is_counted_in_something_wider_than_either_axis() {
        // Two legal counts whose product is not a `u32`. A device's limit is per axis, so this is
        // reachable rather than theoretical.
        let wide = Grid::new(u32::MAX, 2);

        assert_eq!(wide.workgroups(), u64::from(u32::MAX) * 2);
    }

    #[test]
    fn an_axis_of_zero_is_floored_rather_than_dispatched_as_nothing() {
        assert_eq!(Grid::new(0, 0).counts(), (1, 1));
        assert_eq!(Grid::new(8, 0).counts(), (8, 1));
        assert_eq!(Grid::new(0, 8).counts(), (1, 8));
    }

    #[test]
    fn a_grid_that_needs_no_flooring_passes_its_counts_through() {
        assert_eq!(Grid::new(3, 5).counts(), (3, 5));
    }
}
