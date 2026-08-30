#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub x: u32,
    pub y: u32,
}

impl Grid {
    #[must_use]
    pub const fn linear(x: u32) -> Self {
        Self { x, y: 1 }
    }

    #[must_use]
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub const fn workgroups(self) -> u64 {
        (self.x as u64) * (self.y as u64)
    }

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
