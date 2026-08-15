//! How a lane count sits on a subgroup.
//!
//! Three arrangements and one refusal, decided in one place so that every operation gets the same
//! answer. `decisions/DR-0002` is why this is settled at build time rather than on the device.

use super::{LaneError, Lanes, MAX_STRIPS};

/// How a lane count sits on the hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mapping {
    /// The vector is exactly as wide as the subgroup: one element per lane.
    WholeSubgroup,
    /// Several vectors share a subgroup, each reducing within its own cluster.
    ///
    /// The case that would otherwise idle hardware: a `Simd<f32, 8>` on a 32-lane machine runs
    /// four of itself at once.
    Clusters {
        /// Lanes per cluster, which is the vector's own width.
        size: u32,
    },
    /// The vector is wider than the subgroup, so each lane holds several elements.
    ///
    /// Lane `l` holds the elements at `l`, `l + width`, `l + 2·width` — strided, so that every
    /// strip is still a coalesced read.
    Strips {
        /// How many elements each lane holds.
        count: u32,
    },
}

impl Mapping {
    /// How a vector of `lanes` sits on a `subgroup`-wide device.
    ///
    /// **The rule, at run time, so that it is written once.** [`Lanes::mapping`] takes the width as
    /// a const generic — which is what `decisions/DR-0002` is about — and that is right for the
    /// emitter and unusable for anything holding a width it learned at run time. So the callers
    /// that could not reach it wrote the rule again: `runner`'s fuzzer decided it with
    /// `lanes < subgroup`, and `kernels::reduce` with `LANES > subgroup`, and **neither was the
    /// same rule as this one** — a three-lane vector on a 32-wide subgroup is "clustered" to a
    /// comparison and refused by divisibility.
    ///
    /// The mutation gate found both copies within a week of each other, one of them able to delete
    /// a whole finish's coverage without failing anything. Copies of a decision do not diverge
    /// loudly; they diverge in the cases nobody draws.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoMapping`] when `lanes` neither divides nor is a multiple of the width,
    /// [`LaneError::TooManyStrips`] when it is a multiple too large to hold inline.
    pub const fn of(lanes: u32, subgroup: u32) -> Result<Self, LaneError> {
        let no_mapping = Err(LaneError::NoMapping {
            lanes,
            width: subgroup,
        });

        // Zero has to go first and is load-bearing: every integer is a multiple of nothing, so
        // without this the strip arm below would happily compute `0 / width` strips.
        if lanes == 0 || subgroup == 0 {
            return no_mapping;
        }
        if lanes == subgroup {
            return Ok(Self::WholeSubgroup);
        }

        // No `lanes < subgroup` here, though that is what this arm means. The equal case is
        // already gone, so a comparison would be indistinguishable from `<=` — divisibility says
        // the same thing and says it once.
        if subgroup.is_multiple_of(lanes) {
            return Ok(Self::Clusters { size: lanes });
        }
        if !lanes.is_multiple_of(subgroup) {
            return no_mapping;
        }

        let count = lanes / subgroup;
        if count as usize > MAX_STRIPS {
            return Err(LaneError::TooManyStrips {
                strips: count as usize,
                limit: MAX_STRIPS,
            });
        }
        Ok(Self::Strips { count })
    }
}

impl Lanes<'_> {
    /// How many elements each lane holds for a vector of `LANES`.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoMapping`] or [`LaneError::TooManyStrips`] if there is no mapping.
    pub fn strips_for<const LANES: u32>(&self) -> Result<usize, LaneError> {
        match self.mapping::<LANES>()? {
            Mapping::WholeSubgroup | Mapping::Clusters { .. } => Ok(1),
            Mapping::Strips { count } => Ok(count as usize),
        }
    }

    /// Check that `LANES` can map onto this subgroup, and say how.
    ///
    /// The const-generic face of [`Mapping::of`], which is where the rule lives. Kept as its own
    /// name because every operation in this crate asks it that way — `decisions/DR-0002` is why the
    /// width is a const generic here — and because a caller reading `self.mapping::<32>()` should
    /// not have to know which width it is being compared against.
    ///
    /// # Errors
    ///
    /// As [`Mapping::of`].
    pub const fn mapping<const LANES: u32>(&self) -> Result<Mapping, LaneError> {
        Mapping::of(LANES, self.width())
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::module::{Module, Version};

    fn module() -> Module {
        Module::new(Version::V1_3)
    }

    #[test]
    fn a_vector_as_wide_as_the_subgroup_maps_to_the_whole_of_it() {
        let mut module = module();
        let lanes = Lanes::new(&mut module, 32).expect("built");

        assert_eq!(lanes.mapping::<32>(), Ok(Mapping::WholeSubgroup));
        assert_eq!(lanes.strips_for::<32>(), Ok(1));
    }

    #[test]
    fn a_narrower_vector_maps_to_clusters_of_its_own_width() {
        let mut module = module();
        let lanes = Lanes::new(&mut module, 32).expect("built");

        assert_eq!(lanes.mapping::<8>(), Ok(Mapping::Clusters { size: 8 }));
        assert_eq!(lanes.strips_for::<8>(), Ok(1), "still one element per lane");
    }

    #[test]
    fn a_wider_vector_gives_every_lane_more_than_one_element() {
        let mut module = module();
        let lanes = Lanes::new(&mut module, 32).expect("built");

        assert_eq!(lanes.mapping::<64>(), Ok(Mapping::Strips { count: 2 }));
        assert_eq!(lanes.mapping::<128>(), Ok(Mapping::Strips { count: 4 }));
        assert_eq!(lanes.strips_for::<128>(), Ok(4));
    }

    #[test]
    fn a_width_that_neither_divides_nor_multiplies_has_no_mapping() {
        let mut module = module();
        let lanes = Lanes::new(&mut module, 32).expect("built");

        assert_eq!(
            lanes.mapping::<12>(),
            Err(LaneError::NoMapping {
                lanes: 12,
                width: 32
            })
        );
        assert_eq!(
            lanes.mapping::<48>(),
            Err(LaneError::NoMapping {
                lanes: 48,
                width: 32
            }),
            "48 is wider than 32 and not a multiple of it"
        );
    }

    #[test]
    fn a_vector_of_no_lanes_has_no_mapping() {
        // Zero is a multiple of every width, so the strip arm would otherwise compute zero strips
        // and hand back a vector holding nothing.
        let mut module = module();
        let lanes = Lanes::new(&mut module, 32).expect("built");

        assert_eq!(
            lanes.mapping::<0>(),
            Err(LaneError::NoMapping {
                lanes: 0,
                width: 32
            })
        );
    }

    #[test]
    fn more_strips_than_fit_inline_are_refused() {
        let mut module = module();
        let lanes = Lanes::new(&mut module, 32).expect("built");

        assert_eq!(
            lanes.mapping::<512>(),
            Err(LaneError::TooManyStrips {
                strips: 16,
                limit: MAX_STRIPS
            })
        );
    }

    #[test]
    fn exactly_the_inline_maximum_of_strips_is_accepted() {
        // The boundary itself: 256 lanes on a 32-wide subgroup is eight strips, which is
        // `MAX_STRIPS` exactly and must be allowed.
        let mut module = module();
        let lanes = Lanes::new(&mut module, 32).expect("built");

        assert_eq!(
            lanes.mapping::<256>(),
            Ok(Mapping::Strips {
                count: MAX_STRIPS as u32
            })
        );
    }

    #[test]
    fn the_same_lane_count_maps_three_different_ways_across_two_devices() {
        // DR-0002 in one test: 32 lanes is the whole machine on NVIDIA, half of it on a 64-wide
        // AMD part, and 64 lanes is two strips on the first and the whole of the second.
        let mut narrow = module();
        let mut wide = module();
        let on_nvidia = Lanes::new(&mut narrow, 32).expect("built");
        let on_amd = Lanes::new(&mut wide, 64).expect("built");

        assert_eq!(on_nvidia.mapping::<32>(), Ok(Mapping::WholeSubgroup));
        assert_eq!(on_amd.mapping::<32>(), Ok(Mapping::Clusters { size: 32 }));
        assert_eq!(on_nvidia.mapping::<64>(), Ok(Mapping::Strips { count: 2 }));
        assert_eq!(on_amd.mapping::<64>(), Ok(Mapping::WholeSubgroup));
    }
}
