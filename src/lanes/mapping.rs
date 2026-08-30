use super::{LaneError, Lanes, MAX_STRIPS};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mapping {
    WholeSubgroup,
    Clusters { size: u32 },
    Strips { count: u32 },
}

impl Mapping {
    pub const fn of(lanes: u32, subgroup: u32) -> Result<Self, LaneError> {
        let no_mapping = Err(LaneError::NoMapping {
            lanes,
            width: subgroup,
        });

        if lanes == 0 || subgroup == 0 {
            return no_mapping;
        }
        if lanes == subgroup {
            return Ok(Self::WholeSubgroup);
        }

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
    pub fn strips_for<const LANES: u32>(&self) -> Result<usize, LaneError> {
        match self.mapping::<LANES>()? {
            Mapping::WholeSubgroup | Mapping::Clusters { .. } => Ok(1),
            Mapping::Strips { count } => Ok(count as usize),
        }
    }

    pub const fn mapping<const LANES: u32>(&self) -> Result<Mapping, LaneError> {
        Mapping::of(LANES, self.width())
    }
}

#[cfg(test)]
mod tests {
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
