//! Values a pipeline fixes in a module that left them open.
//!
//! A module declares a specialization constant with a default and a `SpecId`; this is the other
//! half — the `VkSpecializationInfo` that replaces it when the pipeline is created. One module,
//! several pipelines, different numbers.
//!
//! # Every value is four bytes
//!
//! Vulkan's map entries carry an offset and a size, so a specialization block may hold values of
//! any width. This one holds `u32`s, because every specialization constant `simdr` emits is a
//! 32-bit scalar and a `f32` goes in as its bits — the same convention `Lanes::splat_bits` uses,
//! for the same reason: one signature that does not need a numeric trait the standard library
//! does not have.

use ash::vk;

/// The specialization constants a pipeline sets, by their `SpecId`.
///
/// An empty one means "use every default", and is what every pipeline in this crate used before
/// specialization existed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Specialization {
    entries: Vec<(u32, u32)>,
}

impl Specialization {
    /// No overrides: every constant keeps the default the module declared.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// What this sets `spec_id` to, if it sets it at all.
    ///
    /// **Read by the dispatch bound rather than by the driver**, which is why it exists. A
    /// specialization constant is a number chosen after the module was built, so an address that
    /// adds one has no literal for `dispatch::extent` to find — and a bound that cannot see a term
    /// counts zero for it, which is the direction that *lets an overrun through*. The value the
    /// pipeline will be built with is here, and now the bound asks.
    ///
    /// `None` means the module's own default stands.
    pub(crate) fn value_of(&self, spec_id: u32) -> Option<u32> {
        self.entries
            .iter()
            .find(|(existing, _)| *existing == spec_id)
            .map(|(_, value)| *value)
    }

    /// Set the constant carrying `spec_id` to `value`.
    ///
    /// Setting the same id twice keeps the last value rather than sending two entries — Vulkan
    /// leaves duplicate ids to the implementation, and "the last one wins" is the reading every
    /// caller expects and none should have to rely on a driver for.
    #[must_use]
    pub fn set(mut self, spec_id: u32, value: u32) -> Self {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|(existing, _)| *existing == spec_id)
        {
            existing.1 = value;
        } else {
            self.entries.push((spec_id, value));
        }
        self
    }

    /// The same, for a value that is a float.
    #[must_use]
    pub fn set_f32(self, spec_id: u32, value: f32) -> Self {
        self.set(spec_id, value.to_bits())
    }

    /// Whether anything is overridden.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many constants are set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The map entries, one per constant, pointing into what [`Specialization::data`] returns.
    pub(super) fn map_entries(&self) -> Vec<vk::SpecializationMapEntry> {
        self.entries
            .iter()
            .enumerate()
            .map(|(index, &(spec_id, _))| {
                vk::SpecializationMapEntry::default()
                    .constant_id(spec_id)
                    .offset((index * size_of::<u32>()) as u32)
                    .size(size_of::<u32>())
            })
            .collect()
    }

    /// The block the entries point into: every value, in order, little-endian.
    pub(super) fn data(&self) -> Vec<u8> {
        self.entries
            .iter()
            .flat_map(|&(_, value)| value.to_le_bytes())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_specialization_sets_nothing() {
        let none = Specialization::none();

        assert!(none.is_empty());
        assert_eq!(none.len(), 0);
        assert!(none.data().is_empty());
        assert!(none.map_entries().is_empty());
    }

    #[test]
    fn each_value_lands_at_its_own_offset() {
        // The offsets are what tie an entry to its four bytes, and getting them wrong hands a
        // driver a value assembled from two neighbours — which is a number, so nothing complains.
        let set = Specialization::none().set(0, 7).set(4, 9);

        let entries = set.map_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].constant_id, 0);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[0].size, 4);
        assert_eq!(entries[1].constant_id, 4, "the id is not the index");
        assert_eq!(entries[1].offset, 4);

        assert_eq!(set.data(), vec![7, 0, 0, 0, 9, 0, 0, 0]);
    }

    #[test]
    fn setting_an_id_twice_replaces_it_rather_than_sending_two_entries() {
        let set = Specialization::none().set(2, 1).set(2, 5);

        assert_eq!(set.len(), 1);
        assert_eq!(set.data(), vec![5, 0, 0, 0]);
    }

    #[test]
    fn a_float_goes_in_as_its_bits() {
        let set = Specialization::none().set_f32(0, 1.5);

        assert_eq!(set.data(), 1.5_f32.to_bits().to_le_bytes().to_vec());
    }
}
