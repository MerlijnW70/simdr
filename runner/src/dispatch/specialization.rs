use ash::vk;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Specialization {
    entries: Vec<(u32, u32)>,
}

impl Specialization {
    #[must_use]
    pub const fn none() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub(crate) fn value_of(&self, spec_id: u32) -> Option<u32> {
        self.entries
            .iter()
            .find(|(existing, _)| *existing == spec_id)
            .map(|(_, value)| *value)
    }

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

    #[must_use]
    pub fn set_f32(self, spec_id: u32, value: f32) -> Self {
        self.set(spec_id, value.to_bits())
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

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
