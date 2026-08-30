use crate::module::Id;

pub const MAX_STRIPS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Strips {
    ids: [Id; MAX_STRIPS],
    count: u8,
}

impl Strips {
    pub(crate) fn new(ids: &[Id]) -> Option<Self> {
        let &last = ids.last()?;
        if ids.len() > MAX_STRIPS {
            return None;
        }

        let mut slots = [last; MAX_STRIPS];
        for (slot, &id) in slots.iter_mut().zip(ids) {
            *slot = id;
        }

        Some(Self {
            ids: slots,
            count: ids.len() as u8,
        })
    }

    pub(crate) fn as_slice(&self) -> &[Id] {
        self.ids.get(..usize::from(self.count)).unwrap_or(&[])
    }

    pub(crate) const fn len(&self) -> usize {
        self.count as usize
    }

    pub(crate) const fn first(&self) -> Id {
        self.ids[0]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector<T, const LANES: u32> {
    strips: Strips,
    element: core::marker::PhantomData<T>,
}

impl<T, const LANES: u32> Vector<T, LANES> {
    pub(crate) fn from_strips(ids: &[Id]) -> Option<Self> {
        Some(Self {
            strips: Strips::new(ids)?,
            element: core::marker::PhantomData,
        })
    }

    #[must_use]
    pub fn strips(&self) -> &[Id] {
        self.strips.as_slice()
    }

    #[must_use]
    pub const fn strip_count(&self) -> usize {
        self.strips.len()
    }

    #[must_use]
    pub fn id(&self) -> Id {
        self.strips.first()
    }

    #[must_use]
    pub const fn lanes() -> u32 {
        LANES
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::lanes::F32;
    use crate::module::{Module, Version};

    fn ids(count: usize) -> Vec<Id> {
        let mut module = Module::new(Version::V1_3);
        (0..count)
            .map(|_| module.alloc_id().expect("an id"))
            .collect()
    }

    #[test]
    fn a_single_strip_vector_is_its_one_id() {
        let ids = ids(1);
        let vector = Vector::<F32, 32>::from_strips(&ids).expect("built");

        assert_eq!(vector.strip_count(), 1);
        assert_eq!(vector.strips(), &ids[..]);
        assert_eq!(vector.id(), ids[0]);
    }

    #[test]
    fn a_multi_strip_vector_keeps_every_id_in_order() {
        let ids = ids(4);
        let vector = Vector::<F32, 128>::from_strips(&ids).expect("built");

        assert_eq!(vector.strip_count(), 4);
        assert_eq!(vector.strips(), &ids[..]);
        assert_eq!(vector.id(), ids[0], "and `id` is the first of them");
    }

    #[test]
    fn slots_past_the_strip_count_are_never_reported() {
        let ids = ids(2);
        let vector = Vector::<F32, 64>::from_strips(&ids).expect("built");

        assert_eq!(vector.strips().len(), 2, "not MAX_STRIPS");
    }

    #[test]
    fn a_value_of_no_strips_cannot_be_built() {
        assert!(Strips::new(&[]).is_none());
        assert!(Vector::<F32, 32>::from_strips(&[]).is_none());
    }

    #[test]
    fn exactly_the_inline_maximum_is_accepted() {
        let ids = ids(MAX_STRIPS);
        let strips = Strips::new(&ids).expect("the boundary, not past it");

        assert_eq!(strips.len(), MAX_STRIPS);
    }

    #[test]
    fn one_more_than_the_maximum_is_refused_rather_than_truncated() {
        let ids = ids(MAX_STRIPS + 1);

        assert!(
            Strips::new(&ids).is_none(),
            "truncating would silently drop elements from every operation downstream"
        );
        assert!(Vector::<F32, 512>::from_strips(&ids).is_none());
    }

    #[test]
    fn a_vector_knows_its_width_without_an_instance() {
        assert_eq!(Vector::<F32, 8>::lanes(), 8);
        assert_eq!(Vector::<F32, 64>::lanes(), 64);
    }
}
