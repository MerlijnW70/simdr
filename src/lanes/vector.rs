//! A value of `LANES` elements, however many of them each lane ends up holding.
//!
//! # Strips
//!
//! When `LANES` matches the subgroup, each lane holds one element and a vector is one id. When
//! `LANES` is a multiple of the subgroup, each lane holds several — lane `l` holds the elements
//! at `l`, `l + width`, `l + 2·width`, and so on. Those are the *strips*, and the vector is that
//! many ids.
//!
//! Strided rather than blocked, deliberately: lane `l` reading element `l + s·width` is a
//! coalesced access, where blocking would have lane `l` read `l·strips + s` and every lane in the
//! subgroup would touch a different cache line on every strip. (Past two strips that choice stops
//! paying — see `notes/FINDINGS.md`.)
//!
//! The strip count is a runtime value, since it depends on the device's width, so it cannot size
//! an array in the type. [`MAX_STRIPS`] is the inline ceiling that keeps these types `Copy` and
//! allocation-free; a wider one is refused by name.

use crate::module::Id;

/// The most elements one lane may hold.
///
/// Eight strips over a 64-lane subgroup is a `Simd<T, 512>`, far past anything the rest of this
/// crate can do usefully. The limit exists so a value stays `Copy` and needs no allocation;
/// raising it is a one-line change.
pub const MAX_STRIPS: usize = 8;

/// One id per strip, inline.
///
/// Shared by [`Vector`] and [`crate::lanes::Predicate`], which are the same shape holding
/// different things. They had a copy each until the prober pointed out that meant two identical
/// bounds checks and only one of them reachable — one construction path, one set of tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Strips {
    /// Slots past `count` repeat the last live one and are never read. The count is the
    /// authority; this keeps the array `Copy` without an `Option` per slot.
    ids: [Id; MAX_STRIPS],
    count: u8,
}

impl Strips {
    /// Build from one id per strip, in strip order.
    ///
    /// `None` if `ids` is empty or longer than [`MAX_STRIPS`].
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
            // Between 1 and MAX_STRIPS, which fits a `u8` many times over.
            count: ids.len() as u8,
        })
    }

    /// The live ids.
    pub(crate) fn as_slice(&self) -> &[Id] {
        self.ids.get(..usize::from(self.count)).unwrap_or(&[])
    }

    /// How many there are.
    pub(crate) const fn len(&self) -> usize {
        self.count as usize
    }

    /// The first, which is the whole of it when there is only one.
    pub(crate) fn first(&self) -> Id {
        // `new` refuses an empty slice, so slot zero is always live. Written without an index so
        // that staying true is not a promise this file has to keep on its own.
        self.ids.first().copied().unwrap_or(self.ids[0])
    }
}

/// A value spread across the subgroup's lanes.
///
/// `LANES` is part of the type, so two vectors of different widths cannot be combined and the
/// mismatch is a compile error. `T` is too, so a float vector cannot be added to an integer one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector<T, const LANES: u32> {
    strips: Strips,
    element: core::marker::PhantomData<T>,
}

impl<T, const LANES: u32> Vector<T, LANES> {
    /// Build from the ids of each strip, in strip order.
    pub(crate) fn from_strips(ids: &[Id]) -> Option<Self> {
        Some(Self {
            strips: Strips::new(ids)?,
            element: core::marker::PhantomData,
        })
    }

    /// The ids this is made of, one per strip.
    #[must_use]
    pub fn strips(&self) -> &[Id] {
        self.strips.as_slice()
    }

    /// How many elements each lane holds.
    #[must_use]
    pub const fn strip_count(&self) -> usize {
        self.strips.len()
    }

    /// The id of the first strip.
    ///
    /// The whole vector when there is only one, which is the common case; an escape hatch into
    /// [`crate::module`] otherwise.
    #[must_use]
    pub fn id(&self) -> Id {
        self.strips.first()
    }

    /// How many lanes this spans.
    #[must_use]
    pub const fn lanes() -> u32 {
        LANES
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
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
