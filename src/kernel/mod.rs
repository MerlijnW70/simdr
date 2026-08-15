//! A whole compute kernel, from the lane API down.
//!
//! [`crate::lanes`] turns lane operations into instructions; this turns a kernel into a module.
//! Buffer bindings, layout decorations, the entry point, the invocation index and the capability
//! declarations are all its business, so a kernel is the few lines that say what to compute:
//!
//! ```
//! # use simdr::kernel::{Kernel, Shape};
//! # use simdr::lanes::F32;
//! # fn build() -> Result<Vec<u32>, simdr::lanes::LaneError> {
//! let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2))?;
//! let value = kernel.load::<8>(0)?;
//! let total = kernel.lanes()?.reduce_sum(value)?;
//! kernel.store_scalar(1, total)?;
//! kernel.finish()
//! # }
//! ```
//!
//! `access` has the reads and writes, and the address arithmetic that decides where each element
//! lives; `plane` has the same thing on two axes; `binding` has the interface every kernel starts
//! from; `shared` has workgroup memory and the barrier; `scatter` has the writes whose address
//! comes from the data. All five are private — what they add appears as methods on [`Kernel`].

mod access;
mod binding;
mod plane;
mod scatter;
mod shared;

pub use self::shared::Shared;

use crate::lanes::{Element, LaneError, Lanes};
use crate::module::{Id, Module};

/// How a kernel is laid out before anything is built into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// The device's subgroup width. See `decisions/DR-0002` for why this is not discoverable
    /// later.
    pub subgroup: u32,
    /// Invocations per workgroup along x.
    ///
    /// For a grid this is the workgroup's *width*: `workgroup × rows` invocations in all.
    pub workgroup: u32,
    /// Invocations per workgroup along y, and `None` for a kernel with no second axis at all.
    ///
    /// `Some(1)` and `None` emit the same `LocalSize`, and they are not the same shape: the first
    /// says the dispatch has a y dimension and one invocation row per workgroup, the second says
    /// there is no y and a row index would be meaningless. Only the first admits
    /// [`Kernel::load_row`].
    pub rows: Option<u32>,
    /// How many storage buffers to bind, in descriptor set 0 at bindings `0..buffers`.
    pub buffers: u32,
}

impl Shape {
    /// A kernel over `buffers` storage buffers, `workgroup` invocations at a time.
    ///
    /// One axis: every address is a single index and the dispatch's y and z are 1.
    #[must_use]
    pub const fn new(subgroup: u32, workgroup: u32, buffers: u32) -> Self {
        Self {
            subgroup,
            workgroup,
            rows: None,
            buffers,
        }
    }

    /// The same, with a second axis: `workgroup × rows` invocations per group.
    ///
    /// A grid kernel addresses `(row, column)` rather than a single index — see
    /// [`Kernel::load_row`]. `rows` may be 1, and that is the common case: one invocation row per
    /// workgroup and one workgroup per image row.
    ///
    /// **`workgroup` is what the subgroups are cut from.** SPIR-V numbers a workgroup's
    /// invocations x-fastest, so subgroups fill along x first; a `workgroup` that is a multiple of
    /// the subgroup width keeps each subgroup inside one row, and one that is not lets a subgroup
    /// straddle two. Nothing here refuses that, because the same is true of a one-axis kernel
    /// whose `workgroup` is not a multiple of the width — but on a grid it silently makes a
    /// row-wise reduction sum parts of two rows.
    #[must_use]
    pub const fn grid(subgroup: u32, workgroup: u32, rows: u32, buffers: u32) -> Self {
        Self {
            subgroup,
            workgroup,
            rows: Some(rows),
            buffers,
        }
    }
}

/// A compute kernel over storage buffers of `T`, under construction.
pub struct Kernel<T: Element> {
    module: Module,
    shape: Shape,
    element: Id,
    element_pointer: Id,
    uint: Id,
    zero: Id,
    buffers: Vec<Id>,
    /// This invocation's position within its workgroup, and which workgroup that is.
    local: Id,
    group: Id,
    /// This invocation's row in the whole dispatch, for a grid kernel, and `None` for a linear
    /// one — where there is no second axis to have a position on.
    row: Option<Id>,
    marker: core::marker::PhantomData<T>,
}

impl<T: Element> Kernel<T> {
    /// Set up the interface and open `main`, ready for loads and lane operations.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the shape is unusable or the module cannot be built.
    pub fn new(shape: Shape) -> Result<Self, LaneError> {
        binding::build::<T>(shape).map(|parts| Self {
            module: parts.module,
            shape,
            element: parts.element,
            element_pointer: parts.element_pointer,
            uint: parts.uint,
            zero: parts.zero,
            buffers: parts.buffers,
            local: parts.local,
            group: parts.group,
            row: parts.row,
            marker: core::marker::PhantomData,
        })
    }

    /// The lane builder, for this kernel's subgroup width.
    ///
    /// Cheap to make and thrown away after each use, which is what lets loads and lane operations
    /// interleave without either borrowing the other for longer than a statement.
    ///
    /// # Errors
    ///
    /// [`LaneError`] if the width is not usable.
    pub fn lanes(&mut self) -> Result<Lanes<'_>, LaneError> {
        Lanes::new(&mut self.module, self.shape.subgroup)
    }

    /// The module underneath, for anything this layer does not cover.
    pub const fn module(&mut self) -> &mut Module {
        &mut self.module
    }

    /// The SPIR-V type of `T`.
    #[must_use]
    pub const fn element(&self) -> Id {
        self.element
    }

    /// How this kernel was shaped.
    #[must_use]
    pub const fn shape(&self) -> Shape {
        self.shape
    }

    /// This invocation's index within its workgroup.
    ///
    /// The slot a workgroup handover writes to: every invocation has a different one, so no two
    /// collide and the write needs no synchronisation of its own. See [`Kernel::store_shared`].
    #[must_use]
    pub const fn local_index(&self) -> Id {
        self.local
    }

    /// Which workgroup this invocation is in — the dispatch's x index.
    ///
    /// **The slot a per-block result belongs at.** Everything else in this interface addresses by
    /// *invocation*: [`Kernel::store_scalar`] writes one value per invocation, and a workgroup of
    /// 64 therefore writes 64 of them. A chained algorithm often wants one value per *workgroup* —
    /// a block's total, its maximum, how many elements it kept — and there was no way to say where
    /// that goes, because the only index available counted invocations.
    ///
    /// It was loaded all along and used internally to work out where a workgroup's run begins; this
    /// exposes the number rather than computing anything new.
    ///
    /// The x index and not a vector of three. `decisions/DR-0006` allows two dispatch axes, and the
    /// y one is [`Kernel::row`] — which is a different question with a different answer, so they
    /// are different functions rather than components of one.
    #[must_use]
    pub const fn workgroup_index(&self) -> Id {
        self.group
    }

    /// The type this kernel's addresses are computed in — a 32-bit *unsigned* integer.
    ///
    /// What a caller building an offset needs. [`Kernel::load_offset_by`] and
    /// [`Kernel::element_pointer_to`] both take an [`Id`] and both add it to an address, so the
    /// value has to be of this type; a caller that asks the module for one itself is
    /// reconstructing a decision this kernel already made, and can reconstruct it differently.
    ///
    /// It came from a mutation survivor. `runner/src/kernels/reduce.rs` wrote
    /// `module().type_int(32, false)` and flipping that `false` changed nothing observable —
    /// `OpIAdd` is sign-agnostic, so a signed spec constant computes the same address. The sign was
    /// untestable because it was never load-bearing, and the honest fix is not a test for it but
    /// not writing it down twice.
    #[must_use]
    pub const fn index_type(&self) -> Id {
        self.uint
    }

    /// Close `main` and hand back the finished module.
    ///
    /// # Errors
    ///
    /// [`LaneError::Build`] if the closing instructions cannot be emitted.
    pub fn finish(mut self) -> Result<Vec<u32>, LaneError> {
        self.module.return_void()?;
        self.module.end_function()?;
        Ok(self.module.finish())
    }

    /// A pointer to one element of a buffer.
    pub(super) const fn element_pointer(&self) -> Id {
        self.element_pointer
    }

    /// The unsigned integer type the addresses are computed in.
    pub(super) const fn uint(&self) -> Id {
        self.uint
    }

    /// The constant zero, which every access chain starts with — buffers hold one struct member.
    pub(super) const fn zero(&self) -> Id {
        self.zero
    }

    /// This invocation's index within its workgroup, and which workgroup that is.
    pub(super) const fn position(&self) -> (Id, Id) {
        (self.local, self.group)
    }

    /// This invocation's row, or `None` for a kernel with no second axis.
    pub(super) const fn row_index(&self) -> Option<Id> {
        self.row
    }

    /// The variable bound at `index`.
    pub(super) fn buffer(&self, index: u32) -> Result<Id, LaneError> {
        self.buffers
            .get(index as usize)
            .copied()
            .ok_or(LaneError::NoSuchBuffer {
                index,
                bound: self.buffers.len() as u32,
            })
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{F32, U32};
    use crate::module::op;
    use crate::spec::{Capability, Decoration};

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    /// Whether `words` declares `capability`.
    fn declares(words: &[u32], capability: Capability) -> bool {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == op::CAPABILITY)
            .any(|instruction| instruction.operands().first() == Some(&capability.word()))
    }

    #[test]
    fn a_reduction_kernel_is_four_lines_and_loads_once() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");
        let total = kernel
            .lanes()
            .expect("lanes")
            .reduce_sum(value)
            .expect("sum");
        kernel.store_scalar(1, total).expect("stored");

        let words = kernel.finish().expect("finished");

        assert_eq!(
            count(&words, op::LOAD),
            3,
            "the buffer, plus the two built-ins the address arithmetic needs"
        );
        assert_eq!(count(&words, op::STORE), 1);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_ADD), 1);
    }

    #[test]
    fn a_full_width_kernel_does_not_declare_the_clustered_capability() {
        // Declaring it would make the module refuse to run on a device that offers everything
        // else, which is worse than noise.
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");
        let total = kernel
            .lanes()
            .expect("lanes")
            .reduce_sum(value)
            .expect("sum");
        kernel.store_scalar(1, total).expect("stored");

        let words = kernel.finish().expect("finished");

        assert!(declares(&words, Capability::GroupNonUniformArithmetic));
        assert!(!declares(&words, Capability::GroupNonUniformClustered));
    }

    #[test]
    fn a_clustered_kernel_does_declare_it() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<8>(0).expect("loaded");
        let total = kernel
            .lanes()
            .expect("lanes")
            .reduce_sum(value)
            .expect("sum");
        kernel.store_scalar(1, total).expect("stored");

        assert!(declares(
            &kernel.finish().expect("finished"),
            Capability::GroupNonUniformClustered
        ));
    }

    #[test]
    fn a_kernel_that_only_scales_declares_no_subgroup_capability_at_all() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");
        let doubled = {
            let mut lanes = kernel.lanes().expect("lanes");
            let two = lanes.splat_bits::<F32, 32>(2.0_f32.to_bits()).expect("two");
            lanes.mul(value, two).expect("scaled")
        };
        kernel.store(1, doubled).expect("stored");

        let words = kernel.finish().expect("finished");

        assert!(declares(&words, Capability::Shader));
        assert!(!declares(&words, Capability::GroupNonUniform));
    }

    #[test]
    fn a_shape_with_no_buffers_describes_nothing_and_is_refused() {
        assert_eq!(
            Kernel::<F32>::new(Shape::new(32, 64, 0)).err(),
            Some(LaneError::BadShape {
                workgroup: 64,
                buffers: 0
            })
        );
    }

    #[test]
    fn a_shape_with_no_invocations_describes_nothing_and_is_refused() {
        assert_eq!(
            Kernel::<F32>::new(Shape::new(32, 0, 2)).err(),
            Some(LaneError::BadShape {
                workgroup: 0,
                buffers: 2
            })
        );
    }

    #[test]
    fn a_shape_whose_subgroup_is_not_a_power_of_two_is_refused_at_the_kernel() {
        // **The field that was not checked.** `Shape` carries four numbers and this validated
        // three of them, so `Shape::new(0, 64, 2)` built a kernel, stored to a buffer and finished
        // a module `spirv-val` accepts — the width never questioned, because the only thing that
        // questioned it was `Lanes::new`, which a kernel with no lane operation never reaches.
        //
        // A width is not a detail of the lane API. `decisions/DR-0002` makes it the number the
        // whole module is specialised to, and a shape naming an impossible one describes nothing
        // in the same way a workgroup of no invocations does.
        for width in [0_u32, 3, 24, 63, u32::MAX] {
            assert_eq!(
                Kernel::<F32>::new(Shape::new(width, 64, 2)).err(),
                Some(LaneError::BadWidth { width }),
                "a subgroup width of {width} was accepted"
            );
        }

        // And the same for a grid, which goes through the same builder.
        assert_eq!(
            Kernel::<F32>::new(Shape::grid(24, 64, 4, 2)).err(),
            Some(LaneError::BadWidth { width: 24 })
        );

        // Every power of two a device could report still builds, including the ones no device
        // does: this refuses what cannot be a width, not what is unlikely to be one.
        for width in [1_u32, 4, 8, 16, 32, 64, 128] {
            assert!(
                Kernel::<F32>::new(Shape::new(width, 64, 2)).is_ok(),
                "a subgroup width of {width} was refused"
            );
        }
    }

    #[test]
    fn the_workgroup_index_is_the_workgroup_built_in_and_not_the_invocation_one() {
        // Two accessors returning ids of the same type, one of which is a plausible wrong answer
        // for the other. Swapping them compiles, validates, and puts every block's result in the
        // slot belonging to a lane — so the assertion traces the id back to the decoration.
        use crate::spec::BuiltIn;

        let kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let group = kernel.workgroup_index();
        let local = kernel.local_index();
        assert_ne!(group, local, "the two positions are not the same value");

        let words = kernel.finish().expect("finished");

        // Which variable each built-in decorates.
        let decorated = |wanted: u32| {
            decode::body(&words)
                .filter(|instruction| instruction.opcode() == op::DECORATE)
                .find_map(|instruction| match instruction.operands() {
                    [target, decoration, built_in]
                        if *decoration == Decoration::BuiltIn.word() && *built_in == wanted =>
                    {
                        Some(*target)
                    }
                    _ => None,
                })
        };

        // `group` is a component extracted from the loaded vector, so the chain to follow is
        // extract <- load <- variable. Both are traced the same way, and only the built-in differs.
        let source_of = |value: Id| {
            let extracted = decode::body(&words)
                .filter(|instruction| instruction.opcode() == op::COMPOSITE_EXTRACT)
                .find_map(|instruction| match instruction.operands() {
                    [_type, id, composite, ..] if *id == value.word() => Some(*composite),
                    _ => None,
                })?;
            decode::body(&words)
                .filter(|instruction| instruction.opcode() == op::LOAD)
                .find_map(|instruction| match instruction.operands() {
                    [_type, id, pointer] if *id == extracted => Some(*pointer),
                    _ => None,
                })
        };

        // Both sides found, before they are compared. Two `None`s are equal, and a version of
        // this test that skipped this would pass for a module containing neither built-in.
        let from_group = source_of(group).expect("workgroup_index traces back to a variable");
        let from_local = source_of(local).expect("local_index traces back to a variable");
        let workgroup_id =
            decorated(BuiltIn::WorkgroupId.word()).expect("WorkgroupId decorates something");
        let local_id = decorated(BuiltIn::LocalInvocationId.word())
            .expect("LocalInvocationId decorates something");
        assert_ne!(workgroup_id, local_id, "two built-ins, two variables");

        assert_eq!(
            from_group, workgroup_id,
            "workgroup_index does not come from WorkgroupId"
        );
        assert_eq!(
            from_local, local_id,
            "local_index does not come from LocalInvocationId"
        );
    }

    #[test]
    fn the_index_type_is_unsigned_because_the_built_ins_are() {
        // `LocalInvocationId` and `WorkgroupId` are vectors of 32-bit *unsigned* integers, so the
        // scalar the addresses are computed in has to match.
        let kernel = Kernel::<F32>::new(Shape::new(32, 64, 1)).expect("built");
        let words = kernel.finish().expect("finished");

        let integers: Vec<Vec<u32>> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::TYPE_INT)
            .map(|instruction| instruction.operands().to_vec())
            .collect();

        assert_eq!(integers.len(), 1, "one integer type, declared once");
        assert_eq!(integers[0][1], 32, "of 32 bits");
        assert_eq!(integers[0][2], 0, "and unsigned");
    }

    #[test]
    fn the_element_type_reaches_the_instructions() {
        let mut kernel = Kernel::<U32>::new(Shape::new(32, 64, 2)).expect("built");
        let value = kernel.load::<32>(0).expect("loaded");
        let total = kernel
            .lanes()
            .expect("lanes")
            .reduce_sum(value)
            .expect("sum");
        kernel.store_scalar(1, total).expect("stored");

        let words = kernel.finish().expect("finished");

        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_I_ADD), 1);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_ADD), 0);
    }
}
