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
//! lives; `binding` has the interface every kernel starts from; `shared` has workgroup memory and
//! the barrier. All three are private — what they add appears as methods on [`Kernel`].

mod access;
mod binding;
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
    /// Invocations per workgroup.
    pub workgroup: u32,
    /// How many storage buffers to bind, in descriptor set 0 at bindings `0..buffers`.
    pub buffers: u32,
}

impl Shape {
    /// A kernel over `buffers` storage buffers, `workgroup` invocations at a time.
    #[must_use]
    pub const fn new(subgroup: u32, workgroup: u32, buffers: u32) -> Self {
        Self {
            subgroup,
            workgroup,
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
    use crate::spec::Capability;

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
