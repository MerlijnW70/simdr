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

mod access;
mod binding;
mod plane;
mod scatter;
mod shared;

pub use self::access::Binding;
pub use self::shared::Shared;

use crate::lanes::{Element, LaneError, Lanes};
use crate::module::{Id, Module};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub subgroup: u32,
    pub workgroup: u32,
    pub rows: Option<u32>,
    pub buffers: u32,
}

impl Shape {
    #[must_use]
    pub const fn new(subgroup: u32, workgroup: u32, buffers: u32) -> Self {
        Self {
            subgroup,
            workgroup,
            rows: None,
            buffers,
        }
    }

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

pub struct Kernel<T: Element> {
    module: Module,
    shape: Shape,
    element: Id,
    element_pointer: Id,
    uint: Id,
    zero: Id,
    buffers: Vec<Id>,
    local: Id,
    group: Id,
    row: Option<Id>,
    marker: core::marker::PhantomData<T>,
}

impl<T: Element> crate::lanes::loops::Emits for Kernel<T> {
    fn module(&mut self) -> &mut Module {
        Self::module(self)
    }
}

impl<T: Element> Kernel<T> {
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

    pub fn lanes(&mut self) -> Result<Lanes<'_>, LaneError> {
        Lanes::new(&mut self.module, self.shape.subgroup)
    }

    pub const fn module(&mut self) -> &mut Module {
        &mut self.module
    }

    pub fn repeat_rolled<F>(
        &mut self,
        times: u32,
        carried_type: Id,
        initial: Id,
        body: F,
    ) -> Result<Id, LaneError>
    where
        F: FnOnce(&mut Self, Id, Id) -> Result<Id, LaneError>,
    {
        crate::lanes::loops::rolled(self, times, carried_type, initial, body)
    }

    pub fn repeat_rolled_many<F>(
        &mut self,
        times: u32,
        carried_type: Id,
        initial: &[Id],
        body: F,
    ) -> Result<Vec<Id>, LaneError>
    where
        F: FnOnce(&mut Self, &[Id], Id) -> Result<Vec<Id>, LaneError>,
    {
        crate::lanes::loops::rolled_many(self, times, carried_type, initial, body)
    }

    #[must_use]
    pub const fn element(&self) -> Id {
        self.element
    }

    #[must_use]
    pub const fn shape(&self) -> Shape {
        self.shape
    }

    #[must_use]
    pub const fn local_index(&self) -> Id {
        self.local
    }

    #[must_use]
    pub const fn workgroup_index(&self) -> Id {
        self.group
    }

    #[must_use]
    pub const fn index_type(&self) -> Id {
        self.uint
    }

    pub fn finish(mut self) -> Result<Vec<u32>, LaneError> {
        self.module.return_void()?;
        self.module.end_function()?;
        Ok(self.module.finish())
    }

    pub(super) const fn element_pointer(&self) -> Id {
        self.element_pointer
    }

    pub(super) const fn uint(&self) -> Id {
        self.uint
    }

    pub(super) const fn zero(&self) -> Id {
        self.zero
    }

    pub(super) const fn position(&self) -> (Id, Id) {
        (self.local, self.group)
    }

    pub(super) const fn row_index(&self) -> Option<Id> {
        self.row
    }

    /// How many descriptors this kernel has declared, which is where the next
    /// one goes.
    pub(super) fn bound(&self) -> u32 {
        self.buffers.len() as u32
    }

    /// Keeps a binding declared after the shape in the same list, so a later
    /// one lands beside it rather than on top of it.
    pub(super) fn remember(&mut self, variable: Id) {
        self.buffers.push(variable);
    }

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
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{F32, LaneError, U32};
    use crate::module::op;
    use crate::spec::{Capability, Decoration};

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    fn declares(words: &[u32], capability: Capability) -> bool {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == op::CAPABILITY)
            .any(|instruction| instruction.operands().first() == Some(&capability.word()))
    }

    #[test]
    fn a_rolled_loop_over_a_kernel_builds_one_body_that_reads_a_buffer() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 32, 2)).expect("built");
        let element = kernel.element();
        let nought = F32::constant_from_bits(kernel.module(), 0.0_f32.to_bits()).expect("nought");

        kernel
            .repeat_rolled(16, element, nought, |kernel, carried, counter| {
                let value = kernel.load_at(0, counter)?;
                Ok(kernel.module().f_add(element, carried, value)?)
            })
            .expect("looped");
        let words = kernel.finish().expect("finished");

        assert_eq!(
            count(&words, op::LOOP_MERGE),
            1,
            "a real loop and not sixteen bodies"
        );
        assert_eq!(count(&words, op::PHI), 2, "the counter and the value");
        assert_eq!(
            count(&words, op::F_ADD),
            1,
            "the body is built once, whatever the trips"
        );
        assert!(
            count(&words, op::ACCESS_CHAIN) >= 1,
            "and it reaches a buffer"
        );
    }

    #[test]
    fn a_rolled_loop_can_carry_several_running_totals_at_once() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 32, 2)).expect("built");
        let element = kernel.element();
        let nought = F32::constant_from_bits(kernel.module(), 0.0_f32.to_bits()).expect("nought");
        let start = vec![nought; 4];

        let out = kernel
            .repeat_rolled_many(16, element, &start, |kernel, carried, counter| {
                let value = kernel.load_at(0, counter)?;
                carried
                    .iter()
                    .map(|one| Ok(kernel.module().f_add(element, *one, value)?))
                    .collect()
            })
            .expect("looped");
        assert_eq!(out.len(), 4, "four totals out for four in");

        let at = kernel.module().constant_u32(0).expect("at");
        kernel.store_at(1, at, out[0]).expect("stored");
        let words = kernel.finish().expect("finished");

        assert_eq!(count(&words, op::LOOP_MERGE), 1, "one loop and not four");
        assert_eq!(count(&words, op::PHI), 5, "the counter and four values");
        assert_eq!(count(&words, op::F_ADD), 4, "one body, four totals");
    }

    #[test]
    fn a_rolled_body_that_carries_the_wrong_number_out_is_refused() {
        let mut kernel = Kernel::<F32>::new(Shape::new(32, 32, 2)).expect("built");
        let element = kernel.element();
        let nought = F32::constant_from_bits(kernel.module(), 0.0_f32.to_bits()).expect("nought");

        let refused = kernel.repeat_rolled_many(4, element, &[nought; 3], |_, carried, _| {
            Ok(carried[..2].to_vec())
        });
        assert!(matches!(
            refused,
            Err(LaneError::BadCarry {
                given: 2,
                wanted: 3
            })
        ));
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
        for width in [0_u32, 3, 24, 63, u32::MAX] {
            assert_eq!(
                Kernel::<F32>::new(Shape::new(width, 64, 2)).err(),
                Some(LaneError::BadWidth { width }),
                "a subgroup width of {width} was accepted"
            );
        }

        assert_eq!(
            Kernel::<F32>::new(Shape::grid(24, 64, 4, 2)).err(),
            Some(LaneError::BadWidth { width: 24 })
        );

        for width in [1_u32, 4, 8, 16, 32, 64, 128] {
            assert!(
                Kernel::<F32>::new(Shape::new(width, 64, 2)).is_ok(),
                "a subgroup width of {width} was refused"
            );
        }
    }

    #[test]
    fn the_workgroup_index_is_the_workgroup_built_in_and_not_the_invocation_one() {
        use crate::spec::BuiltIn;

        let kernel = Kernel::<F32>::new(Shape::new(32, 64, 2)).expect("built");
        let group = kernel.workgroup_index();
        let local = kernel.local_index();
        assert_ne!(group, local, "the two positions are not the same value");

        let words = kernel.finish().expect("finished");

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
