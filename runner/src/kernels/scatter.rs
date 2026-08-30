use super::{shape, whole_subgroup};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{LaneError, U32};

fn histogram_at<const LANES: u32>(
    subgroup: u32,
    workgroup: u32,
    bins: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, workgroup, 2))?;
    let value = kernel.load::<LANES>(0)?;

    let bin = {
        let mut lanes = kernel.lanes()?;
        let ceiling = lanes.splat_bits::<U32, LANES>(bins.saturating_sub(1))?;
        lanes.min(value, ceiling)?
    };

    let one = kernel.module().constant_u32(1)?;
    kernel.atomic_add_at(1, bin.id(), one)?;
    kernel.finish()
}

fn histogram_incrementing_at<const LANES: u32>(
    subgroup: u32,
    workgroup: u32,
    bins: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, workgroup, 2))?;
    let value = kernel.load::<LANES>(0)?;

    let bin = {
        let mut lanes = kernel.lanes()?;
        let ceiling = lanes.splat_bits::<U32, LANES>(bins.saturating_sub(1))?;
        lanes.min(value, ceiling)?
    };

    kernel.atomic_increment_at(1, bin.id())?;
    kernel.finish()
}

pub fn histogram(subgroup: u32, workgroup: u32, bins: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, histogram_at, workgroup, bins)
}

pub fn histogram_incrementing(
    subgroup: u32,
    workgroup: u32,
    bins: u32,
) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, histogram_incrementing_at, workgroup, bins)
}

pub fn claim_slots(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;

    let uint = kernel.index_type();
    let counter = kernel.module().constant_u32(0)?;
    let one = kernel.module().constant_u32(1)?;

    let claimed = kernel.atomic_add_at(1, counter, one)?;

    let slot = kernel.module().i_add(uint, claimed, one)?;
    let local = kernel.local_index();
    let pointer = kernel.element_pointer_to(1, slot)?;
    kernel.module().store(pointer, local)?;

    kernel.finish()
}

pub fn exchange_chain(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;

    let uint = kernel.index_type();
    let slot = kernel.module().constant_u32(0)?;
    let one = kernel.module().constant_u32(1)?;

    let mine = kernel.local_index();
    let displaced = kernel.atomic_exchange_at(1, slot, mine)?;

    let at = kernel.module().i_add(uint, mine, one)?;
    let pointer = kernel.element_pointer_to(1, at)?;
    kernel.module().store(pointer, displaced)?;

    kernel.finish()
}

pub fn atomic_gather(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, atomic_gather_at)
}

fn atomic_gather_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let index = kernel.load::<LANES>(0)?;

    let last = kernel
        .module()
        .constant_u32(super::WORKGROUP_SIZE.saturating_sub(1))?;
    let clamped = {
        let mut lanes = kernel.lanes()?;
        let limit = lanes.from_lane_value::<U32, LANES>(last)?;
        lanes.min(index, limit)?
    };

    let fetched = kernel.atomic_load_at(0, clamped.id())?;
    kernel.store_scalar(1, fetched)?;
    kernel.finish()
}

#[cfg(test)]
mod tests {
    use super::claim_slots;
    use simdr::decode;
    use simdr::module::op;

    #[test]
    fn the_allocator_declares_one_32_bit_integer_type_and_not_two() {
        let words = claim_slots(32).expect("built");

        let integers: Vec<Vec<u32>> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::TYPE_INT)
            .map(|instruction| instruction.operands().to_vec())
            .collect();

        assert_eq!(
            integers.len(),
            1,
            "two integer types of the same width: {integers:?}"
        );
        assert_eq!(
            integers.first().and_then(|operands| operands.get(1)),
            Some(&32)
        );
        assert_eq!(
            integers.first().and_then(|operands| operands.get(2)),
            Some(&0),
            "unsigned, which is what the kernel's own index arithmetic uses"
        );
    }
}
