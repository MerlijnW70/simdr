use super::{shape, whole_subgroup};
use simdr::kernel::Kernel;
use simdr::lanes::{I32, LaneError, U32, Vector};

fn packed_dot_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let packed = kernel.load::<LANES>(0)?;
    let totals = {
        let mut lanes = kernel.lanes()?;
        let products = lanes.dot_signed(packed, packed)?;
        lanes.reinterpret(products)?
    };
    kernel.store(1, totals)?;
    kernel.finish()
}

fn unpacked_dot_at<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let packed = kernel.load::<LANES>(0)?;

    let totals = {
        let mut lanes = kernel.lanes()?;

        let mut total = square_of_byte::<LANES>(&mut lanes, packed, 0)?;
        for byte in 1_u32..4 {
            let squared = square_of_byte::<LANES>(&mut lanes, packed, byte)?;
            total = lanes.add(total, squared)?;
        }

        lanes.reinterpret(total)?
    };

    kernel.store(1, totals)?;
    kernel.finish()
}

fn signed_byte<const LANES: u32>(
    lanes: &mut simdr::lanes::Lanes<'_>,
    packed: Vector<U32, LANES>,
    byte: u32,
) -> Result<Vector<I32, LANES>, LaneError> {
    let up = lanes.splat_bits::<U32, LANES>(24 - byte * 8)?;
    let down = lanes.splat_bits::<U32, LANES>(24)?;

    let raised = lanes.shift_left(packed, up)?;
    let signed = lanes.reinterpret_unsigned(raised)?;
    lanes.shift_right_arithmetic(signed, down)
}

fn square_of_byte<const LANES: u32>(
    lanes: &mut simdr::lanes::Lanes<'_>,
    packed: Vector<U32, LANES>,
    byte: u32,
) -> Result<Vector<I32, LANES>, LaneError> {
    let component = signed_byte::<LANES>(lanes, packed, byte)?;
    lanes.mul(component, component)
}

pub fn byte_component(subgroup: u32, byte: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, byte_component_at, byte)
}

fn byte_component_at<const LANES: u32>(subgroup: u32, byte: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let packed = kernel.load::<LANES>(0)?;
    let component = {
        let mut lanes = kernel.lanes()?;
        let signed = signed_byte::<LANES>(&mut lanes, packed, byte)?;
        lanes.reinterpret(signed)?
    };
    kernel.store(1, component)?;
    kernel.finish()
}

pub fn packed_dot(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, packed_dot_at)
}

pub fn unpacked_dot(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, unpacked_dot_at)
}

fn repeated_packed_at<const LANES: u32>(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let packed = kernel.load::<LANES>(0)?;

    let totals = {
        let mut lanes = kernel.lanes()?;
        let mut total = lanes.splat_bits::<I32, LANES>(0)?;

        for step in 0..times {
            let salt = lanes.splat_bits::<U32, LANES>(step)?;
            let operand = lanes.add(packed, salt)?;
            total = lanes.dot_signed_saturating(operand, operand, total)?;
        }

        lanes.reinterpret(total)?
    };

    kernel.store(1, totals)?;
    kernel.finish()
}

fn repeated_unpacked_at<const LANES: u32>(
    subgroup: u32,
    times: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let packed = kernel.load::<LANES>(0)?;

    let totals = {
        let mut lanes = kernel.lanes()?;
        let mut total = lanes.splat_bits::<I32, LANES>(0)?;

        for step in 0..times {
            let salt = lanes.splat_bits::<U32, LANES>(step)?;
            let operand = lanes.add(packed, salt)?;

            let mut sum = square_of_byte::<LANES>(&mut lanes, operand, 0)?;
            for byte in 1_u32..4 {
                let squared = square_of_byte::<LANES>(&mut lanes, operand, byte)?;
                sum = lanes.add(sum, squared)?;
            }
            total = lanes.add(total, sum)?;
        }

        lanes.reinterpret(total)?
    };

    kernel.store(1, totals)?;
    kernel.finish()
}

pub fn repeated_packed_dot(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, repeated_packed_at, times)
}

pub fn repeated_unpacked_dot(subgroup: u32, times: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, repeated_unpacked_at, times)
}

fn mixed_dot_at<const LANES: u32>(subgroup: u32, offset: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let signed = kernel.load::<LANES>(0)?;
    let unsigned = kernel.load_offset::<LANES>(0, offset)?;

    let totals = {
        let mut lanes = kernel.lanes()?;
        let products = lanes.dot_mixed(signed, unsigned)?;
        lanes.reinterpret(products)?
    };
    kernel.store(1, totals)?;
    kernel.finish()
}

pub fn mixed_dot(subgroup: u32, offset: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, mixed_dot_at, offset)
}
