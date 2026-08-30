use super::{shape, whole_subgroup};
use simdr::kernel::Kernel;
use simdr::lanes::{Element, LaneError, U32};
use simdr::module::{Reduction, op};
use simdr::spec::{Capability, Scope};

pub mod spec_id {
    pub const ADDEND: u32 = 0;
    pub const FACTOR: u32 = 1;
    pub const OFFSET: u32 = 2;
    pub const CLUSTER: u32 = 3;
}

pub fn specialized_add<T: Element, const LANES: u32>(
    subgroup: u32,
    default: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<T>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let raised = {
        let mut lanes = kernel.lanes()?;
        let element = lanes.type_of::<T>()?;
        let addend = lanes
            .module()
            .spec_constant(element, default, spec_id::ADDEND)?;
        let addend = lanes.splat_id::<T, LANES>(addend)?;
        lanes.add(value, addend)?
    };
    kernel.store(1, raised)?;
    kernel.finish()
}

pub fn specialized_affine<const LANES: u32>(
    subgroup: u32,
    factor: u32,
    offset: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let result = {
        let mut lanes = kernel.lanes()?;
        let element = lanes.type_of::<U32>()?;
        let factor = lanes
            .module()
            .spec_constant(element, factor, spec_id::FACTOR)?;
        let offset = lanes
            .module()
            .spec_constant(element, offset, spec_id::OFFSET)?;

        let factor = lanes.splat_id::<U32, LANES>(factor)?;
        let offset = lanes.splat_id::<U32, LANES>(offset)?;
        let scaled = lanes.mul(value, factor)?;
        lanes.add(scaled, offset)?
    };
    kernel.store(1, result)?;
    kernel.finish()
}

pub fn specialized_cluster(subgroup: u32, default: u32) -> Result<Vec<u32>, LaneError> {
    whole_subgroup!(subgroup, specialized_cluster_at, default)
}

fn specialized_cluster_at<const LANES: u32>(
    subgroup: u32,
    default: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let element = kernel.element();

    let total = {
        let module = kernel.module();
        module.require_capability(Capability::GroupNonUniformArithmetic)?;
        module.require_capability(Capability::GroupNonUniformClustered)?;

        let size = module.spec_constant(element, default, spec_id::CLUSTER)?;
        let scope = module.scope(Scope::Subgroup)?;
        module.subgroup_reduce(
            op::GROUP_NON_UNIFORM_I_ADD,
            element,
            scope,
            Reduction::Clustered { size },
            value.id(),
        )?
    };

    let total = kernel.lanes()?.splat_id::<U32, LANES>(total)?;
    kernel.store(1, total)?;
    kernel.finish()
}

pub fn specialized_derived<const LANES: u32>(
    subgroup: u32,
    default: u32,
) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<LANES>(0)?;
    let raised = {
        let mut lanes = kernel.lanes()?;
        let element = lanes.type_of::<U32>()?;
        let base = lanes
            .module()
            .spec_constant(element, default, spec_id::ADDEND)?;
        let two = lanes.module().constant_u32(2)?;
        let doubled = lanes
            .module()
            .spec_constant_op(element, op::I_MUL, &[base, two])?;

        let doubled = lanes.splat_id::<U32, LANES>(doubled)?;
        lanes.add(value, doubled)?
    };
    kernel.store(1, raised)?;
    kernel.finish()
}
