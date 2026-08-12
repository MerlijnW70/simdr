//! Kernels with a value left open until the pipeline is created.
//!
//! Every other kernel here takes its constants as Rust arguments and bakes them in, which means a
//! module per value. These take a `SpecId` instead: one module, and the number arrives at
//! `vkCreateComputePipeline`.
//!
//! What is left open is always a *number*. `decisions/DR-0005` is why it is never a decision the
//! emitter has to make — the lane count and the subgroup width choose between different
//! instructions, and by the time a specialization constant has a value the instructions are long
//! since chosen.

use super::shape;
use simdr::kernel::Kernel;
use simdr::lanes::{Element, LaneError, U32};
use simdr::module::{Reduction, op};
use simdr::spec::{Capability, Scope};

/// Which `SpecId` each open value answers to.
///
/// Named rather than written at both ends: the constant is declared here and set in a test or a
/// caller, and a number that appears in two files with no name is a number that drifts.
pub mod spec_id {
    /// The addend in [`super::specialized_add`].
    pub const ADDEND: u32 = 0;
    /// The multiplier in [`super::specialized_affine`].
    pub const FACTOR: u32 = 1;
    /// The offset in [`super::specialized_affine`].
    pub const OFFSET: u32 = 2;
    /// The cluster size in [`super::specialized_cluster`].
    pub const CLUSTER: u32 = 3;
}

/// `out[i] = in[i] + k`, where `k` is fixed when the pipeline is created.
///
/// # Errors
///
/// [`LaneError`] if the module cannot be built.
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

/// `out[i] = in[i] * factor + offset`, both fixed when the pipeline is created.
///
/// Two open values rather than one, because a specialization block with a single entry cannot
/// tell a right offset calculation from an ignored one.
///
/// # Errors
///
/// As [`specialized_add`].
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

/// A clustered subgroup sum whose **cluster size** is a specialization constant.
///
/// The experiment `notes/NEXT.md` asked for. `ClusterSize` must be a constant *instruction*, and a
/// specialization constant is one — so this may work, and the only way to find out is to build it
/// and hand it to a validator and a driver. `decisions/DR-0005` records what happened.
///
/// Built through the module layer rather than through `Lanes`, because `Lanes` picks the mapping
/// itself from a lane count known at build time — which is the very thing being tested.
///
/// # Errors
///
/// As [`specialized_add`].
pub fn specialized_cluster(subgroup: u32, default: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(shape(subgroup))?;
    let value = kernel.load::<32>(0)?;
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

    let total = kernel.lanes()?.splat_id::<U32, 32>(total)?;
    kernel.store(1, total)?;
    kernel.finish()
}

/// `out[i] = in[i] + factor * 2`, where the doubling happens at pipeline creation.
///
/// The one kernel here that uses `OpSpecConstantOp`: the value added is *derived* from an open
/// constant rather than being one, so nothing in the function body computes it and the result is
/// still a constant by the time the shader is compiled.
///
/// # Errors
///
/// As [`specialized_add`].
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
