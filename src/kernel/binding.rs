//! Setting up a kernel's interface: buffers, layout, entry point, invocation index.
//!
//! All of it is the same every time, which is exactly why it belongs here rather than being
//! copied into each kernel — this is the sixty lines that `Kernel::new` replaces.

use super::Shape;
use crate::lanes::{Element, LaneError};
use crate::module::{Id, Module, Section, Version, op};
use crate::spec::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, MemoryModel, StorageClass,
};

/// A built interface, with `main` open and waiting for a body.
pub(super) struct Parts {
    pub(super) module: Module,
    pub(super) element: Id,
    pub(super) element_pointer: Id,
    pub(super) uint: Id,
    pub(super) zero: Id,
    pub(super) buffers: Vec<Id>,
    /// This invocation's index within its workgroup.
    pub(super) local: Id,
    /// Which workgroup this invocation is in.
    pub(super) group: Id,
    /// This invocation's row across the whole dispatch, for a grid kernel.
    pub(super) row: Option<Id>,
}

/// Build a module holding the interface, and open `main`.
pub(super) fn build<T: Element>(shape: Shape) -> Result<Parts, LaneError> {
    if shape.buffers == 0 || shape.workgroup == 0 {
        return Err(LaneError::BadShape {
            workgroup: shape.workgroup,
            buffers: shape.buffers,
        });
    }
    if shape.rows == Some(0) {
        return Err(LaneError::BadRows { rows: 0 });
    }

    let mut module = Module::new(Version::V1_3);

    let main = module.alloc_id()?;
    module.name(main, "main")?;
    module.require_capability(Capability::Shader)?;
    module.emit(
        Section::MemoryModel,
        op::MEMORY_MODEL,
        &[AddressingModel::Logical.word(), MemoryModel::Glsl450.word()],
    )?;

    let element = T::type_id(&mut module)?;
    let uint = module.type_int(32, false)?;
    let uint3 = module.type_vector(uint, 3)?;

    let element_pointer = module.type_pointer(StorageClass::StorageBuffer, element)?;

    // One struct per buffer rather than one shared: §2.8 lets aggregates repeat precisely so they
    // can be decorated apart, and a caller that later wants different strides per binding should
    // not have to unpick a shared type first.
    // Two permissions rather than one: `element` above declared whatever the *type* needs, and
    // this declares what a storage buffer holding it needs. A device may offer `shaderInt8` and
    // not `storageBuffer8BitAccess`, and only the second is a property of the buffer.
    T::require_in_storage_buffer(&mut module)?;

    let mut buffers = Vec::with_capacity(shape.buffers as usize);
    for binding in 0..shape.buffers {
        let elements = module.type_runtime_array(element)?;
        let block = module.type_struct(&[elements])?;
        // The element's own width, not four. This is the whole of the narrow types' benefit: the
        // stride is what decides how many bytes a dispatch moves, and everything above it is
        // unchanged.
        module.decorate(elements, Decoration::ArrayStride, &[T::STRIDE])?;
        module.decorate(block, Decoration::Block, &[])?;
        module.member_decorate(block, 0, Decoration::Offset, &[0])?;

        let pointer = module.type_pointer(StorageClass::StorageBuffer, block)?;
        let variable = module.global_variable(pointer, StorageClass::StorageBuffer)?;
        module.decorate(variable, Decoration::DescriptorSet, &[0])?;
        module.decorate(variable, Decoration::Binding, &[binding])?;
        buffers.push(variable);
    }

    // Declared, decorated and named in the interface in one call each. Below SPIR-V 1.4 that
    // interface lists Input and Output only, so the buffers are deliberately absent — and it is
    // rendered rather than emitted, because a kernel that reaches for a built-in while its body is
    // being built adds to it long after this line. See `module::entry`.
    let local_id = module.builtin_input(BuiltIn::LocalInvocationId, uint3)?;
    let workgroup_id = module.builtin_input(BuiltIn::WorkgroupId, uint3)?;
    module.name(local_id, "local_id")?;
    module.name(workgroup_id, "workgroup_id")?;

    module.entry_point(ExecutionModel::GlCompute, main, "main")?;
    module.emit(
        Section::ExecutionMode,
        op::EXECUTION_MODE,
        &[
            main.word(),
            ExecutionMode::LocalSize.word(),
            shape.workgroup,
            shape.rows.unwrap_or(1),
            1,
        ],
    )?;

    let zero = module.constant_u32(0)?;

    let void = module.type_void()?;
    let signature = module.type_function(void, &[])?;
    module.begin_function(void, main, FunctionControl::None, signature)?;
    module.label()?;

    // Both built-ins are three-component vectors; a linear dispatch only ever uses x.
    let local_vector = module.load(uint3, local_id)?;
    let local = module.composite_extract(uint, local_vector, &[0])?;
    let group_vector = module.load(uint3, workgroup_id)?;
    let group = module.composite_extract(uint, group_vector, &[0])?;

    let row = row_index(&mut module, shape, uint, local_vector, group_vector)?;

    Ok(Parts {
        module,
        element,
        element_pointer,
        uint,
        zero,
        buffers,
        local,
        group,
        row,
    })
}

/// This invocation's row across the whole dispatch: `group.y × rows + local.y`.
///
/// Emitted once at function entry, because every access on the second axis wants it and the two
/// built-ins are already loaded here.
///
/// **A workgroup one row deep folds to `group.y`.** The `LocalSize` above declares y as 1, so
/// `LocalInvocationId.y` can only be 0 — that is not an assumption about the dispatch, it is what
/// the execution mode this same function emitted says. Two instructions saved on the shape that
/// most grid kernels use.
fn row_index(
    module: &mut Module,
    shape: Shape,
    uint: Id,
    local_vector: Id,
    group_vector: Id,
) -> Result<Option<Id>, LaneError> {
    let rows = match shape.rows {
        None => return Ok(None),
        Some(rows) => rows,
    };

    let group_y = module.composite_extract(uint, group_vector, &[1])?;
    if rows == 1 {
        return Ok(Some(group_y));
    }

    let local_y = module.composite_extract(uint, local_vector, &[1])?;
    let depth = module.constant_u32(rows)?;
    let base = module.i_mul(uint, group_y, depth)?;
    Ok(Some(module.i_add(uint, base, local_y)?))
}
