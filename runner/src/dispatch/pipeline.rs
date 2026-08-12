//! The compute pipeline and the descriptors that point it at its buffers.
//!
//! Built once per run and reused across every timed iteration, which is the other half of making
//! [`crate::Gpu::time`] mean something: creating a pipeline costs far more than running one.

use super::Specialization;
use crate::buffer::Buffer;
use crate::{Error, Gpu};
use ash::vk;

/// A compute pipeline, its layout, and a descriptor set bound to two storage buffers.
pub(super) struct Pipeline {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptors: vk::DescriptorSet,
    shader: vk::ShaderModule,
}

impl Pipeline {
    /// Build everything needed to dispatch `spirv` over the two buffers.
    ///
    /// # Safety
    ///
    /// The buffers must outlive this, and [`Pipeline::destroy`] must run before the device goes.
    /// `bound` is one entry per binding, in binding order: the buffer and how much of it the
    /// shader may see. Two is the common case and the shape every kernel in `kernels/` emits, but
    /// the emitter's `Shape` has always taken a count and there is no reason this could not.
    pub(super) unsafe fn new(
        gpu: &Gpu,
        spirv: &[u32],
        bound: &[(&Buffer, u64)],
        specialization: &Specialization,
    ) -> Result<Self, Error> {
        let device = gpu.device();
        let count = u32::try_from(bound.len()).map_err(|_| Error::NoPipeline)?;
        if count == 0 {
            return Err(Error::NoPipeline);
        }

        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> =
            (0..count).map(storage_binding).collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(count)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;

        let set_layouts = [set_layout];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let sets = unsafe { device.allocate_descriptor_sets(&allocate) }?;
        let descriptors = sets.first().copied().ok_or(Error::NoPipeline)?;

        // The infos have to outlive the `update_descriptor_sets` call, so they are collected
        // first and borrowed second — a `map` producing both at once would drop each one while
        // the write still pointed at it.
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bound
            .iter()
            .map(|&(buffer, bytes)| [buffer_info(buffer.handle, bytes)])
            .collect();
        let writes: Vec<vk::WriteDescriptorSet<'_>> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| storage_write(descriptors, binding as u32, info))
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader = unsafe { device.create_shader_module(&module_info, None) }?;

        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;

        // Declared before the stage that borrows them, and left in scope until the pipeline is
        // created: `SpecializationInfo` holds raw pointers into both, so a temporary would be
        // freed while the driver still had the address.
        let entries = specialization.map_entries();
        let data = specialization.data();
        let info = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&data);

        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main");
        if !specialization.is_empty() {
            // Attached only when there is something to say. An empty block is legal and this
            // keeps the common path byte-identical to what it was before specialization existed.
            stage = stage.specialization_info(&info);
        }

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipelines = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, result)| Error::Vulkan(result))?;
        let handle = pipelines.first().copied().ok_or(Error::NoPipeline)?;

        Ok(Self {
            handle,
            layout,
            set_layout,
            descriptor_pool,
            descriptors,
            shader,
        })
    }

    /// The pipeline to bind.
    pub(super) const fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    /// Its layout, which binding descriptors needs.
    pub(super) const fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    /// The descriptor set pointing at the buffers.
    pub(super) const fn descriptors(&self) -> vk::DescriptorSet {
        self.descriptors
    }

    /// Release everything.
    ///
    /// # Safety
    ///
    /// No submission using this may still be in flight.
    pub(super) unsafe fn destroy(self, gpu: &Gpu) {
        let device = gpu.device();
        unsafe {
            device.destroy_pipeline(self.handle, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_shader_module(self.shader, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

impl Gpu {
    /// Build a pipeline for `spirv` and destroy it, so a caller can time what that costs.
    ///
    /// The same shape as [`Gpu::probe_resident`], and it exists for the same reason: a claim about
    /// where the setup cost goes needs the parts measured apart. `notes/NEXT.md` argued for
    /// specialization constants on the grounds that "one module per parameter value" is expensive
    /// in *pipeline creation* — and pipeline creation had never been timed on its own.
    ///
    /// The two buffers are allocated here and freed here, so what is timed is
    /// `vkCreateShaderModule`, the descriptor plumbing and `vkCreateComputePipeline`, plus two
    /// small allocations that are the same in every call.
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`] if any call fails, [`Error::NoPipeline`] if the driver returns none.
    pub fn probe_pipeline(
        &self,
        spirv: &[u32],
        specialization: &Specialization,
    ) -> Result<(), Error> {
        let bytes = 256;

        // SAFETY: both buffers and the pipeline are created here and destroyed before returning,
        // nothing is submitted, and nothing else ever sees them.
        unsafe {
            let source = Buffer::device_local(self, bytes)?;
            let destination = Buffer::device_local(self, bytes)?;

            let built = Pipeline::new(
                self,
                spirv,
                &[(&source, bytes), (&destination, bytes)],
                specialization,
            );

            let outcome = match built {
                Ok(pipeline) => {
                    pipeline.destroy(self);
                    Ok(())
                }
                Err(error) => Err(error),
            };

            source.destroy(self);
            destination.destroy(self);
            outcome
        }
    }
}

/// One storage-buffer binding, visible to compute.
fn storage_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

/// The whole of `handle` as a descriptor's target.
fn buffer_info(handle: vk::Buffer, bytes: u64) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(handle)
        .offset(0)
        .range(bytes)
}

/// Point `binding` of `set` at `info`.
fn storage_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a [vk::DescriptorBufferInfo],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(info)
}
