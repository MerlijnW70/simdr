//! The compute pipeline and the descriptors that point it at its buffers.
//!
//! Built once per run and reused across every timed iteration, which is the other half of making
//! [`crate::Gpu::time`] mean something: creating a pipeline costs far more than running one.

use super::Specialization;
use crate::buffer::Buffer;
use crate::{Error, Gpu};
use ash::vk;

/// A compute pipeline, its layout, and a descriptor set bound to two storage buffers.
pub(crate) struct Pipeline {
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
    pub(crate) unsafe fn new(
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
        // SAFETY: the device outlives this call, and `bindings` — which the info borrows — is a
        // local that outlives it too. `Pipeline::destroy` releases what is created here.
        let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(count)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: as above; `sizes` is a local the info borrows for the length of the call, and
        // the pool is sized for exactly the one set allocated from it below.
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;

        let set_layouts = [set_layout];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        // SAFETY: the pool was created immediately above with room for one set, and the layout is
        // the one created above from the same binding count.
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
        // SAFETY: `writes` points into `infos`, which is still in scope, and each info names a
        // buffer the caller supplied and guaranteed live. The descriptor set is not in use by any
        // submission — it was allocated a few lines ago and has not been bound.
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: the device is live and `spirv` outlives the call. The words are handed over
        // untouched, which is the whole point — see the crate doc on why this is not `wgpu`.
        let shader = unsafe { device.create_shader_module(&module_info, None) }?;

        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: `set_layouts` holds the layout created above and outlives the call.
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
        // SAFETY: the shader module and layout were created above and are still live, and the
        // specialization info points into `entries` and `data`, which are held in scope for
        // exactly this reason — a temporary would be freed while the driver held its address.
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
    pub(crate) const fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    /// Its layout, which binding descriptors needs.
    pub(crate) const fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    /// The descriptor set pointing at the buffers.
    pub(crate) const fn descriptors(&self) -> vk::DescriptorSet {
        self.descriptors
    }

    /// Release everything.
    ///
    /// # Safety
    ///
    /// No submission using this may still be in flight.
    pub(crate) unsafe fn destroy(self, gpu: &Gpu) {
        let device = gpu.device();
        // SAFETY: `self` is taken by value, so nothing else names these handles, and the caller's
        // contract says no submission using them is in flight. The pipeline goes before the layout
        // and module it was built from, and the descriptor pool before the set layout — destroying
        // a pool frees the sets allocated from it, which is why those are not released separately.
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
    /// Build a pipeline for each entry of `builds` and destroy them, so a caller can time it.
    ///
    /// The same shape as [`Gpu::probe_resident`], and it exists for the same reason: a claim about
    /// where the setup cost goes needs the parts measured apart. `notes/NEXT.md` argued for
    /// specialization constants on the grounds that "one module per parameter value" is expensive
    /// in *pipeline creation*, and pipeline creation had never been timed on its own.
    ///
    /// **A batch rather than one at a time, and that is the whole point.** The two buffers a
    /// descriptor set needs are allocated once here and reused for every pipeline. The first
    /// version of this took a single module and allocated a pair per call — so the number it
    /// produced was pipeline creation *plus two allocations*, and allocation is the larger half.
    /// It reported 485 µs where the pipeline itself is nearer 180, and that wrong number reached
    /// three documents before `runner/examples/reducer.rs` measured a whole reduction and did not
    /// add up.
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`] if any call fails, [`Error::NoPipeline`] if the driver returns none.
    pub fn probe_pipelines(&self, builds: &[(&[u32], &Specialization)]) -> Result<(), Error> {
        let bytes = 256;

        // SAFETY: both buffers and every pipeline are created here and destroyed before returning,
        // nothing is submitted, and nothing else ever sees them.
        unsafe {
            let source = Buffer::device_local(self, bytes)?;
            let destination = Buffer::device_local(self, bytes)?;
            let bound = [(&source, bytes), (&destination, bytes)];

            let mut built = Vec::with_capacity(builds.len());
            let mut outcome = Ok(());
            for (spirv, specialization) in builds {
                match Pipeline::new(self, spirv, &bound, specialization) {
                    Ok(pipeline) => built.push(pipeline),
                    Err(error) => {
                        outcome = Err(error);
                        break;
                    }
                }
            }

            for pipeline in built {
                pipeline.destroy(self);
            }
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
