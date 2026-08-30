mod narrow;

pub use narrow::Narrow;

use self::narrow::{
    EIGHT_BIT_STORAGE, INTEGER_DOT_PRODUCT, SHADER_FLOAT16_INT8, SIXTEEN_BIT_STORAGE,
    SUBGROUP_EXTENDED_TYPES, WANTED,
};
use crate::{Error, Gpu};
use ash::vk;
use simdr::spec::Capability;
use std::ffi::{CStr, c_char};

#[derive(Debug, Clone)]
pub struct Limits {
    pub name: String,
    pub subgroup_size: u32,
    pub subgroup_basic: bool,
    pub subgroup_arithmetic: bool,
    pub subgroup_clustered: bool,
    pub subgroup_shuffle: bool,
    pub subgroup_shuffle_relative: bool,
    pub subgroup_ballot: bool,
    pub subgroup_vote: bool,
    pub narrow: Narrow,
    pub max_workgroup_invocations: u32,
    pub timestamp_period_ns: f32,
}

impl Limits {
    #[must_use]
    pub const fn supports(&self, capability: Capability) -> bool {
        match capability {
            Capability::Shader => true,
            Capability::GroupNonUniform => self.subgroup_basic,
            Capability::GroupNonUniformVote => self.subgroup_vote,
            Capability::GroupNonUniformArithmetic => self.subgroup_arithmetic,
            Capability::GroupNonUniformBallot => self.subgroup_ballot,
            Capability::GroupNonUniformShuffle => self.subgroup_shuffle,
            Capability::GroupNonUniformShuffleRelative => self.subgroup_shuffle_relative,
            Capability::GroupNonUniformClustered => self.subgroup_clustered,
            Capability::Int8 => self.narrow.int8,
            Capability::Int16 => self.narrow.int16,
            Capability::Float16 => self.narrow.float16,
            Capability::StorageBuffer8BitAccess => self.narrow.storage8,
            Capability::StorageBuffer16BitAccess => self.narrow.storage16,
            Capability::DotProduct | Capability::DotProductInput4x8BitPacked => {
                self.narrow.integer_dot_product
            }
        }
    }

    #[must_use]
    pub const fn subgroup_surface(&self) -> bool {
        self.subgroup_basic
            && self.subgroup_arithmetic
            && self.subgroup_clustered
            && self.subgroup_shuffle
            && self.subgroup_shuffle_relative
            && self.subgroup_vote
    }

    #[must_use]
    pub fn unsupported_in(&self, spirv: &[u32]) -> Vec<Capability> {
        let mut missing = Vec::new();
        for instruction in simdr::decode::body(spirv) {
            if instruction.opcode() != simdr::module::op::CAPABILITY {
                continue;
            }
            let Some(&word) = instruction.operands().first() else {
                continue;
            };
            let Some(capability) = Capability::from_word(word) else {
                continue;
            };
            if !self.supports(capability) && !missing.contains(&capability) {
                missing.push(capability);
            }
        }
        missing
    }
}

impl Gpu {
    pub fn open_matching(pattern: Option<&str>) -> Result<Option<Self>, Error> {
        // SAFETY: `load` only reads the loader library from the usual platform location. It is
        // unsafe because dynamic loading is, not because of anything we pass it.
        let entry = match unsafe { ash::Entry::load() } {
            Ok(entry) => entry,
            Err(_) => return Ok(None),
        };

        let application =
            vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
        let instance_info = vk::InstanceCreateInfo::default().application_info(&application);

        // SAFETY: `instance_info` borrows `application`, which outlives this call, and no
        // extensions or layers are requested.
        let instance = unsafe { entry.create_instance(&instance_info, None) }?;

        // SAFETY: the instance was just created and nothing else holds it.
        unsafe { open_on(&entry, Guard::new(instance), pattern) }.map(Some)
    }

    pub fn names() -> Result<Vec<String>, Error> {
        // SAFETY: as in `open_matching`.
        let entry = match unsafe { ash::Entry::load() } {
            Ok(entry) => entry,
            Err(_) => return Ok(Vec::new()),
        };

        let application =
            vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
        let instance_info = vk::InstanceCreateInfo::default().application_info(&application);
        // SAFETY: as in `open_matching`.
        let instance = unsafe { entry.create_instance(&instance_info, None) }?;
        let guard = Guard::new(instance);

        // SAFETY: the instance is live for the whole of this block, and the guard destroys it.
        let names = unsafe { guard.enumerate_physical_devices() }?
            .into_iter()
            .filter(|&physical| {
                // SAFETY: as the enumeration above — the guard holds the instance live for this
                // whole block, and `physical` is a handle that enumeration just returned.
                let families =
                    unsafe { guard.get_physical_device_queue_family_properties(physical) };
                families.iter().any(|family| {
                    family.queue_flags.contains(vk::QueueFlags::COMPUTE) && family.queue_count > 0
                })
            })
            .map(|physical| {
                // SAFETY: as above, and a physical device handle needs no destruction — it is
                // owned by the instance, not by this.
                unsafe { name_of(&guard, physical) }
            })
            .collect();

        drop(guard);
        Ok(names)
    }

    pub fn open() -> Result<Option<Self>, Error> {
        let wanted = std::env::var("SIMDR_DEVICE").ok();
        Self::open_matching(wanted.as_deref())
    }
}

struct Guard {
    instance: Option<ash::Instance>,
}

impl Guard {
    const fn new(instance: ash::Instance) -> Self {
        Self {
            instance: Some(instance),
        }
    }

    fn release(mut self) -> ash::Instance {
        self.instance
            .take()
            .expect("the instance is taken only here, and this consumes the guard")
    }
}

impl std::ops::Deref for Guard {
    type Target = ash::Instance;

    fn deref(&self) -> &ash::Instance {
        self.instance
            .as_ref()
            .expect("the instance is taken only by `release`, which consumes the guard")
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(instance) = self.instance.take() {
            // SAFETY: this guard was the sole owner, and nothing derived from the instance
            // outlives the failed open that is unwinding past here.
            unsafe { instance.destroy_instance(None) };
        }
    }
}

unsafe fn name_of(instance: &ash::Instance, physical: vk::PhysicalDevice) -> String {
    // SAFETY: the caller's contract, and a physical device handle needs no destruction — it is
    // owned by the instance rather than by this.
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    properties
        .device_name_as_c_str()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::from("<unnamed device>"))
}

unsafe fn open_on(
    entry: &ash::Entry,
    instance: Guard,
    pattern: Option<&str>,
) -> Result<Gpu, Error> {
    // SAFETY: this function's own contract says the instance is live, and the guard keeps it so
    // until it is either released into a `Gpu` or destroyed on the way out.
    let candidates = unsafe { instance.enumerate_physical_devices() }?;
    let wanted = pattern.map(str::to_lowercase);

    let compute: Vec<(vk::PhysicalDevice, u32)> = candidates
        .into_iter()
        .filter_map(|physical| {
            // SAFETY: live instance as above, and `physical` came from its own enumeration.
            let families =
                unsafe { instance.get_physical_device_queue_family_properties(physical) };
            let family = families.iter().position(|family| {
                family.queue_flags.contains(vk::QueueFlags::COMPUTE) && family.queue_count > 0
            })?;
            u32::try_from(family).ok().map(|family| (physical, family))
        })
        .collect();

    if compute.is_empty() {
        return Err(Error::NoComputeDevice);
    }

    // SAFETY: as above — the guard holds the instance live, and each handle came from it.
    let named = |physical| unsafe { name_of(&instance, physical) };

    let Some((physical, queue_family)) = compute
        .iter()
        .copied()
        .filter(|&(physical, _)| {
            let Some(wanted) = wanted.as_deref() else {
                return true;
            };
            named(physical).to_lowercase().contains(wanted)
        })
        .max_by_key(|&(physical, _)| {
            // SAFETY: as above.
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            u8::from(properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        })
    else {
        return Err(Error::NoSuchDevice {
            wanted: pattern.unwrap_or_default().to_owned(),
            present: compute
                .iter()
                .map(|&(physical, _)| named(physical))
                .collect(),
        });
    };

    // SAFETY: live instance, and `physical` is one of the handles it enumerated.
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };

    // SAFETY: as above. The device has not been created yet, which is exactly when this must be
    // asked — enabling an extension the device does not have is a failed `create_device`.
    let available = unsafe { instance.enumerate_device_extension_properties(physical) }?;
    let offers = |wanted: &CStr| {
        available
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(wanted))
    };
    let wanted: Vec<&CStr> = WANTED.into_iter().filter(|name| offers(name)).collect();
    let names: Vec<*const c_char> = wanted.iter().map(|name| name.as_ptr()).collect();

    // SAFETY: both ask only that `physical` belong to a live `instance`, which is this function's
    // own precondition and the guard's job for as long as it holds one.
    let narrow = unsafe { narrow::supported(&instance, physical, &offers) };
    // SAFETY: as above, and `queue_family` is the index chosen from this same device's families.
    let limits = unsafe { describe(&instance, physical, queue_family, narrow) };

    let (mut storage8, mut storage16, mut float16int8, mut extended_types, mut dot_product) =
        narrow::to_enable(narrow);

    let mut features = vk::PhysicalDeviceFeatures2::default()
        .features(vk::PhysicalDeviceFeatures::default().shader_int16(narrow.int16));
    if offers(EIGHT_BIT_STORAGE) {
        features = features.push_next(&mut storage8);
    }
    if offers(SIXTEEN_BIT_STORAGE) {
        features = features.push_next(&mut storage16);
    }
    if offers(SHADER_FLOAT16_INT8) {
        features = features.push_next(&mut float16int8);
    }
    if offers(SUBGROUP_EXTENDED_TYPES) {
        features = features.push_next(&mut extended_types);
    }
    if offers(INTEGER_DOT_PRODUCT) {
        features = features.push_next(&mut dot_product);
    }

    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities);
    let queue_infos = [queue_info];
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&names)
        .push_next(&mut features);

    // SAFETY: the instance is live, `physical` is one of its devices, and every extension named
    // in `names` was filtered against what this device reported — enabling one it lacks is the
    // failure this avoids. The feature chain borrows structs that all outlive the call.
    let device = unsafe { instance.create_device(physical, &device_info, None) }?;
    // SAFETY: the device was created immediately above with exactly one queue of this family, so
    // index 0 is the queue that was asked for and it exists.
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    Ok(Gpu {
        device,
        queue,
        queue_family,
        memory_properties,
        limits,
        instance: instance.release(),
        _entry: entry.clone(),
    })
}

unsafe fn describe(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    queue_family: u32,
    narrow: Narrow,
) -> Limits {
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    // SAFETY: `physical` belongs to `instance`, which is this function's stated precondition, and
    // `properties` holds the subgroup struct alive through the `push_next` chain for the call.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };

    let core = properties.properties;

    let name = core
        .device_name_as_c_str()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::from("<unnamed device>"));

    let in_compute = subgroup
        .supported_stages
        .contains(vk::ShaderStageFlags::COMPUTE);
    let has = |operation: vk::SubgroupFeatureFlags| {
        in_compute && subgroup.supported_operations.contains(operation)
    };

    // SAFETY: as above — the device belongs to the instance and the query allocates nothing this
    // has to release.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let valid_bits = families
        .get(queue_family as usize)
        .map_or(0, |family| family.timestamp_valid_bits);
    let timestamp_period_ns = if valid_bits > 0 {
        core.limits.timestamp_period
    } else {
        0.0
    };

    Limits {
        name,
        subgroup_size: subgroup.subgroup_size,
        subgroup_basic: has(vk::SubgroupFeatureFlags::BASIC),
        subgroup_arithmetic: has(vk::SubgroupFeatureFlags::ARITHMETIC),
        subgroup_clustered: has(vk::SubgroupFeatureFlags::CLUSTERED),
        subgroup_shuffle: has(vk::SubgroupFeatureFlags::SHUFFLE),
        subgroup_shuffle_relative: has(vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE),
        subgroup_ballot: has(vk::SubgroupFeatureFlags::BALLOT),
        subgroup_vote: has(vk::SubgroupFeatureFlags::VOTE),
        narrow,
        max_workgroup_invocations: core.limits.max_compute_work_group_invocations,
        timestamp_period_ns,
    }
}

impl Gpu {
    pub(crate) const fn queue_family(&self) -> u32 {
        self.queue_family
    }

    pub(crate) const fn queue(&self) -> vk::Queue {
        self.queue
    }

    pub(crate) const fn device(&self) -> &ash::Device {
        &self.device
    }

    pub(crate) const fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }
}
