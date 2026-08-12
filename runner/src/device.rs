//! Finding a device and opening it.

use crate::{Error, Gpu};
use ash::vk;
use std::ffi::{CStr, c_char};

/// What a device reports about itself.
///
/// `subgroup_size` is the number the whole project turns on: it is how many lanes a `Simd<T, N>`
/// can map onto, it is decided by the hardware rather than by us, and it is only knowable at
/// runtime. 32 on NVIDIA, 32 or 64 on AMD.
#[derive(Debug, Clone)]
pub struct Limits {
    /// The device's own name, as the driver spells it.
    pub name: String,
    /// How many invocations a subgroup holds.
    pub subgroup_size: u32,
    /// Whether `GroupNonUniformArithmetic` — reductions and scans — is usable in compute.
    pub subgroup_arithmetic: bool,
    /// Whether clustered reductions are usable.
    pub subgroup_clustered: bool,
    /// Whether the shuffles are usable.
    pub subgroup_shuffle: bool,
    /// Whether the votes and `ballot` are usable.
    ///
    /// A kernel using them declares `GroupNonUniformBallot`, and a *surplus* capability
    /// declaration fails at pipeline creation rather than at validation — so a test that skips on
    /// this is skipping for a real reason rather than being cautious.
    pub subgroup_ballot: bool,
    /// What the device offers for elements narrower than 32 bits.
    pub narrow: Narrow,
    /// Nanoseconds per timestamp tick, or zero when the device cannot be asked.
    ///
    /// Non-zero means [`crate::Gpu::time`] reports what the *device* spent rather than what the
    /// host observed between a submit and a fence. The difference between the two is scheduling
    /// this harness has no view of, and at a few microseconds of work it dominates.
    pub timestamp_period_ns: f32,
}

/// What a device offers for elements narrower than a lane.
///
/// Four separate permissions, and a device may hold any subset of them. They are listed apart
/// rather than folded into one "supports i8" flag because each gates a different thing, and a
/// kernel that needs only one of them should not be turned away for missing another.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Narrow {
    /// `shaderInt8`: 8-bit integers work in arithmetic.
    pub int8: bool,
    /// `shaderInt16`: 16-bit integers work in arithmetic. A core Vulkan 1.0 feature, unlike the
    /// other three.
    pub int16: bool,
    /// `shaderFloat16`: 16-bit floats work in arithmetic.
    pub float16: bool,
    /// `storageBuffer8BitAccess`: a storage buffer may hold 8-bit types.
    ///
    /// The one that matters for bandwidth. Arithmetic in `i8` over a buffer of `i32` moves just
    /// as many bytes as `i32` throughout.
    pub storage8: bool,
    /// `storageBuffer16BitAccess`: a storage buffer may hold 16-bit types.
    pub storage16: bool,
    /// `shaderSubgroupExtendedTypes`: the subgroup operations accept 8- and 16-bit types.
    ///
    /// **This one leaves no trace in the module.** There is no SPIR-V capability for it, so a
    /// module reducing over `i8` is byte-for-byte what it would be if the feature existed
    /// everywhere — it validates, and then a device without the feature refuses the pipeline.
    pub subgroup_extended_types: bool,
}

impl Narrow {
    /// Whether an `i8` or `u8` kernel reading an 8-bit buffer can run here at all.
    #[must_use]
    pub const fn byte_kernel(self) -> bool {
        self.int8 && self.storage8
    }

    /// The same for the 16-bit integer types.
    #[must_use]
    pub const fn short_kernel(self) -> bool {
        self.int16 && self.storage16
    }

    /// The same for `f16`.
    #[must_use]
    pub const fn half_kernel(self) -> bool {
        self.float16 && self.storage16
    }
}

impl Gpu {
    /// Open the first device that can run compute work.
    ///
    /// Returns `Ok(None)` when the loader is present but no device is — a machine without a GPU is
    /// a normal state for a test suite to find, not an error to fail on. A loader that will not
    /// load at all *is* an error, because it means the environment is broken rather than bare.
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`] if a call fails, [`Error::NoComputeDevice`] if devices exist but none
    /// offers a compute queue.
    /// Open a device whose name contains `pattern`, case-insensitively.
    ///
    /// The reason this exists: a machine with two GPUs has two *subgroup widths*, and the whole
    /// lane API turns on that number. `decisions/DR-0002` argues that a module is built for one
    /// width, and until there was a way to ask for the other device, only one width had ever run.
    ///
    /// `None` keeps the old behaviour — prefer a discrete GPU — and [`Gpu::open`] passes whatever
    /// `SIMDR_DEVICE` holds, so an entire test run can be pointed at the other device without a
    /// line of it knowing.
    ///
    /// # Errors
    ///
    /// As [`Gpu::open`]. A pattern matching nothing gives `Ok(None)`, the same as no device at
    /// all: a machine that lacks the part being asked for is a normal state for a suite to find.
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
        match unsafe { open_on(&entry, Guard::new(instance), pattern) } {
            Ok(gpu) => Ok(Some(gpu)),
            Err(Error::NoComputeDevice) if pattern.is_some() => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// The names of every device that could run compute work here.
    ///
    /// What `simdr probe --all` lists, and what a caller needs before it can pass one of them to
    /// [`Gpu::open_matching`].
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`] if a call fails. A machine with no loader gives an empty list rather than
    /// an error, for the same reason [`Gpu::open`] gives `Ok(None)`.
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
                let families =
                    unsafe { guard.get_physical_device_queue_family_properties(physical) };
                families.iter().any(|family| {
                    family.queue_flags.contains(vk::QueueFlags::COMPUTE) && family.queue_count > 0
                })
            })
            .map(|physical| {
                let properties = unsafe { guard.get_physical_device_properties(physical) };
                properties
                    .device_name_as_c_str()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| String::from("<unnamed device>"))
            })
            .collect();

        drop(guard);
        Ok(names)
    }

    /// Open the first device that can run compute work, or the one `SIMDR_DEVICE` names.
    pub fn open() -> Result<Option<Self>, Error> {
        // Vulkan 1.1 is the floor: it is where subgroup operations and
        // `VkPhysicalDeviceSubgroupProperties` became core, and this crate exists to exercise
        // those. `open_matching` is where that is asked for.
        let wanted = std::env::var("SIMDR_DEVICE").ok();
        Self::open_matching(wanted.as_deref())
    }
}

/// Owns an instance until a [`Gpu`] takes it over, and destroys it if one never does.
///
/// Every early return between creating an instance and building the device around it has to
/// release it, and the version of this that threaded the instance back through the error type
/// worked but made `Result`'s error variant enormous — clippy was right to object.
struct Guard {
    instance: Option<ash::Instance>,
}

impl Guard {
    const fn new(instance: ash::Instance) -> Self {
        Self {
            instance: Some(instance),
        }
    }

    /// Hand the instance over, disarming the guard.
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

/// Pick a device on `instance` and build a [`Gpu`] around it.
///
/// The guard destroys the instance on any early return, so nothing here has to remember to.
///
/// # Safety
///
/// `instance` must hold a live instance the caller has not destroyed.
unsafe fn open_on(
    entry: &ash::Entry,
    instance: Guard,
    pattern: Option<&str>,
) -> Result<Gpu, Error> {
    let candidates = unsafe { instance.enumerate_physical_devices() }?;
    let wanted = pattern.map(str::to_lowercase);

    let Some((physical, queue_family)) = candidates
        .into_iter()
        .filter_map(|physical| {
            let families =
                unsafe { instance.get_physical_device_queue_family_properties(physical) };
            let family = families.iter().position(|family| {
                family.queue_flags.contains(vk::QueueFlags::COMPUTE) && family.queue_count > 0
            })?;
            u32::try_from(family).ok().map(|family| (physical, family))
        })
        .filter(|&(physical, _)| {
            // A name filter, when there is one. Substring rather than exact: nobody types
            // "AMD Radeon(TM) Graphics" correctly twice.
            let Some(wanted) = wanted.as_deref() else {
                return true;
            };
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            properties
                .device_name_as_c_str()
                .is_ok_and(|name| name.to_string_lossy().to_lowercase().contains(wanted))
        })
        // Prefer a discrete GPU when nothing was asked for. With a pattern this still applies, and
        // only among the devices that matched it.
        .max_by_key(|&(physical, _)| {
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            u8::from(properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        })
    else {
        return Err(Error::NoComputeDevice);
    };

    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };

    // Which of the narrow-type extensions this device has. Enabling one it does not have is a
    // failed `create_device`, so the list is filtered rather than assumed — and the same query
    // decides what `Limits` reports, so the two cannot drift apart.
    let available = unsafe { instance.enumerate_device_extension_properties(physical) }?;
    let offers = |wanted: &CStr| {
        available
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(wanted))
    };
    let wanted: Vec<&CStr> = [
        EIGHT_BIT_STORAGE,
        SIXTEEN_BIT_STORAGE,
        SHADER_FLOAT16_INT8,
        SUBGROUP_EXTENDED_TYPES,
    ]
    .into_iter()
    .filter(|name| offers(name))
    .collect();
    let names: Vec<*const c_char> = wanted.iter().map(|name| name.as_ptr()).collect();

    // Asked for in its own scope, then *asked for again* as an enable list below. The alternative
    // — filling one set of structs from the query and handing the same ones to `create_device` —
    // is fewer lines and does not compile: the chain holds a mutable borrow of each struct for as
    // long as the chain lives, so nothing can read a flag out of one until the create call is
    // done with it.
    let narrow = unsafe { supported_narrow(&instance, physical, &offers) };
    let limits = unsafe { describe(&instance, physical, queue_family, narrow) };

    let mut storage8 = vk::PhysicalDevice8BitStorageFeatures::default()
        .storage_buffer8_bit_access(narrow.storage8);
    let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default()
        .storage_buffer16_bit_access(narrow.storage16);
    let mut float16int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
        .shader_int8(narrow.int8)
        .shader_float16(narrow.float16);
    let mut extended_types = vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures::default()
        .shader_subgroup_extended_types(narrow.subgroup_extended_types);

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

    let priorities = [1.0_f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities);
    let queue_infos = [queue_info];
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&names)
        // `push_next` of a `PhysicalDeviceFeatures2` is how the core features are passed when a
        // chain is used; `enabled_features` must stay null, and setting both is invalid.
        .push_next(&mut features);

    let device = unsafe { instance.create_device(physical, &device_info, None) }?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    Ok(Gpu {
        device,
        queue,
        queue_family,
        memory_properties,
        limits,
        // Only now, once nothing left can fail: from here the `Gpu`'s own `Drop` owns it.
        instance: instance.release(),
        _entry: entry.clone(),
    })
}

/// Read the properties this crate cares about off a physical device.
///
/// # Safety
///
/// `physical` must belong to `instance`.
unsafe fn describe(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    queue_family: u32,
    narrow: Narrow,
) -> Limits {
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };

    // Copied out first: `push_next` keeps `properties` holding a mutable borrow of `subgroup`, so
    // nothing can read the subgroup fields until `properties` is last used — which is here.
    let core = properties.properties;

    let name = core
        .device_name_as_c_str()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::from("<unnamed device>"));

    // A subgroup operation is only usable if the *compute* stage is among the supported ones —
    // a device may report the operation and not offer it where we need it.
    let in_compute = subgroup
        .supported_stages
        .contains(vk::ShaderStageFlags::COMPUTE);
    let has = |operation: vk::SubgroupFeatureFlags| {
        in_compute && subgroup.supported_operations.contains(operation)
    };

    // Timestamps need two things and both are optional: a non-zero period on the device, and a
    // queue family that reports valid bits. A queue with zero valid bits accepts the write and
    // returns nothing useful, which is worse than refusing.
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
        subgroup_arithmetic: has(vk::SubgroupFeatureFlags::ARITHMETIC),
        subgroup_clustered: has(vk::SubgroupFeatureFlags::CLUSTERED),
        subgroup_shuffle: has(vk::SubgroupFeatureFlags::SHUFFLE),
        subgroup_ballot: has(vk::SubgroupFeatureFlags::BALLOT),
        narrow,
        timestamp_period_ns,
    }
}

/// Which narrow-type features this device reports.
///
/// # Safety
///
/// `physical` must belong to `instance`.
unsafe fn supported_narrow(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    offers: &impl Fn(&CStr) -> bool,
) -> Narrow {
    let mut storage8 = vk::PhysicalDevice8BitStorageFeatures::default();
    let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut float16int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut extended_types = vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures::default();

    // Only the structs whose extension the device has. Chaining one it does not know is not an
    // error the loader reports — it is a struct the driver skips, leaving the flags at zero, which
    // reads as "unsupported" and would be indistinguishable from the truth.
    let mut features = vk::PhysicalDeviceFeatures2::default();
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

    unsafe { instance.get_physical_device_features2(physical, &mut features) };

    // As in `describe`: the core features are copied out at the chain's last use, and only then
    // can the chained structs be read.
    let core = features.features;
    Narrow {
        int8: float16int8.shader_int8 == vk::TRUE,
        int16: core.shader_int16 == vk::TRUE,
        float16: float16int8.shader_float16 == vk::TRUE,
        storage8: storage8.storage_buffer8_bit_access == vk::TRUE,
        storage16: storage16.storage_buffer16_bit_access == vk::TRUE,
        subgroup_extended_types: extended_types.shader_subgroup_extended_types == vk::TRUE,
    }
}

/// `VK_KHR_8bit_storage` — a storage buffer may hold 8-bit types.
///
/// Written out as a C string rather than taken from `ash`'s generated modules: an extension with
/// no commands may or may not get one, and a name spelt here is a name the compiler checks the
/// null-termination of. A wrong one silently disables the feature rather than failing.
const EIGHT_BIT_STORAGE: &CStr = c"VK_KHR_8bit_storage";
/// `VK_KHR_16bit_storage` — the same for 16-bit types.
const SIXTEEN_BIT_STORAGE: &CStr = c"VK_KHR_16bit_storage";
/// `VK_KHR_shader_float16_int8` — `f16` and `i8` arithmetic.
const SHADER_FLOAT16_INT8: &CStr = c"VK_KHR_shader_float16_int8";
/// `VK_KHR_shader_subgroup_extended_types` — subgroup operations over narrow types.
const SUBGROUP_EXTENDED_TYPES: &CStr = c"VK_KHR_shader_subgroup_extended_types";

impl Gpu {
    /// The queue family this device's queue came from.
    pub(crate) const fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// The queue kernels are submitted on.
    pub(crate) const fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// The logical device.
    pub(crate) const fn device(&self) -> &ash::Device {
        &self.device
    }

    /// The memory types this device offers.
    pub(crate) const fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }
}
