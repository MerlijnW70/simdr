//! Finding a device and opening it.

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

/// What a device reports about itself.
///
/// `subgroup_size` is the number the whole project turns on: it is how many lanes a `Simd<T, N>`
/// can map onto, it is decided by the implementation rather than by us, and it is only knowable at
/// runtime. Measured here: 32 on an NVIDIA RTX 4080, 64 on an integrated AMD Radeon, and 8 on
/// Mesa's lavapipe, which runs on the CPU.
#[derive(Debug, Clone)]
pub struct Limits {
    /// The device's own name, as the driver spells it.
    pub name: String,
    /// How many invocations a subgroup holds.
    pub subgroup_size: u32,
    /// Whether the subgroup instructions exist at all — `GroupNonUniform`, Vulkan's `BASIC` bit.
    ///
    /// **Every kernel here that touches a lane declares this capability**, and nothing reported it
    /// until the capabilities and the feature bits were laid side by side. It is offered by every
    /// device that offers any of the others, which is exactly why it went unnoticed.
    pub subgroup_basic: bool,
    /// Whether `GroupNonUniformArithmetic` — reductions and scans — is usable in compute.
    pub subgroup_arithmetic: bool,
    /// Whether clustered reductions are usable.
    pub subgroup_clustered: bool,
    /// Whether the arbitrary shuffles are usable — `OpGroupNonUniformShuffle` and `ShuffleXor`.
    pub subgroup_shuffle: bool,
    /// Whether the **relative** shuffles are usable — up and down by a delta.
    ///
    /// A separate feature bit and a separate capability, and the one the whole scan rests on: the
    /// clustered ladder is `log2(cluster)` `ShuffleUp`s, and `Lanes::shift_up`/`shift_down` are
    /// nothing else. This was missing while every one of those kernels declared
    /// `GroupNonUniformShuffleRelative` and their tests gated on the *arbitrary* shuffle.
    pub subgroup_shuffle_relative: bool,
    /// Whether `ballot` is usable.
    ///
    /// A kernel using it declares `GroupNonUniformBallot`, and a *surplus* capability declaration
    /// fails at pipeline creation rather than at validation — so a test that skips on this is
    /// skipping for a real reason rather than being cautious.
    pub subgroup_ballot: bool,
    /// Whether the **votes** are usable — `any`, `all`, and the vote about a value.
    ///
    /// A separate feature bit from the ballot, and a separate capability:
    /// `GroupNonUniformVote` against `GroupNonUniformBallot`. This was missing from these limits
    /// while three kernels used votes and their tests gated on the ballot instead — right on every
    /// device here, because no implementation offers one without the other, and a claim about the
    /// wrong feature all the same.
    pub subgroup_vote: bool,
    /// What the device offers for elements narrower than 32 bits.
    pub narrow: Narrow,
    /// The most invocations one workgroup may hold — `maxComputeWorkGroupInvocations`.
    ///
    /// The ceiling on `Shape::workgroup`, and on `workgroup × rows` for a grid. Asking for more
    /// fails at pipeline creation with no useful message, which is why it is reported here rather
    /// than discovered: a caller sweeping workgroup sizes needs to know where to stop.
    pub max_workgroup_invocations: u32,
    /// Nanoseconds per timestamp tick, or zero when the device cannot be asked.
    ///
    /// Non-zero means [`crate::Gpu::time`] reports what the *device* spent rather than what the
    /// host observed between a submit and a fence. The difference between the two is scheduling
    /// this harness has no view of, and at a few microseconds of work it dominates.
    pub timestamp_period_ns: f32,
}

impl Limits {
    /// Whether this device offers what `capability` needs.
    ///
    /// **The correspondence, written down once.** A module declares capabilities; a device offers
    /// feature bits; and until this existed the two were matched by whoever wrote each test —
    /// which is how three kernels using votes came to be gated on the *ballot*, and how every
    /// kernel in the library declared `GroupNonUniform` while nothing reported whether the device
    /// had it. Both were right on all three devices here, because no implementation offers one of
    /// these without the others. Being right by luck is what the audit was looking for.
    #[must_use]
    pub const fn supports(&self, capability: Capability) -> bool {
        match capability {
            // Any device that runs compute at all.
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

    /// Whether this device offers every subgroup feature the generated programs can reach.
    ///
    /// **The gate the fuzzer needs, and the one it was writing by hand.** A generated program may
    /// end in a reduction (`Arithmetic`), fold a cluster (`Clustered`), butterfly (`Shuffle`),
    /// shift (`ShuffleRelative`) or vote (`Vote`) — and the gate named the first three. The two it
    /// left out are offered by every device here, which is why nothing noticed; naming them is the
    /// difference between a gate that is right and one that is lucky.
    #[must_use]
    pub const fn subgroup_surface(&self) -> bool {
        self.subgroup_basic
            && self.subgroup_arithmetic
            && self.subgroup_clustered
            && self.subgroup_shuffle
            && self.subgroup_shuffle_relative
            && self.subgroup_vote
    }

    /// What `spirv` declares that this device does not offer.
    ///
    /// Read out of the module rather than out of the caller's memory: `OpCapability` is the
    /// module's own statement of what it needs, and a pipeline built from a module the device
    /// cannot satisfy fails with a message naming neither the capability nor the kernel.
    ///
    /// Empty means every declared capability is offered — *not* that the dispatch will work, since
    /// a feature can be present and the module still wrong. It is the necessary half of the
    /// condition, and the half a test can check before it decides to skip.
    ///
    /// A capability this crate does not know is ignored rather than reported: it cannot have been
    /// declared by this emitter, so it came from somewhere with its own reasons.
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
    /// Returns `Ok(None)` when the loader is present but no matching device is — a machine without
    /// a GPU, or without the part being asked for, is a normal state for a test suite to find
    /// rather than an error to fail on. A loader that will not load at all is not an error either,
    /// for the same reason: the environment is bare, not broken.
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`] if a call fails, [`Error::NoComputeDevice`] if devices exist and none
    /// offers a compute queue — which is reported only when no `pattern` was given, since a
    /// pattern that matches nothing is the `Ok(None)` case above.
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
    /// What `simdr list` prints, and what a caller needs before it can pass one of them to
    /// [`Gpu::open_matching`]. (It said `simdr probe --all` for as long as that spelling had not
    /// existed — the subcommand was `list` from the day it was written.)
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
    ///
    /// The whole suite goes through here, which is what lets a run be pointed at the other device
    /// in this machine without a line of it knowing: `SIMDR_DEVICE=radeon` matches a substring,
    /// case-insensitively, and `simdr list` names what there is to match.
    ///
    /// # Errors
    ///
    /// As [`Gpu::open_matching`].
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
    // SAFETY: this function's own contract says the instance is live, and the guard keeps it so
    // until it is either released into a `Gpu` or destroyed on the way out.
    let candidates = unsafe { instance.enumerate_physical_devices() }?;
    let wanted = pattern.map(str::to_lowercase);

    let Some((physical, queue_family)) = candidates
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
        .filter(|&(physical, _)| {
            // A name filter, when there is one. Substring rather than exact: nobody types
            // "AMD Radeon(TM) Graphics" correctly twice.
            let Some(wanted) = wanted.as_deref() else {
                return true;
            };
            // SAFETY: as above.
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            properties
                .device_name_as_c_str()
                .is_ok_and(|name| name.to_string_lossy().to_lowercase().contains(wanted))
        })
        // Prefer a discrete GPU when nothing was asked for. With a pattern this still applies, and
        // only among the devices that matched it.
        .max_by_key(|&(physical, _)| {
            // SAFETY: as above.
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            u8::from(properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
        })
    else {
        return Err(Error::NoComputeDevice);
    };

    // SAFETY: live instance, and `physical` is one of the handles it enumerated.
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };

    // Which of the narrow-type extensions this device has. Enabling one it does not have is a
    // failed `create_device`, so the list is filtered rather than assumed — and the same query
    // decides what `Limits` reports, so the two cannot drift apart.
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

    // Asked for in its own scope, then *asked for again* as an enable list below. The alternative
    // — filling one set of structs from the query and handing the same ones to `create_device` —
    // is fewer lines and does not compile: the chain holds a mutable borrow of each struct for as
    // long as the chain lives, so nothing can read a flag out of one until the create call is
    // done with it.
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
        // `push_next` of a `PhysicalDeviceFeatures2` is how the core features are passed when a
        // chain is used; `enabled_features` must stay null, and setting both is invalid.
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
    // SAFETY: `physical` belongs to `instance`, which is this function's stated precondition, and
    // `properties` holds the subgroup struct alive through the `push_next` chain for the call.
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
