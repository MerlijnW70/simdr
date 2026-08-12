//! What a device offers for elements narrower than a lane, and how to ask.
//!
//! Split from [`super`] because it answers a different question. That file finds a device and
//! opens it; this one is the six Vulkan features `decisions/DR-0004` turns on, the four extensions
//! that carry them, and the query that fills them in.
//!
//! The six are listed apart rather than folded into one "supports i8" flag because each gates a
//! different thing, and a kernel needing only one of them should not be turned away for missing
//! another.

use ash::vk;
use std::ffi::CStr;

/// What a device offers for elements narrower than a lane.
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
    /// `shaderIntegerDotProduct`: `OpSDot` and its relatives exist.
    ///
    /// Whether the packed 4×8-bit form is *accelerated* is a separate thing the device reports,
    /// and [`Narrow::packed_dot_accelerated`] carries it. An implementation may support the
    /// instruction and lower it to the four multiplies it replaces, which is correct and is not
    /// the reason anyone reaches for it.
    pub integer_dot_product: bool,
    /// `integerDotProduct4x8BitPackedSignedAccelerated`: and the hardware does it in one go.
    ///
    /// Reported rather than required. A kernel using `OpSDot` runs either way; this is what says
    /// whether it is worth using, and it is the only one of these flags that is a *performance*
    /// statement rather than a legality one.
    pub packed_dot_accelerated: bool,
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

/// `VK_KHR_8bit_storage` — a storage buffer may hold 8-bit types.
///
/// Written out as a C string rather than taken from `ash`'s generated modules: an extension with
/// no commands may or may not get one, and a name spelt here is a name the compiler checks the
/// null-termination of. A wrong one silently disables the feature rather than failing.
pub(super) const EIGHT_BIT_STORAGE: &CStr = c"VK_KHR_8bit_storage";
/// `VK_KHR_16bit_storage` — the same for 16-bit types.
pub(super) const SIXTEEN_BIT_STORAGE: &CStr = c"VK_KHR_16bit_storage";
/// `VK_KHR_shader_float16_int8` — `f16` and `i8` arithmetic.
pub(super) const SHADER_FLOAT16_INT8: &CStr = c"VK_KHR_shader_float16_int8";
/// `VK_KHR_shader_subgroup_extended_types` — subgroup operations over narrow types.
pub(super) const SUBGROUP_EXTENDED_TYPES: &CStr = c"VK_KHR_shader_subgroup_extended_types";
/// `VK_KHR_shader_integer_dot_product` — `OpSDot` and its relatives.
pub(super) const INTEGER_DOT_PRODUCT: &CStr = c"VK_KHR_shader_integer_dot_product";

/// Every extension the narrow types need, for a caller filtering by what the device offers.
pub(super) const WANTED: [&CStr; 5] = [
    EIGHT_BIT_STORAGE,
    SIXTEEN_BIT_STORAGE,
    SHADER_FLOAT16_INT8,
    SUBGROUP_EXTENDED_TYPES,
    INTEGER_DOT_PRODUCT,
];

/// Which narrow-type features this device reports.
///
/// # Safety
///
/// `physical` must belong to `instance`.
pub(super) unsafe fn supported(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    offers: &impl Fn(&CStr) -> bool,
) -> Narrow {
    let mut storage8 = vk::PhysicalDevice8BitStorageFeatures::default();
    let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut float16int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut extended_types = vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures::default();
    let mut dot_product = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();

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
    if offers(INTEGER_DOT_PRODUCT) {
        features = features.push_next(&mut dot_product);
    }

    unsafe { instance.get_physical_device_features2(physical, &mut features) };

    // The core features are copied out at the chain's last use, and only then can the chained
    // structs be read: `push_next` leaves `features` holding a mutable borrow of each.
    let core = features.features;
    let integer_dot_product = dot_product.shader_integer_dot_product == vk::TRUE;

    // Whether the packed form is *accelerated* is a device **property**, not a feature, so it
    // arrives through a different call — and it is worth asking only if the instruction exists.
    let packed_dot_accelerated =
        integer_dot_product && unsafe { packed_dot_is_accelerated(instance, physical) };

    Narrow {
        int8: float16int8.shader_int8 == vk::TRUE,
        int16: core.shader_int16 == vk::TRUE,
        float16: float16int8.shader_float16 == vk::TRUE,
        storage8: storage8.storage_buffer8_bit_access == vk::TRUE,
        storage16: storage16.storage_buffer16_bit_access == vk::TRUE,
        subgroup_extended_types: extended_types.shader_subgroup_extended_types == vk::TRUE,
        integer_dot_product,
        packed_dot_accelerated,
    }
}

/// Whether the device does a packed 4×8-bit signed dot product in hardware.
///
/// A *property* rather than a feature, so it comes from `vkGetPhysicalDeviceProperties2` and not
/// from the feature chain. Nothing depends on it — a kernel using `OpSDot` runs either way — and
/// it is the only thing here that answers "is this worth doing" rather than "is this allowed".
///
/// # Safety
///
/// `physical` must belong to `instance`.
unsafe fn packed_dot_is_accelerated(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut dot = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
    {
        // Scoped, so the chain's borrow of `dot` ends before `dot` is read. A `drop` would not do
        // it — the struct is `Copy`, so dropping a copy releases nothing.
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut dot);
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    }

    dot.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE
}

/// The feature structs to hand `vkCreateDevice`, enabling exactly what `narrow` reports.
///
/// Returned as the four structs rather than a chain, because a chain borrows them and the caller
/// has to own them for as long as the create call runs.
pub(super) fn to_enable(
    narrow: Narrow,
) -> (
    vk::PhysicalDevice8BitStorageFeatures<'static>,
    vk::PhysicalDevice16BitStorageFeatures<'static>,
    vk::PhysicalDeviceShaderFloat16Int8Features<'static>,
    vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures<'static>,
    vk::PhysicalDeviceShaderIntegerDotProductFeatures<'static>,
) {
    (
        vk::PhysicalDevice8BitStorageFeatures::default()
            .storage_buffer8_bit_access(narrow.storage8),
        vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(narrow.storage16),
        vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_int8(narrow.int8)
            .shader_float16(narrow.float16),
        vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures::default()
            .shader_subgroup_extended_types(narrow.subgroup_extended_types),
        vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
            .shader_integer_dot_product(narrow.integer_dot_product),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kernel_needs_both_arithmetic_and_storage() {
        // The reason these are separate flags: a device offering one without the other cannot run
        // a narrow kernel end to end, and folding them into a single "supports i8" would have
        // reported that it could.
        let arithmetic_only = Narrow {
            int8: true,
            ..Narrow::default()
        };
        let storage_only = Narrow {
            storage8: true,
            ..Narrow::default()
        };
        let both = Narrow {
            int8: true,
            storage8: true,
            ..Narrow::default()
        };

        assert!(!arithmetic_only.byte_kernel());
        assert!(!storage_only.byte_kernel());
        assert!(both.byte_kernel());
    }

    #[test]
    fn the_16_bit_types_share_a_storage_flag_and_not_an_arithmetic_one() {
        // `storageBuffer16BitAccess` covers `i16`, `u16` and `f16` together; the arithmetic is
        // gated by `shaderInt16` and `shaderFloat16` separately. A device with one and not the
        // other is the case this distinguishes.
        let integers = Narrow {
            int16: true,
            storage16: true,
            ..Narrow::default()
        };

        assert!(integers.short_kernel());
        assert!(!integers.half_kernel(), "no shaderFloat16");
    }

    #[test]
    fn nothing_is_offered_by_a_device_that_reports_nothing() {
        let none = Narrow::default();

        assert!(!none.byte_kernel());
        assert!(!none.short_kernel());
        assert!(!none.half_kernel());
    }

    #[test]
    fn every_wanted_extension_is_named_once() {
        let mut names: Vec<&str> = WANTED
            .iter()
            .map(|name| name.to_str().unwrap_or_default())
            .collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();

        assert_eq!(names.len(), count, "an extension is listed twice");
        assert!(names.iter().all(|name| name.starts_with("VK_KHR_")));
    }

    #[test]
    fn the_enable_list_carries_exactly_what_was_reported() {
        let narrow = Narrow {
            int8: true,
            int16: false,
            float16: true,
            storage8: true,
            storage16: false,
            subgroup_extended_types: true,
            integer_dot_product: true,
            packed_dot_accelerated: false,
        };

        let (storage8, storage16, float16int8, extended, dot) = to_enable(narrow);

        assert_eq!(storage8.storage_buffer8_bit_access, vk::TRUE);
        assert_eq!(storage16.storage_buffer16_bit_access, vk::FALSE);
        assert_eq!(float16int8.shader_int8, vk::TRUE);
        assert_eq!(float16int8.shader_float16, vk::TRUE);
        assert_eq!(extended.shader_subgroup_extended_types, vk::TRUE);
        assert_eq!(dot.shader_integer_dot_product, vk::TRUE);
    }

    #[test]
    fn acceleration_is_reported_and_never_enabled() {
        // `packed_dot_accelerated` is a device *property*, not a feature: there is nothing to turn
        // on. A version of `to_enable` that tried to pass it would be passing a performance hint
        // as a permission, and the two are in different Vulkan structures for a reason.
        let accelerated = Narrow {
            integer_dot_product: true,
            packed_dot_accelerated: true,
            ..Narrow::default()
        };
        let lowered = Narrow {
            integer_dot_product: true,
            packed_dot_accelerated: false,
            ..Narrow::default()
        };

        let (.., accelerated_dot) = to_enable(accelerated);
        let (.., lowered_dot) = to_enable(lowered);

        assert_eq!(
            accelerated_dot.shader_integer_dot_product, lowered_dot.shader_integer_dot_product,
            "acceleration must not change what is enabled — only whether it is worth using"
        );
    }
}
