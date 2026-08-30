use ash::vk;
use std::ffi::CStr;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Narrow {
    pub int8: bool,
    pub int16: bool,
    pub float16: bool,
    pub storage8: bool,
    pub storage16: bool,
    pub subgroup_extended_types: bool,
    pub integer_dot_product: bool,
    pub packed_dot_accelerated: bool,
}

impl Narrow {
    #[must_use]
    pub const fn byte_kernel(self) -> bool {
        self.int8 && self.storage8
    }

    #[must_use]
    pub const fn short_kernel(self) -> bool {
        self.int16 && self.storage16
    }

    #[must_use]
    pub const fn half_kernel(self) -> bool {
        self.float16 && self.storage16
    }
}

pub(super) const EIGHT_BIT_STORAGE: &CStr = c"VK_KHR_8bit_storage";
pub(super) const SIXTEEN_BIT_STORAGE: &CStr = c"VK_KHR_16bit_storage";
pub(super) const SHADER_FLOAT16_INT8: &CStr = c"VK_KHR_shader_float16_int8";
pub(super) const SUBGROUP_EXTENDED_TYPES: &CStr = c"VK_KHR_shader_subgroup_extended_types";
pub(super) const INTEGER_DOT_PRODUCT: &CStr = c"VK_KHR_shader_integer_dot_product";

pub(super) const WANTED: [&CStr; 5] = [
    EIGHT_BIT_STORAGE,
    SIXTEEN_BIT_STORAGE,
    SHADER_FLOAT16_INT8,
    SUBGROUP_EXTENDED_TYPES,
    INTEGER_DOT_PRODUCT,
];

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

    // SAFETY: `physical` belongs to `instance`, which is this function's stated precondition, and
    // every struct in the `push_next` chain is a local that outlives the call. Only structs whose
    // extension the device reported were chained in above — asking for one it does not have is
    // what this filtering exists to avoid.
    unsafe { instance.get_physical_device_features2(physical, &mut features) };

    let core = features.features;
    let integer_dot_product = dot_product.shader_integer_dot_product == vk::TRUE;

    let packed_dot_accelerated =
        // SAFETY: `packed_dot_is_accelerated` asks what this function's own contract already
        // asks — that `physical` belong to `instance`.
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

unsafe fn packed_dot_is_accelerated(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> bool {
    let mut dot = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
    {
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut dot);
        // SAFETY: `physical` belongs to `instance` by this function's contract, and `dot` is a
        // local the chain borrows for no longer than this scope.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    }

    dot.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE
}

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
