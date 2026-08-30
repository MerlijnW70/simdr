#![allow(
    dead_code,
    unused_imports,
    reason = "each test binary compiles this file and uses a different subset of it — and a               re-export nobody in *this* binary names is an unused import rather than dead code,               which is a second lint saying the same thing about the same arrangement"
)]

#[path = "../../../tests/common/spirv_val.rs"]
mod spirv_val;
pub use spirv_val::{VULKAN_1_1, expect_valid, validate, validator};

use runner::Gpu;

pub fn device(label: &str) -> Option<Gpu> {
    match Gpu::open() {
        Ok(Some(gpu)) => Some(gpu),
        Ok(None) => {
            eprintln!("SKIPPED {label}: no Vulkan device");
            None
        }
        Err(error @ runner::Error::NoSuchDevice { .. }) => {
            panic!("SIMDR_DEVICE names a device that is not here — {error}")
        }
        Err(error) => {
            eprintln!("SKIPPED {label}: could not open a device — {error}");
            None
        }
    }
}

pub fn runnable(gpu: &Gpu, label: &str, modules: &[&[u32]]) -> bool {
    assert!(
        !modules.is_empty(),
        "runnable({label:?}) was given no modules to ask about, which gates on nothing"
    );

    for spirv in modules {
        let missing = gpu.limits().unsupported_in(spirv);
        if !missing.is_empty() {
            eprintln!("SKIPPED {label}: this device does not offer {missing:?}");
            return false;
        }
    }
    true
}

pub fn elements(width: u32, lanes: u32) -> usize {
    let strips = (lanes / width.max(1)).max(1) as usize;
    runner::kernels::WORKGROUP_SIZE as usize * strips
}

pub fn ramp(count: usize) -> Vec<f32> {
    (0..count).map(|index| index as f32).collect()
}

pub fn grouped_sums(count: usize, group: usize) -> Vec<f32> {
    (0..count)
        .map(|lane| {
            let first = lane / group * group;
            (first..(first + group).min(count))
                .map(|other| other as f32)
                .sum()
        })
        .collect()
}
