use crate::buffer::Buffer;
use crate::{Error, Gpu};
use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub device_local: bool,
    pub largest_device_heap: u64,
    pub resident: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryType {
    pub index: u32,
    pub device_local: bool,
    pub host_visible: bool,
    pub host_coherent: bool,
    pub host_cached: bool,
}

impl Gpu {
    #[must_use]
    pub fn memory_types(&self) -> Vec<MemoryType> {
        let properties = self.memory_properties();
        let has = |flags: vk::MemoryPropertyFlags, wanted: vk::MemoryPropertyFlags| {
            flags.contains(wanted)
        };

        properties
            .memory_types
            .iter()
            .take(properties.memory_type_count as usize)
            .enumerate()
            .map(|(index, kind)| MemoryType {
                index: index as u32,
                device_local: has(kind.property_flags, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                host_visible: has(kind.property_flags, vk::MemoryPropertyFlags::HOST_VISIBLE),
                host_coherent: has(kind.property_flags, vk::MemoryPropertyFlags::HOST_COHERENT),
                host_cached: has(kind.property_flags, vk::MemoryPropertyFlags::HOST_CACHED),
            })
            .collect()
    }

    pub fn probe_memory(&self, bytes: u64) -> Result<Placement, Error> {
        self.probe_resident(bytes, 1)
    }

    pub fn probe_resident(&self, bytes: u64, count: u32) -> Result<Placement, Error> {
        let wanted = count.max(1);

        // SAFETY: every buffer is created here and destroyed before returning, and nothing else
        // ever sees them. They are held together on purpose: allocating and freeing one at a time
        // is the measurement this exists to replace.
        let device_local = unsafe {
            let mut held = Vec::with_capacity(wanted as usize);
            let mut all_local = true;

            for _ in 0..wanted {
                match Buffer::device_local(self, bytes.max(1)) {
                    Ok(buffer) => {
                        all_local &= buffer.is_device_local();
                        held.push(buffer);
                    }
                    Err(error) => {
                        for buffer in held {
                            buffer.destroy(self);
                        }
                        return Err(error);
                    }
                }
            }

            for buffer in held {
                buffer.destroy(self);
            }
            all_local
        };

        let properties = self.memory_properties();
        let largest_device_heap = properties
            .memory_heaps
            .iter()
            .take(properties.memory_heap_count as usize)
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .max()
            .unwrap_or(0);

        Ok(Placement {
            device_local,
            largest_device_heap,
            resident: wanted,
        })
    }
}
