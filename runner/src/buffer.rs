use crate::{Error, Gpu};
use ash::vk;

pub(crate) struct Buffer {
    pub(crate) handle: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: u64,
    mappable: bool,
    coherent: bool,
    device_local: bool,
}

impl Buffer {
    pub(crate) const fn is_device_local(&self) -> bool {
        self.device_local
    }

    pub(crate) const fn host_writable(&self) -> bool {
        self.mappable && self.coherent
    }
}

impl Buffer {
    pub(crate) unsafe fn staging(gpu: &Gpu, bytes: u64) -> Result<Self, Error> {
        // SAFETY: `preferring` asks exactly what this function's own contract asks — a live device
        // and a caller who will destroy what comes back. Forwarding discharges nothing, so there
        // is nothing new to argue here beyond that the two contracts are the same one.
        unsafe {
            Self::preferring(
                gpu,
                bytes,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::HOST_CACHED,
            )
        }
    }

    pub(crate) unsafe fn device_local(gpu: &Gpu, bytes: u64) -> Result<Self, Error> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        // SAFETY: as `staging` — `new` asks what this function's contract already asks, and the
        // fallback below is the same call with different flags, so it inherits the same argument.
        match unsafe { Self::new(gpu, bytes, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL) } {
            Ok(buffer) => Ok(buffer),
            // SAFETY: as immediately above. Nothing was allocated by the failed attempt — `new`
            // destroys the buffer it created before returning an error — so this starts clean.
            Err(Error::NoHostVisibleMemory) => unsafe {
                Self::new(
                    gpu,
                    bytes,
                    usage,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            },
            Err(other) => Err(other),
        }
    }

    pub(crate) unsafe fn shared(gpu: &Gpu, bytes: u64) -> Result<Self, Error> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        // SAFETY: as `staging` — the same contract forwarded to the same function.
        match unsafe {
            Self::preferring(
                gpu,
                bytes,
                usage,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        } {
            Ok(buffer) => Ok(buffer),
            // SAFETY: `device_local` asks the same of its caller as this does, and the failed
            // attempt above left nothing allocated to leak.
            Err(Error::NoHostVisibleMemory) => unsafe { Self::device_local(gpu, bytes) },
            Err(other) => Err(other),
        }
    }

    unsafe fn new(
        gpu: &Gpu,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        wanted: vk::MemoryPropertyFlags,
    ) -> Result<Self, Error> {
        // SAFETY: as `staging`. An empty `preferred` asks for no tiebreak, which is a value
        // rather than a weaker precondition.
        unsafe { Self::preferring(gpu, bytes, usage, wanted, vk::MemoryPropertyFlags::empty()) }
    }

    unsafe fn preferring(
        gpu: &Gpu,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        wanted: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Result<Self, Error> {
        let device = gpu.device();

        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the device outlives this call — the caller holds the `Gpu` that owns it — and
        // `info` describes a buffer with no external handles or queue families to get wrong.
        let handle = unsafe { device.create_buffer(&info, None) }?;

        // SAFETY: `handle` was created immediately above by this device and has not been
        // destroyed, which is all this query requires of it.
        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let permitted = requirements.memory_type_bits;

        let candidates = [
            memory_type(gpu, permitted, wanted | preferred),
            memory_type(gpu, permitted, wanted),
        ];

        let mut allocated = None;
        let mut refusal = Error::NoHostVisibleMemory;
        for (memory_type, offered) in candidates.into_iter().flatten() {
            if allocated.is_some() {
                break;
            }
            let allocation = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            // SAFETY: the size comes from the device's own requirements for this buffer and the
            // type index from `memory_type`, which only ever returns indices the device reported
            // and that `requirements.memory_type_bits` permits for it.
            match unsafe { device.allocate_memory(&allocation, None) } {
                Ok(memory) => allocated = Some((memory, offered)),
                Err(result) => refusal = Error::Vulkan(result),
            }
        }

        let Some((memory, offered)) = allocated else {
            // SAFETY: the buffer was created above, nothing was ever bound to it — every
            // allocation attempt failed — and nothing else holds the handle.
            unsafe { device.destroy_buffer(handle, None) };
            return Err(refusal);
        };

        // SAFETY: both handles are this device's and were made above; the memory is freshly
        // allocated, so nothing is bound to it yet, and its size is the buffer's own requirement.
        if let Err(result) = unsafe { device.bind_buffer_memory(handle, memory, 0) } {
            // SAFETY: the bind failed, so nothing refers to either object and neither has been
            // handed to a caller. Freeing memory that nothing is bound to is the ordinary case.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(handle, None);
            }
            return Err(Error::Vulkan(result));
        }

        Ok(Self {
            handle,
            memory,
            bytes,
            mappable: offered.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
            coherent: offered.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
            device_local: offered.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
        })
    }

    pub(crate) const fn capacity(&self) -> usize {
        (self.bytes / size_of::<u32>() as u64) as usize
    }

    pub(crate) unsafe fn write(&self, gpu: &Gpu, words: &[u32]) -> Result<(), Error> {
        // SAFETY: `words` is a slice, so its pointer is valid for `words.len()` reads, and the
        // caller's obligation about dispatches is passed straight through.
        unsafe { self.write_words(gpu, words.as_ptr(), words.len()) }
    }

    pub(crate) unsafe fn write_floats(&self, gpu: &Gpu, values: &[f32]) -> Result<(), Error> {
        // SAFETY: `f32` and `u32` have the same size and alignment, so a `*const f32` is a valid
        // `*const u32` for the same count — and the bytes are *copied* rather than read as a
        // number, which is exactly what `f32::to_bits` does one element at a time. The caller's
        // obligation about dispatches is passed straight through.
        unsafe { self.write_words(gpu, values.as_ptr().cast::<u32>(), values.len()) }
    }

    unsafe fn write_words(&self, gpu: &Gpu, source: *const u32, count: usize) -> Result<(), Error> {
        if !self.mappable {
            return Err(Error::NotMappable);
        }
        if count > self.capacity() {
            return Err(Error::TooLarge {
                words: count,
                capacity: self.capacity(),
            });
        }

        let device = gpu.device();
        // SAFETY: the memory is this buffer's own and is host-visible — the check above refused
        // it otherwise — and it is not already mapped: every mapping in this file is unmapped
        // before the function that made it returns.
        let mapped =
            unsafe { device.map_memory(self.memory, 0, self.bytes, vk::MemoryMapFlags::empty()) }?;

        // SAFETY: the mapping covers `self.bytes` and the check above refused anything longer.
        unsafe {
            std::ptr::copy_nonoverlapping(source, mapped.cast::<u32>(), count);
            device.unmap_memory(self.memory);
        }
        Ok(())
    }

    pub(crate) unsafe fn read(&self, gpu: &Gpu, count: usize) -> Result<Vec<u32>, Error> {
        if !self.mappable {
            return Err(Error::NotMappable);
        }
        if count > self.capacity() {
            return Err(Error::TooLarge {
                words: count,
                capacity: self.capacity(),
            });
        }

        let device = gpu.device();
        // SAFETY: as the write side — this buffer's own host-visible memory, not already mapped.
        let mapped =
            unsafe { device.map_memory(self.memory, 0, self.bytes, vk::MemoryMapFlags::empty()) }?;

        let mut words = vec![0_u32; count];
        // SAFETY: the mapping covers `self.bytes` and the check above refused anything longer.
        unsafe {
            std::ptr::copy_nonoverlapping(mapped.cast::<u32>(), words.as_mut_ptr(), count);
            device.unmap_memory(self.memory);
        }
        Ok(words)
    }

    pub(crate) unsafe fn destroy(self, gpu: &Gpu) {
        let device = gpu.device();
        // SAFETY: `self` is taken by value, so no other `Buffer` names these handles, and this
        // function's own contract says nothing is still using them. The buffer goes before the
        // memory it is bound to, which is the order Vulkan requires.
        unsafe {
            device.destroy_buffer(self.handle, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn memory_type(
    gpu: &Gpu,
    permitted: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Option<(u32, vk::MemoryPropertyFlags)> {
    let properties = gpu.memory_properties();

    properties
        .memory_types
        .iter()
        .take(properties.memory_type_count as usize)
        .enumerate()
        .find(|&(index, memory_type)| {
            let allowed = permitted & (1 << index) != 0;
            allowed && memory_type.property_flags.contains(wanted)
        })
        .and_then(|(index, memory_type)| {
            u32::try_from(index)
                .ok()
                .map(|index| (index, memory_type.property_flags))
        })
}
