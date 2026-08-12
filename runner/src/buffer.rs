//! Buffers, in the two kinds a measurement needs.
//!
//! A [`Buffer::staging`] one is host-visible: the CPU writes it and reads it directly. A
//! [`Buffer::device_local`] one is not visible to the host at all, and on a discrete GPU that is
//! the difference between VRAM and system memory across PCIe.
//!
//! **That distinction is why this file exists in its current shape.** Everything was host-visible
//! until the benchmark reported the same number for every kernel, and the reason turned out to be
//! that all of them were reading across the bus at 24 GB/s rather than out of the 4080's ~700 —
//! see `notes/FINDINGS.md`. A kernel is now handed device-local memory and the host's copies are
//! separate submissions, so what gets timed is the kernel.

use crate::{Error, Gpu};
use ash::vk;

/// A buffer, its memory, and how big it is.
pub(crate) struct Buffer {
    pub(crate) handle: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: u64,
    mappable: bool,
    device_local: bool,
}

impl Buffer {
    /// Whether this actually landed in device-local memory.
    ///
    /// Not the same question as whether it was *asked* for: [`Buffer::device_local`] falls back to
    /// host-visible when no device-local type accepts a storage buffer, and a benchmark that
    /// assumed the request was honoured would report the bus and call it VRAM. This crate has
    /// made that mistake once already.
    pub(crate) const fn is_device_local(&self) -> bool {
        self.device_local
    }
}

impl Buffer {
    /// A buffer the host can write and read, for moving data in and out.
    ///
    /// # Safety
    ///
    /// The caller destroys it with [`Buffer::destroy`] before the device goes away.
    /// **Cached memory is preferred, and finding out why cost a measurement.** Host-visible memory
    /// without `HOST_CACHED` is typically write-combined: sequential writes coalesce and go at
    /// full speed, and every *read* is an uncached fetch with no prefetching and no line reuse.
    /// [`Buffer::read`] memcpys out of this mapping on the way home from every dispatch.
    ///
    /// Asking only for `HOST_VISIBLE | HOST_COHERENT` and taking the first match got an uncached
    /// type on an RTX 4080 while a cached one sat one index later, and host transfers ran at
    /// ~370 MB/s on a bus good for tens of gigabytes. `preferred` is what makes the better type
    /// win where it exists, without refusing a device that has no such type at all.
    pub(crate) unsafe fn staging(gpu: &Gpu, bytes: u64) -> Result<Self, Error> {
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

    /// A buffer the kernel reads and writes, in the fastest memory the device offers.
    ///
    /// Falls back to host-visible when no device-local type accepts a storage buffer, which is
    /// the normal state of an integrated part — there the two are the same memory anyway.
    ///
    /// # Safety
    ///
    /// As [`Buffer::staging`].
    pub(crate) unsafe fn device_local(gpu: &Gpu, bytes: u64) -> Result<Self, Error> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        match unsafe { Self::new(gpu, bytes, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL) } {
            Ok(buffer) => Ok(buffer),
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

    /// Allocate a buffer of `bytes` with `usage`, in memory offering `wanted`.
    ///
    /// # Safety
    ///
    /// As [`Buffer::staging`].
    unsafe fn new(
        gpu: &Gpu,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        wanted: vk::MemoryPropertyFlags,
    ) -> Result<Self, Error> {
        unsafe { Self::preferring(gpu, bytes, usage, wanted, vk::MemoryPropertyFlags::empty()) }
    }

    /// The same, taking a memory type that also offers `preferred` where one exists.
    ///
    /// Two passes rather than one. `wanted` is a requirement and `preferred` is a tiebreak, and
    /// keeping them apart is what lets a caller say "cached if you have it" without narrowing the
    /// devices it runs on.
    ///
    /// # Safety
    ///
    /// As [`Buffer::staging`].
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
        let handle = unsafe { device.create_buffer(&info, None) }?;

        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let permitted = requirements.memory_type_bits;
        let chosen = memory_type(gpu, permitted, wanted | preferred)
            .or_else(|| memory_type(gpu, permitted, wanted));

        let Some((memory_type, offered)) = chosen else {
            unsafe { device.destroy_buffer(handle, None) };
            return Err(Error::NoHostVisibleMemory);
        };

        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(result) => {
                unsafe { device.destroy_buffer(handle, None) };
                return Err(Error::Vulkan(result));
            }
        };

        if let Err(result) = unsafe { device.bind_buffer_memory(handle, memory, 0) } {
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
            // What the memory type *offers*, not what was asked for. A device-local request that
            // fell back to host-visible still yields mappable memory, and saying otherwise would
            // turn a fallback into a panic later.
            mappable: offered.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
            device_local: offered.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
        })
    }

    /// How many whole words fit.
    pub(crate) const fn capacity(&self) -> usize {
        (self.bytes / size_of::<u32>() as u64) as usize
    }

    /// Copy `words` into the buffer, which must be mappable and at least as large.
    ///
    /// Words rather than a typed slice: `f32` and `u32` are both four bytes, and a buffer does
    /// not care which one a kernel will read out of it. The typed wrappers live on [`Gpu`].
    ///
    /// **The length is checked here rather than assumed.** It used to be assumed, and the comment
    /// said so: "the caller's slice is no longer than that because this crate always allocates
    /// from the same element count it writes". That was true while `Gpu::run` was the only caller
    /// — it allocates `input.len()` and writes `input`. `Session` broke it: its staging buffer is
    /// sized to the *largest* binding and `Session::write` takes a slice from outside. A caller
    /// passing more words than that would have memcpyd past the end of a mapping, from safe code,
    /// in a crate whose whole claim is that it cannot.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if `words` does not fit, [`Error::NotMappable`] if the buffer lives on
    /// the device.
    ///
    /// # Safety
    ///
    /// No dispatch may be reading the buffer at the time.
    pub(crate) unsafe fn write(&self, gpu: &Gpu, words: &[u32]) -> Result<(), Error> {
        if !self.mappable {
            return Err(Error::NotMappable);
        }
        if words.len() > self.capacity() {
            return Err(Error::TooLarge {
                words: words.len(),
                capacity: self.capacity(),
            });
        }

        let device = gpu.device();
        let mapped =
            unsafe { device.map_memory(self.memory, 0, self.bytes, vk::MemoryMapFlags::empty()) }?;

        // SAFETY: the mapping covers `self.bytes` and the check above refused anything longer.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), mapped.cast::<u32>(), words.len());
            device.unmap_memory(self.memory);
        }
        Ok(())
    }

    /// Read `count` words back out.
    ///
    /// Bounded for the same reason as [`Buffer::write`], and the read side is the worse of the
    /// two: reading past a mapping hands the caller whatever was next in the address space and
    /// looks like a plausible answer.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] if `count` is past the end, [`Error::NotMappable`] if the buffer lives
    /// on the device.
    ///
    /// # Safety
    ///
    /// The dispatch and any copy that wrote the buffer must have completed.
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

    /// Release the buffer and its memory.
    ///
    /// # Safety
    ///
    /// Nothing may still be using it.
    pub(crate) unsafe fn destroy(self, gpu: &Gpu) {
        let device = gpu.device();
        unsafe {
            device.destroy_buffer(self.handle, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// A memory type allowed for `permitted` that offers everything in `wanted`, and what it offers.
///
/// The second half matters: a type that satisfies the request may carry other properties too, and
/// the caller records what it actually got rather than what it asked for.
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
