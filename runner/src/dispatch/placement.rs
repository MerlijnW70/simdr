//! Where the runner's buffers actually end up, as opposed to where they were asked to go.
//!
//! [`crate::buffer::Buffer::device_local`] falls back to host-visible when no device-local type
//! accepts a storage buffer, silently and by design. A benchmark that never checked would report
//! the bus and call it VRAM, and this project did exactly that for two passes before anyone asked.
//!
//! The line above used to end "— an integrated part has no separate memory to fall back *from*",
//! which sounded obvious and is not true here. The integrated Radeon on this machine offers a
//! device-local type that is **not** host-visible, so the fallback has never once fired; the
//! 4080 offers one that **is** host-visible, which is the opposite of the same guess.
//! `runner/examples/memtypes.rs` prints both tables, and it is the reason
//! [`crate::buffer::Buffer::shared`] asks the device instead of reasoning about it.

use crate::buffer::Buffer;
use crate::{Error, Gpu};
use ash::vk;

/// Where the runner's buffers actually end up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    /// Whether a device-local request of that size was honoured.
    pub device_local: bool,
    /// The largest device-local heap the device reports, in bytes.
    ///
    /// A working set approaching this is a reason to distrust a timing: the driver may be moving
    /// pages rather than the kernel being slow.
    pub largest_device_heap: u64,
    /// How many buffers of that size were resident at once when the question was asked.
    ///
    /// One means the answer is about an allocation in isolation, which is a weaker claim than it
    /// looks — see [`Gpu::probe_resident`].
    pub resident: u32,
}

/// One memory type the device offers, as a caller can read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryType {
    /// Its index, which is what `vkAllocateMemory` takes.
    pub index: u32,
    /// Fast for the device to read.
    pub device_local: bool,
    /// The host can map it.
    pub host_visible: bool,
    /// Writes are visible without an explicit flush.
    pub host_coherent: bool,
    /// **Cached on the host side.**
    ///
    /// The one that decides whether reading a mapping back is fast. Host-visible memory without
    /// this is typically write-combined: sequential writes coalesce and go at full speed, and
    /// every *read* is an uncached fetch with no prefetch and no line reuse. A download path that
    /// maps such a buffer and memcpys out of it runs at a fraction of the bus.
    pub host_cached: bool,
}

impl Gpu {
    /// Every memory type the device offers.
    ///
    /// Diagnostic rather than load-bearing: a buffer picks a type by asking for the flags it
    /// wants, and this is how a caller finds out what was on the menu. It exists because a
    /// throughput measurement that does not know which memory it measured is not a measurement —
    /// and one such measurement was off by nine times.
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

    /// Where a buffer of `bytes` would land, and how much device-local memory exists.
    ///
    /// Allocates and immediately frees, so it answers the question rather than assuming it.
    ///
    /// **It answers for one buffer.** A run holds three of that size, and on a system where the
    /// driver may migrate an allocation under pressure, a request honoured in isolation says
    /// nothing about what happens when all three are resident. That gap is real and not yet
    /// closed.
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`] if the allocation fails outright — which is itself an answer, and a
    /// different one from landing somewhere unexpected.
    pub fn probe_memory(&self, bytes: u64) -> Result<Placement, Error> {
        self.probe_resident(bytes, 1)
    }

    /// The same question, with `count` buffers of that size **resident at the same time**.
    ///
    /// This is what a run actually holds. `Gpu::run` allocates a staging buffer and two
    /// device-local ones, so a working set of *n* bytes puts three *n*-byte allocations on the
    /// device at once, and a request honoured in isolation says nothing about the third one.
    ///
    /// The gap was written into [`Gpu::probe_memory`]'s own documentation as open and unclosed
    /// for some time. It is the last cheap hypothesis for the large-working-set cliff this project
    /// has recorded three sightings of and no explanation for: if the driver starts placing the
    /// second or third buffer in host memory, a kernel reading it crosses the bus every access and
    /// the collapse would look exactly as it does.
    ///
    /// **A negative result is worth as much as a positive one here.** If all three land
    /// device-local at a size where the timing has already fallen apart, that hypothesis is dead
    /// too and the record should say so.
    ///
    /// # Errors
    ///
    /// [`Error::Vulkan`] if an allocation fails outright — which is itself an answer, and a
    /// different one from landing somewhere unexpected.
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
