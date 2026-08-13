//! Buffers, in the three kinds a measurement needs.
//!
//! A [`Buffer::staging`] one is host-visible: the CPU writes it and reads it directly. A
//! [`Buffer::device_local`] one is the fastest memory for a kernel, and on a discrete GPU that is
//! the difference between VRAM and system memory across PCIe. A [`Buffer::shared`] one asks for
//! both at once — device-local *and* host-writable — and takes plain device-local memory where the
//! device has no such type, so it never narrows what runs.
//!
//! The third kind exists because the first two make the host copy everything twice: into staging,
//! then across into the kernel's buffer. Where one type is both, the first write lands in the
//! right place. That is worth about a third of a held reduction over 4 MB, and it is worth
//! *nothing* — worse, 62% on a 4080 — to a call that allocates its buffers each time, because
//! allocating out of that memory costs more than the copy it saves. [`Buffer::shared`] says which
//! callers should ask.
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
    coherent: bool,
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

    /// Whether the host can write this buffer directly and the device will see it.
    ///
    /// **Both halves are required.** Mappable alone is not enough: nothing here flushes a mapping,
    /// so a host-visible type without `HOST_COHERENT` would take the write and hide it.
    ///
    /// This is how a caller skips the staging hop where there is nothing to hop over, and it is a
    /// question to ask rather than predict. The guess this crate started with — that an integrated
    /// part shares its memory with the host and a discrete card cannot — is wrong in **both**
    /// directions here: the integrated Radeon offers a device-local type that is not host-visible,
    /// and the RTX 4080 offers one that is. `runner/examples/memtypes.rs` prints the tables.
    ///
    /// Only a buffer that asked for such memory can answer yes; see [`Buffer::shared`].
    pub(crate) const fn host_writable(&self) -> bool {
        self.mappable && self.coherent
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

    /// A buffer the kernel reads and writes, in the fastest memory the device offers.
    ///
    /// Falls back to host-visible when no device-local type accepts a storage buffer. That
    /// fallback has not fired on any device here: `runner/examples/memtypes.rs` shows a
    /// device-local type without `HOST_VISIBLE` on both the discrete card **and** the integrated
    /// Radeon, which is worth saying because the obvious guess — that an integrated part must
    /// share its memory with the host — is wrong on this machine. See [`Buffer::shared`] for the
    /// buffer that does ask for both.
    ///
    /// # Safety
    ///
    /// As [`Buffer::staging`].
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

    /// The same buffer, in memory the host can also write where the device offers such a type.
    ///
    /// **This is how an upload copy stops existing rather than getting faster.** A kernel's input
    /// normally arrives in two hops: the host writes staging memory, then the device copies
    /// staging into the buffer the kernel reads. Where one type is both device-local and
    /// host-coherent — a BAR window on a discrete card, plain RAM on a part that has only one kind
    /// of memory — the host can write the kernel's buffer itself and the second hop has nothing
    /// left to do.
    ///
    /// Host-visibility is a *preference*, not a requirement, so this never narrows the devices
    /// that run: with no such type, or with one whose heap will not fit `bytes`, this is
    /// [`Buffer::device_local`] and the caller keeps staging. [`Buffer::host_writable`] is how the
    /// caller tells which it got, and it must be asked rather than assumed — the answer is a
    /// property of the device, and guessing it from "discrete" or "integrated" gets it wrong.
    ///
    /// # Who should ask for it
    ///
    /// **Anything that allocates once and uploads many times**, which is [`crate::Reducer`] and
    /// [`crate::Session`]. There it is worth about a third of a 4 MB reduction on both devices
    /// here.
    ///
    /// **Not a call that allocates per use.** `Gpu::run_chain` asked for this for one afternoon
    /// and `Gpu::sum` over 2²⁰ went from ~2153 µs to ~3492 µs on an RTX 4080 — 62% slower, for a
    /// change that removes a copy. Allocating out of this memory costs more than the copy it
    /// saves, and the 8 192-element case proves it is the allocation: 32 KB to upload, nothing to
    /// gain from the transfer, still 22% slower.
    ///
    /// # Safety
    ///
    /// As [`Buffer::staging`].
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
            // No device-local type at all: same fallback as `device_local`, for the same reason.
            // SAFETY: `device_local` asks the same of its caller as this does, and the failed
            // attempt above left nothing allocated to leak.
            Err(Error::NoHostVisibleMemory) => unsafe { Self::device_local(gpu, bytes) },
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
        // SAFETY: as `staging`. An empty `preferred` asks for no tiebreak, which is a value
        // rather than a weaker precondition.
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
        // SAFETY: the device outlives this call — the caller holds the `Gpu` that owns it — and
        // `info` describes a buffer with no external handles or queue families to get wrong.
        let handle = unsafe { device.create_buffer(&info, None) }?;

        // SAFETY: `handle` was created immediately above by this device and has not been
        // destroyed, which is all this query requires of it.
        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let permitted = requirements.memory_type_bits;

        // Two candidates, best first, and the fallback is an *allocation* fallback rather than a
        // selection one. A type that suits the buffer is not the same as a type that fits it: the
        // memory a host can write and a device can read at full speed is usually a BAR window, and
        // a BAR window is often 256 MB against several gigabytes of plain device memory. Choosing
        // it and then failing would turn a preference into a size limit.
        let candidates = [
            memory_type(gpu, permitted, wanted | preferred),
            memory_type(gpu, permitted, wanted),
        ];

        let mut allocated = None;
        // Kept from the last attempt, so a device that has no matching type at all and one whose
        // heap is full report differently. Where both candidates are the same index — no preferred
        // type exists — the second attempt repeats the first, which costs one failed call on a
        // path that is already failing and less code than noticing.
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
            // What the memory type *offers*, not what was asked for. A device-local request that
            // fell back to host-visible still yields mappable memory, and saying otherwise would
            // turn a fallback into a panic later.
            mappable: offered.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
            // Coherent, not merely visible. Nothing in this crate flushes a mapping, so a
            // host-visible type without `HOST_COHERENT` is one the host may write and the device
            // may not see — which is a wrong answer rather than a slow one.
            coherent: offered.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
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
        // SAFETY: `words` is a slice, so its pointer is valid for `words.len()` reads, and the
        // caller's obligation about dispatches is passed straight through.
        unsafe { self.write_words(gpu, words.as_ptr(), words.len()) }
    }

    /// The same, from a slice of `f32`.
    ///
    /// **Because the bits are already the right bits.** `Reducer::sum` takes `&[f32]` and the
    /// buffer holds words, so it built a `Vec<u32>` of the whole input on every call — a four
    /// megabyte allocation and copy to reinterpret bits that `f32::to_bits` is *defined* as
    /// reinterpreting. `runner/examples/reducer.rs` costed that at **52%** of a reduction over 2²⁰
    /// elements: the largest single item in the call, and it computed nothing.
    ///
    /// # Errors
    ///
    /// As [`Buffer::write`].
    ///
    /// # Safety
    ///
    /// As [`Buffer::write`]: no dispatch may be reading the buffer at the time.
    pub(crate) unsafe fn write_floats(&self, gpu: &Gpu, values: &[f32]) -> Result<(), Error> {
        // SAFETY: `f32` and `u32` have the same size and alignment, so a `*const f32` is a valid
        // `*const u32` for the same count — and the bytes are *copied* rather than read as a
        // number, which is exactly what `f32::to_bits` does one element at a time. The caller's
        // obligation about dispatches is passed straight through.
        unsafe { self.write_words(gpu, values.as_ptr().cast::<u32>(), values.len()) }
    }

    /// Map, copy `count` words from `source`, unmap.
    ///
    /// # Safety
    ///
    /// `source` must be valid for `count` reads of `u32`, and no dispatch may be reading the
    /// buffer at the time.
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

    /// Release the buffer and its memory.
    ///
    /// # Safety
    ///
    /// Nothing may still be using it.
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
