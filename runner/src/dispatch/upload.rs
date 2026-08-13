//! Getting a caller's input to the buffer a kernel reads, by the shortest route the device allows.
//!
//! There are two routes and the device picks, not the caller. Normally the host writes staging
//! memory and the device copies staging into the kernel's buffer — two crossings of the same
//! megabytes. Where the kernel's buffer is itself host-writable ([`Buffer::shared`]) the first
//! write lands in the right place and the copy has nothing to do.
//!
//! # Why this is one function and not three call sites
//!
//! Three paths upload: the held reducer, a [`crate::Session`] write, and the one-shot chain. Each
//! one wrote its own `if` over the same two lines for about ten minutes, which is long enough to
//! see how they would end up disagreeing — a fix to one, a device quirk handled in another. The
//! branch is the interesting part, so it lives once and the call sites say only what they are
//! moving.
//!
//! The return value is what [`crate::Gpu::replay`] already takes: `Some` copy to record, or `None`
//! because there is nothing left to copy.

use super::chain::Staged;
use crate::buffer::Buffer;
use crate::{Error, Gpu};

/// Put `words` where `source` will be read from, staging only if the device requires it.
///
/// # Safety
///
/// Both buffers must be live, large enough for `words`, and the device idle with respect to
/// them — the same requirement as writing either one directly.
pub(crate) unsafe fn deliver<'a>(
    gpu: &Gpu,
    words: &[u32],
    staging: &'a Buffer,
    source: &'a Buffer,
) -> Result<Option<Staged<'a>>, Error> {
    unsafe {
        route(staging, source, words.len(), |target| {
            target.write(gpu, words)
        })
    }
}

/// The same for a slice of `f32`, whose bits are already the bits the device wants.
///
/// # Safety
///
/// As [`deliver`].
pub(crate) unsafe fn deliver_floats<'a>(
    gpu: &Gpu,
    values: &[f32],
    staging: &'a Buffer,
    source: &'a Buffer,
) -> Result<Option<Staged<'a>>, Error> {
    unsafe {
        route(staging, source, values.len(), |target| {
            target.write_floats(gpu, values)
        })
    }
}

/// Choose the buffer, hand it to `write`, and report what still has to be copied.
///
/// # Safety
///
/// As [`deliver`].
unsafe fn route<'a>(
    staging: &'a Buffer,
    source: &'a Buffer,
    count: usize,
    write: impl FnOnce(&Buffer) -> Result<(), Error>,
) -> Result<Option<Staged<'a>>, Error> {
    // Asked of the buffer, never inferred from the kind of device. A discrete card with a
    // resizable BAR answers yes and an integrated part with a device-local-only memory type
    // answers no, which is the opposite of the guess — `runner/examples/memtypes.rs` prints both.
    //
    // The answer, and the size that follows from it, come from `super::step::upload_bytes`. That
    // is not indirection for its own sake: this file allocates and writes memory, so it is excused
    // from the mutation gate, and a decision left here would be one the gate never tries to break.
    // What is left below is which buffer to hand to `write`, which no arithmetic can get subtly
    // wrong — it is either the right buffer or a failing test on every device.
    let Some(bytes) = super::step::upload_bytes(source.host_writable(), count) else {
        write(source)?;
        return Ok(None);
    };

    write(staging)?;
    Ok(Some(Staged {
        from: staging,
        to: source,
        bytes,
    }))
}
