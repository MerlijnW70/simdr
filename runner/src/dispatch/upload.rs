use super::chain::Staged;
use crate::buffer::Buffer;
use crate::{Error, Gpu};

pub(crate) unsafe fn deliver<'a>(
    gpu: &Gpu,
    words: &[u32],
    staging: &'a Buffer,
    source: &'a Buffer,
) -> Result<Option<Staged<'a>>, Error> {
    // SAFETY: `route` and `Buffer::write` ask exactly what this function's own contract asks —
    // live buffers, big enough, device idle with respect to them. Nothing new is discharged here.
    unsafe {
        route(staging, source, words.len(), |target| {
            target.write(gpu, words)
        })
    }
}

pub(crate) unsafe fn deliver_floats<'a>(
    gpu: &Gpu,
    values: &[f32],
    staging: &'a Buffer,
    source: &'a Buffer,
) -> Result<Option<Staged<'a>>, Error> {
    // SAFETY: as `deliver`. `write_floats` differs from `write` only in how it reads the
    // caller's slice, and both write the same number of bytes for the same number of elements.
    unsafe {
        route(staging, source, values.len(), |target| {
            target.write_floats(gpu, values)
        })
    }
}

unsafe fn route<'a>(
    staging: &'a Buffer,
    source: &'a Buffer,
    count: usize,
    write: impl FnOnce(&Buffer) -> Result<(), Error>,
) -> Result<Option<Staged<'a>>, Error> {
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
