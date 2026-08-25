//! Shared harness for the measurement examples.
//!
//! Not an example of its own — Cargo compiles `examples/*.rs` as separate binaries and leaves
//! directories alone, so each of those declares `mod common;`. The same arrangement as
//! `runner/tests/common/mod.rs`, and the allow below follows from it: a helper only some of the
//! examples use is dead code in the rest.
//!
//! # Why this exists
//!
//! [`runner::Timing`] and [`runner::Gpu::time_repeated`] already say that a measurement is a
//! spread rather than a number, and four examples already used them. Twelve did not, and every one
//! of those twelve prints a *comparison* — a ratio, a difference, a column against another column.
//!
//! What that cost is on the record. `runner/examples/occupancy.rs` printed a table that was read
//! off once and reported as a 12-20% finding, which five repeats then refused to reproduce.
//! `runner/examples/scanner.rs` printed 13 002 µs for a scan over 65 536 elements — against 1 653
//! µs for the same scan over sixteen times as many — and three runs afterwards put the true figure
//! near **130 µs**. A hundredfold error, in a results table, with nothing beside it to say so.
//!
//! `Gpu::time_repeated` covers the device clock. This covers the host clock, which is where the
//! costs that turn out to dominate actually live: allocation, pipeline creation, a submission and
//! its fence, a whole round trip.

#![allow(
    dead_code,
    reason = "each example binary compiles this file and uses a different subset of it"
)]

use runner::Timing;
use std::error::Error;
use std::time::Instant;

/// How many times a measurement is repeated before one of its numbers is printed.
///
/// Five, which is what `runner/examples/sweep.rs` chose and what the rest now match. It is enough
/// to catch the failure mode these examples actually have — one sample landing an order of
/// magnitude off the others — and cheap enough that no example got materially slower for it.
pub const SAMPLES: u32 = 5;

/// Time `batch` by the host clock, `repeats` times over.
///
/// [`runner::Gpu::time_repeated`] is this for a kernel on the device clock; the shape is
/// deliberately the same. `batch` is whatever the caller wants one sample to be — usually a run of
/// iterations — and the durations come back per *batch*, so the caller divides by whatever it put
/// in one.
///
/// A `repeats` of zero runs the batch once. A caller asking for no measurement wants a
/// measurement, not a panic.
///
/// # Errors
///
/// Whatever `batch` returns, on the first repeat that fails. A benchmark whose subject broke
/// partway through has no number worth printing, and averaging the repeats that happened to
/// succeed would print one anyway.
pub fn host<E>(
    repeats: u32,
    mut batch: impl FnMut() -> Result<(), E>,
) -> Result<Timing, Box<dyn Error>>
where
    E: Into<Box<dyn Error>>,
{
    let mut samples = Vec::with_capacity(repeats.max(1) as usize);
    for _ in 0..repeats.max(1) {
        let started = Instant::now();
        batch().map_err(Into::into)?;
        samples.push(started.elapsed());
    }

    Timing::of(&samples).ok_or_else(|| "at least one repeat produced no sample".into())
}

/// Summarise `repeats` runs of a measurement that reports its own duration.
///
/// [`host`] is for work the host has to time from outside. This is for the device clock, which
/// hands back a `Duration` of its own: [`runner::Gpu::time_repeated`] already does this for
/// `Gpu::time`, and this covers the calls that have no `_repeated` twin — `time_grid`, and anything
/// else measured by timestamps rather than by `Instant`.
///
/// # Errors
///
/// Whatever `once` returns, on the first repeat that fails.
pub fn samples<E>(
    repeats: u32,
    mut once: impl FnMut() -> Result<std::time::Duration, E>,
) -> Result<Timing, Box<dyn Error>>
where
    E: Into<Box<dyn Error>>,
{
    let mut taken = Vec::with_capacity(repeats.max(1) as usize);
    for _ in 0..repeats.max(1) {
        taken.push(once().map_err(Into::into)?);
    }

    Timing::of(&taken).ok_or_else(|| "at least one repeat produced no sample".into())
}

/// A ratio of two medians, marked when *either* side wandered.
///
/// Stricter than marking the two sides alone, and deliberately so: a steady numerator over a
/// wandering denominator is a wandering ratio, and the ratio is the column people quote.
pub fn ratio(numerator: Timing, denominator: Timing) -> String {
    let steady = numerator.is_steady() && denominator.is_steady();
    format!(
        "{:.2}x{}",
        numerator.median.as_secs_f64() / denominator.median.as_secs_f64().max(f64::MIN_POSITIVE),
        if steady { "" } else { "!" }
    )
}

/// Microseconds per iteration, with `!` appended when the repeats behind it did not agree.
///
/// The mark is the whole point of the helper. A number that came from repeats which disagreed by
/// more than a fifth is not evidence, and the reader has to be told beside the number rather than
/// in a paragraph under the table.
pub fn marked(timing: Timing, iterations: u32) -> String {
    format!(
        "{}{}",
        micros(timing.median / iterations.max(1)),
        mark(timing)
    )
}

/// Just the mark, for callers that format the number themselves.
pub fn mark(timing: Timing) -> &'static str {
    if timing.is_steady() { "" } else { "!" }
}

/// Microseconds, which is the scale these land on.
pub fn micros(duration: std::time::Duration) -> String {
    format!("{:.1} us", duration.as_secs_f64() * 1e6)
}

/// The legend every table carrying a `!` needs under it.
pub const LEGEND: &str =
    "`!` marks a number whose repeats disagreed by more than a fifth: not evidence.";
