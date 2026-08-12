//! Several measurements of the same thing, and how far apart they were.
//!
//! A benchmark that reports one number reads like a result whether or not it is one. The sweep
//! that refuted the cache-capacity story looked perfectly authoritative in its first run and moved
//! its cliff by 8 MB in its second — and nothing in the output said so, because the output was a
//! single figure per point.
//!
//! So a measurement here is a *spread*. When the fastest and slowest of a handful of repeats are
//! close, the number means something; when they are not, the reader can see that without having
//! to run it twice and remember.

use std::time::Duration;

/// What repeated timings of one thing came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timing {
    /// The fastest repeat — the closest to the work with nothing else in the way.
    pub best: Duration,
    /// The middle one.
    pub median: Duration,
    /// The slowest.
    pub worst: Duration,
    /// How many repeats it came from.
    pub repeats: usize,
}

impl Timing {
    /// Summarise `samples`, which must not be empty.
    ///
    /// Returns `None` for an empty slice rather than inventing a zero: a measurement of nothing
    /// is not a measurement of zero.
    ///
    /// There is no `is_empty` guard because there does not need to be one — `first()?` already
    /// returns `None`, and the guard was a second copy of that decision which no test could ever
    /// distinguish from its absence. A mutation run flipped it to `if false` and nothing changed,
    /// which is what an unkillable branch looks like from the outside.
    #[must_use]
    pub fn of(samples: &[Duration]) -> Option<Self> {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();

        Some(Self {
            best: *sorted.first()?,
            median: *sorted.get(sorted.len() / 2)?,
            worst: *sorted.last()?,
            repeats: sorted.len(),
        })
    }

    /// How much slower the worst repeat was than the best, as a ratio.
    ///
    /// One means every repeat agreed. Anything much above it means the number below is not about
    /// the kernel, and the reader should be told before they quote it.
    ///
    /// A best of **zero** reports one rather than infinity. A device whose clock has not moved has
    /// told us nothing about stability, and `inf` would read as the loudest possible warning about
    /// the least informative possible measurement.
    #[must_use]
    pub fn spread(&self) -> f64 {
        let best = self.best.as_secs_f64();
        if best <= 0.0 {
            return 1.0;
        }
        self.worst.as_secs_f64() / best
    }

    /// Whether the repeats agreed closely enough to be worth quoting.
    ///
    /// A fifth apart is generous for a GPU measurement and still catches the kind of wandering
    /// that made the sweep untrustworthy.
    #[must_use]
    pub fn is_steady(&self) -> bool {
        self.spread() <= 1.2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn micros(values: &[u64]) -> Vec<Duration> {
        values.iter().map(|&us| Duration::from_micros(us)).collect()
    }

    #[test]
    fn a_measurement_of_nothing_is_not_a_measurement_of_zero() {
        assert!(Timing::of(&[]).is_none());
    }

    #[test]
    fn a_best_of_zero_reports_a_spread_of_one_rather_than_infinity() {
        // The division's denominator, and nothing covered it. A device whose clock has not moved
        // has said nothing about stability, and `inf` would read as the loudest possible warning
        // about the least informative possible measurement — `is_steady` would then call a
        // perfectly ordinary zero-length sample unsteady.
        let stopped = Timing::of(&[Duration::ZERO]).expect("one sample");
        assert_eq!(stopped.spread(), 1.0);
        assert!(stopped.is_steady());

        // And a zero best beside a non-zero worst, which is the case that would actually divide.
        let partly = Timing::of(&[Duration::ZERO, Duration::from_micros(50)]).expect("samples");
        assert_eq!(partly.best, Duration::ZERO);
        assert_eq!(partly.spread(), 1.0);
        assert!(partly.spread().is_finite());
    }

    #[test]
    fn one_sample_is_its_own_best_median_and_worst() {
        let timing = Timing::of(&micros(&[100])).expect("one sample");

        assert_eq!(timing.best, Duration::from_micros(100));
        assert_eq!(timing.median, Duration::from_micros(100));
        assert_eq!(timing.worst, Duration::from_micros(100));
        assert_eq!(timing.repeats, 1);
    }

    #[test]
    fn the_summary_does_not_depend_on_the_order_they_arrived_in() {
        let forwards = Timing::of(&micros(&[10, 20, 30, 40, 50])).expect("samples");
        let backwards = Timing::of(&micros(&[50, 40, 30, 20, 10])).expect("samples");

        assert_eq!(forwards, backwards);
        assert_eq!(forwards.median, Duration::from_micros(30));
    }

    #[test]
    fn a_steady_measurement_is_one_whose_repeats_agree() {
        let steady = Timing::of(&micros(&[100, 105, 110])).expect("samples");

        assert!(steady.is_steady());
        assert!((steady.spread() - 1.1).abs() < 0.001);
    }

    #[test]
    fn a_wandering_measurement_announces_itself() {
        // The shape the sweep actually produced: one repeat an order of magnitude off the others.
        let wandering = Timing::of(&micros(&[300, 320, 3_100])).expect("samples");

        assert!(!wandering.is_steady());
        assert!(wandering.spread() > 10.0);
        assert_eq!(
            wandering.median,
            Duration::from_micros(320),
            "the median resists the outlier, which is why it is reported beside the best"
        );
    }

    #[test]
    fn the_boundary_of_steadiness_is_where_it_says_it_is() {
        let just_inside = Timing::of(&micros(&[100, 120])).expect("samples");
        let just_outside = Timing::of(&micros(&[100, 121])).expect("samples");

        assert!(just_inside.is_steady());
        assert!(!just_outside.is_steady());
    }
}
