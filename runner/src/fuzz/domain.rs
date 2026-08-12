//! Which element type a generated program computes in, and what each operation means there.
//!
//! # Why floats can be compared exactly
//!
//! Floating-point addition is not associative, and a subgroup reduction combines lanes in an
//! order the specification does not fix — so comparing arbitrary `f32` sums exactly would be
//! comparing against one arbitrary order.
//!
//! The fuzzer sidesteps that rather than papering over it with a tolerance: **every float value it
//! generates is a small integer**. Integers below 2²⁴ are exactly representable in `f32`, and so
//! are their sums and products as long as the running total stays under that bound. Within that
//! range float arithmetic is exact and therefore associative, and the answer does not depend on
//! the order at all.
//!
//! **What that does not cover, and is worth saying plainly:** rounding, denormals, infinities and
//! NaN. Those need fixed tests with a reasoned expectation, not a fuzzer — a random program has
//! no way to say what the right answer *is* near a rounding boundary. What this does cover is the
//! half that could be silently wrong: instruction selection, the mapping, and the fact that
//! `OpFAdd` and `OpIAdd` are different instructions reached by the same source.

/// The element type a program computes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// 32-bit unsigned integers, wrapping.
    Unsigned,
    /// 32-bit signed integers, wrapping.
    ///
    /// Not a duplicate of [`Domain::Unsigned`]: the comparison and the extremes reach different
    /// instructions — `OpSGreaterThan` and `OpGroupNonUniformSMax` against their `U` counterparts
    /// — and a value with the top bit set orders the opposite way between them. That is precisely
    /// the mistake a shared code path makes and a fuzzer catches.
    Signed,
    /// 32-bit floats holding small integers, where arithmetic is exact.
    Float,
}

/// Every domain, for a caller that wants to sweep them.
pub const ALL_DOMAINS: [Domain; 3] = [Domain::Unsigned, Domain::Signed, Domain::Float];

impl Domain {
    /// The largest value the generator may produce in this domain.
    ///
    /// Floats stop well below 2²⁴ so that a sum over a few hundred of them stays exact; integers
    /// are allowed to be larger because wrapping is defined and the reference wraps too.
    #[must_use]
    pub const fn ceiling(self) -> u32 {
        match self {
            Self::Unsigned | Self::Signed => 4_096,
            Self::Float => 256,
        }
    }

    /// Whether values in this domain may be negative.
    ///
    /// The generator uses it to reach below zero, which is the half of the signed domain that
    /// differs from the unsigned one at all.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::Signed | Self::Float)
    }

    /// Encode a small integer as this domain's bit pattern.
    #[must_use]
    pub fn encode(self, value: u32) -> u32 {
        match self {
            // Both integer domains hold the value as-is; the generator keeps them below `ceiling`,
            // so the top bit is clear and the two encodings coincide.
            Self::Unsigned | Self::Signed => value,
            Self::Float => (value as f32).to_bits(),
        }
    }

    /// Encode a possibly-negative integer, for the domains that have one.
    ///
    /// Unsigned takes the magnitude — a negative value there is not a smaller number, it is a
    /// number near `u32::MAX`, and generating one would make every sum wrap for reasons that say
    /// nothing about the emitter.
    #[must_use]
    pub fn encode_signed(self, value: i32) -> u32 {
        match self {
            Self::Unsigned => value.unsigned_abs(),
            Self::Signed => u32::from_ne_bytes(value.to_ne_bytes()),
            Self::Float => (value as f32).to_bits(),
        }
    }

    /// Add, in this domain.
    #[must_use]
    pub fn add(self, left: u32, right: u32) -> u32 {
        match self {
            Self::Unsigned | Self::Signed => left.wrapping_add(right),
            Self::Float => (f32::from_bits(left) + f32::from_bits(right)).to_bits(),
        }
    }

    /// Multiply, in this domain.
    #[must_use]
    pub fn mul(self, left: u32, right: u32) -> u32 {
        match self {
            Self::Unsigned | Self::Signed => left.wrapping_mul(right),
            Self::Float => (f32::from_bits(left) * f32::from_bits(right)).to_bits(),
        }
    }

    /// Is `left` strictly greater than `right`, in this domain?
    ///
    /// Ordered for floats, which is what `OpFOrdGreaterThan` gives and what the lane API emits.
    /// Signed and unsigned genuinely disagree here whenever the top bit is set, which is the point
    /// of having both.
    #[must_use]
    pub fn greater(self, left: u32, right: u32) -> bool {
        match self {
            Self::Unsigned => left > right,
            Self::Signed => (left as i32) > (right as i32),
            Self::Float => f32::from_bits(left) > f32::from_bits(right),
        }
    }

    /// The smaller of two values.
    #[must_use]
    pub fn min(self, left: u32, right: u32) -> u32 {
        if self.greater(left, right) {
            right
        } else {
            left
        }
    }

    /// The value a `min` reduction starts from: larger than anything the generator produces.
    ///
    /// Not `zero`. A minimum folded from zero would return zero whenever every element is
    /// positive, which is most of the time and looks entirely plausible.
    #[must_use]
    pub fn largest(self) -> u32 {
        match self {
            Self::Unsigned => u32::MAX,
            Self::Signed => u32::from_ne_bytes(i32::MAX.to_ne_bytes()),
            Self::Float => f32::INFINITY.to_bits(),
        }
    }

    /// The value a `max` reduction starts from: smaller than anything the generator produces.
    #[must_use]
    pub fn smallest(self) -> u32 {
        match self {
            Self::Unsigned => 0,
            Self::Signed => u32::from_ne_bytes(i32::MIN.to_ne_bytes()),
            Self::Float => f32::NEG_INFINITY.to_bits(),
        }
    }

    /// The larger of two values.
    #[must_use]
    pub fn max(self, left: u32, right: u32) -> u32 {
        if self.greater(left, right) {
            left
        } else {
            right
        }
    }

    /// The additive identity's bit pattern.
    #[must_use]
    pub fn zero(self) -> u32 {
        self.encode(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_arithmetic_wraps_and_float_arithmetic_does_not() {
        assert_eq!(Domain::Unsigned.add(u32::MAX, 1), 0);

        let big = Domain::Float.encode(1_000_000);
        let sum = Domain::Float.add(big, Domain::Float.encode(1));
        assert_eq!(f32::from_bits(sum), 1_000_001.0);
    }

    #[test]
    fn a_float_sum_within_the_ceiling_is_exact_whatever_the_order() {
        // The claim the whole float mode rests on: at these magnitudes addition is associative,
        // because every partial sum is an integer below 2^24 and therefore exact.
        let values: Vec<u32> = (0..=Domain::Float.ceiling())
            .map(|value| Domain::Float.encode(value))
            .collect();

        let forwards = values.iter().fold(Domain::Float.zero(), |total, &value| {
            Domain::Float.add(total, value)
        });
        let backwards = values
            .iter()
            .rev()
            .fold(Domain::Float.zero(), |total, &value| {
                Domain::Float.add(total, value)
            });

        assert_eq!(forwards, backwards);

        let expected = Domain::Float.ceiling() * (Domain::Float.ceiling() + 1) / 2;
        assert_eq!(f32::from_bits(forwards), expected as f32);
    }

    #[test]
    fn a_float_sum_past_the_ceiling_is_not_exact_which_is_why_there_is_one() {
        // The counter-example that justifies `ceiling`. Far above 2^24 an addition of one is lost
        // entirely, so order would start to matter and the comparison would be meaningless.
        let huge = Domain::Float.encode(1);
        let past = (2.0_f32.powi(25)).to_bits();

        assert_eq!(Domain::Float.add(past, huge), past, "the one vanished");
    }

    #[test]
    fn comparison_is_ordered_for_floats() {
        let one = Domain::Float.encode(1);
        let two = Domain::Float.encode(2);

        assert!(Domain::Float.greater(two, one));
        assert!(!Domain::Float.greater(one, two));
        assert_eq!(Domain::Float.max(one, two), two);
    }

    #[test]
    fn zero_is_the_additive_identity_in_every_domain() {
        for domain in ALL_DOMAINS {
            let value = domain.encode(7);
            assert_eq!(domain.add(value, domain.zero()), value);
        }
    }

    #[test]
    fn signed_and_unsigned_order_the_same_bits_the_opposite_way() {
        // The reason `Signed` is a domain and not a duplicate. `-1` is `0xFFFF_FFFF`, which is the
        // largest unsigned value and the second-smallest signed one, and the two reach different
        // SPIR-V instructions for exactly this.
        let minus_one = Domain::Signed.encode_signed(-1);
        let one = Domain::Signed.encode(1);

        assert!(Domain::Signed.greater(one, minus_one), "1 > -1");
        assert!(
            Domain::Unsigned.greater(minus_one, one),
            "the same bits, read unsigned, are the larger"
        );
    }

    #[test]
    fn the_unsigned_domain_refuses_to_hold_a_negative() {
        // A negative there is not a small number, it is a huge one, and generating it would make
        // every sum wrap for a reason that says nothing about the emitter.
        assert_eq!(Domain::Unsigned.encode_signed(-7), 7);
        assert_eq!(Domain::Signed.encode_signed(-7) as i32, -7);
        assert_eq!(f32::from_bits(Domain::Float.encode_signed(-7)), -7.0);
    }

    #[test]
    fn the_extremes_bracket_everything_the_generator_can_produce() {
        for domain in ALL_DOMAINS {
            let top = domain.encode(domain.ceiling());
            let bottom = domain.encode_signed(-(domain.ceiling() as i32));

            assert!(domain.greater(domain.largest(), top), "{domain:?} largest");
            assert!(
                domain.greater(bottom, domain.smallest()) || bottom == domain.smallest(),
                "{domain:?} smallest"
            );
        }
    }

    #[test]
    fn min_and_max_are_each_others_opposite() {
        for domain in ALL_DOMAINS {
            let low = domain.encode(3);
            let high = domain.encode(9);

            assert_eq!(domain.max(low, high), high);
            assert_eq!(domain.min(low, high), low);
        }
    }
}
