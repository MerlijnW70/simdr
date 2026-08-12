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
//!
//! # The narrow integers
//!
//! `i8`, `u8`, `i16` and `u16` are here and are the same argument with a different modulus: their
//! arithmetic wraps at 8 or 16 bits, wrapping is *defined*, and the reference wraps identically.
//! So the exactness comes for free and what is being checked is instruction selection again —
//! `OpSConvert` against `OpUConvert`, `SMax` against `UMax`, and a buffer whose stride is one byte
//! rather than four.
//!
//! **`f16` is deliberately absent.** A half represents integers exactly only up to 2048, and a sum
//! over sixty-four lanes leaves that range at once — so the argument the float domain rests on
//! does not hold, and a tolerance would be checking something else. `runner/tests/narrow.rs` tests
//! `f16` against expectations reasoned from the format instead.

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
    /// 8-bit unsigned integers, wrapping at 256.
    UnsignedByte,
    /// 8-bit signed integers, wrapping at 128.
    Byte,
    /// 16-bit unsigned integers.
    UnsignedShort,
    /// 16-bit signed integers.
    Short,
}

/// Every domain, for a caller that wants to sweep them.
pub const ALL_DOMAINS: [Domain; 7] = [
    Domain::Unsigned,
    Domain::Signed,
    Domain::Float,
    Domain::UnsignedByte,
    Domain::Byte,
    Domain::UnsignedShort,
    Domain::Short,
];

impl Domain {
    /// How many bits an element occupies.
    ///
    /// The number every operation below is written in terms of, rather than one match arm per
    /// domain per operation. Seven domains and eight operations would be fifty-six arms; this is
    /// eight functions and one table.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::Unsigned | Self::Signed | Self::Float => 32,
            Self::UnsignedShort | Self::Short => 16,
            Self::UnsignedByte | Self::Byte => 8,
        }
    }

    /// Whether this is the float domain, where arithmetic is not modular.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float)
    }

    /// The mask that keeps a value inside this domain's width.
    const fn mask(self) -> u32 {
        match self.bits() {
            32 => u32::MAX,
            bits => (1 << bits) - 1,
        }
    }

    /// `value` cut down to this domain's width.
    const fn truncate(self, value: u32) -> u32 {
        value & self.mask()
    }

    /// `value` read as a signed number of this domain's width, widened to `i32`.
    ///
    /// §2.2.1's rule, applied on the host: an `i8` of `0xff` is −1 and not 255, and a comparison
    /// that skipped this would order the narrow domains the way the wide ones do.
    const fn signed_value(self, value: u32) -> i32 {
        let spare = 32 - self.bits();
        ((value << spare) as i32) >> spare
    }

    /// The largest value the generator may produce in this domain.
    ///
    /// Floats stop well below 2²⁴ so that a sum over a few hundred of them stays exact. The
    /// integers are allowed to be larger because wrapping is defined and the reference wraps too —
    /// but a narrow domain still keeps its constants inside its own width, or every generated
    /// constant would be the same truncated value.
    #[must_use]
    pub const fn ceiling(self) -> u32 {
        match self {
            Self::Unsigned | Self::Signed => 4_096,
            Self::Float => 256,
            Self::UnsignedShort | Self::Short => 1_024,
            Self::UnsignedByte | Self::Byte => 32,
        }
    }

    /// Whether values in this domain may be negative.
    ///
    /// The generator uses it to reach below zero, which is the half of the signed domains that
    /// differs from the unsigned ones at all.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::Signed | Self::Float | Self::Byte | Self::Short)
    }

    /// Encode a small integer as this domain's bit pattern.
    #[must_use]
    pub fn encode(self, value: u32) -> u32 {
        if self.is_float() {
            (value as f32).to_bits()
        } else {
            self.truncate(value)
        }
    }

    /// Encode a possibly-negative integer, for the domains that have one.
    ///
    /// An unsigned domain takes the magnitude — a negative value there is not a smaller number, it
    /// is a number near its maximum, and generating one would make every sum wrap for reasons that
    /// say nothing about the emitter.
    #[must_use]
    pub fn encode_signed(self, value: i32) -> u32 {
        if self.is_float() {
            return (value as f32).to_bits();
        }
        if self.is_signed() {
            self.truncate(u32::from_ne_bytes(value.to_ne_bytes()))
        } else {
            self.truncate(value.unsigned_abs())
        }
    }

    /// Add, in this domain.
    #[must_use]
    pub fn add(self, left: u32, right: u32) -> u32 {
        if self.is_float() {
            (f32::from_bits(left) + f32::from_bits(right)).to_bits()
        } else {
            self.truncate(left.wrapping_add(right))
        }
    }

    /// Multiply, in this domain.
    #[must_use]
    pub fn mul(self, left: u32, right: u32) -> u32 {
        if self.is_float() {
            (f32::from_bits(left) * f32::from_bits(right)).to_bits()
        } else {
            self.truncate(left.wrapping_mul(right))
        }
    }

    /// Is `left` strictly greater than `right`, in this domain?
    ///
    /// Ordered for floats, which is what `OpFOrdGreaterThan` gives and what the lane API emits.
    /// Signed and unsigned genuinely disagree here whenever the top bit is set, which is the point
    /// of having both at every width.
    #[must_use]
    pub fn greater(self, left: u32, right: u32) -> bool {
        if self.is_float() {
            return f32::from_bits(left) > f32::from_bits(right);
        }
        if self.is_signed() {
            self.signed_value(left) > self.signed_value(right)
        } else {
            self.truncate(left) > self.truncate(right)
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

    /// The larger of two values.
    #[must_use]
    pub fn max(self, left: u32, right: u32) -> u32 {
        if self.greater(left, right) {
            left
        } else {
            right
        }
    }

    /// The value a `min` reduction starts from: larger than anything the generator produces.
    ///
    /// Not `zero`. A minimum folded from zero would return zero whenever every element is
    /// positive, which is most of the time and looks entirely plausible.
    #[must_use]
    pub fn largest(self) -> u32 {
        if self.is_float() {
            return f32::INFINITY.to_bits();
        }
        if self.is_signed() {
            self.mask() >> 1
        } else {
            self.mask()
        }
    }

    /// The value a `max` reduction starts from: smaller than anything the generator produces.
    #[must_use]
    pub fn smallest(self) -> u32 {
        if self.is_float() {
            return f32::NEG_INFINITY.to_bits();
        }
        if self.is_signed() {
            self.truncate(1 << (self.bits() - 1))
        } else {
            0
        }
    }

    /// The additive identity's bit pattern.
    #[must_use]
    pub fn zero(self) -> u32 {
        self.encode(0)
    }
}

#[cfg(test)]
mod tests;
