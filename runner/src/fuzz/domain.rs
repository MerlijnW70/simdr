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
//! # `f16`, and the paragraph that said it was absent
//!
//! A half represents integers exactly only up to **2048**, and a sum over sixty-four lanes leaves
//! that range at once — so the argument the wider float domain rests on does not hold for it. This
//! file said the domain was therefore "deliberately absent" for as long as [`Domain::Half`] sat
//! twenty lines below, which is the drift this project keeps finding: a reason that outlived its
//! conclusion.
//!
//! The reasoning was right and the conclusion was too strong. What it argues for is not skipping
//! the domain but **noticing** when a round leaves the range: [`Domain::exact_limit`] says where
//! that is, and the reference refuses such a round rather than comparing two roundings. Every
//! `Half` round that is compared at all is compared exactly, and the refused ones are counted.
//! `runner/tests/narrow.rs` covers the rounding itself, against expectations reasoned from the
//! format.

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
    /// 16-bit floats holding small integers, where arithmetic is exact.
    ///
    /// The domain `notes/NEXT.md` said could not be fuzzed. Its reasoning was right and its
    /// conclusion was too strong: a half represents integers exactly only to **2048**, so a sum
    /// over a few hundred lanes leaves that range and a tolerance would be checking the rounding
    /// rather than the emitter.
    ///
    /// What that argues for is not skipping the domain but *noticing* when a round leaves the
    /// range. [`Domain::exact_limit`] says where that is and the reference refuses the round
    /// rather than comparing it — so every `Half` round that is compared at all is compared
    /// exactly, and the ones that cannot be are counted rather than quietly loosened.
    Half,
}

/// Every domain, for a caller that wants to sweep them.
pub const ALL_DOMAINS: [Domain; 8] = [
    Domain::Unsigned,
    Domain::Signed,
    Domain::Float,
    Domain::UnsignedByte,
    Domain::Byte,
    Domain::UnsignedShort,
    Domain::Short,
    Domain::Half,
];

impl Domain {
    /// How many bits an element occupies.
    ///
    /// The number every operation below is written in terms of, rather than one match arm per
    /// domain per operation. Eight domains and eight operations would be sixty-four arms; this is
    /// eight functions and one table.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::Unsigned | Self::Signed | Self::Float => 32,
            Self::UnsignedShort | Self::Short | Self::Half => 16,
            Self::UnsignedByte | Self::Byte => 8,
        }
    }

    /// Whether this is a float domain, where arithmetic rounds instead of wrapping.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float | Self::Half)
    }

    /// This domain's bits, read as the number they stand for.
    ///
    /// The one place the two float widths differ. Everything else is written in terms of `f32`,
    /// because a half converts to one exactly — there is no second rounding hiding in here.
    fn decode(self, bits: u32) -> f32 {
        if matches!(self, Self::Half) {
            simdr::half::to_f32(bits as u16)
        } else {
            f32::from_bits(bits)
        }
    }

    /// The inverse: a number written back as this domain's bits.
    fn encode_float(self, value: f32) -> u32 {
        if matches!(self, Self::Half) {
            u32::from(simdr::half::from_f32(value))
        } else {
            value.to_bits()
        }
    }

    /// This domain's bits as a number, for a caller that wants to reason about magnitude.
    ///
    /// Integer domains answer with the value they stand for, so one bound serves both kinds.
    #[must_use]
    pub fn as_f32(self, bits: u32) -> f32 {
        if self.is_float() {
            self.decode(bits)
        } else if self.is_signed() {
            self.signed_value(bits) as f32
        } else {
            self.truncate(bits) as f32
        }
    }

    /// The magnitude past which this domain stops counting integers exactly, if it has one.
    ///
    /// `None` for the integer domains: wrapping is exact and the reference wraps the same way.
    /// The floats do have one — 2²⁴ for a single and **2¹¹** for a half — and past it a sum is
    /// rounded, so comparing it against a host sum compares two roundings rather than the mapping.
    ///
    /// The single's limit had never been checked, only assumed. It is checked now, for both.
    #[must_use]
    pub const fn exact_limit(self) -> Option<f32> {
        match self {
            Self::Float => Some(16_777_216.0),
            Self::Half => Some(2_048.0),
            _ => None,
        }
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
            // Small, so that a sum over a few hundred lanes usually stays under 2048 and the
            // round can be checked exactly. Rounds that still leave the range are refused.
            Self::Half => 8,
            Self::UnsignedByte | Self::Byte => 32,
        }
    }

    /// Whether values in this domain may be negative.
    ///
    /// The generator uses it to reach below zero, which is the half of the signed domains that
    /// differs from the unsigned ones at all.
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::Signed | Self::Float | Self::Byte | Self::Short | Self::Half
        )
    }

    /// Encode a small integer as this domain's bit pattern.
    #[must_use]
    pub fn encode(self, value: u32) -> u32 {
        if self.is_float() {
            self.encode_float(value as f32)
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
            return self.encode_float(value as f32);
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
            self.encode_float(self.decode(left) + self.decode(right))
        } else {
            self.truncate(left.wrapping_add(right))
        }
    }

    /// Multiply, in this domain.
    #[must_use]
    pub fn mul(self, left: u32, right: u32) -> u32 {
        if self.is_float() {
            self.encode_float(self.decode(left) * self.decode(right))
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
            return self.decode(left) > self.decode(right);
        }
        if self.is_signed() {
            self.signed_value(left) > self.signed_value(right)
        } else {
            self.truncate(left) > self.truncate(right)
        }
    }

    /// Whether two values are equal, the way the device's comparison is.
    ///
    /// Decoded rather than compared as bits, for the same reason [`Domain::greater`] is: a float
    /// domain's `+0.0` and `-0.0` are two bit patterns and one value, and `OpFOrdEqual` says they
    /// are equal. Nothing in this corpus produces a negative zero — which is exactly why the
    /// comparison is written the way the hardware does it rather than the way the corpus would let
    /// it get away with.
    ///
    /// A NaN is equal to nothing, itself included, and `f32`'s own `==` already says so.
    #[must_use]
    pub fn equals(self, left: u32, right: u32) -> bool {
        if self.is_float() {
            return self.decode(left) == self.decode(right);
        }
        self.truncate(left) == self.truncate(right)
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
            return self.encode_float(f32::INFINITY);
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
            return self.encode_float(f32::NEG_INFINITY);
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
