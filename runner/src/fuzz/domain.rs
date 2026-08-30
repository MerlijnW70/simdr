use super::program::Fold;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitShift {
    Left,
    RightLogical,
    RightArithmetic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Unsigned,
    Signed,
    Float,
    UnsignedByte,
    Byte,
    UnsignedShort,
    Short,
    Half,
}

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
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::Unsigned | Self::Signed | Self::Float => 32,
            Self::UnsignedShort | Self::Short | Self::Half => 16,
            Self::UnsignedByte | Self::Byte => 8,
        }
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float | Self::Half)
    }

    fn decode(self, bits: u32) -> f32 {
        if matches!(self, Self::Half) {
            simdr::half::to_f32(bits as u16)
        } else {
            f32::from_bits(bits)
        }
    }

    fn encode_float(self, value: f32) -> u32 {
        if matches!(self, Self::Half) {
            u32::from(simdr::half::from_f32(value))
        } else {
            value.to_bits()
        }
    }

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

    #[must_use]
    pub const fn exact_limit(self) -> Option<f32> {
        match self {
            Self::Float => Some(16_777_216.0),
            Self::Half => Some(2_048.0),
            _ => None,
        }
    }

    const fn mask(self) -> u32 {
        match self.bits() {
            32 => u32::MAX,
            bits => (1 << bits) - 1,
        }
    }

    const fn truncate(self, value: u32) -> u32 {
        value & self.mask()
    }

    const fn signed_value(self, value: u32) -> i32 {
        let spare = 32 - self.bits();
        ((value << spare) as i32) >> spare
    }

    #[must_use]
    pub const fn ceiling(self) -> u32 {
        match self {
            Self::Unsigned | Self::Signed => 4_096,
            Self::Float => 256,
            Self::UnsignedShort | Self::Short => 1_024,
            Self::Half => 8,
            Self::UnsignedByte | Self::Byte => 32,
        }
    }

    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::Signed | Self::Float | Self::Byte | Self::Short | Self::Half
        )
    }

    #[must_use]
    pub fn encode(self, value: u32) -> u32 {
        if self.is_float() {
            self.encode_float(value as f32)
        } else {
            self.truncate(value)
        }
    }

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

    #[must_use]
    pub fn add(self, left: u32, right: u32) -> u32 {
        if self.is_float() {
            self.encode_float(self.decode(left) + self.decode(right))
        } else {
            self.truncate(left.wrapping_add(right))
        }
    }

    #[must_use]
    pub fn mul(self, left: u32, right: u32) -> u32 {
        if self.is_float() {
            self.encode_float(self.decode(left) * self.decode(right))
        } else {
            self.truncate(left.wrapping_mul(right))
        }
    }

    #[must_use]
    /// The fold itself, and the value a lane with nothing before it takes.
    /// Both have to come from the domain, because the identity of a minimum is
    /// the largest value the *type* holds and not the largest `u32`.
    pub fn fold(self, fold: Fold, left: u32, right: u32) -> u32 {
        match fold {
            Fold::Product => self.mul(left, right),
            Fold::Min => self.min(left, right),
            Fold::Max => self.max(left, right),
            Fold::And => self.bitand(left, right),
            Fold::Or => self.bitor(left, right),
            Fold::Xor => self.bitxor(left, right),
        }
    }

    pub fn identity(self, fold: Fold) -> u32 {
        match fold {
            Fold::Product => self.encode(1),
            Fold::Min => self.largest(),
            Fold::Max => self.smallest(),
            Fold::And => self.truncate(u32::MAX),
            Fold::Or | Fold::Xor => 0,
        }
    }

    pub fn sub(self, left: u32, right: u32) -> u32 {
        if self.is_float() {
            self.encode_float(self.decode(left) - self.decode(right))
        } else {
            self.truncate(left.wrapping_sub(right))
        }
    }

    /// Clamped to this domain rather than to `u32`, so an `i8` stops at 127 and
    /// not at the top of the word it is carried in.
    pub fn saturating_add(self, left: u32, right: u32) -> u32 {
        if self.is_signed() {
            let sum = i64::from(self.signed_value(left)) + i64::from(self.signed_value(right));
            self.clamp_signed(sum)
        } else {
            let sum = u64::from(left) + u64::from(right);
            self.truncate(sum.min(u64::from(self.largest())) as u32)
        }
    }

    pub fn saturating_sub(self, left: u32, right: u32) -> u32 {
        if self.is_signed() {
            let difference =
                i64::from(self.signed_value(left)) - i64::from(self.signed_value(right));
            self.clamp_signed(difference)
        } else {
            self.truncate(left.saturating_sub(right))
        }
    }

    fn clamp_signed(self, value: i64) -> u32 {
        let high = i64::from(self.signed_value(self.largest()));
        let low = i64::from(self.signed_value(self.smallest()));
        self.truncate(value.clamp(low, high) as u32)
    }

    pub fn bitand(self, left: u32, right: u32) -> u32 {
        self.truncate(left & right)
    }

    pub fn bitor(self, left: u32, right: u32) -> u32 {
        self.truncate(left | right)
    }

    pub fn bitxor(self, left: u32, right: u32) -> u32 {
        self.truncate(left ^ right)
    }

    pub fn not(self, bits: u32) -> u32 {
        self.truncate(!bits)
    }

    pub fn floor(self, bits: u32) -> u32 {
        self.encode_float(self.decode(bits).floor())
    }

    pub fn ceil(self, bits: u32) -> u32 {
        self.encode_float(self.decode(bits).ceil())
    }

    pub fn trunc(self, bits: u32) -> u32 {
        self.encode_float(self.decode(bits).trunc())
    }

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

    #[must_use]
    pub fn equals(self, left: u32, right: u32) -> bool {
        if self.is_float() {
            return self.decode(left) == self.decode(right);
        }
        self.truncate(left) == self.truncate(right)
    }

    #[must_use]
    pub fn min(self, left: u32, right: u32) -> u32 {
        if self.greater(left, right) {
            right
        } else {
            left
        }
    }

    #[must_use]
    pub fn max(self, left: u32, right: u32) -> u32 {
        if self.greater(left, right) {
            left
        } else {
            right
        }
    }

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

    #[must_use]
    pub fn zero(self) -> u32 {
        self.encode(0)
    }

    #[must_use]
    pub fn abs(self, bits: u32) -> u32 {
        if self.is_float() {
            return self.encode_float(self.decode(bits).abs());
        }
        self.truncate(self.signed_value(bits).unsigned_abs())
    }

    #[must_use]
    pub fn bit_shift(self, kind: BitShift, bits: u32, by: u32) -> u32 {
        match kind {
            BitShift::Left => self.truncate(self.truncate(bits).checked_shl(by).unwrap_or(0)),
            BitShift::RightLogical => self.truncate(bits).checked_shr(by).unwrap_or(0),
            BitShift::RightArithmetic => {
                self.truncate((self.signed_value(bits) >> by.min(31)) as u32)
            }
        }
    }
}

#[cfg(test)]
mod tests;
