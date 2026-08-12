//! How a dot product reads its operands' bits.
//!
//! `OpSDot` takes two integers and a *format*, and the format says whether those integers are the
//! numbers themselves or a container holding several narrower ones. There is exactly one format
//! defined, and it is the one both devices here accelerate: four 8-bit components packed into a
//! 32-bit integer.
//!
//! The number was read out of `spirv.core.grammar.json` (1.6.7) on 2026-08-12, by the recipe in
//! `decisions/DR-0001`.

use crate::encode::Word;

/// How to read the bits of a dot product's operands.
///
/// The operand is **optional** in the grammar: left off, the operands are ordinary integer vectors
/// and each component is its own element. Supplied, they are scalars whose bytes are the
/// components. This crate always supplies it, because the packed form is the one with hardware
/// behind it — and because a `Vector<U32, N>` is what a kernel already has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedVectorFormat {
    /// Four 8-bit components in a 32-bit integer, least significant first.
    ///
    /// Whether those components are read as signed or unsigned is the *instruction's* business —
    /// `OpSDot` against `OpUDot` — not the format's. That split is why a format of "4x8 bit" says
    /// nothing about sign and why the two instructions exist at all.
    FourEightBit,
}

impl PackedVectorFormat {
    /// The literal this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::FourEightBit => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_format_matches_the_khronos_grammar() {
        assert_eq!(PackedVectorFormat::FourEightBit.word(), 0);
    }

    #[test]
    fn zero_is_a_value_and_not_an_absence() {
        // Worth stating because the operand is optional in the grammar and its only value is zero.
        // An emitter that treated "format 0" as "no format" would leave the operand off, and the
        // instruction would then read its operands as *vectors* rather than as packed scalars —
        // which is a different computation on the same words.
        assert_eq!(PackedVectorFormat::FourEightBit.word(), 0);
        assert_eq!(
            size_of_val(&PackedVectorFormat::FourEightBit.word()),
            size_of::<Word>()
        );
    }
}
