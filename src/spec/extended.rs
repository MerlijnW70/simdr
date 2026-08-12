//! The GLSL.std.450 extended instruction set.
//!
//! Core SPIR-V has no scalar `min`, `max`, `abs` or `sqrt`. It has arithmetic, comparison and
//! selection, and everything else arrives through an *extended instruction set* — a named
//! collection imported by [`crate::module::Module::ext_inst_import`] and then reached through
//! `OpExtInst`, which carries the set's id and a literal instruction number.
//!
//! GLSL.std.450 is the set every Vulkan implementation is required to support. It needs no
//! capability and no `OpExtension`: an import is enough, which is why nothing here declares one.
//!
//! The numbers were read out of Khronos' `extinst.glsl.std.450.grammar.json` (version 100,
//! revision 2) on 2026-08-12, by the recipe in `decisions/DR-0001`. That file is a *different*
//! grammar from `spirv.core.grammar.json` and its numbers live in their own space: `40` here is
//! `FMax`, and `40` in the core grammar is `OpTypeSampler`. Nothing catches a value read from the
//! wrong file, because both are well-formed.

use crate::encode::Word;

/// An instruction in the GLSL.std.450 set.
///
/// Only the ones something in this crate emits are listed — the set has eighty-odd, most of them
/// transcendentals, matrix operations and packing helpers this crate has no vectors for. Adding
/// one means running the DR-0001 recipe for it rather than reading the number off this list's
/// pattern; they are not contiguous by family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Glsl {
    /// `FAbs` — the magnitude of a float.
    FAbs,
    /// `SAbs` — the magnitude of a signed integer.
    ///
    /// There is no `UAbs`, because an unsigned integer is already its own magnitude. That absence
    /// is why [`crate::lanes::Signed`] exists as a separate trait rather than `abs` being one more
    /// constant on [`crate::lanes::Element`].
    SAbs,
    /// `FMin` — the smaller of two floats.
    FMin,
    /// `UMin` — the smaller of two unsigned integers.
    UMin,
    /// `SMin` — the smaller of two signed integers.
    SMin,
    /// `FMax` — the larger of two floats.
    FMax,
    /// `UMax` — the larger of two unsigned integers.
    UMax,
    /// `SMax` — the larger of two signed integers.
    SMax,
    /// `FClamp` — a float held between two bounds.
    FClamp,
    /// `UClamp` — an unsigned integer held between two bounds.
    UClamp,
    /// `SClamp` — a signed integer held between two bounds.
    SClamp,
    /// `Sqrt` — the square root of a float.
    Sqrt,
    /// `InverseSqrt` — one over the square root, in one instruction rather than two.
    InverseSqrt,
    /// `Exp` — e raised to a float.
    Exp,
    /// `Log` — the natural logarithm of a float.
    Log,
    /// `Fma` — `a * b + c`, rounded once.
    ///
    /// The single rounding is the point and also the caveat: this is *not* the same value as an
    /// `OpFMul` followed by an `OpFAdd`, which rounds twice. It is usually more accurate and it is
    /// never identical, so a kernel and a CPU reference that must agree bit for bit have to make
    /// the same choice.
    Fma,
}

impl Glsl {
    /// The name a module imports this set under.
    ///
    /// The string is matched exactly by the implementation, dots and all.
    pub const SET_NAME: &'static str = "GLSL.std.450";

    /// The literal instruction number `OpExtInst` carries.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::FAbs => 4,
            Self::SAbs => 5,
            Self::Exp => 27,
            Self::Log => 28,
            Self::Sqrt => 31,
            Self::InverseSqrt => 32,
            Self::FMin => 37,
            Self::UMin => 38,
            Self::SMin => 39,
            Self::FMax => 40,
            Self::UMax => 41,
            Self::SMax => 42,
            Self::FClamp => 43,
            Self::UClamp => 44,
            Self::SClamp => 45,
            Self::Fma => 50,
        }
    }

    /// How many operands the instruction takes.
    ///
    /// Not used to emit — [`crate::module::Module::ext_inst`] takes whatever it is given — but it
    /// is what the test below checks the arity against, and a caller assembling operands
    /// programmatically has no other way to ask.
    #[must_use]
    pub const fn operands(self) -> usize {
        match self {
            Self::FAbs | Self::SAbs | Self::Sqrt | Self::InverseSqrt | Self::Exp | Self::Log => 1,
            Self::FMin | Self::UMin | Self::SMin | Self::FMax | Self::UMax | Self::SMax => 2,
            Self::FClamp | Self::UClamp | Self::SClamp | Self::Fma => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_instruction_matches_the_khronos_grammar() {
        // extinst.glsl.std.450.grammar.json, version 100 revision 2.
        assert_eq!(Glsl::FAbs.word(), 4);
        assert_eq!(Glsl::SAbs.word(), 5);
        assert_eq!(Glsl::Exp.word(), 27);
        assert_eq!(Glsl::Log.word(), 28);
        assert_eq!(Glsl::Sqrt.word(), 31);
        assert_eq!(Glsl::InverseSqrt.word(), 32);
        assert_eq!(Glsl::FMin.word(), 37);
        assert_eq!(Glsl::UMin.word(), 38);
        assert_eq!(Glsl::SMin.word(), 39);
        assert_eq!(Glsl::FMax.word(), 40);
        assert_eq!(Glsl::UMax.word(), 41);
        assert_eq!(Glsl::SMax.word(), 42);
        assert_eq!(Glsl::FClamp.word(), 43);
        assert_eq!(Glsl::UClamp.word(), 44);
        assert_eq!(Glsl::SClamp.word(), 45);
        assert_eq!(Glsl::Fma.word(), 50);
    }

    #[test]
    fn the_set_name_is_the_string_the_implementation_matches() {
        assert_eq!(Glsl::SET_NAME, "GLSL.std.450");
    }

    #[test]
    fn no_two_instructions_share_a_number() {
        // A transposed pair here would emit a well-formed `OpExtInst` computing something else —
        // `SMin` for `UMin` differs only on values above 2³¹, which no small test would reach.
        let every = [
            Glsl::FAbs,
            Glsl::SAbs,
            Glsl::Exp,
            Glsl::Log,
            Glsl::Sqrt,
            Glsl::InverseSqrt,
            Glsl::FMin,
            Glsl::UMin,
            Glsl::SMin,
            Glsl::FMax,
            Glsl::UMax,
            Glsl::SMax,
            Glsl::FClamp,
            Glsl::UClamp,
            Glsl::SClamp,
            Glsl::Fma,
        ];
        let mut numbers: Vec<Word> = every.iter().map(|instruction| instruction.word()).collect();
        numbers.sort_unstable();
        let count = numbers.len();
        numbers.dedup();

        assert_eq!(numbers.len(), count);
    }

    #[test]
    fn the_families_agree_on_how_many_operands_they_take() {
        // The three-of-a-kind families are the ones a wrong arity would hide in: `SClamp` taking
        // two operands assembles as an `SClamp` with a missing bound only if the *length* is also
        // wrong, and the length comes from the operand slice.
        for one in [Glsl::FAbs, Glsl::SAbs, Glsl::Sqrt, Glsl::Exp] {
            assert_eq!(one.operands(), 1);
        }
        for two in [Glsl::FMin, Glsl::UMin, Glsl::SMin, Glsl::FMax] {
            assert_eq!(two.operands(), 2);
        }
        for three in [Glsl::FClamp, Glsl::UClamp, Glsl::SClamp, Glsl::Fma] {
            assert_eq!(three.operands(), 3);
        }
    }
}
