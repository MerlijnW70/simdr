//! Converting between `f32` and the 16-bit float SPIR-V calls `Float16`.
//!
//! Rust has no stable `f16`, and this crate has no dependencies to borrow one from — so the two
//! conversions are here, written from IEEE 754's binary16 rather than adapted from anywhere.
//!
//! # Why the emitter needs them at all
//!
//! [`crate::lanes::Lanes::splat_bits`] takes a constant as bits, which for `f32` is
//! `1.5_f32.to_bits()` and for a half is *sixteen* bits nothing in the standard library can
//! produce. Without [`from_f32`] an `F16` kernel could not be given a constant, which is not much
//! of a type.
//!
//! # What is checked
//!
//! Every one of the 65 536 half bit patterns is round-tripped through [`to_f32`] and back in the
//! tests below, so the pair agrees with itself exhaustively rather than on a sample. That catches
//! a wrong shift in either direction; what it cannot catch is *both* being wrong the same way, so
//! the rounding cases are also checked against values worked out from the format.

/// The `f32` a half's bit pattern denotes.
///
/// Exact in every case: binary16 has fewer exponent and mantissa bits than binary32, so every half
/// — including the subnormals and both infinities — is a float exactly.
#[must_use]
pub const fn to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero, keeping its sign: -0.0 and 0.0 are different bit patterns in both formats.
            return f32::from_bits(sign);
        }

        // Subnormal. A half subnormal is `mantissa × 2⁻²⁴`, and every one of them is a *normal*
        // `f32` — so it has to be renormalised rather than copied across, which is the step a
        // conversion written by pattern-matching the normal case gets wrong.
        let highest = 31 - mantissa.leading_zeros();
        let exponent = 127 - 24 + highest;
        let fraction = (mantissa << (23 - highest)) & 0x007f_ffff;
        return f32::from_bits(sign | (exponent << 23) | fraction);
    }

    if exponent == 0x1f {
        // Infinity, or a NaN whose payload is carried across so it stays a NaN.
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }

    f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13))
}

/// The half nearest `value`, rounding ties to even.
///
/// Round-to-nearest-even is what every IEEE 754 operation defaults to, and what a device converting
/// the same number would give. Truncating instead would be a systematic bias towards zero that
/// only shows up in the last bit — which is exactly the size of error a half has to spare.
///
/// Values too large for the format become an infinity of the right sign; values too small become a
/// zero of the right sign. A NaN stays a NaN.
#[must_use]
pub const fn from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        // A NaN. Its payload is truncated, and a payload that truncates to zero would become an
        // *infinity* — so the low bit is forced, which keeps a NaN a NaN.
        return sign | 0x7c00 | (((mantissa >> 13) as u16) | 1);
    }

    let shifted = exponent - 127 + 15;

    if shifted >= 0x1f {
        // Larger than the format holds. `f32::MAX` becomes infinity rather than the largest half,
        // which is what rounding to nearest gives.
        return sign | 0x7c00;
    }

    if shifted <= 0 {
        // Below the smallest normal half. Everything under 2⁻²⁵ rounds to zero; between there and
        // 2⁻¹⁴ the result is a half subnormal, reached by shifting the *explicit* leading one back
        // in and rounding at whatever bit position is left.
        if shifted < -10 {
            return sign;
        }

        let with_implicit = mantissa | 0x0080_0000;
        let shift = (14 - shifted) as u32;
        let truncated = with_implicit >> shift;
        let half = truncated as u16;

        let round_bit = (with_implicit >> (shift - 1)) & 1;
        let sticky = with_implicit & ((1 << (shift - 1)) - 1);
        if round_bit == 1 && (sticky != 0 || (truncated & 1) == 1) {
            // The carry can reach the exponent field, which is correct: the largest subnormal
            // rounding up *is* the smallest normal, and the bit layout makes that automatic.
            return sign | (half + 1);
        }
        return sign | half;
    }

    let truncated = ((shifted as u16) << 10) | ((mantissa >> 13) as u16);
    let round_bit = (mantissa >> 12) & 1;
    let sticky = mantissa & 0x0fff;
    if round_bit == 1 && (sticky != 0 || (truncated & 1) == 1) {
        // As above: a mantissa of all ones rounding up carries into the exponent, and an exponent
        // of 30 carrying out becomes infinity. Both fall out of the layout rather than needing a
        // case.
        return sign | (truncated + 1);
    }
    sign | truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_half_survives_a_round_trip_through_f32() {
        // Exhaustive, because it can be: 65 536 patterns is nothing, and a sample would leave the
        // subnormals and the boundaries to luck. NaNs are excluded from the equality because their
        // payload is truncated on the way back and comparing them as numbers is always false —
        // that they stay NaN is asserted instead.
        for bits in 0..=u16::MAX {
            let wide = to_f32(bits);
            let exponent = (bits >> 10) & 0x1f;
            let mantissa = bits & 0x03ff;

            if exponent == 0x1f && mantissa != 0 {
                assert!(wide.is_nan(), "{bits:#06x} should widen to a NaN");
                assert!(is_nan_half(from_f32(wide)), "{bits:#06x} should stay one");
                continue;
            }

            assert_eq!(
                from_f32(wide),
                bits,
                "{bits:#06x} widened to {wide} and came back as {:#06x}",
                from_f32(wide)
            );
        }
    }

    /// Whether a half bit pattern is a NaN — the format's rule, since `u16` has no opinion.
    const fn is_nan_half(bits: u16) -> bool {
        (bits >> 10) & 0x1f == 0x1f && bits & 0x03ff != 0
    }

    #[test]
    fn the_ordinary_values_are_the_ones_the_format_says() {
        // Worked out from the layout rather than read off an implementation: 1.0 is exponent 15
        // with an empty mantissa, and 2.0 is one exponent higher.
        assert_eq!(from_f32(0.0), 0x0000);
        assert_eq!(from_f32(-0.0), 0x8000);
        assert_eq!(from_f32(1.0), 0x3c00);
        assert_eq!(from_f32(-1.0), 0xbc00);
        assert_eq!(from_f32(2.0), 0x4000);
        assert_eq!(from_f32(0.5), 0x3800);

        assert_eq!(to_f32(0x3c00), 1.0);
        assert_eq!(to_f32(0x4000), 2.0);
        assert_eq!(to_f32(0xbc00), -1.0);
    }

    #[test]
    fn the_edges_of_the_format_are_where_the_format_says() {
        // Largest finite half: 65504. Smallest normal: 2⁻¹⁴. Smallest subnormal: 2⁻²⁴.
        assert_eq!(to_f32(0x7bff), 65504.0);
        assert_eq!(from_f32(65504.0), 0x7bff);
        assert_eq!(to_f32(0x0400), 2.0_f32.powi(-14));
        assert_eq!(to_f32(0x0001), 2.0_f32.powi(-24));
        assert_eq!(from_f32(2.0_f32.powi(-24)), 0x0001);
    }

    #[test]
    fn a_value_too_large_becomes_an_infinity_rather_than_the_largest_half() {
        // 65520 is the midpoint between 65504 and the next value the format would have had, so
        // round-to-nearest sends it *up*, out of range. A conversion that clamped would give
        // 65504 and be wrong in a way no round-trip test would see.
        assert_eq!(from_f32(65520.0), 0x7c00);
        assert_eq!(from_f32(f32::MAX), 0x7c00);
        assert_eq!(from_f32(f32::INFINITY), 0x7c00);
        assert_eq!(from_f32(f32::NEG_INFINITY), 0xfc00);
        assert!(to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn a_value_too_small_becomes_a_zero_of_the_right_sign() {
        assert_eq!(from_f32(1.0e-30), 0x0000);
        assert_eq!(
            from_f32(-1.0e-30),
            0x8000,
            "the sign survives the underflow"
        );
    }

    #[test]
    fn rounding_goes_to_nearest_and_ties_go_to_even() {
        // 2049 sits exactly between two representable halves (the format has 11 bits of precision
        // there, so it steps by two), and the tie rounds to the even neighbour: 2048.
        assert_eq!(to_f32(from_f32(2049.0)), 2048.0);
        // 2051 is likewise a tie, and the even neighbour is 2052 this time.
        assert_eq!(to_f32(from_f32(2051.0)), 2052.0);
        // And a value that is not a tie goes to whichever is nearer.
        assert_eq!(to_f32(from_f32(2050.0)), 2050.0);

        // Truncation instead of rounding would give 2048 for all three, which is the bias this
        // pins down.
        assert_ne!(to_f32(from_f32(2051.0)), 2048.0);
    }

    /// Rounding *into* the subnormal range, which the round trip above cannot reach.
    ///
    /// Fourteen mutants survived this file before this test existed, every one of them in the
    /// subnormal path's round-and-sticky arithmetic. The exhaustive round trip could not touch
    /// them: it only ever hands `from_f32` a value that *came from* a half, and those are exactly
    /// representable, so no rounding ever happens. An exhaustive test over the wrong domain is
    /// still a test over the wrong domain.
    ///
    /// Every expectation here is worked out from the format: a half subnormal is `n × 2⁻²⁴`, so
    /// the question at each value is which multiple of `2⁻²⁴` it is nearest, and ties go to the
    /// even `n`.
    #[test]
    fn a_value_between_two_subnormals_rounds_to_the_nearer_and_ties_to_even() {
        let step = 2.0_f32.powi(-24);

        // Exactly representable, for a floor to compare against.
        assert_eq!(from_f32(step), 0x0001);
        assert_eq!(from_f32(step * 2.0), 0x0002);

        // Nearer one side than the other.
        assert_eq!(from_f32(step * 1.4), 0x0001);
        assert_eq!(from_f32(step * 1.6), 0x0002);

        // Ties. 1.5 goes up to the even 2; 2.5 goes down to the even 2.
        assert_eq!(from_f32(step * 1.5), 0x0002, "a tie rounds to even");
        assert_eq!(
            from_f32(step * 2.5),
            0x0002,
            "and so does this one, downwards"
        );
        assert_eq!(from_f32(step * 3.5), 0x0004);
    }

    #[test]
    fn the_bottom_of_the_subnormal_range_rounds_rather_than_being_cut_off() {
        let step = 2.0_f32.powi(-24);

        // Half the smallest subnormal is a tie between zero and it, and zero is the even one.
        assert_eq!(from_f32(step * 0.5), 0x0000);
        // Three quarters is nearer the smallest subnormal than it is to zero.
        assert_eq!(from_f32(step * 0.75), 0x0001, "this must not flush to zero");
        // And a quarter is nearer zero.
        assert_eq!(from_f32(step * 0.25), 0x0000);
    }

    #[test]
    fn the_largest_subnormal_rounding_up_becomes_the_smallest_normal() {
        // 1023 × 2⁻²⁴ is the largest subnormal and 1024 × 2⁻²⁴ is 2⁻¹⁴, the smallest normal. A
        // value between them at the tie rounds to the even 1024 — and the carry has to reach the
        // *exponent* field, which the bit layout does for free. A fold that clamped inside the
        // mantissa would give 0x03ff and be wrong by one step at the one place the format changes
        // shape.
        let step = 2.0_f32.powi(-24);

        assert_eq!(from_f32(step * 1023.0), 0x03ff, "the largest subnormal");
        assert_eq!(
            from_f32(step * 1023.5),
            0x0400,
            "the tie carries into normal"
        );
        assert_eq!(from_f32(step * 1024.0), 0x0400, "the smallest normal");
        assert_eq!(to_f32(0x0400), 2.0_f32.powi(-14));
    }

    #[test]
    fn the_first_value_too_large_for_the_format_is_an_infinity_and_not_a_nan() {
        // The overflow test is `>=`, and off by one it lets `shifted == 31` through to the normal
        // path — where an exponent of 31 with a *non-zero* mantissa is a NaN rather than an
        // infinity. 65536 has a zero mantissa and so survives the mistake; 70000 does not.
        assert_eq!(from_f32(65536.0), 0x7c00);
        assert_eq!(from_f32(70000.0), 0x7c00, "an overflow is an infinity");
        assert_eq!(from_f32(-70000.0), 0xfc00);
        assert!(to_f32(from_f32(70000.0)).is_infinite());
        assert!(!to_f32(from_f32(70000.0)).is_nan());
    }

    #[test]
    fn a_nan_stays_a_nan_in_both_directions() {
        assert!(to_f32(0x7e00).is_nan());
        assert!(to_f32(0xfe00).is_nan());

        let narrowed = from_f32(f32::NAN);
        assert!(to_f32(narrowed).is_nan());

        // A NaN whose payload lives entirely in the bits that get truncated must not become an
        // infinity, which is the one way this conversion can turn a NaN into a number.
        let low_payload = f32::from_bits(0x7f80_0001);
        assert!(low_payload.is_nan());
        assert!(to_f32(from_f32(low_payload)).is_nan());

        // And a *quiet* NaN stays quiet. The quiet bit is the top of the mantissa in both formats,
        // so carrying the payload across means shifting it down thirteen places — shift it the
        // other way and the top bits fall off the end, leaving a signalling NaN that is still a
        // NaN and still passes every assertion above.
        let quiet = f32::from_bits(0x7fc0_0000);
        assert!(quiet.is_nan());
        assert_eq!(
            from_f32(quiet) & 0x0200,
            0x0200,
            "the quiet bit did not survive the narrowing"
        );
    }

    #[test]
    fn the_subnormals_are_evenly_spaced_all_the_way_down() {
        // A half subnormal is `mantissa × 2⁻²⁴`, so widening one is a multiplication and the
        // spacing is constant. A renormalisation with the wrong shift would bend that line, and
        // the round-trip test above would still pass because it would bend both ways.
        let step = 2.0_f32.powi(-24);
        for mantissa in 1..0x0400_u16 {
            assert_eq!(
                to_f32(mantissa),
                f32::from(mantissa) * step,
                "subnormal {mantissa:#06x}"
            );
        }
    }
}
