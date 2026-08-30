#[must_use]
pub const fn to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x03ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }

        let highest = 31 - mantissa.leading_zeros();
        let exponent = 127 - 24 + highest;
        let fraction = (mantissa << (23 - highest)) & 0x007f_ffff;
        return f32::from_bits(sign | (exponent << 23) | fraction);
    }

    if exponent == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }

    f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13))
}

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
        return sign | 0x7c00 | (((mantissa >> 13) as u16) | 1);
    }

    let shifted = exponent - 127 + 15;

    if shifted >= 0x1f {
        return sign | 0x7c00;
    }

    if shifted <= 0 {
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
            return sign | (half + 1);
        }
        return sign | half;
    }

    let truncated = ((shifted as u16) << 10) | ((mantissa >> 13) as u16);
    let round_bit = (mantissa >> 12) & 1;
    let sticky = mantissa & 0x0fff;
    if round_bit == 1 && (sticky != 0 || (truncated & 1) == 1) {
        return sign | (truncated + 1);
    }
    sign | truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_half_survives_a_round_trip_through_f32() {
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

    const fn is_nan_half(bits: u16) -> bool {
        (bits >> 10) & 0x1f == 0x1f && bits & 0x03ff != 0
    }

    #[test]
    fn the_ordinary_values_are_the_ones_the_format_says() {
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
        assert_eq!(to_f32(0x7bff), 65504.0);
        assert_eq!(from_f32(65504.0), 0x7bff);
        assert_eq!(to_f32(0x0400), 2.0_f32.powi(-14));
        assert_eq!(to_f32(0x0001), 2.0_f32.powi(-24));
        assert_eq!(from_f32(2.0_f32.powi(-24)), 0x0001);
    }

    #[test]
    fn a_value_too_large_becomes_an_infinity_rather_than_the_largest_half() {
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
        assert_eq!(to_f32(from_f32(2049.0)), 2048.0);
        assert_eq!(to_f32(from_f32(2051.0)), 2052.0);
        assert_eq!(to_f32(from_f32(2050.0)), 2050.0);

        assert_ne!(to_f32(from_f32(2051.0)), 2048.0);
    }

    #[test]
    fn a_value_between_two_subnormals_rounds_to_the_nearer_and_ties_to_even() {
        let step = 2.0_f32.powi(-24);

        assert_eq!(from_f32(step), 0x0001);
        assert_eq!(from_f32(step * 2.0), 0x0002);

        assert_eq!(from_f32(step * 1.4), 0x0001);
        assert_eq!(from_f32(step * 1.6), 0x0002);

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

        assert_eq!(from_f32(step * 0.5), 0x0000);
        assert_eq!(from_f32(step * 0.75), 0x0001, "this must not flush to zero");
        assert_eq!(from_f32(step * 0.25), 0x0000);
    }

    #[test]
    fn the_largest_subnormal_rounding_up_becomes_the_smallest_normal() {
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

        let low_payload = f32::from_bits(0x7f80_0001);
        assert!(low_payload.is_nan());
        assert!(to_f32(from_f32(low_payload)).is_nan());

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
