use super::{ALL_DOMAINS, Domain};

const NARROW: [Domain; 4] = [
    Domain::UnsignedByte,
    Domain::Byte,
    Domain::UnsignedShort,
    Domain::Short,
];

#[test]
fn integer_arithmetic_wraps_at_the_domains_own_width() {
    assert_eq!(Domain::Unsigned.add(u32::MAX, 1), 0);
    assert_eq!(Domain::UnsignedByte.add(255, 1), 0, "8 bits");
    assert_eq!(Domain::UnsignedShort.add(65_535, 1), 0, "16 bits");

    assert_eq!(Domain::Byte.add(127, 1), 128, "0x80, which reads as -128");
    assert_eq!(Domain::Byte.signed_value(Domain::Byte.add(127, 1)), -128);
}

#[test]
fn float_arithmetic_does_not_wrap() {
    let big = Domain::Float.encode(1_000_000);
    let sum = Domain::Float.add(big, Domain::Float.encode(1));

    assert_eq!(f32::from_bits(sum), 1_000_001.0);
}

#[test]
fn a_float_sum_within_the_ceiling_is_exact_whatever_the_order() {
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
        assert_eq!(domain.add(value, domain.zero()), value, "{domain:?}");
    }
}

#[test]
fn signed_and_unsigned_order_the_same_bits_the_opposite_way_at_every_width() {
    for (signed, unsigned) in [
        (Domain::Signed, Domain::Unsigned),
        (Domain::Short, Domain::UnsignedShort),
        (Domain::Byte, Domain::UnsignedByte),
    ] {
        let minus_one = signed.encode_signed(-1);
        let one = signed.encode(1);

        assert!(signed.greater(one, minus_one), "{signed:?}: 1 > -1");
        assert!(
            unsigned.greater(minus_one, one),
            "{unsigned:?}: the same bits, read unsigned, are the larger"
        );
    }
}

#[test]
fn an_unsigned_domain_refuses_to_hold_a_negative() {
    assert_eq!(Domain::Unsigned.encode_signed(-7), 7);
    assert_eq!(Domain::UnsignedByte.encode_signed(-7), 7);
    assert_eq!(Domain::Signed.encode_signed(-7) as i32, -7);
    assert_eq!(
        Domain::Byte.signed_value(Domain::Byte.encode_signed(-7)),
        -7
    );
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
fn min_and_max_are_each_others_opposite_in_every_domain() {
    for domain in ALL_DOMAINS {
        let low = domain.encode(3);
        let high = domain.encode(9);

        assert_eq!(domain.max(low, high), high, "{domain:?}");
        assert_eq!(domain.min(low, high), low, "{domain:?}");
    }
}

#[test]
fn a_narrow_domain_never_produces_a_value_outside_its_width() {
    for domain in NARROW {
        let limit = 1_u32 << domain.bits();
        let values = [
            domain.encode(domain.ceiling()),
            domain.encode_signed(-(domain.ceiling() as i32)),
            domain.add(domain.encode(200), domain.encode(200)),
            domain.mul(domain.encode(200), domain.encode(200)),
            domain.largest(),
            domain.smallest(),
            domain.zero(),
        ];

        for value in values {
            assert!(value < limit, "{domain:?} produced {value:#x}");
        }
    }
}

#[test]
fn the_ceiling_of_a_narrow_domain_fits_inside_it() {
    for domain in NARROW {
        assert!(
            domain.ceiling() < (1 << domain.bits()),
            "{domain:?} generates values it cannot hold"
        );
    }
}

#[test]
fn every_domain_is_in_the_sweep() {
    assert_eq!(ALL_DOMAINS.len(), 8);
    for domain in [
        Domain::Unsigned,
        Domain::Signed,
        Domain::Float,
        Domain::UnsignedByte,
        Domain::Byte,
        Domain::UnsignedShort,
        Domain::Short,
        Domain::Half,
    ] {
        assert!(ALL_DOMAINS.contains(&domain), "{domain:?} is not swept");
    }
}

#[test]
fn only_the_float_domains_have_a_limit_and_they_have_the_right_ones() {
    assert_eq!(Domain::Float.exact_limit(), Some(16_777_216.0));
    assert_eq!(Domain::Half.exact_limit(), Some(2_048.0));

    for domain in ALL_DOMAINS {
        assert_eq!(
            domain.exact_limit().is_some(),
            domain.is_float(),
            "{domain:?}"
        );
    }
}

#[test]
fn the_limits_are_where_the_formats_actually_stop_counting() {
    let single = 16_777_216.0_f32;
    assert_eq!(single + 1.0, single, "2²⁴ + 1 is not an f32");

    let limit = Domain::Half.exact_limit().expect("a half has one");
    assert_eq!(
        simdr::half::to_f32(simdr::half::from_f32(limit)),
        limit,
        "2¹¹ does not survive a half"
    );
    assert_ne!(
        simdr::half::to_f32(simdr::half::from_f32(limit + 1.0)),
        limit + 1.0,
        "2¹¹ + 1 survived a half, so the limit is in the wrong place"
    );
}

#[test]
fn as_f32_reads_each_domain_the_way_that_domain_means_it() {
    assert_eq!(Domain::Float.as_f32(Domain::Float.encode(1_000)), 1_000.0);
    assert_eq!(Domain::Half.as_f32(Domain::Half.encode(1_000)), 1_000.0);

    assert_eq!(Domain::Byte.as_f32(0xff), -1.0);
    assert_eq!(Domain::UnsignedByte.as_f32(0xff), 255.0);
    assert_eq!(Domain::Short.as_f32(0xffff), -1.0);
    assert_eq!(Domain::UnsignedShort.as_f32(0xffff), 65_535.0);
    assert_eq!(Domain::Signed.as_f32(0xffff_ffff), -1.0);
    assert_eq!(Domain::Unsigned.as_f32(0xffff_ffff), 4_294_967_295.0);
}

#[test]
fn as_f32_agrees_with_the_ordering_the_same_domain_uses() {
    for domain in ALL_DOMAINS {
        for (left, right) in [(1_u32, 2_u32), (0, 7), (3, 3)] {
            let (left, right) = (domain.encode(left), domain.encode(right));
            assert_eq!(
                domain.greater(left, right),
                domain.as_f32(left) > domain.as_f32(right),
                "{domain:?} orders {left:#x} and {right:#x} two different ways"
            );
        }
    }
}

#[test]
fn the_reduction_identities_sit_outside_the_values_a_domain_generates() {
    for domain in ALL_DOMAINS {
        let ceiling = domain.encode(domain.ceiling());

        assert!(
            !domain.greater(ceiling, domain.largest()),
            "{domain:?} can generate a value above its `largest`"
        );
        assert!(
            domain.greater(ceiling, domain.smallest()),
            "{domain:?} generates values at or below its `smallest`"
        );

        if domain.is_signed() {
            let below = domain.encode_signed(-(domain.ceiling() as i32));
            assert!(
                domain.greater(below, domain.smallest()),
                "{domain:?} is signed and its `smallest` is not below its negatives"
            );
        }
    }
}

#[test]
fn an_unsigned_domain_reaches_its_whole_width_and_a_signed_one_stops_at_its_sign_bit() {
    assert_eq!(Domain::Unsigned.largest(), u32::MAX);
    assert_eq!(Domain::UnsignedByte.largest(), 0xff);
    assert_eq!(Domain::UnsignedShort.largest(), 0xffff);

    assert_eq!(Domain::Signed.largest() as i32, i32::MAX);
    assert_eq!(Domain::Byte.signed_value(Domain::Byte.largest()), 127);
    assert_eq!(Domain::Short.signed_value(Domain::Short.largest()), 32_767);

    assert_eq!(Domain::Unsigned.smallest(), 0);
    assert_eq!(Domain::UnsignedByte.smallest(), 0);
    assert_eq!(Domain::Byte.signed_value(Domain::Byte.smallest()), -128);
    assert_eq!(
        Domain::Short.signed_value(Domain::Short.smallest()),
        -32_768
    );
}

#[test]
fn equality_is_numeric_for_the_floats_and_bitwise_for_the_integers() {
    assert!(
        Domain::Float.equals(0.0_f32.to_bits(), (-0.0_f32).to_bits()),
        "+0.0 and -0.0 are one value and two bit patterns"
    );
    assert_ne!(
        0.0_f32.to_bits(),
        (-0.0_f32).to_bits(),
        "the test above says nothing unless the bits differ"
    );

    let nan = f32::NAN.to_bits();
    assert!(!Domain::Float.equals(nan, nan), "a NaN is equal to nothing");

    assert!(
        Domain::Unsigned.equals(nan, nan),
        "an integer is equal to itself whatever its bits would mean elsewhere"
    );

    assert!(
        Domain::UnsignedByte.equals(0x0000_00FF, 0xFFFF_00FF),
        "a byte domain compares eight bits"
    );
    assert!(!Domain::UnsignedByte.equals(0x0000_00FF, 0x0000_00FE));
}
