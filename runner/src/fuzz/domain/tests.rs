//! What each domain's arithmetic has to be true of.
//!
//! Split from the implementation because there are seven domains and the properties are stated
//! once and swept over all of them — which reads as a list of claims rather than as a list of
//! cases, and is the shape that catches a new domain that forgot one.

use super::{ALL_DOMAINS, Domain};

/// The four domains whose arithmetic wraps at fewer than 32 bits.
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

    // And the signed ones wrap at the same place, because the bits are the same bits.
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
        assert_eq!(domain.add(value, domain.zero()), value, "{domain:?}");
    }
}

#[test]
fn signed_and_unsigned_order_the_same_bits_the_opposite_way_at_every_width() {
    // The reason `Signed` is a domain and not a duplicate, and the reason each width has both.
    // `-1` is all ones, which is the largest unsigned value and the second-smallest signed one,
    // and the two reach different SPIR-V instructions for exactly this.
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
    // A negative there is not a small number, it is a large one, and generating it would make
    // every sum wrap for a reason that says nothing about the emitter.
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
    // The invariant the whole width-driven implementation rests on: nothing a domain returns has
    // bits above its own. If it did, the value would round-trip through the buffer as something
    // else — the buffer holds one byte per element and the reference would be comparing against
    // a number the device never saw.
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
    // A ceiling above the width would make every generated constant collapse to the same few
    // truncated values, and the generator would look busy while exploring nothing.
    for domain in NARROW {
        assert!(
            domain.ceiling() < (1 << domain.bits()),
            "{domain:?} generates values it cannot hold"
        );
    }
}

#[test]
fn every_domain_is_in_the_sweep() {
    // `ALL_DOMAINS` is what the fuzz tests iterate, so a domain missing from it is a domain that
    // is never fuzzed while appearing to be supported.
    assert_eq!(ALL_DOMAINS.len(), 7);
    for domain in [
        Domain::Unsigned,
        Domain::Signed,
        Domain::Float,
        Domain::UnsignedByte,
        Domain::Byte,
        Domain::UnsignedShort,
        Domain::Short,
    ] {
        assert!(ALL_DOMAINS.contains(&domain), "{domain:?} is not swept");
    }
}
