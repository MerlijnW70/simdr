use super::{Integer, LaneError, Lanes, U32, Vector};

impl Lanes<'_> {
    /// `a + b`, clamped to the type's range rather than wrapped.
    ///
    /// SPIR-V has no saturating integer add, so this is a sequence. An unsigned
    /// one is two instructions and exact at any width: `a + min(b, !a)`, where
    /// `!a` is the largest addend that still fits. A signed one costs more,
    /// because it has to know which end it overflowed towards.
    pub fn saturating_add<T: Integer, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        if T::SIGNED {
            let sum = self.add(left, right)?;

            // Overflow exactly when the sum's sign differs from both operands'.
            let from_left = self.xor(left, sum)?;
            let from_right = self.xor(right, sum)?;
            let both = self.and(from_left, from_right)?;

            self.clamp_to_the_end_it_passed::<T, LANES>(left, both, sum)
        } else {
            let room = self.not(left)?;
            let fits = self.min(right, room)?;
            self.add(left, fits)
        }
    }

    /// `a - b`, clamped to the type's range rather than wrapped.
    ///
    /// Unsigned, this is `a - min(a, b)`: subtracting no more than is there
    /// leaves zero rather than a wrapped maximum.
    pub fn saturating_sub<T: Integer, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        right: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        if T::SIGNED {
            let difference = self.sub(left, right)?;

            // Overflow exactly when the operands differ in sign and the
            // difference took the sign of the right one.
            let between = self.xor(left, right)?;
            let from_left = self.xor(left, difference)?;
            let both = self.and(between, from_left)?;

            self.clamp_to_the_end_it_passed::<T, LANES>(left, both, difference)
        } else {
            let taken = self.min(left, right)?;
            self.sub(left, taken)
        }
    }

    /// `wrapped` where `overflowed`'s top bit is clear, and otherwise the end
    /// of the range that `left` was heading towards: the maximum when it was
    /// positive, the minimum when it was negative.
    fn clamp_to_the_end_it_passed<T: Integer, const LANES: u32>(
        &mut self,
        left: Vector<T, LANES>,
        overflowed: Vector<T, LANES>,
        wrapped: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let sign = self.splat_bits::<U32, LANES>(T::BITS - 1)?;
        let zero = self.splat_bits::<T, LANES>(0)?;

        // All ones when `left` was negative, all zeros when it was not, so the
        // exclusive-or below picks the minimum or the maximum without a branch.
        let spread = self.shift_right_arithmetic(left, sign)?;
        let largest = self.splat_bits::<T, LANES>(largest_of::<T>())?;
        let limit = self.xor(spread, largest)?;

        let past = self.less_than(overflowed, zero)?;
        self.select(past, limit, wrapped)
    }
}

/// The largest value the type holds, as the bits it is written from: every bit
/// set but the sign.
fn largest_of<T: Integer>() -> u32 {
    let sign = 1_u32 << (T::BITS - 1);
    sign.wrapping_sub(1)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{I8, I16, I32, U8, U16, U32};
    use crate::module::{Module, Version, op};

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    #[test]
    fn the_largest_value_of_each_width_is_the_one_that_width_holds() {
        assert_eq!(largest_of::<I8>(), 127);
        assert_eq!(largest_of::<I16>(), 32_767);
        assert_eq!(largest_of::<I32>(), 0x7fff_ffff);
    }

    #[test]
    fn every_integer_declares_the_signedness_its_own_name_gives_it() {
        let declared = [
            ("i8", I8::SIGNED, I8::BITS),
            ("i16", I16::SIGNED, I16::BITS),
            ("i32", I32::SIGNED, I32::BITS),
            ("u8", U8::SIGNED, U8::BITS),
            ("u16", U16::SIGNED, U16::BITS),
            ("u32", U32::SIGNED, U32::BITS),
        ];

        for (name, signed, bits) in declared {
            assert_eq!(
                signed,
                name.starts_with('i'),
                "{name} disagrees with its own name about its sign"
            );
            assert_eq!(
                bits,
                name[1..]
                    .parse::<u32>()
                    .expect("a width follows the letter"),
                "{name} disagrees with its own name about its width"
            );
        }
    }

    #[test]
    fn each_width_reports_the_bits_its_stride_holds() {
        assert_eq!(U8::BITS, 8);
        assert_eq!(U16::BITS, 16);
        assert_eq!(U32::BITS, 32);
    }

    #[test]
    fn an_unsigned_saturating_add_is_a_complement_a_minimum_and_an_add() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 32>(7).expect("splat");

        lanes.saturating_add(value, value).expect("added");

        let words = module.finish();
        assert_eq!(count(&words, op::NOT), 1);
        assert_eq!(
            count(&words, op::EXT_INST),
            1,
            "one UMin from the extended set"
        );
        assert_eq!(count(&words, op::I_ADD), 1);
        assert_eq!(
            count(&words, op::SELECT),
            0,
            "the unsigned sequence needs no comparison and no pick"
        );
    }

    #[test]
    fn an_unsigned_saturating_sub_is_a_minimum_and_a_subtraction() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 32>(7).expect("splat");

        lanes.saturating_sub(value, value).expect("subtracted");

        let words = module.finish();
        assert_eq!(count(&words, op::EXT_INST), 1);
        assert_eq!(count(&words, op::I_SUB), 1);
        assert_eq!(count(&words, op::NOT), 0, "nothing to complement here");
        assert_eq!(count(&words, op::SELECT), 0);
    }

    #[test]
    fn a_signed_saturating_add_detects_its_overflow_and_picks_an_end() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<I32, 32>(7).expect("splat");

        lanes.saturating_add(value, value).expect("added");

        let words = module.finish();
        assert_eq!(count(&words, op::I_ADD), 1, "the wrapping sum");
        assert_eq!(
            count(&words, op::BITWISE_XOR),
            3,
            "two to compare signs against the sum, one to turn the spread into an end"
        );
        assert_eq!(count(&words, op::BITWISE_AND), 1);
        assert_eq!(count(&words, op::SHIFT_RIGHT_ARITHMETIC), 1);
        assert_eq!(count(&words, op::S_LESS_THAN), 1, "signed, not unsigned");
        assert_eq!(count(&words, op::SELECT), 1);
    }

    #[test]
    fn a_signed_saturating_sub_subtracts_rather_than_adds() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<I32, 32>(7).expect("splat");

        lanes.saturating_sub(value, value).expect("subtracted");

        let words = module.finish();
        assert_eq!(count(&words, op::I_SUB), 1);
        assert_eq!(count(&words, op::I_ADD), 0, "no addition anywhere");
        assert_eq!(count(&words, op::SELECT), 1);
    }

    #[test]
    fn the_two_signednesses_are_two_different_sequences() {
        let emitted = |build: fn(&mut Lanes<'_>)| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            build(&mut lanes);
            let words = module.finish();
            (count(&words, op::SELECT), count(&words, op::NOT))
        };

        let unsigned = emitted(|lanes| {
            let value = lanes.splat_bits::<U32, 32>(7).expect("splat");
            lanes.saturating_add(value, value).expect("added");
        });
        let signed = emitted(|lanes| {
            let value = lanes.splat_bits::<I32, 32>(7).expect("splat");
            lanes.saturating_add(value, value).expect("added");
        });

        assert_eq!(unsigned, (0, 1), "a complement and no pick");
        assert_eq!(signed, (1, 0), "a pick and no complement");
    }

    #[test]
    fn a_strip_mined_saturating_add_saturates_every_strip() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let wide = lanes.splat_bits::<U32, 128>(7).expect("splat");

        let saturated = lanes.saturating_add(wide, wide).expect("added");

        assert_eq!(saturated.strip_count(), 4);
        assert_eq!(count(&module.finish(), op::I_ADD), 4);
    }

    #[test]
    fn the_narrow_integers_saturate_at_their_own_width() {
        for build in [
            (|lanes: &mut Lanes<'_>| {
                let value = lanes.splat_bits::<U8, 32>(7).expect("splat");
                lanes.saturating_add(value, value).expect("added");
            }) as fn(&mut Lanes<'_>),
            |lanes: &mut Lanes<'_>| {
                let value = lanes.splat_bits::<I16, 32>(7).expect("splat");
                lanes.saturating_sub(value, value).expect("subtracted");
            },
        ] {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            build(&mut lanes);
            let words = module.finish();
            assert!(!words.is_empty(), "a narrow saturation emitted nothing");
        }
    }
}
