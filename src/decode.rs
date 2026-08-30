use crate::encode::Word;

pub const HEADER_WORDS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction<'a> {
    opcode: u16,
    operands: &'a [Word],
}

impl<'a> Instruction<'a> {
    #[must_use]
    pub const fn opcode(&self) -> u16 {
        self.opcode
    }

    #[must_use]
    pub const fn operands(&self) -> &'a [Word] {
        self.operands
    }

    #[must_use]
    pub const fn word_count(&self) -> usize {
        self.operands.len() + 1
    }
}

#[derive(Debug, Clone)]
pub struct Instructions<'a> {
    rest: &'a [Word],
}

impl<'a> Iterator for Instructions<'a> {
    type Item = Instruction<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (&first, tail) = self.rest.split_first()?;

        let count = (first >> 16) as usize;
        let opcode = (first & 0xffff) as u16;

        let operand_count = count.checked_sub(1)?;
        let operands = tail.get(..operand_count)?;

        self.rest = tail.get(operand_count..)?;
        Some(Instruction { opcode, operands })
    }
}

#[must_use]
pub fn instructions(words: &[Word]) -> Instructions<'_> {
    Instructions { rest: words }
}

#[must_use]
pub fn body(words: &[Word]) -> Instructions<'_> {
    Instructions {
        rest: words.get(HEADER_WORDS..).unwrap_or(&[]),
    }
}

#[must_use]
pub fn opcodes(words: &[Word]) -> Vec<u16> {
    body(words).map(|instruction| instruction.opcode).collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::encode;

    #[test]
    fn an_instruction_reads_back_the_way_it_was_written() {
        let mut words = Vec::new();
        encode::instruction(&mut words, 21, &[32, 0]).expect("fits");

        let read: Vec<_> = instructions(&words).collect();

        assert_eq!(read.len(), 1);
        assert_eq!(read[0].opcode(), 21);
        assert_eq!(read[0].operands(), &[32, 0]);
        assert_eq!(read[0].word_count(), 3);
    }

    #[test]
    fn a_stream_of_several_reads_back_in_order() {
        let mut words = Vec::new();
        encode::instruction(&mut words, 1, &[]).expect("fits");
        encode::instruction(&mut words, 2, &[7]).expect("fits");
        encode::instruction(&mut words, 3, &[8, 9]).expect("fits");

        let read: Vec<u16> = instructions(&words)
            .map(|instruction| instruction.opcode())
            .collect();

        assert_eq!(read, vec![1, 2, 3]);
    }

    #[test]
    fn an_instruction_with_no_operands_reads_back_as_empty_rather_than_stopping_the_walk() {
        let mut words = Vec::new();
        encode::instruction(&mut words, 253, &[]).expect("fits");
        encode::instruction(&mut words, 56, &[]).expect("fits");

        let read: Vec<_> = instructions(&words).collect();

        assert_eq!(read.len(), 2);
        assert!(read[0].operands().is_empty());
    }

    #[test]
    fn a_truncated_stream_stops_rather_than_reading_past_the_end() {
        let words = vec![(4 << 16) | 21, 32];

        let read: Vec<_> = instructions(&words).collect();

        assert!(read.is_empty(), "the truncated instruction is not yielded");
    }

    #[test]
    fn arbitrary_words_are_walked_without_panicking_or_looping() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            state = state
                .wrapping_mul(0x5851_F42D_4C95_7F2D)
                .wrapping_add(0x1405_7B7E_F767_814F);
            (state >> 32) as u32
        };

        for length in 0..64_usize {
            let words: Vec<Word> = (0..length).map(|_| next()).collect();

            let mut seen = 0_usize;
            for instruction in instructions(&words) {
                seen += instruction.word_count();
                assert!(
                    instruction.word_count() >= 1,
                    "an instruction consumed nothing"
                );
            }
            assert!(seen <= length, "the walk read {seen} of {length} words");

            let _ = opcodes(&words);
        }
    }

    #[test]
    fn a_zero_word_count_stops_the_walk_instead_of_looping_forever() {
        let words = vec![0, 0, 0];

        let read: Vec<_> = instructions(&words).collect();

        assert!(read.is_empty());
    }

    #[test]
    fn the_body_of_a_module_skips_its_header() {
        let mut words = vec![encode::MAGIC, 0x0001_0000, 0, 2, 0];
        encode::instruction(&mut words, 17, &[1]).expect("fits");

        assert_eq!(opcodes(&words), vec![17]);
    }

    #[test]
    fn a_stream_too_short_to_hold_a_header_yields_nothing() {
        assert!(opcodes(&[encode::MAGIC, 0]).is_empty());
    }
}
