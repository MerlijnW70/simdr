//! Reading a word stream back as instructions.
//!
//! The encoder's tests were asserting on hand-counted word offsets, and two of them were wrong
//! about how long an instruction was rather than about what it contained. Offsets are the
//! encoding's business; a test wants to say "then an `OpTypePointer`". This is what lets it.
//!
//! It is also a real capability rather than test scaffolding: reading a module back is what a
//! round-trip property needs, and eventually what a disassembler would be built on.

use crate::encode::Word;

/// The five-word header a module begins with (§2.3).
pub const HEADER_WORDS: usize = 5;

/// One instruction, borrowed from the stream it was read out of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction<'a> {
    opcode: u16,
    operands: &'a [Word],
}

impl<'a> Instruction<'a> {
    /// Its opcode.
    #[must_use]
    pub const fn opcode(&self) -> u16 {
        self.opcode
    }

    /// Everything after the opcode word.
    #[must_use]
    pub const fn operands(&self) -> &'a [Word] {
        self.operands
    }

    /// How many words it occupied, the opcode word included.
    #[must_use]
    pub const fn word_count(&self) -> usize {
        self.operands.len() + 1
    }
}

/// An iterator over a stream of instructions.
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

        // A zero count would not advance, and a count past the end means the stream is truncated.
        // Both stop the walk rather than panicking: this reads bytes that may not be ours.
        let operand_count = count.checked_sub(1)?;
        let operands = tail.get(..operand_count)?;

        self.rest = tail.get(operand_count..)?;
        Some(Instruction { opcode, operands })
    }
}

/// Walk a stream that has no header — a single section, say.
#[must_use]
pub fn instructions(words: &[Word]) -> Instructions<'_> {
    Instructions { rest: words }
}

/// Walk a whole module, skipping its header.
///
/// A stream shorter than a header yields nothing rather than failing: there are no instructions
/// in it either way.
#[must_use]
pub fn body(words: &[Word]) -> Instructions<'_> {
    Instructions {
        rest: words.get(HEADER_WORDS..).unwrap_or(&[]),
    }
}

/// Every opcode in `words`, in order, skipping the header.
///
/// The shorthand a test reaches for when it cares about the shape of a module and not the
/// operands.
#[must_use]
pub fn opcodes(words: &[Word]) -> Vec<u16> {
    body(words).map(|instruction| instruction.opcode).collect()
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
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
        // An instruction claiming four words, with only two present.
        let words = vec![(4 << 16) | 21, 32];

        let read: Vec<_> = instructions(&words).collect();

        assert!(read.is_empty(), "the truncated instruction is not yielded");
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
