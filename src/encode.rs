//! The word-level encoding a SPIR-V module is made of.
//!
//! Everything above this module thinks in instructions; this one is the only place that knows a
//! module is a stream of 32-bit words. Section numbers cite the SPIR-V specification, unified
//! revision — `§2.2` is "Terms", `§2.3` "Physical Layout".

use core::fmt;

/// A single 32-bit word. A SPIR-V module is a stream of these and nothing else.
pub type Word = u32;

/// The magic number a module begins with (§2.3).
///
/// A consumer reads this first to learn the module's endianness: seeing it byte-reversed means
/// the words need swapping. We always emit it in our own order, so a reader on the same machine
/// sees it upright.
pub const MAGIC: Word = 0x0723_0203;

/// Something a caller asked for that cannot be expressed in the encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// An instruction was longer than the 16-bit word count in its first word can describe.
    ///
    /// Reachable from a long enough literal string, so it is refused rather than truncated: a
    /// truncated instruction is a module that reads as valid and means something else.
    InstructionTooLong {
        /// The opcode that could not be emitted.
        opcode: u16,
        /// How many words it would have needed, the opcode word included.
        words: usize,
    },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InstructionTooLong { opcode, words } => write!(
                f,
                "instruction {opcode} needs {words} words and the encoding allows {}",
                u16::MAX
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Append one instruction to `out`.
///
/// The first word packs the length and the opcode together (§2.2); the length counts itself, so
/// an instruction with no operands is one word. Passing more operands than that word can count
/// yields [`EncodeError::InstructionTooLong`] and leaves `out` untouched.
pub fn instruction(out: &mut Vec<Word>, opcode: u16, operands: &[Word]) -> Result<(), EncodeError> {
    // The operand count comes from a caller and the +1 is the opcode word, so this is the one
    // addition here that could overflow on a 16-bit-ish target; saturating is enough because the
    // conversion below refuses anything near the limit anyway.
    let words = operands.len().saturating_add(1);
    let count =
        u16::try_from(words).map_err(|_| EncodeError::InstructionTooLong { opcode, words })?;

    out.push((Word::from(count) << 16) | Word::from(opcode));
    out.extend_from_slice(operands);
    Ok(())
}

/// Append a literal string operand (§2.2.1).
///
/// UTF-8, NUL-terminated, packed low byte first, zero-padded to a word boundary. A string whose
/// byte length is already a multiple of four still gains a whole word of zeros — the terminator
/// is part of the literal, not padding that may be elided, and a consumer that finds four
/// non-zero bytes in the last word keeps reading into the next operand.
pub fn literal_string(operands: &mut Vec<Word>, text: &str) {
    let mut word: Word = 0;
    let mut filled: u32 = 0;

    for &byte in text.as_bytes() {
        // `filled` is 0..=3 here, so the shift is at most 24 and cannot overflow.
        word |= Word::from(byte) << (8 * filled);
        filled += 1;
        if filled == 4 {
            operands.push(word);
            word = 0;
            filled = 0;
        }
    }

    // Unconditional: this is the terminator when the last chunk was short, and the whole extra
    // word when the length divided evenly.
    operands.push(word);
}

/// How many words [`literal_string`] will append for `text`.
///
/// Callers that need an instruction's length before building it — to decide whether it will fit —
/// use this rather than encoding the string twice.
#[must_use]
pub fn literal_string_words(text: &str) -> usize {
    // One word per four bytes, plus one for the terminator that is always emitted. Written with
    // the terminator byte folded in so the "already a multiple of four" case needs no branch.
    text.len() / 4 + 1
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports. The crate-level denials are about what a caller
    // can provoke in shipped code, and they would otherwise make every assertion a lint error.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;

    #[test]
    fn an_instructions_first_word_counts_itself() {
        let mut out = Vec::new();
        instruction(&mut out, 17, &[1]).expect("two words is well inside the limit");

        assert_eq!(out, vec![(2 << 16) | 17, 1]);
    }

    #[test]
    fn an_instruction_with_no_operands_is_one_word() {
        let mut out = Vec::new();
        instruction(&mut out, 253, &[]).expect("one word is well inside the limit");

        assert_eq!(out, vec![(1 << 16) | 253]);
    }

    #[test]
    fn an_instruction_too_long_to_count_is_refused_rather_than_truncated() {
        let operands = vec![0; usize::from(u16::MAX)];
        let mut out = Vec::new();

        let refused = instruction(&mut out, 5, &operands);

        assert_eq!(
            refused,
            Err(EncodeError::InstructionTooLong {
                opcode: 5,
                words: 65_536
            })
        );
        assert!(
            out.is_empty(),
            "a refused instruction leaves nothing behind"
        );
    }

    #[test]
    fn the_longest_encodable_instruction_is_accepted() {
        // One less operand than the test above: exactly u16::MAX words including the opcode.
        let operands = vec![0; usize::from(u16::MAX) - 1];
        let mut out = Vec::new();

        instruction(&mut out, 5, &operands).expect("this is the boundary, not past it");

        assert_eq!(out.len(), usize::from(u16::MAX));
    }

    #[test]
    fn a_string_packs_its_bytes_low_byte_first() {
        let mut operands = Vec::new();
        literal_string(&mut operands, "abc");

        // 'a' = 0x61 lands in the low byte, and the NUL terminator in the high one.
        assert_eq!(operands, vec![0x0063_6261]);
    }

    #[test]
    fn a_string_whose_length_is_a_multiple_of_four_still_gets_a_terminator_word() {
        let mut operands = Vec::new();
        literal_string(&mut operands, "main");

        assert_eq!(operands, vec![0x6e69_616d, 0x0000_0000]);
    }

    #[test]
    fn an_empty_string_is_one_word_of_zeros() {
        let mut operands = Vec::new();
        literal_string(&mut operands, "");

        assert_eq!(operands, vec![0]);
    }

    #[test]
    fn a_strings_predicted_length_matches_what_it_encodes_to() {
        for text in ["", "a", "ab", "abc", "main", "abcde", "GLSL.std.450"] {
            let mut operands = Vec::new();
            literal_string(&mut operands, text);

            assert_eq!(
                literal_string_words(text),
                operands.len(),
                "prediction disagreed with encoding for {text:?}"
            );
        }
    }

    #[test]
    fn a_multibyte_character_is_counted_in_bytes_not_characters() {
        // Three bytes of UTF-8 plus a terminator: one word of text, one of padding-and-NUL.
        let mut operands = Vec::new();
        literal_string(&mut operands, "é€");

        assert_eq!(operands.len(), 2);
        assert_eq!(literal_string_words("é€"), 2);
    }
}
