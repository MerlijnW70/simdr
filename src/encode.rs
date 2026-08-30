use core::fmt;

pub type Word = u32;

pub const MAGIC: Word = 0x0723_0203;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    InstructionTooLong { opcode: u16, words: usize },
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

pub fn instruction(out: &mut Vec<Word>, opcode: u16, operands: &[Word]) -> Result<(), EncodeError> {
    let words = operands.len().saturating_add(1);
    let count =
        u16::try_from(words).map_err(|_| EncodeError::InstructionTooLong { opcode, words })?;

    out.push((Word::from(count) << 16) | Word::from(opcode));
    out.extend_from_slice(operands);
    Ok(())
}

pub fn literal_string(operands: &mut Vec<Word>, text: &str) {
    let mut word: Word = 0;
    let mut filled: u32 = 0;

    for &byte in text.as_bytes() {
        word |= Word::from(byte) << (8 * filled);
        filled += 1;
        if filled == 4 {
            operands.push(word);
            word = 0;
            filled = 0;
        }
    }

    operands.push(word);
}

#[must_use]
pub fn literal_string_words(text: &str) -> usize {
    text.len() / 4 + 1
}

#[cfg(test)]
mod tests {
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
        let operands = vec![0; usize::from(u16::MAX) - 1];
        let mut out = Vec::new();

        instruction(&mut out, 5, &operands).expect("this is the boundary, not past it");

        assert_eq!(out.len(), usize::from(u16::MAX));
    }

    #[test]
    fn a_string_packs_its_bytes_low_byte_first() {
        let mut operands = Vec::new();
        literal_string(&mut operands, "abc");

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
        let mut operands = Vec::new();
        literal_string(&mut operands, "é€");

        assert_eq!(operands.len(), 2);
        assert_eq!(literal_string_words("é€"), 2);
    }
}
