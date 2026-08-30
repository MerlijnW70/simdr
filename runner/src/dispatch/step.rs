#[derive(Debug, Clone, Copy)]
pub struct Pass<'words> {
    pub spirv: &'words [u32],
    pub workgroups: u32,
}

impl<'words> Pass<'words> {
    #[must_use]
    pub const fn new(spirv: &'words [u32], workgroups: u32) -> Self {
        Self { spirv, workgroups }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ends {
    Forward,
    Back,
}

impl Ends {
    pub(crate) const fn of(index: usize) -> Self {
        if index.is_multiple_of(2) {
            Self::Forward
        } else {
            Self::Back
        }
    }

    pub(crate) const fn order<T: Copy>(self, source: T, destination: T) -> (T, T) {
        match self {
            Self::Forward => (source, destination),
            Self::Back => (destination, source),
        }
    }
}

pub(crate) const fn upload_bytes(host_writable: bool, words: usize) -> Option<u64> {
    if host_writable {
        return None;
    }

    let words = if words == 0 { 1 } else { words };
    Some((words * size_of::<u32>()) as u64)
}

pub(crate) const fn answer_in_destination(passes: usize) -> bool {
    passes % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_pass_reads_what_the_host_filled() {
        assert_eq!(Ends::of(0), Ends::Forward);
        assert_eq!(Ends::of(0).order('a', 'b'), ('a', 'b'));
    }

    #[test]
    fn every_pass_after_it_swaps() {
        let bound: Vec<(char, char)> = (0..5).map(|i| Ends::of(i).order('a', 'b')).collect();

        assert_eq!(
            bound,
            vec![('a', 'b'), ('b', 'a'), ('a', 'b'), ('b', 'a'), ('a', 'b')],
            "a pass must read what the one before it wrote"
        );
    }

    #[test]
    fn what_one_pass_writes_is_what_the_next_reads() {
        for index in 0..8_usize {
            let (_, written) = Ends::of(index).order("source", "destination");
            let (read, _) = Ends::of(index + 1).order("source", "destination");

            assert_eq!(written, read, "between pass {index} and {}", index + 1);
        }
    }

    #[test]
    fn no_pass_reads_and_writes_the_same_buffer() {
        for index in 0..8_usize {
            let (read, written) = Ends::of(index).order("source", "destination");
            assert_ne!(read, written, "pass {index}");
        }
    }

    #[test]
    fn the_answer_is_wherever_the_last_pass_wrote() {
        for passes in 1..10_usize {
            let (_, written) = Ends::of(passes - 1).order("source", "destination");
            let expected = written == "destination";

            assert_eq!(answer_in_destination(passes), expected, "{passes} passes");
        }
    }

    #[test]
    fn an_odd_chain_ends_in_the_destination_and_an_even_one_in_the_source() {
        assert!(answer_in_destination(1));
        assert!(!answer_in_destination(2));
        assert!(answer_in_destination(15), "the 2^20 reduction");
        assert!(!answer_in_destination(8), "the 8192 one");
    }

    #[test]
    fn a_chain_of_none_leaves_the_answer_where_the_host_put_it() {
        assert!(!answer_in_destination(0));
    }

    #[test]
    fn a_host_writable_source_leaves_nothing_to_copy() {
        assert_eq!(upload_bytes(true, 1024), None);
        assert_eq!(upload_bytes(true, 0), None);
    }

    #[test]
    fn a_staged_upload_copies_one_word_per_word() {
        assert_eq!(upload_bytes(false, 1), Some(4));
        assert_eq!(upload_bytes(false, 1024), Some(4096));
        assert_eq!(upload_bytes(false, 1 << 20), Some(4 << 20));
    }

    #[test]
    fn an_empty_staged_upload_copies_one_word_rather_than_none() {
        assert_eq!(upload_bytes(false, 0), Some(4));
    }

    #[test]
    fn the_two_ends_of_a_chain_are_decided_independently() {
        for passes in 0..8_usize {
            for words in [0_usize, 1, 4096] {
                assert_eq!(upload_bytes(true, words), None, "{passes} {words}");
                assert!(upload_bytes(false, words).is_some(), "{passes} {words}");
            }
        }
    }
}
