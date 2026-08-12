//! What a chain runs, and which end of the pair each pass reads.
//!
//! Split from [`super::chain`], which records and submits. This file decides *what* to record and
//! contains no `unsafe`, which is the difference that matters: `tests/integrity.rs` excuses the
//! FFI half of `runner` from mutation because a mutant there kills the process rather than failing
//! a test, and a buffer pair chosen the wrong way round is not that — it is a wrong answer, and it
//! belongs inside the gate.
//!
//! # The ping-pong
//!
//! Every kernel this crate emits binds buffer 0 read and buffer 1 written, baked into the module.
//! Two device buffers are enough to chain any number of passes: pass 0 reads A and writes B, pass 1
//! reads B and writes A, and so on. The *module* never knows — only the descriptor set the pipeline
//! was built with changes.
//!
//! This replaced a device-to-device copy of B back into A after every pass. That copy was measured
//! at 22% of a held reduction over 2²⁰ elements, of which only a third was the data and two thirds
//! were the pair of pipeline barriers around it. Alternating removes the copy, and one of the two
//! barriers with it.
//!
//! **It is not the speed-up that predicted.** Removing the second barrier saved about 2 µs of the
//! 19 the pair cost, so paired against the old build on the same machine there is no measurable
//! difference on an RTX 4080 or on lavapipe — and **5.5%** on the integrated Radeon, where
//! bandwidth is scarce enough for 4 MB of copying to show. It is kept for being simpler and never
//! slower, not for being faster. `notes/FINDINGS.md` has the runs.
//!
//! **The price is that the answer moves.** A chain of an odd number of passes leaves it in B and an
//! even number leaves it in A, which is [`answer_in_destination`] and is the only arithmetic here
//! that a caller can get wrong.

/// One dispatch of a chain.
#[derive(Debug, Clone, Copy)]
pub struct Pass<'words> {
    /// The module to run.
    pub spirv: &'words [u32],
    /// How many workgroups of it.
    pub workgroups: u32,
}

impl<'words> Pass<'words> {
    /// A pass running `workgroups` groups of `spirv`.
    ///
    /// Each pass reads what the one before it wrote. **A pass must write everything the pass after
    /// it reads** — there is no copy filling the rest in, so a region this pass leaves alone holds
    /// what the pass *two* before it put there rather than what the one before it did. For a
    /// halving fold that is automatic: a pass folding `2h` into `h` writes all `h` that the next
    /// one reads.
    #[must_use]
    pub const fn new(spirv: &'words [u32], workgroups: u32) -> Self {
        Self { spirv, workgroups }
    }
}

/// Which way round a pass has its two buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ends {
    /// Reads the source, writes the destination.
    Forward,
    /// Reads the destination, writes the source.
    Back,
}

impl Ends {
    /// How pass `index` is bound.
    ///
    /// Pass zero reads the buffer the host filled, so it is [`Ends::Forward`]; every pass after it
    /// swaps.
    pub(crate) const fn of(index: usize) -> Self {
        if index.is_multiple_of(2) {
            Self::Forward
        } else {
            Self::Back
        }
    }

    /// The two buffers in binding order — 0 read, 1 written — given the source and the destination.
    pub(crate) const fn order<T: Copy>(self, source: T, destination: T) -> (T, T) {
        match self {
            Self::Forward => (source, destination),
            Self::Back => (destination, source),
        }
    }
}

/// Whether a chain of `passes` leaves its answer in the destination buffer.
///
/// Pass `i` writes the destination when `i` is even, so the last pass — `passes - 1` — writes it
/// exactly when `passes` is odd.
///
/// **A chain of none is `false`**, and that is not a degenerate case being tidied away: nothing ran,
/// so the answer is the input, and the input is in the source.
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
        // The property the whole scheme rests on, asserted directly rather than inferred from the
        // pattern above: pass i's *write* end is pass i+1's *read* end, for every i.
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
        // Derived independently of `answer_in_destination`, so the two have to agree rather than
        // being the same expression twice.
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
}
