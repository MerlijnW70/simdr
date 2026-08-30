use simdr::{decode, encode};

struct Rng(u64);

impl Rng {
    const fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    const fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

fn instruction(rng: &mut Rng) -> (u16, Vec<u32>) {
    let opcode = rng.below(0xFFFF) as u16;

    let count = rng.below(24) as usize;
    let operands = (0..count).map(|_| rng.next() as u32).collect();

    (opcode, operands)
}

#[test]
fn every_generated_instruction_reads_back_the_way_it_was_written() {
    let mut rng = Rng(0);

    for round in 0..4_096 {
        let (opcode, operands) = instruction(&mut rng);

        let mut words = Vec::new();
        encode::instruction(&mut words, opcode, &operands).expect("fits");

        let mut read = decode::instructions(&words);
        let back = read
            .next()
            .unwrap_or_else(|| panic!("round {round}: nothing decoded"));

        assert_eq!(back.opcode(), opcode, "round {round}");
        assert_eq!(back.operands(), operands.as_slice(), "round {round}");
        assert_eq!(back.word_count(), operands.len() + 1, "round {round}");
        assert!(read.next().is_none(), "round {round}: trailing instruction");
    }
}

#[test]
fn a_stream_of_many_instructions_reads_back_in_order() {
    let mut rng = Rng(1);
    let written: Vec<(u16, Vec<u32>)> = (0..256).map(|_| instruction(&mut rng)).collect();

    let mut words = Vec::new();
    for (opcode, operands) in &written {
        encode::instruction(&mut words, *opcode, operands).expect("fits");
    }

    let read: Vec<(u16, Vec<u32>)> = decode::instructions(&words)
        .map(|instruction| (instruction.opcode(), instruction.operands().to_vec()))
        .collect();

    assert_eq!(read, written);
}

#[test]
fn a_literal_string_occupies_exactly_the_words_it_claims_to() {
    for text in [
        "",
        "a",
        "ab",
        "abc",
        "abcd",
        "abcde",
        "main",
        "local_id",
        "workgroup_id",
        "a rather longer name than anything this crate emits",
    ] {
        let mut operands = Vec::new();
        encode::literal_string(&mut operands, text);

        assert_eq!(
            operands.len(),
            encode::literal_string_words(text),
            "{text:?} was sized wrongly"
        );

        let bytes: Vec<u8> = operands
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let end = bytes
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or_else(|| panic!("{text:?} was not terminated"));

        assert_eq!(
            std::str::from_utf8(bytes.get(..end).unwrap_or_default()),
            Ok(text),
            "{text:?} did not come back"
        );
    }
}

#[test]
fn a_truncated_stream_stops_rather_than_reading_past_the_end() {
    let mut words = Vec::new();
    encode::instruction(&mut words, 21, &[32, 0]).expect("fits");

    for length in 0..words.len() {
        let short = words.get(..length).unwrap_or_default();
        let count = decode::instructions(short).count();
        assert!(count <= 1, "a {length}-word stream yielded {count}");
    }
}

#[test]
fn an_instruction_too_long_to_encode_is_refused_rather_than_truncated() {
    let mut words = vec![0xDEAD_BEEF];
    let operands = vec![0_u32; 0x1_0000];

    assert!(encode::instruction(&mut words, 21, &operands).is_err());
    assert_eq!(words, vec![0xDEAD_BEEF], "the stream was written to anyway");
}
