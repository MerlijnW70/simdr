use super::{Domain, Finish, Rng, generate};
use crate::fuzz::{ALL_DOMAINS, BitShift, Op};

#[test]
fn a_whole_subgroup_program_reaches_the_finish_that_carries_a_phi() {
    let reached = (0..512_u64).any(|seed| {
        let program = generate(&mut Rng::new(seed), Domain::Unsigned, 32, 64);
        program.lanes == 32 && matches!(program.finish, Finish::SumOrMax { .. })
    });

    assert!(
        reached,
        "no whole-subgroup program in 512 seeds ends in `SumOrMax`, so nothing generated here \
         carries a value out of a branch on the mapping the sweep spends most of its time in"
    );
}

/// Draws unsigned programs only, so the float-only operations cannot appear
/// here; [`the_generator_reaches_the_operations_only_a_float_has`] covers those.
#[test]
fn the_generator_reaches_every_operation_it_knows() {
    let mut seen: Vec<&'static str> = Vec::new();
    for seed in 0..512_u64 {
        let program = generate(&mut Rng::new(seed), Domain::Unsigned, 32, 64);
        for step in &program.steps {
            let name = match step {
                Op::AddConstant(_) => "add",
                Op::MulConstant(_) => "mul",
                Op::ClampBelow(_) => "clamp",
                Op::MinConstant(_) => "min",
                Op::MaxConstant(_) => "max",
                Op::ClampBoth { .. } => "clamp-both",
                Op::RepeatAdd { .. } => "repeat",
                Op::RolledAdd { .. } => "rolled",
                Op::RolledCounterAdd { .. } => "counter",
                Op::ButterflyAdd(_) => "butterfly",
                Op::AddIfAnyAbove { .. } => "branch",
                Op::AddIfAllEqual { .. } => "agree",
                Op::SelectEqual { .. } => "equal",
                Op::RotateUp(_) => "rotate",
                Op::ShiftUp => "shift",
                Op::ShiftDown => "shift-down",
                Op::BroadcastLane(_) => "broadcast",
                Op::BitShift {
                    kind: BitShift::Left,
                    ..
                } => "bit-left",
                Op::BitShift {
                    kind: BitShift::RightLogical,
                    ..
                } => "bit-right-logical",
                Op::BitShift {
                    kind: BitShift::RightArithmetic,
                    ..
                } => "bit-right-arithmetic",
                Op::Absolute => "abs",
                Op::FusedMulAdd { .. } => "fma",
                Op::AddIfAllAbove { .. } => "all-above",
                Op::SubConstant(_) => "sub",
                Op::SaturatingAddConstant(_) => "saturating-add",
                Op::SaturatingSubConstant(_) => "saturating-sub",
                Op::AndConstant(_) => "and",
                Op::OrConstant(_) => "or",
                Op::XorConstant(_) => "xor",
                Op::NotValue => "not",
                Op::Floor => "floor",
                Op::Ceil => "ceil",
                Op::Trunc => "trunc",
            };
            if !seen.contains(&name) {
                seen.push(name);
            }
        }
    }

    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![
            "add",
            "agree",
            "all-above",
            "and",
            "bit-left",
            "bit-right-arithmetic",
            "bit-right-logical",
            "branch",
            "broadcast",
            "butterfly",
            "clamp",
            "clamp-both",
            "counter",
            "equal",
            "max",
            "min",
            "mul",
            "not",
            "or",
            "repeat",
            "rolled",
            "rotate",
            "saturating-add",
            "saturating-sub",
            "shift",
            "shift-down",
            "sub",
            "xor",
        ],
        "the generator never produced some of its own vocabulary in 512 seeds"
    );
}

#[test]
fn every_element_gated_operation_is_reached_where_it_exists() {
    let reaches = |domain: Domain, wanted: fn(&Op) -> bool| {
        (0..512_u64)
            .flat_map(|seed| generate(&mut Rng::new(seed), domain, 32, 64).steps)
            .any(|step| wanted(&step))
    };

    assert!(
        reaches(Domain::Signed, |step| matches!(step, Op::Absolute)),
        "no signed program in 512 seeds took a magnitude"
    );
    assert!(
        reaches(Domain::Half, |step| matches!(step, Op::Absolute)),
        "no half program in 512 seeds took a magnitude, and `f16` is a `Signed` element"
    );
    assert!(
        reaches(Domain::Float, |step| matches!(step, Op::FusedMulAdd { .. })),
        "no float program in 512 seeds fused a multiply and an add"
    );

    for domain in [
        Domain::Unsigned,
        Domain::UnsignedByte,
        Domain::UnsignedShort,
    ] {
        assert!(
            !reaches(domain, |step| matches!(step, Op::Absolute)),
            "{domain:?} was offered a magnitude and `Lanes::abs` takes `T: Signed`"
        );
    }
    for domain in ALL_DOMAINS {
        if matches!(domain, Domain::Float) {
            continue;
        }
        assert!(
            !reaches(domain, |step| matches!(step, Op::FusedMulAdd { .. })),
            "{domain:?} was offered a fused multiply-add and `Lanes::fma` takes `F32`"
        );
    }
}

#[test]
fn the_all_vote_is_drawn_low_enough_to_pass_and_high_enough_to_fail() {
    let mut lowest = u32::MAX;
    let mut highest = 0;
    let mut seen = 0;

    for seed in 0..512_u64 {
        for step in generate(&mut Rng::new(seed), Domain::Unsigned, 32, 64).steps {
            if let Op::AddIfAllAbove { when_all_above, .. } = step {
                lowest = lowest.min(when_all_above);
                highest = highest.max(when_all_above);
                seen += 1;
            }
        }
    }

    assert!(seen > 0, "the `all` vote was never generated at all");
    assert_eq!(
        lowest, 0,
        "no threshold was drawn that every element of the corpus clears, so the vote never passes          and the step is an identity"
    );
    assert!(
        highest > 0,
        "every threshold drawn was zero, so the vote always passes and the step is `AddConstant`          with a vote in front of it"
    );
}

#[test]
fn a_generated_clamp_never_has_its_bounds_crossed() {
    let mut seen = 0;
    for seed in 0..512_u64 {
        for domain in [Domain::Unsigned, Domain::Signed, Domain::Float] {
            for step in &generate(&mut Rng::new(seed), domain, 32, 64).steps {
                if let Op::ClampBoth { low, high } = *step {
                    assert!(low <= high, "seed {seed} drew [{low}, {high}]");
                    seen += 1;
                }
            }
        }
    }

    assert!(
        seen > 0,
        "no clamp was generated at all, so this checked nothing"
    );
}

#[test]
fn a_whole_subgroup_program_does_ask_for_shuffles_and_votes() {
    let mut reached = false;
    for seed in 0..512_u64 {
        let program = generate(&mut Rng::new(seed), Domain::Float, 32, 64);
        if program.lanes != 32 {
            continue;
        }
        if program.steps.iter().any(|step| {
            matches!(
                step,
                Op::ButterflyAdd(_) | Op::ShiftUp | Op::AddIfAnyAbove { .. }
            )
        }) {
            reached = true;
            break;
        }
    }

    assert!(
        reached,
        "no 32-lane program in 512 seeds got a shuffle or a vote, \
         so the whole-subgroup mapping is being treated as clustered"
    );
}

#[test]
fn every_generated_program_dispatches_and_reads_something() {
    for seed in 0..512_u64 {
        let program = generate(&mut Rng::new(seed), Domain::Unsigned, 32, 64);

        assert!(
            program.workgroups() >= 1,
            "seed {seed} dispatches {} workgroups",
            program.workgroups()
        );
        assert!(
            program.input_len() >= 1,
            "seed {seed} reads {} elements",
            program.input_len()
        );
    }
}

#[test]
fn the_generator_reaches_every_butterfly_distance() {
    let mut seen: Vec<u32> = Vec::new();
    for seed in 0..512_u64 {
        let program = generate(&mut Rng::new(seed), Domain::Unsigned, 32, 64);
        for step in &program.steps {
            if let Op::ButterflyAdd(mask) = *step
                && !seen.contains(&mask)
            {
                seen.push(mask);
            }
        }
    }

    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![1, 2, 4, 8],
        "the butterfly should pair lanes 1, 2, 4 and 8 apart, and reached {seen:?}"
    );
}

#[test]
fn no_generated_butterfly_reaches_outside_its_subgroup() {
    for width in [4_u32, 8, 16, 32, 64] {
        let mut reached = 0;
        for seed in 0..256_u64 {
            let program = generate(&mut Rng::new(seed), Domain::Unsigned, width, 64);
            for step in &program.steps {
                if let Op::ButterflyAdd(mask) = *step {
                    assert!(
                        mask < width,
                        "a subgroup of {width} was given a butterfly of {mask}, \
                         which pairs a lane with one in the next subgroup"
                    );
                    reached += 1;
                }
            }
        }

        assert!(
            reached > 0,
            "no butterfly at all was generated for a subgroup of {width}, \
             so this checked nothing there"
        );
    }
}

#[test]
fn the_generator_reaches_every_finish() {
    let mut sums = 0;
    let mut maxes = 0;
    let mut mins = 0;
    let mut chosen = 0;
    let mut scans = 0;
    let mut exclusive = 0;

    for seed in 0..512_u64 {
        match generate(&mut Rng::new(seed), Domain::Float, 32, 64).finish {
            Finish::Sum => sums += 1,
            Finish::Max => maxes += 1,
            Finish::Min => mins += 1,
            Finish::SumOrMax { .. } => chosen += 1,
            Finish::Scan => scans += 1,
            Finish::ScanExclusive => exclusive += 1,
        }
    }

    for (name, count) in [
        ("sum", sums),
        ("max", maxes),
        ("min", mins),
        ("sum-or-max", chosen),
        ("scan", scans),
        ("exclusive scan", exclusive),
    ] {
        assert!(count > 0, "no program in 512 seeds finished with {name}");
    }
}

#[test]
fn the_generator_is_splitmix64() {
    let mut rng = Rng::new(0);
    let drawn: Vec<u64> = (0..4).map(|_| rng.next()).collect();

    assert_eq!(
        drawn,
        vec![
            0xE220_A839_7B1D_CDAF,
            0x6E78_9E6A_A1B9_65F4,
            0x06C4_5D18_8009_454F,
            0xF88B_B8A8_724C_81EC,
        ],
        "this is no longer SplitMix64, and every seed recorded against it now points \
         at a different program"
    );
}

#[test]
fn the_random_stream_spreads_across_its_range() {
    fn spread_enough(buckets: &[u32], draws: u32, what: &str) {
        let floor = draws / buckets.len() as u32 / 10;
        assert!(
            buckets.iter().all(|&count| count > floor),
            "{what} do not spread (floor {floor}): {buckets:?}"
        );
    }

    const DRAWS: u32 = 8_192;

    let mut rng = Rng::new(0);
    let mut high = [0_u32; 8];
    let mut low = [0_u32; 8];
    for _ in 0..DRAWS {
        let value = rng.next();
        if let Some(bucket) = high.get_mut((value >> 61) as usize) {
            *bucket += 1;
        }
        if let Some(bucket) = low.get_mut((value & 0b111) as usize) {
            *bucket += 1;
        }
    }
    spread_enough(&high, DRAWS, "the top three bits");
    spread_enough(&low, DRAWS, "the low three bits");

    let mut rng = Rng::new(1);
    let mut drawn = [0_u32; 5];
    for _ in 0..DRAWS {
        if let Some(bucket) = drawn.get_mut(rng.below(5) as usize) {
            *bucket += 1;
        }
    }
    spread_enough(&drawn, DRAWS, "the values `below(5)` returns");
}

#[test]
fn the_rotate_is_generated_for_a_clustered_vector_and_for_a_whole_one() {
    let mut clustered = 0;
    let mut whole = 0;

    for seed in 0..512_u64 {
        let program = generate(&mut Rng::new(seed), Domain::Unsigned, 32, 64);
        if !program
            .steps
            .iter()
            .any(|step| matches!(step, Op::RotateUp(_)))
        {
            continue;
        }
        assert!(
            program.lanes <= 32,
            "seed {seed} put a rotate in a strip-mined program: {program:?}"
        );
        if program.lanes < 32 {
            clustered += 1;
        } else {
            whole += 1;
        }
    }

    assert!(clustered > 0, "no clustered rotate in 512 seeds");
    assert!(
        whole > 0,
        "no rotate over a subgroup-wide vector in 512 seeds — the pool for `lanes == subgroup` \
         is not being reached"
    );
}

#[test]
fn the_bit_shifts_reach_every_integer_domain_and_no_float_one() {
    for domain in ALL_DOMAINS {
        let shifted = (0..512_u64)
            .flat_map(|seed| generate(&mut Rng::new(seed), domain, 32, 64).steps)
            .filter(|step| matches!(step, Op::BitShift { .. }))
            .count();

        if domain.is_float() {
            assert_eq!(
                shifted, 0,
                "{domain:?} was offered a bit shift, and `spirv-val` rejects the module it builds"
            );
        } else {
            assert!(
                shifted > 0,
                "no program in 512 seeds shifted bits in {domain:?}, so the domain has the \
                 instruction and nothing generated reaches it"
            );
        }
    }
}

#[test]
fn a_generated_shift_stays_inside_the_element_and_reaches_its_top() {
    for domain in ALL_DOMAINS {
        if domain.is_float() {
            continue;
        }

        let mut furthest = 0;
        for seed in 0..512_u64 {
            for step in &generate(&mut Rng::new(seed), domain, 32, 64).steps {
                if let Op::BitShift { by, .. } = *step {
                    assert!(
                        by < domain.bits(),
                        "{domain:?} drew a shift of {by} into a {}-bit element, which SPIR-V \
                         leaves undefined",
                        domain.bits()
                    );
                    furthest = furthest.max(by);
                }
            }
        }

        assert_eq!(
            furthest,
            domain.bits() - 1,
            "{domain:?} never drew a shift to the top of its element, so nothing generated here \
             puts a bit where the two right shifts differ"
        );
    }
}

#[test]
fn the_two_right_shifts_disagree_once_the_top_bit_is_set() {
    for domain in ALL_DOMAINS {
        if domain.is_float() {
            continue;
        }

        let top = domain.bits() - 1;
        let raised = domain.bit_shift(BitShift::Left, 1, top);
        let logical = domain.bit_shift(BitShift::RightLogical, raised, top);
        let arithmetic = domain.bit_shift(BitShift::RightArithmetic, raised, top);

        assert_eq!(
            logical, 1,
            "{domain:?} filled a logical shift with something other than zeros"
        );
        assert_ne!(
            logical, arithmetic,
            "{domain:?} gives the same answer to both right shifts even with the top bit set, so \
             the corpus can never tell the two instructions apart"
        );
    }
}

#[test]
fn the_generator_reaches_the_operations_only_a_float_has() {
    let mut seen: Vec<&'static str> = Vec::new();
    for seed in 0..512_u64 {
        let program = generate(&mut Rng::new(seed), Domain::Float, 32, 64);
        for step in &program.steps {
            let name = match step {
                Op::Absolute => "abs",
                Op::FusedMulAdd { .. } => "fma",
                Op::Floor => "floor",
                Op::Ceil => "ceil",
                Op::Trunc => "trunc",
                _ => continue,
            };
            if !seen.contains(&name) {
                seen.push(name);
            }
        }
    }

    seen.sort_unstable();
    assert_eq!(
        seen,
        vec!["abs", "ceil", "floor", "fma", "trunc"],
        "the float pool names operations the generator never draws, so the sweep would never \
         reach them and nothing else here would say so"
    );
}
