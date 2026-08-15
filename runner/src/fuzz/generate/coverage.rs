//! Whether the generator explores, reaches its operands, and runs.
//!
//! A `#[cfg(test)]` module of its own because it answers a different question from the tests
//! beside `generate`. Those ask whether a generated program is *valid*; these ask whether the
//! generator is doing its job at all — and five real gaps came out of that distinction in one
//! night.
//!
//! Each is a separate assertion because each was invisible from the others:
//!
//! - **It must explore.** A degraded random stream produces the same few programs forever and
//!   still reports thousands of agreements.
//! - **It must reach its operands.** Every operation appearing says nothing about what is inside
//!   them; the butterfly only ever used distance zero and nothing noticed.
//! - **It must run.** A program dispatching zero workgroups agrees with everything, and the sweep
//!   counts the round as checked.
//! - **Its rules hold both ways.** A narrow program having no shuffle was asserted; a wide program
//!   having one was not, so treating every vector as clustered stayed green.

use super::{Domain, Finish, Rng, generate};
use crate::fuzz::{ALL_DOMAINS, BitShift, Op};

/// The finish that carries a value out of a branch, on the mapping that is most of the sweep.
///
/// **A coverage assertion the mutation gate asked for.** `Finish::SumOrMax` is withheld from
/// clustered programs, and the comparison that withholds it was one character away from withholding
/// it from *whole-subgroup* programs too — every test passed, because a program without it still
/// builds, still runs and still agrees. What is lost is the `OpPhi` across a branch, which this
/// module's header names as the failure no other layer catches.
///
/// So the claim is made where it can fail: over a sweep of seeds, a vector exactly the subgroup's
/// width reaches that finish.
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

/// Every operation the generator knows about, reached across a sweep of seeds.
///
/// The gap a mutation run found: replacing the `^` in the generator's finaliser with `&`
/// degrades the random stream badly — an AND biases hard toward zero — and the whole suite
/// stayed green, because nothing anywhere asserted that the fuzzer *explores*. A fuzzer that
/// generates the same three programs forever still reports thousands of agreements.
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
                // Named apart rather than counted as one "shift". The two right shifts agree on
                // every value whose top bit is clear, so a run that only ever drew one of them
                // would look identical to a run that drew both — the coverage question this
                // module exists to ask.
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
            "repeat",
            "rolled",
            "rotate",
            "shift",
            "shift-down",
        ],
        "the generator never produced some of its own vocabulary in 512 seeds"
    );
    // **`abs` and `fma` are absent from that list on purpose.** This sweep is over
    // `Domain::Unsigned`, which has neither — a magnitude needs `Signed` and a fused multiply-add
    // needs `F32`. The list is what *this* domain can draw, and the test below is what checks the
    // ones it cannot.
}

/// Every element-gated operation is reached in a domain that has it.
///
/// The companion to the sweep above, which is over one domain and would go on passing if the second
/// axis stopped offering anything at all. Each of these is drawn only where its element type allows
/// it, so each needs a domain of its own to be seen in — and a gate that silently narrowed to
/// nothing looks exactly like a vocabulary that was never widened.
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

    // And the negatives, which are the half that fails quietly. An unsigned element has no
    // magnitude and every domain but one has no fused multiply-add; offering either builds a module
    // `spirv-val` rejects, and the sweep would report it as a refusal rather than as a mistake.
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

/// The `all` vote fires, and does not fire always.
///
/// **The operand is the whole test.** `AddIfAnyAbove` draws its threshold straddling the corpus so
/// both arms are reached; an `all` vote over a few hundred distinct elements passes far less often
/// at the same threshold, so this one draws from the bottom of the range. A step that never fires is
/// an identity, and an identity agrees with every reference including a wrong one — while a step
/// that *always* fires is an unconditional add, which is `AddConstant` with extra instructions.
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

/// The bounds of a generated clamp are in order.
///
/// Not a coverage question but the same shape of one: `*Clamp` with its bounds crossed is
/// undefined, so a generator that ever drew `high` below `low` would produce rounds whose
/// "disagreement" is the specification's silence rather than a bug. The relationship is a rule the
/// operand drawing has to keep, and nothing else here would notice it breaking.
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

/// A whole-subgroup program *does* get the subgroup-wide operations.
///
/// The mirror of the test above it, and the half that was missing. `lanes < subgroup` mutated
/// to `<=` makes a full-width vector count as clustered, so the generator stops offering it
/// shuffles and votes — on the one mapping those matter most. The negative test still passed,
/// because it only asks that *narrow* programs have none.
///
/// A rule worth testing in one direction is usually worth testing in both.
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

/// Every generated program actually runs something.
///
/// The third gap of this family, and the most embarrassing: a program dispatching **zero**
/// workgroups computes nothing, the reference computes nothing, and two empty answers agree.
/// The sweep counts that as a round checked. `groups: 1 + rng.below(2)` mutated to `1 -` makes
/// half of every run test nothing at all and report success.
///
/// A fuzzer must explore, must reach its own operands, and must *run*. Each of those had to be
/// said separately, because passing the first two says nothing about the third.
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

/// Which butterfly distances the generator reaches.
///
/// The second gap of this shape a mutation run found. `1 << rng.below(4)` mutated to `>>`
/// gives a mask of 0 or 1 forever — a butterfly of distance 0 pairs a lane with itself, which
/// is a valid program the reference agrees with, so everything stayed green while the fuzzer
/// stopped exercising the shuffle at all.
///
/// Reaching every *operation* is not the same as reaching every operation's *operands*.
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

/// And never a distance that leaves the subgroup.
///
/// The bug a third device found. `1 << below(4)` gives 8, which is inside a 32-wide subgroup and
/// is the *width* of an 8-wide one — and a shuffle across the boundary is undefined, so the
/// fuzzer reported a disagreement that was its own.
///
/// Checked at every width the dispatcher can build for, because the rule is about the relationship
/// and not about any one of them.
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

/// The same for how a program ends.
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

/// `Rng` is SplitMix64, and here is what SplitMix64 produces.
///
/// A golden test, which is usually a smell and is right here for two reasons.
///
/// **It is a specified algorithm.** SplitMix64 has published constants and a published output
/// sequence, so pinning it states something true and checkable rather than freezing an arbitrary
/// internal choice. These four values were computed from the published formula
/// — `z = (x += γ); z = (z ^ z>>30) * C₁; z = (z ^ z>>27) * C₂; return z ^ z>>31` — in a separate
/// implementation and agree with the sequence the reference publishes for seed zero. Two
/// transcriptions agreeing is the check; neither was read off this file.
///
/// **The seeds mean something.** `notes/FINDINGS.md` records which seed found which bug. Change
/// the generator and every one of those becomes a number pointing at a different program. That
/// should be a failing test rather than a silent loss.
///
/// Four mutations of the finaliser's inner rounds survived every distribution test written for it:
/// the algorithm is good enough that mangling one round still spreads three bits over eight
/// thousand draws. Distribution tests catch a *collapsed* generator. Only this catches a
/// *different* one.
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
        "this is no longer SplitMix64, and every seed recorded in notes/FINDINGS.md now points \
         at a different program"
    );
}

/// The stream itself, rather than what is built from it.
///
/// A generator whose values cluster produces a corpus that looks large and covers little.
/// Deliberately loose — this is not a statistical test of `SplitMix64`, it is a check that the
/// finaliser was not replaced by something that throws bits away.
///
/// # The first version of this test looked at the wrong end of the word
///
/// It bucketed by `next() >> 61` — the *top* three bits. Every consumer of this generator goes
/// through `below(n)`, which is `next() % n`, and a modulus reads the **low** bits. So the test
/// was checking bits nobody uses, and four mutations of the finaliser's inner rounds sailed
/// straight past it while degrading exactly the half that matters.
///
/// Both ends are checked now, and the middle by way of `below` itself.
#[test]
fn the_random_stream_spreads_across_its_range() {
    /// A tenth of even coverage: far below what a working generator gives, far above a broken one.
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

    // And the thing every caller actually uses. `below(5)` exercises a modulus that is not a power
    // of two, which is the case a shift-flavoured mutation distorts most.
    let mut rng = Rng::new(1);
    let mut drawn = [0_u32; 5];
    for _ in 0..DRAWS {
        if let Some(bucket) = drawn.get_mut(rng.below(5) as usize) {
            *bucket += 1;
        }
    }
    spread_enough(&drawn, DRAWS, "the values `below(5)` returns");
}

/// The rotate is reached at **both** widths that allow it, not only the narrow one.
///
/// The mutation gate found this too: sending `lanes == subgroup` down the strip-mined pool changed
/// nothing any test could see, because every program still built and still agreed — it only stopped
/// the rotate ever being generated for a subgroup-wide vector. A pool that quietly loses an
/// operation looks exactly like a fuzzer that keeps agreeing.
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
        // A rotate over a strip-mined vector is refused by the lane API, so the generator must not
        // draw one. Asserted rather than panicked: this crate's lints keep a bare `panic!` out of
        // everything but the harness, and a failing claim is an assertion.
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

/// A float domain never gets a bit shift, and every integer one does.
///
/// **Both halves, because the rule has two directions and only one of them is obvious.** A
/// generator that never offered the shifts at all would pass the negative half forever, which is
/// the state this file's header describes: a fuzzer that stops generating a shape looks exactly
/// like a fuzzer that keeps agreeing.
///
/// The positive half is the one that would have caught the mistake actually worth making — gating
/// on the wrong side of `is_float`, which builds, runs, and quietly moves the whole extension into
/// the two domains that cannot use it.
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

/// A generated shift stays inside its element's own width, and reaches the top of it.
///
/// Two claims about one draw, and they pull against each other on purpose. SPIR-V leaves a shift by
/// at least the operand's width **undefined**, so the range has a ceiling the specification fixes —
/// and the interesting half of the range is right below that ceiling, because `OpShiftRightLogical`
/// and `OpShiftRightArithmetic` agree on every value whose top bit is clear. A draw that played it
/// safe with small amounts would satisfy the first claim and make the two right shifts
/// indistinguishable, which is the `ButterflyAdd` mistake read backwards.
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

/// The left shift puts a bit at the top, and the right shifts then disagree about it.
///
/// The claim the whole extension rests on, made against `Domain` rather than against a device: for
/// every integer domain there is a value and a distance where `OpShiftRightLogical` and
/// `OpShiftRightArithmetic` give different answers. Without it the two instructions are one
/// instruction with two names, and generating both proves nothing.
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
