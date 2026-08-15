//! Random lane programs, run on the device and checked against the CPU reference.
//!
//! The other test files check things somebody thought of. This one does not: it generates
//! programs from a seed, works out what each must return by interpreting the same program on the
//! CPU, and compares. A disagreement is a real finding wherever it lands — the emitter, the
//! mapping, the reference, or the driver.
//!
//! Seeds are fixed, so a failure names the seed that produced it and re-running reproduces it
//! exactly. `SIMDR_FUZZ_ROUNDS` searches harder.
//!
//! # Why the gates here ask for the whole surface
//!
//! Every other file asks the *module* what it needs — `common::runnable` reads its `OpCapability`
//! list, so a gate cannot name the wrong feature. That works because those tests know which kernel
//! they are about to run. These do not: the program is drawn from a seed, and which capabilities it
//! declares depends on what the draw produced. Gating per round would skip an unknown subset and
//! report a coverage number over whatever was left, which is worse than skipping the sweep.
//!
//! So `Limits::subgroup_surface` — the union of everything a generated program can reach — is the
//! honest gate here, and it is the one place in this suite where a union is the right shape.
//! `shaderSubgroupExtendedTypes` is asked separately for the narrow domains, because no capability
//! in any module can express it.

mod common;

use common::device;
use runner::fuzz::{self, ALL_DOMAINS, Domain, Finish, Op, Outcome, Program};
use runner::kernels::WORKGROUP_SIZE;

/// How many programs each domain checks.
///
/// Small enough to sit in a normal `cargo test` and large enough to cover every mapping several
/// times over.
const ROUNDS: u64 = 256;

/// Input that makes a wrong answer obvious: every element distinct, none of them zero, and all of
/// them inside the domain's exactly-representable range.
///
/// In the domains that have negatives, every fourth element is one. That is the only thing that
/// separates the signed path from the unsigned one — `OpSGreaterThan` and `OpUGreaterThan` agree
/// on every value with the top bit clear — so a corpus of positives would run the signed sweep and
/// prove nothing new.
fn corpus(domain: Domain, len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| {
            let magnitude = index as i32 % domain.ceiling() as i32 + 1;
            if domain.is_signed() && index % 4 == 3 {
                domain.encode_signed(-magnitude)
            } else {
                domain.encode(magnitude as u32)
            }
        })
        .collect()
}

/// How many rounds to run, taking `SIMDR_FUZZ_ROUNDS` if it is set.
///
/// **A value that will not parse is an error rather than the default.** This variable is how a
/// longer search is asked for, and `and_then(|value| value.parse().ok())` sent every misspelling —
/// `100_000`, `1e6`, a stray space — quietly back to 256. Somebody would have watched a 30 000-round
/// sweep they never ran, and the report at the end says how many rounds it *did*, so the two would
/// have had to be read together to notice.
fn rounds() -> u64 {
    let Some(value) = std::env::var_os("SIMDR_FUZZ_ROUNDS") else {
        return ROUNDS;
    };
    let text = value.to_string_lossy();
    text.trim().parse().unwrap_or_else(|error| {
        panic!("SIMDR_FUZZ_ROUNDS is {text:?}, which is not a number of rounds: {error}")
    })
}

/// One program, run to find out whether **this device** miscompiles the clustered ladder.
///
/// An integrated AMD Radeon does. It answers this one wrongly, and — bisected in
/// `notes/FINDINGS.md` — it takes the *process* down inside `vkCreateComputePipelines` for a
/// four-step program whose three-step prefix compiles, whose reduction and maximum forms compile,
/// and which `spirv-val` accepts. A driver that faults compiling a valid module has a defect
/// whatever the module says; an RTX 4080 and lavapipe compile and run every one of them correctly.
///
/// Probed rather than matched on a device name: a driver update that fixes it should restore the
/// coverage without anyone editing a list, and a second device that has it should lose the
/// coverage without anyone noticing it there first. The probe has to be a program the device gets
/// *wrong* rather than one it faults on — nothing can catch a fault to report it.
fn miscompiles_clustered_scans(gpu: &runner::Gpu, limits: &runner::Limits) -> bool {
    // The probe is an 8-bit program, so a device that cannot hold a byte in a buffer cannot be
    // asked. None is known; the alternative would be a probe whose failure mode is the fault.
    if !limits.narrow.byte_kernel() {
        return false;
    }

    let domain = Domain::UnsignedByte;
    let program = Program {
        domain,
        subgroup: limits.subgroup_size,
        workgroup: WORKGROUP_SIZE,
        groups: 1,
        lanes: (limits.subgroup_size / 2).max(1),
        steps: vec![Op::MaxConstant(3)],
        finish: Finish::ScanExclusive,
    };
    // Every value is above the max's operand, so the step is the identity and only the instruction
    // is under test.
    let input: Vec<u32> = (0..program.input_len())
        .map(|index| domain.encode(index as u32 % 9 + 10))
        .collect();

    match fuzz::check(gpu, &program, &input) {
        Ok(Outcome::Disagreed { .. }) => true,
        // Refused or unrepresentable means the probe proved nothing, and neither does an error;
        // in all three the sweeps run as usual and would report a disagreement themselves.
        Ok(_) | Err(_) => false,
    }
}

/// Whether `program` is the shape [`miscompiles_clustered_scans`] found this device getting wrong.
///
/// **Three widenings, and each one was the filter being too clever.** It began as "8-bit, and a
/// max": the probe reproduced with one. Then a `[MinConstant(0)]` in the same shape faulted the
/// process, so it became the 8-bit family; then a 16-bit program disagreed after a rolled loop, so
/// it became every narrow type; then an `f32` program faulted, and the bisection showed the
/// *inclusive* direction faulting too. What survives all four is the ladder itself.
///
/// So the rule is two conditions: a vector narrower than the subgroup, and a scan. Every reduction,
/// shuffle, vote and strip-mined scan in every domain is still checked on this device — the loss is
/// the mapping the driver cannot compile, and it is counted where the report can see it.
fn defective_here(program: &Program) -> bool {
    program.lanes < program.subgroup
        && matches!(program.finish, Finish::Scan | Finish::ScanExclusive)
}

/// What a sweep did, rather than only how many times it agreed.
struct Swept {
    /// Rounds the device and the reference were compared on, and agreed.
    checked: u64,
    /// Rounds the mapping refused to build — a lane count with no mapping, say.
    refused: u64,
    /// Rounds whose arithmetic left the range the domain counts exactly, so nothing was compared.
    unrepresentable: u64,
    /// Of the rounds that agreed, how many ended in a **scan**.
    ///
    /// Counted apart because the guard below — most rounds proved something — is satisfied by the
    /// reductions alone: "the sweep ran" and "the sweep exercised a prefix" are different claims
    /// and only one of them was being made. Every mapping ends up here now, the clustered one
    /// included, where the generator used to offer a reduction instead.
    scans: u64,
    /// Rounds whose clustered scan was replaced by a sum, this device being unable to compile one.
    ///
    /// Zero on every device that passes [`miscompiles_clustered_scans`], which is every device here
    /// but one. Reported rather than hidden: a skip that nobody sees is a coverage loss that looks
    /// like coverage.
    defective: u64,
}

impl Swept {
    /// Report the counts, and insist that most rounds actually compared something.
    ///
    /// **The number that matters is `checked`.** A domain that refused every round, or whose
    /// arithmetic left its exact range every round, is indistinguishable from one that always
    /// agreed if only the failures are reported — and `Domain::Half` makes that a live risk rather
    /// than a hypothetical, because a sum over a few hundred halves leaves 2048 at once.
    fn expect_mostly_checked(&self, domain: Domain, rounds: u64) {
        eprintln!(
            "fuzz {domain:?} over {rounds} seeds: {} agreed of which {} scans, {} refused, {} unrepresentable, {} clustered scans replaced by a sum",
            self.checked, self.scans, self.refused, self.unrepresentable, self.defective
        );
        assert!(
            self.checked > rounds / 2,
            "most {domain:?} rounds proved nothing: {} checked of {rounds}",
            self.checked
        );
        assert!(
            self.scans > 0,
            "no {domain:?} round compared a scan, so the prefix path proved nothing"
        );
    }
}

/// Run `rounds` generated programs in `domain` and report how many were actually checked.
///
/// `defect` says whether this device failed [`miscompiles_clustered_scans`]; when it did, the
/// rounds that would end in a clustered scan **end in a sum instead** rather than being dropped.
/// A third of the seeds generate one, and dropping them would take their steps — the loops, the
/// votes, the narrow conversions — down with the tail this driver cannot compile.
fn sweep(gpu: &runner::Gpu, domain: Domain, subgroup: u32, rounds: u64, defect: bool) -> Swept {
    let mut checked = 0;
    let mut refused = 0;
    let mut unrepresentable = 0;
    let mut scans = 0;
    let mut defective = 0;

    for seed in 0..rounds {
        let mut rng = fuzz::Rng::new(seed);
        let mut program = fuzz::generate(&mut rng, domain, subgroup, WORKGROUP_SIZE);

        if defect && defective_here(&program) {
            defective += 1;
            program.finish = Finish::Sum;
        }

        let input = corpus(domain, program.input_len());
        let is_scan = matches!(
            program.finish,
            fuzz::Finish::Scan | fuzz::Finish::ScanExclusive
        );

        match fuzz::check(gpu, &program, &input).expect("dispatched") {
            Outcome::Agreed => {
                checked += 1;
                scans += u64::from(is_scan);
            }
            Outcome::Refused(_) => refused += 1,
            Outcome::Unrepresentable => unrepresentable += 1,
            Outcome::Disagreed {
                program,
                expected,
                actual,
                at,
            } => {
                panic!(
                    "{domain:?} seed {seed} disagreed at index {at}: expected {}, got {}\n{}",
                    expected.get(at).copied().unwrap_or_default(),
                    actual.get(at).copied().unwrap_or_default(),
                    describe(&program)
                );
            }
        }
    }

    Swept {
        checked,
        refused,
        unrepresentable,
        scans,
        defective,
    }
}

#[test]
fn generated_integer_programs_agree_with_the_cpu_reference() {
    let Some(gpu) = device("fuzz-u32") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_surface() {
        eprintln!("SKIPPED fuzz-u32: the device lacks part of the subgroup surface");
        return;
    }

    let rounds = rounds();
    let defect = miscompiles_clustered_scans(&gpu, &limits);
    let swept = sweep(&gpu, Domain::Unsigned, limits.subgroup_size, rounds, defect);

    swept.expect_mostly_checked(Domain::Unsigned, rounds);
}

/// The same sweep over `f32`.
///
/// Exact comparison, not a tolerance: every value the generator produces is a small integer, and
/// at those magnitudes float arithmetic is exact and therefore order-independent. See
/// `runner/src/fuzz/domain.rs` for the argument, and for what it deliberately does not cover.
#[test]
fn generated_float_programs_agree_with_the_cpu_reference() {
    let Some(gpu) = device("fuzz-f32") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_surface() {
        eprintln!("SKIPPED fuzz-f32: the device lacks part of the subgroup surface");
        return;
    }

    let rounds = rounds();
    let defect = miscompiles_clustered_scans(&gpu, &limits);
    let swept = sweep(&gpu, Domain::Float, limits.subgroup_size, rounds, defect);

    swept.expect_mostly_checked(Domain::Float, rounds);
}

/// The same sweep over `i32`.
///
/// A separate domain rather than a flag on the integer one: the comparison and the extremes reach
/// `OpSGreaterThan` and `OpGroupNonUniformSMax` where the unsigned sweep reaches their `U`
/// counterparts, and the two disagree on exactly the values this corpus supplies.
#[test]
fn generated_signed_programs_agree_with_the_cpu_reference() {
    let Some(gpu) = device("fuzz-i32") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_surface() {
        eprintln!("SKIPPED fuzz-i32: the device lacks part of the subgroup surface");
        return;
    }

    let rounds = rounds();
    let defect = miscompiles_clustered_scans(&gpu, &limits);
    let swept = sweep(&gpu, Domain::Signed, limits.subgroup_size, rounds, defect);

    swept.expect_mostly_checked(Domain::Signed, rounds);
}

/// The five narrow domains, which need three device features the wide ones do not.
///
/// `i8`, `u8`, `i16` and `u16` had direct device tests and no fuzzing at all — the least-checked
/// surface in the tree by this project's own standard, and the one where the two conversions
/// (`OpSConvert` against `OpUConvert`) and the two extremes (`SMax` against `UMax`) are reached by
/// the same source line.
///
/// The buffer is where they differ from everything else here: a stride of one byte means four
/// elements share a word, and `fuzz::check` packs and unpacks at that boundary.
///
/// **`f16` joined them on 2026-08-13.** It had been excluded on the grounds that a half counts
/// integers only to 2048, so a sum over a few hundred lanes leaves the range and a tolerance would
/// be checking the rounding rather than the emitter. That reasoning was right; the conclusion —
/// leave the domain out — was one step too far. A round that leaves the range is *refused* now
/// instead of loosened, so every `Half` round compared is compared exactly, and the ones that
/// cannot be are counted where `expect_mostly_checked` can see them.
#[test]
fn generated_narrow_programs_agree_with_the_cpu_reference() {
    let Some(gpu) = device("fuzz-narrow") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_surface() {
        eprintln!("SKIPPED fuzz-narrow: the device lacks part of the subgroup surface");
        return;
    }
    // The one that leaves no trace in the module: without it a device accepts every one of these
    // programs at validation and refuses the pipeline.
    if !limits.narrow.subgroup_extended_types {
        eprintln!("SKIPPED fuzz-narrow: no shaderSubgroupExtendedTypes");
        return;
    }

    let rounds = rounds();
    let defect = miscompiles_clustered_scans(&gpu, &limits);
    if defect {
        eprintln!(
            "fuzz-narrow: this driver miscompiles the clustered ladder, so those rounds are \
             counted and skipped — see notes/FINDINGS.md"
        );
    }

    for (domain, needed) in [
        (Domain::UnsignedByte, limits.narrow.byte_kernel()),
        (Domain::Byte, limits.narrow.byte_kernel()),
        (Domain::UnsignedShort, limits.narrow.short_kernel()),
        (Domain::Short, limits.narrow.short_kernel()),
        (Domain::Half, limits.narrow.half_kernel()),
    ] {
        if !needed {
            eprintln!("SKIPPED fuzz-narrow {domain:?}: the device cannot hold it in a buffer");
            continue;
        }

        let swept = sweep(&gpu, domain, limits.subgroup_size, rounds, defect);
        swept.expect_mostly_checked(domain, rounds);
    }
}

/// The strip-mined end, which randomness alone reaches rarely.
#[test]
fn strip_mined_programs_agree_in_every_domain() {
    let Some(gpu) = device("fuzz-strips") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_surface() {
        eprintln!("SKIPPED fuzz-strips: the device lacks part of the subgroup surface");
        return;
    }

    // Scans over a strip-mined vector are the case with a carry between strips, which is the one
    // shape of scan that no single instruction covers. Counted so this cannot pass by reaching
    // none of them.
    let mut scanned = 0_u64;

    for domain in ALL_DOMAINS {
        for seed in 0..64_u64 {
            let mut rng = fuzz::Rng::new(seed.wrapping_mul(0x5851_F42D_4C95_7F2D));
            let mut program =
                fuzz::generate(&mut rng, domain, limits.subgroup_size, WORKGROUP_SIZE);

            program.lanes = limits.subgroup_size * (2 + (seed % 3) as u32).min(4);
            let input = corpus(domain, program.input_len());
            let is_scan = matches!(
                program.finish,
                fuzz::Finish::Scan | fuzz::Finish::ScanExclusive
            );

            match fuzz::check(&gpu, &program, &input).expect("dispatched") {
                Outcome::Agreed => scanned += u64::from(is_scan),
                Outcome::Refused(_) | Outcome::Unrepresentable => {}
                Outcome::Disagreed {
                    program,
                    expected,
                    actual,
                    at,
                } => {
                    panic!(
                        "{domain:?} seed {seed} disagreed at index {at}: expected {}, got {}\n{}",
                        expected.get(at).copied().unwrap_or_default(),
                        actual.get(at).copied().unwrap_or_default(),
                        describe(&program)
                    );
                }
            }
        }
    }

    eprintln!("fuzz-strips: {scanned} strip-mined scans agreed");
    assert!(
        scanned > 0,
        "no strip-mined scan was compared, so the carry between strips proved nothing"
    );
}

/// The clustered end, which randomness alone reaches rarely and which is a *ladder* rather than an
/// instruction.
///
/// A vector narrower than the subgroup packs several of itself into it, and SPIR-V has no clustered
/// scan — so `Lanes::prefix_sum` builds `log2(cluster)` steps of shuffle, compare and select. Every
/// step of that is a chance to include a lane belonging to the cluster next door, and the failure
/// it produces is a scan that agrees in the first cluster of every subgroup and is wrong in all the
/// others. That is exactly the shape a hand-written test at one cluster width is worst at seeing.
///
/// The scan is forced rather than waited for: a generated program reaches this shape by chance only
/// twice in five.
#[test]
fn clustered_programs_agree_in_every_domain() {
    let Some(gpu) = device("fuzz-clusters") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_surface() {
        eprintln!("SKIPPED fuzz-clusters: the device lacks part of the subgroup surface");
        return;
    }
    if limits.subgroup_size < 2 {
        eprintln!("SKIPPED fuzz-clusters: nothing is narrower than a one-lane subgroup");
        return;
    }

    // Every program here is the shape this driver cannot compile, so there is nothing left to run
    // — and the whole test is skipped rather than passing on an empty count.
    if miscompiles_clustered_scans(&gpu, &limits) {
        eprintln!(
            "SKIPPED fuzz-clusters: this driver miscompiles the clustered ladder, and faults \
             compiling some of it — see notes/FINDINGS.md"
        );
        return;
    }

    let mut scanned = 0_u64;

    for domain in ALL_DOMAINS {
        for seed in 0..64_u64 {
            let mut rng = fuzz::Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let mut program =
                fuzz::generate(&mut rng, domain, limits.subgroup_size, WORKGROUP_SIZE);

            // Both directions, and every cluster width the device can hold: a ladder of one step
            // and a ladder of four are different modules, and the mask is what separates them.
            program.finish = if seed % 2 == 0 {
                fuzz::Finish::Scan
            } else {
                fuzz::Finish::ScanExclusive
            };
            program.lanes = (limits.subgroup_size >> (1 + seed % 3)).max(1);

            let input = corpus(domain, program.input_len());

            match fuzz::check(&gpu, &program, &input).expect("dispatched") {
                Outcome::Agreed => scanned += 1,
                Outcome::Refused(_) | Outcome::Unrepresentable => {}
                Outcome::Disagreed {
                    program,
                    expected,
                    actual,
                    at,
                } => {
                    panic!(
                        "{domain:?} seed {seed} disagreed at index {at}: expected {}, got {}\n{}",
                        expected.get(at).copied().unwrap_or_default(),
                        actual.get(at).copied().unwrap_or_default(),
                        describe(&program)
                    );
                }
            }
        }
    }

    eprintln!("fuzz-clusters: {scanned} clustered scans agreed");
    assert!(
        scanned > 0,
        "no clustered scan was compared, so the ladder proved nothing"
    );
}

/// The fuzzer's own test: a deliberately wrong reference must be caught.
///
/// Without this, the whole file could be passing because `check` never reports a disagreement.
/// Same discipline as the validator's teeth test, and it found the same class of problem there.
#[test]
fn the_fuzzer_notices_when_the_answer_is_wrong() {
    let Some(gpu) = device("fuzz-teeth") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED fuzz-teeth: no subgroup arithmetic");
        return;
    }

    for domain in ALL_DOMAINS {
        // Seeds are searched rather than fixed. Seed 1 was enough on a 32-wide subgroup and is not
        // on a 64-wide one: the program it generates there finishes with a minimum that the
        // perturbed element does not reach, so the answer is the same either way and the test
        // asserted nothing while passing. A program *insensitive* to one element is a legitimate
        // program; what has to exist is a sensitive one.
        let mut caught = false;
        for seed in 0..64_u64 {
            let mut rng = fuzz::Rng::new(seed);
            let program = fuzz::generate(&mut rng, domain, limits.subgroup_size, WORKGROUP_SIZE);
            let input = corpus(domain, program.input_len());

            // The device gets the right input and the reference a different one: where the program
            // depends on the element that changed, they must part ways.
            let mut wrong = input.clone();
            if let Some(first) = wrong.first_mut() {
                *first = domain.encode(200);
            }

            // **A refused program is the next seed's turn, not a failure.** The generator draws a
            // lane count from a fixed list, and on a four-wide subgroup a vector of 64 is sixteen
            // strips — more than `MAX_STRIPS`, so the emitter refuses it by name. Every sweep in
            // this file already treats that as `Outcome::Refused`; this test expected it to build,
            // and only ever ran where the seeds happened not to draw one. Found by running at
            // width 4 after the vocabulary changed which programs the seeds produce.
            let Ok(spirv) = program.build() else {
                continue;
            };
            let actual = gpu
                .run_u32(&spirv, &input, program.workgroups())
                .expect("ran");
            let expected = fuzz::reference(&program, &wrong);

            if expected.values != actual {
                caught = true;
                break;
            }
        }

        assert!(
            caught,
            "no program in 64 seeds noticed a perturbed input in {domain:?}, \
             so the comparison this file rests on cannot fail"
        );
    }
}

/// The same teeth, held against a scan specifically.
///
/// The test above generates freely and stops at the first program sensitive to a perturbed input,
/// so it may never reach a scan — and a scan is the newest and most intricate thing the reference
/// models. Here the finish is forced, which also makes the sensitivity trivial to argue: a prefix
/// depends on every element before it, so changing the first one changes every answer after it.
///
/// **Both directions, because they differ in exactly one place.** An exclusive scan leaves each
/// element's own contribution out, so a reference that quietly returned the inclusive answer would
/// be wrong at every position and right at none — and would still pass a test that only ran the
/// inclusive form.
#[test]
fn the_fuzzer_notices_when_a_scan_is_wrong() {
    let Some(gpu) = device("fuzz-teeth-scan") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic {
        eprintln!("SKIPPED fuzz-teeth-scan: no subgroup arithmetic");
        return;
    }

    // Two mappings, because they are two different modules: one subgroup instruction, and a ladder
    // of shuffles and selects. A reference that modelled the first correctly says nothing about the
    // second — which is how the clustered scan came to be checked by hand-written tests only.
    let widths = [limits.subgroup_size, (limits.subgroup_size / 2).max(1)];

    for finish in [fuzz::Finish::Scan, fuzz::Finish::ScanExclusive] {
        for lanes in widths {
            for domain in ALL_DOMAINS {
                let mut rng = fuzz::Rng::new(7);
                let mut program =
                    fuzz::generate(&mut rng, domain, limits.subgroup_size, WORKGROUP_SIZE);
                program.finish = finish;
                program.lanes = lanes;

                let input = corpus(domain, program.input_len());
                let mut wrong = input.clone();
                if let Some(first) = wrong.first_mut() {
                    *first = domain.encode(200);
                }

                let spirv = program.build().expect("built");
                let actual = gpu
                    .run_u32(&spirv, &input, program.workgroups())
                    .expect("ran");
                let expected = fuzz::reference(&program, &wrong);

                assert_ne!(
                    expected.values, actual,
                    "a {finish:?} over {lanes} lanes in {domain:?} did not notice a changed \
                     element, so the comparison that guards every scan in this file cannot fail"
                );
            }
        }
    }
}

/// A program, in a form that can be pasted back into a test.
fn describe(program: &Program) -> String {
    format!(
        "  {:?}, subgroup {}, workgroup {}, groups {}, lanes {}\n  steps {:?}\n  finish {:?}",
        program.domain,
        program.subgroup,
        program.workgroup,
        program.groups,
        program.lanes,
        program.steps,
        program.finish
    )
}

#[test]
fn the_vote_on_a_value_is_compared_on_a_corpus_that_makes_it_pass() {
    // **The corpus that every other test here deliberately avoids.** `corpus` makes every element
    // distinct, because that is what makes a wrong answer obvious — and it means the vote about a
    // value never passes, so `Op::AddIfAllEqual` is generated and does nothing in every round.
    //
    // The mutation gate found it: flipping the reference's condition to `false` changed no sweep's
    // answer. A step that cannot pass is a step nobody is checking.
    let Some(gpu) = device("fuzz-agree") else {
        return;
    };
    let limits = gpu.limits().clone();
    if !limits.subgroup_surface() {
        eprintln!("SKIPPED fuzz-agree: the device lacks part of the subgroup surface");
        return;
    }

    for domain in ALL_DOMAINS {
        let program = Program {
            domain,
            subgroup: limits.subgroup_size,
            workgroup: WORKGROUP_SIZE,
            groups: 1,
            lanes: limits.subgroup_size,
            steps: vec![Op::AddIfAllEqual { add: 3 }],
            finish: Finish::Sum,
        };

        // Uniform: every subgroup agrees, so every lane adds. Nothing else in this file produces
        // this shape.
        let agreeing: Vec<u32> = vec![domain.encode(7); program.input_len()];
        // And one lane of the first subgroup differing, so the same program takes the other branch
        // in one subgroup and not in the rest.
        let mut split = agreeing.clone();
        if let Some(odd) = split.get_mut(1) {
            *odd = domain.encode(8);
        }

        for (name, input) in [("agreeing", &agreeing), ("split", &split)] {
            match fuzz::check(&gpu, &program, input).expect("dispatched") {
                Outcome::Agreed => {}
                Outcome::Refused(why) => eprintln!("fuzz-agree {domain:?} {name}: refused {why}"),
                Outcome::Unrepresentable => {
                    eprintln!("fuzz-agree {domain:?} {name}: outside the exact range");
                }
                Outcome::Disagreed {
                    expected,
                    actual,
                    at,
                    ..
                } => panic!(
                    "{domain:?} disagreed on the {name} corpus at index {at}: expected {}, got {}",
                    expected.get(at).copied().unwrap_or_default(),
                    actual.get(at).copied().unwrap_or_default()
                ),
            }
        }
    }
}
