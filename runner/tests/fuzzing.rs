//! Random lane programs, run on the device and checked against the CPU reference.
//!
//! The other test files check things somebody thought of. This one does not: it generates
//! programs from a seed, works out what each must return by interpreting the same program on the
//! CPU, and compares. A disagreement is a real finding wherever it lands — the emitter, the
//! mapping, the reference, or the driver.
//!
//! Seeds are fixed, so a failure names the seed that produced it and re-running reproduces it
//! exactly. `SIMDR_FUZZ_ROUNDS` searches harder.

mod common;

use common::device;
use runner::fuzz::{self, ALL_DOMAINS, Domain, Outcome, Program};
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
fn rounds() -> u64 {
    std::env::var("SIMDR_FUZZ_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(ROUNDS)
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
    /// reductions alone. A scan is refused for a clustered vector and is the newest thing here, so
    /// "the sweep ran" and "the sweep exercised a prefix" are different claims and only one of
    /// them was being made.
    scans: u64,
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
            "fuzz {domain:?} over {rounds} seeds: {} agreed of which {} scans, {} refused, {} unrepresentable",
            self.checked, self.scans, self.refused, self.unrepresentable
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
fn sweep(gpu: &runner::Gpu, domain: Domain, subgroup: u32, rounds: u64) -> Swept {
    let mut checked = 0;
    let mut refused = 0;
    let mut unrepresentable = 0;
    let mut scans = 0;

    for seed in 0..rounds {
        let mut rng = fuzz::Rng::new(seed);
        let program = fuzz::generate(&mut rng, domain, subgroup, WORKGROUP_SIZE);
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
    }
}

#[test]
fn generated_integer_programs_agree_with_the_cpu_reference() {
    let Some(gpu) = device("fuzz-u32") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_arithmetic || !limits.subgroup_clustered || !limits.subgroup_shuffle {
        eprintln!("SKIPPED fuzz-u32: the device lacks part of the subgroup surface");
        return;
    }

    let rounds = rounds();
    let swept = sweep(&gpu, Domain::Unsigned, limits.subgroup_size, rounds);

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

    if !limits.subgroup_arithmetic || !limits.subgroup_clustered || !limits.subgroup_shuffle {
        eprintln!("SKIPPED fuzz-f32: the device lacks part of the subgroup surface");
        return;
    }

    let rounds = rounds();
    let swept = sweep(&gpu, Domain::Float, limits.subgroup_size, rounds);

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

    if !limits.subgroup_arithmetic || !limits.subgroup_clustered || !limits.subgroup_shuffle {
        eprintln!("SKIPPED fuzz-i32: the device lacks part of the subgroup surface");
        return;
    }

    let rounds = rounds();
    let swept = sweep(&gpu, Domain::Signed, limits.subgroup_size, rounds);

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

    if !limits.subgroup_arithmetic || !limits.subgroup_clustered || !limits.subgroup_shuffle {
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

        let swept = sweep(&gpu, domain, limits.subgroup_size, rounds);
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

    if !limits.subgroup_arithmetic || !limits.subgroup_shuffle {
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

            let spirv = program.build().expect("built");
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

    for finish in [fuzz::Finish::Scan, fuzz::Finish::ScanExclusive] {
        for domain in ALL_DOMAINS {
            let mut rng = fuzz::Rng::new(7);
            let mut program =
                fuzz::generate(&mut rng, domain, limits.subgroup_size, WORKGROUP_SIZE);
            program.finish = finish;
            // A scan has no clustered form, so the vector has to be at least the subgroup.
            program.lanes = limits.subgroup_size;

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
                "a {finish:?} in {domain:?} did not notice a changed element, so the comparison                  that guards every scan in this file cannot fail"
            );
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
