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

/// Run `rounds` generated programs in `domain` and report how many were actually checked.
fn sweep(gpu: &runner::Gpu, domain: Domain, subgroup: u32, rounds: u64) -> (u64, u64) {
    let mut checked = 0;
    let mut refused = 0;

    for seed in 0..rounds {
        let mut rng = fuzz::Rng::new(seed);
        let program = fuzz::generate(&mut rng, domain, subgroup, WORKGROUP_SIZE);
        let input = corpus(domain, program.input_len());

        match fuzz::check(gpu, &program, &input).expect("dispatched") {
            Outcome::Agreed => checked += 1,
            Outcome::Refused(_) => refused += 1,
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

    (checked, refused)
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
    let (checked, refused) = sweep(&gpu, Domain::Unsigned, limits.subgroup_size, rounds);

    eprintln!("fuzz u32: {checked} agreed, {refused} refused, over {rounds} seeds");
    assert!(
        checked > rounds / 2,
        "most rounds were refused rather than checked, so this proved little"
    );
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
    let (checked, refused) = sweep(&gpu, Domain::Float, limits.subgroup_size, rounds);

    eprintln!("fuzz f32: {checked} agreed, {refused} refused, over {rounds} seeds");
    assert!(checked > rounds / 2);
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
    let (checked, refused) = sweep(&gpu, Domain::Signed, limits.subgroup_size, rounds);

    eprintln!("fuzz i32: {checked} agreed, {refused} refused, over {rounds} seeds");
    assert!(checked > rounds / 2);
}

/// The four narrow integer domains, which need three device features the wide ones do not.
///
/// `i8`, `u8`, `i16` and `u16` had direct device tests and no fuzzing at all — the least-checked
/// surface in the tree by this project's own standard, and the one where the two conversions
/// (`OpSConvert` against `OpUConvert`) and the two extremes (`SMax` against `UMax`) are reached by
/// the same source line.
///
/// The buffer is where they differ from everything else here: a stride of one byte means four
/// elements share a word, and `fuzz::check` packs and unpacks at that boundary.
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
    ] {
        if !needed {
            eprintln!("SKIPPED fuzz-narrow {domain:?}: the device cannot hold it in a buffer");
            continue;
        }

        let (checked, refused) = sweep(&gpu, domain, limits.subgroup_size, rounds);
        eprintln!("fuzz {domain:?}: {checked} agreed, {refused} refused, over {rounds} seeds");
        assert!(
            checked > rounds / 2,
            "most {domain:?} rounds were refused rather than checked, so this proved little"
        );
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

    for domain in ALL_DOMAINS {
        for seed in 0..64_u64 {
            let mut rng = fuzz::Rng::new(seed.wrapping_mul(0x5851_F42D_4C95_7F2D));
            let mut program =
                fuzz::generate(&mut rng, domain, limits.subgroup_size, WORKGROUP_SIZE);

            program.lanes = limits.subgroup_size * (2 + (seed % 3) as u32).min(4);
            let input = corpus(domain, program.input_len());

            match fuzz::check(&gpu, &program, &input).expect("dispatched") {
                Outcome::Agreed | Outcome::Refused(_) => {}
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

            if expected != actual {
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
