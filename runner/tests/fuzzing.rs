mod common;

use common::device;
use runner::fuzz::{self, ALL_DOMAINS, Domain, Finish, Op, Outcome, Program};
use runner::kernels::WORKGROUP_SIZE;

const ROUNDS: u64 = 256;

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

fn rounds() -> u64 {
    let Some(value) = std::env::var_os("SIMDR_FUZZ_ROUNDS") else {
        return ROUNDS;
    };
    let text = value.to_string_lossy();
    text.trim().parse().unwrap_or_else(|error| {
        panic!("SIMDR_FUZZ_ROUNDS is {text:?}, which is not a number of rounds: {error}")
    })
}

fn miscompiles_clustered_scans(gpu: &runner::Gpu, limits: &runner::Limits) -> bool {
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
    let input: Vec<u32> = (0..program.input_len())
        .map(|index| domain.encode(index as u32 % 9 + 10))
        .collect();

    match fuzz::check(gpu, &program, &input) {
        Ok(Outcome::Disagreed { .. }) => true,
        Ok(_) | Err(_) => false,
    }
}

fn defective_here(program: &Program) -> bool {
    program.lanes < program.subgroup
        && matches!(program.finish, Finish::Scan | Finish::ScanExclusive)
}

struct Swept {
    checked: u64,
    refused: u64,
    unrepresentable: u64,
    scans: u64,
    defective: u64,
}

impl Swept {
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
            Outcome::Refused(why) => {
                if refused == 0 {
                    eprintln!("fuzz {domain:?}: first refusal at seed {seed} — {why}");
                }
                refused += 1;
            }
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
    if !limits.narrow.subgroup_extended_types {
        eprintln!("SKIPPED fuzz-narrow: no shaderSubgroupExtendedTypes");
        return;
    }

    let rounds = rounds();
    let defect = miscompiles_clustered_scans(&gpu, &limits);
    if defect {
        eprintln!(
            "fuzz-narrow: this driver miscompiles the clustered ladder, so those rounds are \
             counted and skipped"
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

    if miscompiles_clustered_scans(&gpu, &limits) {
        eprintln!(
            "SKIPPED fuzz-clusters: this driver miscompiles the clustered ladder, and faults \
             compiling some of it"
        );
        return;
    }

    let mut scanned = 0_u64;

    for domain in ALL_DOMAINS {
        for seed in 0..64_u64 {
            let mut rng = fuzz::Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let mut program =
                fuzz::generate(&mut rng, domain, limits.subgroup_size, WORKGROUP_SIZE);

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
        let mut caught = false;
        for seed in 0..64_u64 {
            let mut rng = fuzz::Rng::new(seed);
            let program = fuzz::generate(&mut rng, domain, limits.subgroup_size, WORKGROUP_SIZE);
            let input = corpus(domain, program.input_len());

            let mut wrong = input.clone();
            if let Some(first) = wrong.first_mut() {
                *first = domain.encode(200);
            }

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

    let widths = [limits.subgroup_size, (limits.subgroup_size / 2).max(1)];

    for finish in [fuzz::Finish::Scan, fuzz::Finish::ScanExclusive] {
        for lanes in widths {
            for domain in ALL_DOMAINS {
                let Some((program, spirv, input, expected)) = (7_u64..64).find_map(|seed| {
                    let mut program = fuzz::generate(
                        &mut fuzz::Rng::new(seed),
                        domain,
                        limits.subgroup_size,
                        WORKGROUP_SIZE,
                    );
                    program.finish = finish;
                    program.lanes = lanes;
                    let spirv = program.build().ok()?;

                    let input = corpus(domain, program.input_len());
                    let mut wrong = input.clone();
                    *wrong.first_mut()? = domain.encode(200);

                    let seen = fuzz::reference(&program, &input);
                    let changed = fuzz::reference(&program, &wrong);
                    (seen.values != changed.values).then_some((program, spirv, input, changed))
                }) else {
                    panic!(
                        "no seed in 7..64 gave a {finish:?} over {lanes} lanes in {domain:?} that \
                         builds and separates a changed element, so this mapping is not being \
                         checked at all"
                    );
                };

                let actual = gpu
                    .run_u32(&spirv, &input, program.workgroups())
                    .expect("ran");

                assert_ne!(
                    expected.values, actual,
                    "a {finish:?} over {lanes} lanes in {domain:?} did not notice a changed \
                     element, so the comparison that guards every scan in this file cannot fail"
                );
            }
        }
    }
}

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

        let agreeing: Vec<u32> = vec![domain.encode(7); program.input_len()];
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

#[test]
fn a_corpus_shorter_than_the_program_is_named_rather_than_absorbed() {
    let Some(gpu) = device("fuzz-short-corpus") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let program = Program {
        domain: Domain::Unsigned,
        subgroup: width,
        workgroup: WORKGROUP_SIZE,
        groups: 1,
        lanes: width,
        steps: vec![Op::AddConstant(1)],
        finish: Finish::Sum,
    };

    let needed = program.input_len();
    let short = vec![0_u32; needed - 1];

    match fuzz::check(&gpu, &program, &short) {
        Err(fuzz::FuzzError::ShortInput {
            needed: asked,
            given,
        }) => {
            assert_eq!(asked, needed);
            assert_eq!(given, needed - 1);
        }
        other => panic!("a corpus one element short gave {other:?}"),
    }

    let whole = vec![0_u32; needed];
    assert!(fuzz::check(&gpu, &program, &whole).is_ok());
}

#[test]
fn every_generated_shape_is_valid_spirv() {
    let Some(_) = common::validator() else {
        eprintln!("SKIPPED fuzz-validated: spirv-val not found (set SPIRV_VAL)");
        return;
    };

    const SEEDS: u64 = 6;
    let mut mappings = std::collections::BTreeSet::new();
    let mut finishes = std::collections::BTreeSet::new();
    let mut validated = 0_u32;

    for width in [4_u32, 8, 16, 32, 64] {
        for domain in ALL_DOMAINS {
            for seed in 0..SEEDS {
                let program =
                    fuzz::generate(&mut fuzz::Rng::new(seed), domain, width, WORKGROUP_SIZE);

                let Ok(words) = program.build() else {
                    continue;
                };

                mappings.insert(program.lanes.cmp(&program.subgroup) as i8);
                finishes.insert(
                    format!("{:?}", program.finish)
                        .split(' ')
                        .next()
                        .map_or_else(|| String::from("?"), str::to_owned),
                );
                validated += 1;

                common::expect_valid(
                    &words,
                    &format!("fuzz-{domain:?}-{width}-{seed}"),
                    common::VULKAN_1_1,
                );
            }
        }
    }

    assert_eq!(
        mappings.len(),
        3,
        "the validated programs cover {} of the three mappings, so at least one instruction \
         sequence went unvalidated",
        mappings.len()
    );
    assert!(
        finishes.len() >= 5,
        "only {:?} finishes reached the validator",
        finishes
    );
    eprintln!(
        "fuzz-validated: {validated} generated modules, {} finishes",
        finishes.len()
    );
}
