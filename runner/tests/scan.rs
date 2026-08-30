mod common;

use common::{device, ramp};
use runner::kernels::{self, WORKGROUP_SIZE};

fn inclusive(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .scan(0.0_f32, |running, value| {
            *running += value;
            Some(*running)
        })
        .collect()
}

#[test]
fn a_workgroup_scan_returns_what_the_cpu_would() {
    let Some(gpu) = device("scan") else { return };
    eprintln!("device: {}", gpu.limits().name);

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scan::scan_workgroup::<simdr::lanes::F32>(width).expect("built");
    let input = ramp(WORKGROUP_SIZE as usize);

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    assert_eq!(output, inclusive(&input));
}

#[test]
fn every_element_is_checked_and_not_just_the_total() {
    let Some(gpu) = device("scan elements") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scan::scan_workgroup::<simdr::lanes::F32>(width).expect("built");
    let input = ramp(WORKGROUP_SIZE as usize);
    let expected = inclusive(&input);

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
        assert_eq!(got, want, "element {index} of {}", output.len());
    }
    assert_eq!(output.len(), expected.len());
}

#[test]
fn the_subgroup_boundary_is_where_a_workgroup_scan_goes_wrong() {
    let Some(gpu) = device("scan boundary") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scan::scan_workgroup::<simdr::lanes::F32>(width).expect("built");
    let input = vec![1.0_f32; WORKGROUP_SIZE as usize];

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let expected: Vec<f32> = (1..=WORKGROUP_SIZE).map(|index| index as f32).collect();
    assert_eq!(output, expected, "subgroup width {width}");
}

#[test]
fn a_scan_of_zeros_stays_zero_and_a_scan_of_one_element_is_that_element() {
    let Some(gpu) = device("scan zeros") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scan::scan_workgroup::<simdr::lanes::F32>(width).expect("built");

    let zeros = vec![0.0_f32; WORKGROUP_SIZE as usize];
    assert_eq!(
        gpu.run(&spirv, &zeros, 1).expect("dispatched"),
        zeros,
        "a scan of nothing is nothing, however many subgroups it took"
    );

    let mut one = vec![0.0_f32; WORKGROUP_SIZE as usize];
    if let Some(first) = one.first_mut() {
        *first = 7.0;
    }
    let output = gpu.run(&spirv, &one, 1).expect("dispatched");
    assert!(
        output
            .iter()
            .all(|value| (*value - 7.0).abs() < f32::EPSILON),
        "one non-zero at the front should carry all the way across: {output:?}"
    );
}

#[test]
fn each_block_writes_its_own_total_to_its_own_slot() {
    let Some(gpu) = device("scan blocks") else {
        return;
    };
    eprintln!("device: {}", gpu.limits().name);

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scan::scan_blocks::<simdr::lanes::F32>(width).expect("built");
    let block = WORKGROUP_SIZE as usize;

    for blocks in [1_usize, 2, 5] {
        let input: Vec<f32> = (0..blocks * block)
            .map(|index| (index / block + 1) as f32)
            .collect();

        let mut session = gpu
            .session(&spirv, &[input.len(), input.len(), blocks])
            .expect("session");
        session.write(0, &bits(&input)).expect("uploaded");
        session.dispatch(blocks as u32, 1).expect("dispatched");

        let scanned = floats(&session.read(1, input.len()).expect("read scan"));
        let totals = floats(&session.read(2, blocks).expect("read totals"));

        let expected: Vec<f32> = input.chunks(block).flat_map(inclusive).collect();
        assert_eq!(scanned, expected, "{blocks} blocks at width {width}");

        let wanted: Vec<f32> = (0..blocks).map(|b| ((b + 1) * block) as f32).collect();
        assert_eq!(totals, wanted, "{blocks} block totals at width {width}");
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn floats(words: &[u32]) -> Vec<f32> {
    words.iter().map(|word| f32::from_bits(*word)).collect()
}

#[test]
fn three_dispatches_scan_further_than_one_workgroup_reaches() {
    let Some(gpu) = device("scan long") else {
        return;
    };
    eprintln!("device: {}", gpu.limits().name);

    let width = gpu.limits().subgroup_size;
    let block = WORKGROUP_SIZE as usize;

    let first = kernels::scan::scan_blocks::<simdr::lanes::F32>(width).expect("built");
    let middle =
        kernels::scan::scan_workgroup_exclusive::<simdr::lanes::F32>(width).expect("built");
    let last = kernels::scan::add_offsets::<simdr::lanes::F32>(width).expect("built");

    for blocks in [2_usize, 3, 8, 64] {
        let elements = blocks * block;
        let input = vec![1.0_f32; elements];

        let mut scan = gpu
            .session(&first, &[elements, elements, blocks])
            .expect("session");
        scan.write(0, &bits(&input)).expect("uploaded");
        scan.dispatch(blocks as u32, 1).expect("dispatched");
        let scanned = scan.read(1, elements).expect("read");
        let totals = scan.read(2, blocks).expect("read");

        let mut offsets = gpu.session(&middle, &[block, block]).expect("session");
        offsets.write(0, &totals).expect("uploaded");
        offsets.dispatch(1, 1).expect("dispatched");
        let owed = offsets.read(1, blocks).expect("read");

        let mut add = gpu
            .session(&last, &[elements, blocks, elements])
            .expect("session");
        add.write(0, &scanned).expect("uploaded");
        add.write(1, &owed).expect("uploaded");
        add.dispatch(blocks as u32, 1).expect("dispatched");
        let output = floats(&add.read(2, elements).expect("read"));

        assert_eq!(
            output,
            inclusive(&input),
            "{blocks} blocks ({elements} elements) at width {width}"
        );
    }
}

#[test]
fn a_long_scan_of_uneven_values_matches_the_cpu_element_for_element() {
    let Some(gpu) = device("scan long uneven") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let block = WORKGROUP_SIZE as usize;
    let blocks = 16_usize;
    let elements = blocks * block;

    let first = kernels::scan::scan_blocks::<simdr::lanes::F32>(width).expect("built");
    let middle =
        kernels::scan::scan_workgroup_exclusive::<simdr::lanes::F32>(width).expect("built");
    let last = kernels::scan::add_offsets::<simdr::lanes::F32>(width).expect("built");

    let input: Vec<f32> = (0..elements).map(|index| (index % 7) as f32).collect();

    let mut scan = gpu
        .session(&first, &[elements, elements, blocks])
        .expect("session");
    scan.write(0, &bits(&input)).expect("uploaded");
    scan.dispatch(blocks as u32, 1).expect("dispatched");
    let scanned = scan.read(1, elements).expect("read");
    let totals = scan.read(2, blocks).expect("read");

    let mut offsets = gpu.session(&middle, &[block, block]).expect("session");
    offsets.write(0, &totals).expect("uploaded");
    offsets.dispatch(1, 1).expect("dispatched");
    let owed = offsets.read(1, blocks).expect("read");

    let mut add = gpu
        .session(&last, &[elements, blocks, elements])
        .expect("session");
    add.write(0, &scanned).expect("uploaded");
    add.write(1, &owed).expect("uploaded");
    add.dispatch(blocks as u32, 1).expect("dispatched");
    let output = floats(&add.read(2, elements).expect("read"));

    let expected = inclusive(&input);
    for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
        assert_eq!(got, want, "element {index} of {elements} at width {width}");
    }
}

#[test]
fn a_held_scanner_matches_the_cpu_at_every_depth_of_recursion() {
    let Some(gpu) = device("scanner") else { return };
    eprintln!("device: {}", gpu.limits().name);

    let block = WORKGROUP_SIZE as usize;
    for (elements, levels) in [
        (block, 1_usize),
        (block * 2, 1),
        (block * block, 1),
        (block * block * 2, 2),
        (1 << 20, 3),
    ] {
        let mut scanner = gpu.scanner(elements).expect("built");
        assert_eq!(scanner.elements(), elements);
        assert_eq!(
            scanner.dispatches(),
            2 * levels + 1,
            "{elements} elements should need {levels} levels"
        );

        let input = vec![1.0_f32; elements];
        let output = scanner.scan(&input).expect("scanned");

        let expected: Vec<f32> = (1..=elements).map(|index| index as f32).collect();
        assert_eq!(output.len(), expected.len());
        for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "element {index} of {elements}");
        }
    }
}

#[test]
fn a_held_scanner_agrees_with_the_cpu_on_values_that_differ_everywhere() {
    let Some(gpu) = device("scanner uneven") else {
        return;
    };

    let block = WORKGROUP_SIZE as usize;
    let elements = block * block * 3;
    let mut scanner = gpu.scanner(elements).expect("built");

    let input: Vec<f32> = (0..elements).map(|index| (index % 13) as f32).collect();
    let output = scanner.scan(&input).expect("scanned");

    let expected = inclusive(&input);
    for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
        assert_eq!(got, want, "element {index} of {elements}");
    }
}

#[test]
fn a_scanner_is_reusable_and_refuses_a_length_it_was_not_built_for() {
    let Some(gpu) = device("scanner reuse") else {
        return;
    };

    let elements = WORKGROUP_SIZE as usize * 4;
    let mut scanner = gpu.scanner(elements).expect("built");

    let ones = vec![1.0_f32; elements];
    let twos = vec![2.0_f32; elements];

    let first = scanner.scan(&ones).expect("scanned");
    let second = scanner.scan(&twos).expect("scanned");
    let third = scanner.scan(&ones).expect("scanned");

    assert_eq!(first, inclusive(&ones));
    assert_eq!(second, inclusive(&twos));
    assert_eq!(
        third, first,
        "the third call must not have kept the second's"
    );

    assert!(matches!(
        scanner.scan(&ones[..elements - 1]),
        Err(runner::Error::TooLarge { .. })
    ));
}

#[test]
fn a_length_that_is_not_a_whole_number_of_workgroups_is_refused_before_anything_is_built() {
    let Some(gpu) = device("scanner length") else {
        return;
    };

    for elements in [0_usize, 1, 63, 65, 100] {
        assert!(
            matches!(gpu.scanner(elements), Err(runner::Error::BadLength(_))),
            "{elements} was accepted"
        );
    }
}

#[test]
fn a_strip_mined_scan_scans_each_subgroups_own_vector() {
    let Some(gpu) = device("scan strips") else {
        return;
    };
    eprintln!("device: {}", gpu.limits().name);

    let width = gpu.limits().subgroup_size as usize;
    let workgroup = WORKGROUP_SIZE as usize;
    let lanes = 128_u32;
    let strips = (lanes as usize).div_ceil(width);
    if !(lanes as usize).is_multiple_of(width) || strips > 8 {
        eprintln!("SKIPPED scan strips: 128 lanes does not strip-mine onto {width}");
        return;
    }

    let spirv = kernels::scan::scan_strips::<128>(gpu.limits().subgroup_size).expect("built");
    let elements = workgroup * strips;
    let input: Vec<f32> = (0..elements).map(|index| (index % 9) as f32).collect();

    let output = gpu.run(&spirv, &input, 1).expect("dispatched");

    let mut expected = vec![0.0_f32; elements];
    for local in 0..workgroup {
        let subgroup = local / width;
        let lane = local % width;

        let at = |position: usize| {
            (subgroup * width) + (position % width) + (position / width) * workgroup
        };

        let mut running = 0.0_f32;
        for strip in 0..strips {
            let position = strip * width + lane;
            running = (0..=position).map(|earlier| input[at(earlier)]).sum();
            expected[at(position)] = running;
        }
        let _ = running;
    }

    for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
        assert_eq!(
            got, want,
            "element {index} of {elements}, {strips} strips at width {width}"
        );
    }
}

#[test]
fn a_clustered_scan_scans_each_cluster_independently() {
    let Some(gpu) = device("scan clusters") else {
        return;
    };
    eprintln!("device: {}", gpu.limits().name);

    let width = gpu.limits().subgroup_size as usize;
    let workgroup = WORKGROUP_SIZE as usize;
    let input: Vec<f32> = (0..workgroup)
        .map(|index| (index % 5) as f32 + 1.0)
        .collect();

    for cluster in [2_usize, 4, 8] {
        if cluster >= width {
            eprintln!("SKIPPED clusters of {cluster}: not narrower than a {width}-wide subgroup");
            continue;
        }

        let spirv = kernels::scan::scan_clusters(gpu.limits().subgroup_size, cluster as u32)
            .expect("built");

        let output = gpu.run(&spirv, &input, 1).expect("dispatched");

        let expected: Vec<f32> = input
            .chunks(cluster)
            .flat_map(|chunk| {
                chunk.iter().scan(0.0_f32, |running, value| {
                    *running += value;
                    Some(*running)
                })
            })
            .collect();

        for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
            assert_eq!(
                got, want,
                "element {index} of {workgroup}, clusters of {cluster} at width {width}"
            );
        }
    }
}

#[test]
fn a_clustered_exclusive_scan_leaves_each_lanes_own_element_out() {
    let Some(gpu) = device("scan clusters exclusive") else {
        return;
    };
    eprintln!("device: {}", gpu.limits().name);

    let width = gpu.limits().subgroup_size as usize;
    let workgroup = WORKGROUP_SIZE as usize;
    let input: Vec<f32> = (0..workgroup)
        .map(|index| (index % 5) as f32 + 1.0)
        .collect();

    for cluster in [2_usize, 4, 8] {
        if cluster >= width {
            eprintln!("SKIPPED clusters of {cluster}: not narrower than a {width}-wide subgroup");
            continue;
        }

        let spirv =
            kernels::scan::scan_clusters_exclusive(gpu.limits().subgroup_size, cluster as u32)
                .expect("built");

        let output = gpu.run(&spirv, &input, 1).expect("dispatched");

        let expected: Vec<f32> = input
            .chunks(cluster)
            .flat_map(|chunk| {
                chunk.iter().scan(0.0_f32, |running, value| {
                    let before = *running;
                    *running += value;
                    Some(before)
                })
            })
            .collect();

        for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
            assert_eq!(
                got, want,
                "element {index} of {workgroup}, clusters of {cluster} at width {width}"
            );
        }
    }
}

#[test]
fn a_mapped_scanner_runs_the_map_on_the_device_and_agrees_with_doing_it_here() {
    let Some(gpu) = device("scanner mapped") else {
        return;
    };
    eprintln!("device: {}", gpu.limits().name);

    let width = gpu.limits().subgroup_size;
    let square = kernels::square(width).expect("built");
    let elements = WORKGROUP_SIZE as usize * 4;

    let input: Vec<f32> = (0..elements).map(|index| (index % 3) as f32).collect();

    let mut fused = gpu.scanner_of(elements, &square).expect("built");
    let mut plain = gpu.scanner(elements).expect("built");

    assert_eq!(
        fused.dispatches(),
        plain.dispatches() + 1,
        "the map is one more dispatch and nothing else"
    );

    let squares: Vec<f32> = input.iter().map(|value| value * value).collect();
    let expected = inclusive(&squares);

    let through_the_host = plain.scan(&squares).expect("scanned");
    let on_the_device = fused.scan(&input).expect("scanned");

    assert_eq!(
        on_the_device, expected,
        "the fused scan of squares is wrong"
    );
    assert_eq!(
        on_the_device, through_the_host,
        "the two routes computed different numbers"
    );
}

#[test]
fn a_mapped_scanner_is_correct_at_every_depth_of_recursion() {
    let Some(gpu) = device("scanner mapped deep") else {
        return;
    };

    let width = gpu.limits().subgroup_size;
    let square = kernels::square(width).expect("built");
    let block = WORKGROUP_SIZE as usize;

    for elements in [block, block * block, block * block * 2] {
        let input: Vec<f32> = (0..elements).map(|index| (index % 3) as f32).collect();
        let squares: Vec<f32> = input.iter().map(|value| value * value).collect();

        let mut fused = gpu.scanner_of(elements, &square).expect("built");
        let output = fused.scan(&input).expect("scanned");

        let expected = inclusive(&squares);
        for (index, (got, want)) in output.iter().zip(&expected).enumerate() {
            assert_eq!(got, want, "element {index} of {elements}");
        }
    }
}

#[test]
fn a_one_shot_scan_agrees_with_a_held_one_and_refuses_the_same_lengths() {
    let Some(gpu) = device("one-shot scan") else {
        return;
    };

    let block = WORKGROUP_SIZE as usize;
    for elements in [block, block * 4, block * block * 2] {
        let input: Vec<f32> = (0..elements).map(|index| (index % 11) as f32).collect();

        let once = gpu.scan(&input).expect("scanned");
        let held = gpu
            .scanner(elements)
            .expect("built")
            .scan(&input)
            .expect("scanned");

        assert_eq!(once, inclusive(&input), "{elements} elements");
        assert_eq!(once, held, "the two routes disagree at {elements}");
    }

    for elements in [0_usize, 1, 63, 65, 100] {
        let input = vec![1.0_f32; elements];
        assert!(
            matches!(gpu.scan(&input), Err(runner::Error::BadLength(_))),
            "{elements} was accepted"
        );
    }
}
