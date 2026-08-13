//! The prefix sum, run on a device and compared against the CPU element by element.
//!
//! **A scan is a stricter test of the lane mapping than a reduction is.** A reduction sums the same
//! set whatever order the lanes are in, so a mapping that pairs the wrong lanes still returns the
//! right total. A scan does not: element `i` of the answer depends on exactly which elements the
//! hardware considers to come before `i`, so a shuffled mapping produces a wrong number at almost
//! every position while still summing to the same grand total at the end.
//!
//! That last clause is the trap this file is written around. Checking only the final element would
//! pass for a mapping that is wrong everywhere else, so every element is compared.

mod common;

use common::{device, ramp};
use runner::kernels::{self, WORKGROUP_SIZE};

/// The answer, on the host: element `i` is the sum of everything up to and including it.
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
    // The mapping test. A scan whose lanes are paired wrongly still ends at the same grand total,
    // so the last element agreeing proves nothing on its own — and this asserts the middle first,
    // so a failure reports the position rather than the sum.
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
    // A scan that never combined across subgroups would restart at every boundary: correct within
    // each subgroup and short by the running total at the start of the next. With ones as input
    // the two are trivially told apart — a working scan reads 1, 2, 3, … all the way across, and a
    // restarting one drops back to 1 at the boundary.
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
    // The identity cases, which a version that added an uninitialised offset would fail. Shared
    // memory is not zeroed by anyone — `notes/FINDINGS.md` records that two drivers hand back
    // zeros and lavapipe does not — so an offset read before it was written shows up here.
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
    // **What `Kernel::workgroup_index` and `Kernel::store_at` were added for.** Checked on a device
    // and across several workgroups, because the ways this goes wrong all look plausible with one:
    // every block writing to slot zero, or to a lane-derived slot, is correct at one workgroup and
    // wrong at two.
    //
    // Three bindings, so this goes through a `Session` rather than `Gpu::run` — the one-shot path
    // binds two and writes the last, and this kernel writes two of the three.
    let Some(gpu) = device("scan blocks") else {
        return;
    };
    eprintln!("device: {}", gpu.limits().name);

    let width = gpu.limits().subgroup_size;
    let spirv = kernels::scan::scan_blocks::<simdr::lanes::F32>(width).expect("built");
    let block = WORKGROUP_SIZE as usize;

    for blocks in [1_usize, 2, 5] {
        // Block `b` holds `b + 1` in every slot, so its total is `(b + 1) * 64` and no two blocks
        // agree — a total written to the wrong slot is a wrong number rather than a coincidence.
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

        // Each block scans from its own start, not from the start of the buffer.
        let expected: Vec<f32> = input.chunks(block).flat_map(inclusive).collect();
        assert_eq!(scanned, expected, "{blocks} blocks at width {width}");

        let wanted: Vec<f32> = (0..blocks).map(|b| ((b + 1) * block) as f32).collect();
        assert_eq!(totals, wanted, "{blocks} block totals at width {width}");
    }
}

/// A slice of floats as the words a buffer holds.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The other way round.
fn floats(words: &[u32]) -> Vec<f32> {
    words.iter().map(|word| f32::from_bits(*word)).collect()
}

#[test]
fn three_dispatches_scan_further_than_one_workgroup_reaches() {
    // **The composition, by hand.** `scan_blocks` leaves each block scanned from its own start;
    // the block totals scanned exclusively say what each block owes the blocks before it; and
    // `add_offsets` pays it. Three dispatches over four buffers, which is what a held `Scanner`
    // will do for a caller — this is the arithmetic being shown to work before anything wraps it.
    //
    // The middle step is one workgroup, so this reaches 64 blocks: 4096 elements. Beyond that the
    // totals need scanning by the same three steps again, one level up.
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
        // Values that make a misplaced offset obvious: every element is 1, so the answer is
        // 1, 2, 3, … all the way across and any block starting from the wrong offset is visibly
        // out of step rather than merely wrong in the last digit.
        let input = vec![1.0_f32; elements];

        // Blocks scanned, and each block's total.
        let mut scan = gpu
            .session(&first, &[elements, elements, blocks])
            .expect("session");
        scan.write(0, &bits(&input)).expect("uploaded");
        scan.dispatch(blocks as u32, 1).expect("dispatched");
        let scanned = scan.read(1, elements).expect("read");
        let totals = scan.read(2, blocks).expect("read");

        // The totals, scanned exclusively: what each block owes the ones before it.
        let mut offsets = gpu.session(&middle, &[block, block]).expect("session");
        offsets.write(0, &totals).expect("uploaded");
        offsets.dispatch(1, 1).expect("dispatched");
        let owed = offsets.read(1, blocks).expect("read");

        // Paid.
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
    // The same composition over values that differ everywhere, so a block that took the wrong
    // offset cannot coincidentally agree. Kept inside 2^24 so the float sums stay exact and the
    // comparison can be equality rather than a tolerance.
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
    // **The recursion, at each of its depths.** 4 096 elements is one level and the three
    // dispatches composed by hand above; 262 144 is two; 2²⁰ is three. The interesting boundary is
    // between them — a `Scanner` that handled one level and quietly dropped the rest would be
    // right at 4 096 and wrong by whole blocks at 262 144.
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

        // Ones, so the answer is 1, 2, 3, … and a block that took the wrong offset is visibly out
        // of step rather than wrong in the last digit. Exact in `f32` to 2²⁴, which 2²⁰ is inside.
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
    // Ones make a misplaced offset obvious and also make every block's total identical, so a
    // scanner that mixed two blocks' totals up would still be right. These do not.
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
    // The point of holding one. A second call must not see the first's data — the buffers are
    // reused, so a scan that read past its input would find the previous answer sitting there.
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
