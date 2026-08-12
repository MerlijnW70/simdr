//! One module, several pipelines, different answers.
//!
//! That sentence is the whole feature and it is the only thing a validator cannot check: a module
//! with a specialization constant validates identically whether or not anything ever replaces the
//! value, and the default is a perfectly good answer. So the test that means something is the one
//! that runs the **same words** twice and gets two different numbers out.

mod common;

use common::device;
use runner::Specialization;
use runner::kernels::{self, WORKGROUP_SIZE, specialized::spec_id};
use simdr::lanes::U32;

/// The invocations one workgroup runs, as a `usize`.
fn count() -> usize {
    WORKGROUP_SIZE as usize
}

/// A ramp, which makes an elementwise mistake visible at every element.
fn ramp() -> Vec<u32> {
    (0..count() as u32).collect()
}

#[test]
fn one_module_gives_two_answers_under_two_specializations() {
    let Some(gpu) = device("specialize") else {
        return;
    };
    let limits = gpu.limits().clone();

    // Built once. Everything below dispatches these same words.
    let spirv = kernels::specialized_add::<U32, 32>(limits.subgroup_size, 1).expect("built");
    let input = ramp();

    let by_ten = gpu
        .run_specialized(
            &spirv,
            &input,
            1,
            &Specialization::none().set(spec_id::ADDEND, 10),
        )
        .expect("dispatched");
    let by_seven = gpu
        .run_specialized(
            &spirv,
            &input,
            1,
            &Specialization::none().set(spec_id::ADDEND, 7),
        )
        .expect("dispatched");

    assert_eq!(
        by_ten,
        input.iter().map(|value| value + 10).collect::<Vec<u32>>()
    );
    assert_eq!(
        by_seven,
        input.iter().map(|value| value + 7).collect::<Vec<u32>>()
    );
    assert_ne!(
        by_ten, by_seven,
        "the same words produced the same answer, so nothing was specialized"
    );
}

#[test]
fn an_unspecialized_pipeline_uses_the_default_the_module_declared() {
    // The other half, and the one a wrong `SpecId` would pass: if nothing replaces the constant,
    // the default has to survive. A module whose constant was decorated with an id nobody sets is
    // indistinguishable from an ordinary constant, which is exactly the failure mode.
    let Some(gpu) = device("specialize-default") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::specialized_add::<U32, 32>(limits.subgroup_size, 5).expect("built");
    let input = ramp();

    let defaulted = gpu.run_u32(&spirv, &input, 1).expect("dispatched");
    let overridden = gpu
        .run_specialized(
            &spirv,
            &input,
            1,
            &Specialization::none().set(spec_id::ADDEND, 5),
        )
        .expect("dispatched");

    assert_eq!(
        defaulted,
        input.iter().map(|value| value + 5).collect::<Vec<u32>>()
    );
    assert_eq!(
        defaulted, overridden,
        "setting a constant to its own default must change nothing"
    );
}

#[test]
fn two_constants_are_told_apart_by_their_ids_and_not_by_their_order() {
    // The failure a single-constant test cannot see: entries carry a `constant_id` and an offset
    // into the data block, and swapping the two would give an answer that is still a number.
    // `factor` and `offset` are chosen so that transposing them is visible — 3x + 100 and
    // 100x + 3 differ everywhere except x = 1.
    let Some(gpu) = device("specialize-two") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::specialized_affine::<32>(limits.subgroup_size, 1, 0).expect("built");
    let input = ramp();

    let output = gpu
        .run_specialized(
            &spirv,
            &input,
            1,
            &Specialization::none()
                .set(spec_id::FACTOR, 3)
                .set(spec_id::OFFSET, 100),
        )
        .expect("dispatched");

    let expected: Vec<u32> = input.iter().map(|value| value * 3 + 100).collect();
    let transposed: Vec<u32> = input.iter().map(|value| value * 100 + 3).collect();

    assert_eq!(output, expected);
    assert_ne!(output, transposed, "the two ids were swapped");
}

#[test]
fn the_order_the_entries_are_set_in_does_not_matter() {
    let Some(gpu) = device("specialize-order") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::specialized_affine::<32>(limits.subgroup_size, 1, 0).expect("built");
    let input = ramp();

    let forwards = gpu
        .run_specialized(
            &spirv,
            &input,
            1,
            &Specialization::none()
                .set(spec_id::FACTOR, 2)
                .set(spec_id::OFFSET, 9),
        )
        .expect("dispatched");
    let backwards = gpu
        .run_specialized(
            &spirv,
            &input,
            1,
            &Specialization::none()
                .set(spec_id::OFFSET, 9)
                .set(spec_id::FACTOR, 2),
        )
        .expect("dispatched");

    assert_eq!(forwards, backwards);
    assert_eq!(
        forwards,
        input
            .iter()
            .map(|value| value * 2 + 9)
            .collect::<Vec<u32>>()
    );
}

#[test]
fn a_derived_constant_is_computed_from_the_value_the_pipeline_supplied() {
    // `OpSpecConstantOp` doubling an open constant. If the derivation were evaluated against the
    // *default* instead of the supplied value, this would add 2 rather than 20 — and the module
    // would still be valid, so only a dispatch says which happened.
    let Some(gpu) = device("specialize-derived") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::specialized_derived::<32>(limits.subgroup_size, 1).expect("built");
    let input = ramp();

    let output = gpu
        .run_specialized(
            &spirv,
            &input,
            1,
            &Specialization::none().set(spec_id::ADDEND, 10),
        )
        .expect("dispatched");

    assert_eq!(
        output,
        input.iter().map(|value| value + 20).collect::<Vec<u32>>(),
        "the derived constant should be twice the supplied 10, not twice the default 1"
    );
}

/// Can the *cluster size* be deferred to pipeline creation?
///
/// `notes/NEXT.md` asked, on the grounds that `ClusterSize` must be a constant instruction and a
/// specialization constant is one. This is the answer, observed rather than argued —
/// `decisions/DR-0005` writes it up.
///
/// The reference is computed from whatever size the pipeline was given, so a driver that silently
/// used the *default* instead would fail here rather than agreeing with itself.
#[test]
fn a_clustered_reduction_takes_its_cluster_size_from_the_pipeline() {
    let Some(gpu) = device("specialize-cluster") else {
        return;
    };
    let limits = gpu.limits().clone();

    if !limits.subgroup_clustered || !limits.subgroup_arithmetic {
        eprintln!("SKIPPED specialize-cluster: no clustered subgroup arithmetic");
        return;
    }
    if limits.subgroup_size != 32 {
        eprintln!("SKIPPED specialize-cluster: the sizes below assume a 32-wide subgroup");
        return;
    }

    // The default is 32 — the whole subgroup — so every override below asks for something the
    // default would not have given.
    let spirv = kernels::specialized_cluster(32, 32).expect("built");
    let input = ramp();

    for cluster in [4_u32, 8, 16] {
        let output = gpu
            .run_specialized(
                &spirv,
                &input,
                1,
                &Specialization::none().set(spec_id::CLUSTER, cluster),
            )
            .expect("dispatched");

        let cluster = cluster as usize;
        let expected: Vec<u32> = (0..count())
            .map(|lane| {
                let first = lane / cluster * cluster;
                (first..first + cluster).map(|index| index as u32).sum()
            })
            .collect();

        assert_eq!(output, expected, "cluster size {cluster}");
    }

    // And the default, so that "it read the specialization" is distinguishable from "it ignored
    // the operand and reduced the whole subgroup" — which is what a 32-wide answer looks like.
    let defaulted = gpu.run_u32(&spirv, &input, 1).expect("dispatched");
    let whole: Vec<u32> = (0..count())
        .map(|lane| {
            let first = lane / 32 * 32;
            (first..first + 32).map(|index| index as u32).sum()
        })
        .collect();
    assert_eq!(defaulted, whole);
}

#[test]
fn a_specialization_naming_an_id_the_module_does_not_have_is_ignored() {
    // Vulkan says an entry whose `constant_id` matches nothing is ignored. Worth pinning because
    // the alternative — a driver refusing the pipeline — would make a caller's spare entry a
    // crash, and code that sets a superset of what a module declares is an easy thing to write.
    let Some(gpu) = device("specialize-unknown") else {
        return;
    };
    let limits = gpu.limits().clone();

    let spirv = kernels::specialized_add::<U32, 32>(limits.subgroup_size, 4).expect("built");
    let input = ramp();

    let output = gpu
        .run_specialized(
            &spirv,
            &input,
            1,
            &Specialization::none()
                .set(spec_id::ADDEND, 6)
                .set(99, 12345),
        )
        .expect("a spare entry should not be a failure");

    assert_eq!(
        output,
        input.iter().map(|value| value + 6).collect::<Vec<u32>>()
    );
}
