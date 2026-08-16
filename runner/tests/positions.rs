//! `Lanes::position`, at all three mappings, on a real device.
//!
//! A lane's index **within its own vector** is the one thing a butterfly network needs and the lane
//! API could not answer. [`simdr::lanes::Lanes::butterfly`] hands a lane the value at `l ^ mask`,
//! and every algorithm built on that — a Walsh–Hadamard or Fourier transform, a bitonic sort, a
//! hand-rolled scan — then has to know whether it is the low or the high half of the pair, which is
//! one bit of its own position. Without it the exchange is symmetric and the algorithm cannot be
//! written.
//!
//! # Why the assertions never say which invocation holds what
//!
//! Because nothing may. A reference that mapped invocation *i* to lane `i % width` would be
//! asserting that subgroups are cut from consecutive workgroup indices, which Vulkan guarantees only
//! for a pipeline that asked for full subgroups — and `decisions/DR-0002` records this project
//! deciding not to require that extension. So the checks here are the ones that hold whatever the
//! implementation does with its lanes:
//!
//! * every position is inside the vector;
//! * the positions **tile** it — each of `0..LANES` appears exactly as often as it must;
//! * and `position ^ butterfly(position, mask) == mask`, everywhere, which is the contract the
//!   algorithm actually rests on.
//!
//! The third is the one worth having. The first two would pass for a `position` that returned the
//! subgroup lane and never masked it into its cluster; the third would not.

mod common;

use common::{VULKAN_1_1, device, validate};
use runner::Gpu;
use runner::kernels::WORKGROUP_SIZE;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{LaneError, Mapping, U32};

/// Invocations per workgroup.
const WORKGROUP: u32 = WORKGROUP_SIZE;

/// Every position, written to the buffer at each invocation's own slots.
fn positions<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, WORKGROUP, 2))?;
    let mine = {
        let mut lanes = kernel.lanes()?;
        lanes.position::<LANES>()?
    };
    kernel.store::<LANES>(1, mine)?;
    kernel.finish()
}

/// The same, after a butterfly at `mask` — so the two buffers can be compared elementwise.
fn paired<const LANES: u32>(subgroup: u32, mask: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, WORKGROUP, 2))?;
    let partner = {
        let mut lanes = kernel.lanes()?;
        let mine = lanes.position::<LANES>()?;
        lanes.butterfly(mine, mask)?
    };
    kernel.store::<LANES>(1, partner)?;
    kernel.finish()
}

/// One lane count, built and run and held to the three properties above.
///
/// Returns the mapping it ran at, so the caller can report which of the three were reached — a run
/// that exercised one mapping three times and called it three mappings is the failure this whole
/// file is written against.
fn check<const LANES: u32>(gpu: &Gpu, subgroup: u32) -> Option<Mapping> {
    let mapping = Mapping::of(LANES, subgroup).ok()?;

    let spirv = match positions::<LANES>(subgroup) {
        Ok(spirv) => spirv,
        Err(refused) => {
            eprintln!("SKIPPED position at {LANES} lanes: {refused}");
            return None;
        }
    };
    if let Err(complaint) = validate(&spirv, &format!("position-{LANES}"), VULKAN_1_1) {
        panic!("spirv-val rejected position at {LANES} lanes: {complaint}");
    }
    if !gpu.limits().unsupported_in(&spirv).is_empty() {
        eprintln!("SKIPPED position at {LANES} lanes: unsupported");
        return None;
    }

    let strips = match mapping {
        Mapping::Strips { count } => count,
        _ => 1,
    };
    let slots = (WORKGROUP * strips) as usize;
    let empty = vec![0_u32; slots];
    let mine = gpu.run_u32(&spirv, &empty, 1).expect("dispatched");

    // Inside the vector.
    assert!(
        mine.iter().all(|&at| at < LANES),
        "{LANES} lanes at width {subgroup}: a position landed outside the vector — {mine:?}"
    );

    // And tiling it: every position exactly as often as the count demands, which is what rules out
    // a subgroup lane index that was never masked into its cluster.
    let due = slots / LANES as usize;
    for position in 0..LANES {
        let seen = mine.iter().filter(|&&at| at == position).count();
        assert_eq!(
            seen, due,
            "{LANES} lanes at width {subgroup}: position {position} appears {seen} times, not {due}"
        );
    }

    // The contract a butterfly network rests on. Masks are bounded by the vector *and* by the
    // subgroup: a strip-mined vector's butterfly shuffles inside each strip, so a mask that would
    // cross between them is refused rather than answered, and asking for one here would be
    // asserting the refusal is wrong.
    let reach = LANES.min(subgroup);
    let mut masks = 0;
    let mut mask = 1;
    while mask < reach {
        let spirv = paired::<LANES>(subgroup, mask).expect("built");
        if let Err(complaint) = validate(&spirv, &format!("butterfly-{LANES}-{mask}"), VULKAN_1_1) {
            panic!("spirv-val rejected a butterfly of a position: {complaint}");
        }
        let partner = gpu.run_u32(&spirv, &empty, 1).expect("dispatched");

        for (at, (&here, &there)) in mine.iter().zip(&partner).enumerate() {
            assert_eq!(
                here ^ there,
                mask,
                "{LANES} lanes at width {subgroup}: slot {at} holds position {here} and its \
                 partner at mask {mask} holds {there}, which differ in {}",
                here ^ there
            );
        }
        masks += 1;
        mask <<= 1;
    }

    // A vector of one has no pairs, and every wider one has at least one.
    assert_eq!(
        masks,
        reach.trailing_zeros(),
        "{LANES} lanes at width {subgroup}: {masks} butterfly masks were checked"
    );

    Some(mapping)
}

/// Every lane count this device has a mapping for, and the mappings they reached.
#[test]
fn a_lane_knows_where_it_is_in_its_own_vector() {
    let Some(gpu) = device("position") else {
        return;
    };
    let width = gpu.limits().subgroup_size;
    println!("{} — subgroup {width}", gpu.limits().name);

    let mut reached = Vec::new();
    macro_rules! at {
        ($($lanes:literal),+ $(,)?) => {
            $(if let Some(mapping) = check::<$lanes>(&gpu, width) {
                println!("  {:>3} lanes → {mapping:?}", $lanes);
                reached.push(mapping);
            })+
        };
    }
    at!(1, 2, 4, 8, 16, 32, 64, 128);

    assert!(
        reached
            .iter()
            .any(|mapping| matches!(mapping, Mapping::WholeSubgroup)),
        "no lane count matched this device's width, which every power of two up to 128 should"
    );
    assert!(
        reached
            .iter()
            .any(|mapping| matches!(mapping, Mapping::Clusters { .. })),
        "nothing clustered, so the masked form never ran"
    );
    assert!(
        reached
            .iter()
            .any(|mapping| matches!(mapping, Mapping::Strips { .. })),
        "nothing strip-mined, so the per-strip form never ran"
    );
}
