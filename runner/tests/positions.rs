mod common;

use common::{VULKAN_1_1, device, validate};
use runner::Gpu;
use runner::kernels::WORKGROUP_SIZE;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{LaneError, Mapping, U32};

const WORKGROUP: u32 = WORKGROUP_SIZE;

fn positions<const LANES: u32>(subgroup: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::new(subgroup, WORKGROUP, 2))?;
    let mine = {
        let mut lanes = kernel.lanes()?;
        lanes.position::<LANES>()?
    };
    kernel.store::<LANES>(1, mine)?;
    kernel.finish()
}

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

    assert!(
        mine.iter().all(|&at| at < LANES),
        "{LANES} lanes at width {subgroup}: a position landed outside the vector — {mine:?}"
    );

    let due = slots / LANES as usize;
    for position in 0..LANES {
        let seen = mine.iter().filter(|&&at| at == position).count();
        assert_eq!(
            seen, due,
            "{LANES} lanes at width {subgroup}: position {position} appears {seen} times, not {due}"
        );
    }

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

    assert_eq!(
        masks,
        reach.trailing_zeros(),
        "{LANES} lanes at width {subgroup}: {masks} butterfly masks were checked"
    );

    Some(mapping)
}

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
