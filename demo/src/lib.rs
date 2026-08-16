//! Procedural worlds on the GPU, generated from nothing but the coordinates.
//!
//! A throwaway demonstration. See `README.md` for the isolation contract and for how to delete it.
//!
//! # Why this is a test and not a picture
//!
//! `notes/FINDINGS.md` records the entry requirement the last sandbox was built to: **a workload is
//! only a test if something can disagree with it.** A procedural world usually fails that — its
//! answer is "does it look right", which no CPU can check and no test can assert.
//!
//! So everything here is **integer arithmetic**. A value-noise heightmap, a cave mask packed one
//! bit per layer, and an escape-time fractal in fixed point are all exactly reproducible on the
//! host, bit for bit, at every width and on every device. The pictures are a side effect; the point
//! is that a device and a CPU can be held to the same number.
//!
//! # What the engine's own rules cost, and what they bought
//!
//! **No branches.** `decisions/DR-0003` refuses a per-lane branch, and procedural generation is
//! written with them everywhere — *if the density is above the threshold, place stone*. Every one
//! of those is a comparison and a `select` here, which is what the hardware would have done with
//! the branch anyway: a divergent branch runs both sides and masks.
//!
//! **No exclusive-or.** `src/module/op.rs` declares no opcode for one, and the two bitwise
//! operations it does declare are not on the lane API — so the hash below mixes with **multiply,
//! add and shift** and nothing else. That is a real constraint and it shaped the code: `mix` is a
//! multiply-shift-add mixer rather than the xorshift everybody writes.
//!
//! **No division and no subtraction.** A difference is `add(a, mul(b, -1))` — one extra instruction
//! a driver folds — and a divide by a power of two is a shift. Both appear below.
//!
//! # The coordinates come from the invocation
//!
//! A `Vector<T, LANES>` at `LANES == subgroup` is one element per *invocation*, so a value splatted
//! from `Kernel::local_index` is a different number in every lane: the lane's own column. That is
//! the whole trick that makes generation-from-nothing work — no buffer of coordinates is uploaded
//! and the input buffer is never read.

use runner::kernels::WORKGROUP_SIZE;
use runner::{Error, Gpu, Grid};
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{LaneError, Lanes, U32, Vector};
use simdr::spec::Capability;

/// Invocations per workgroup — the engine's own, so a workgroup covers 64 columns.
pub const WORKGROUP: u32 = WORKGROUP_SIZE;

/// The width every world here is generated at, in columns.
///
/// A multiple of [`WORKGROUP`], because a grid dispatches whole workgroups across and a remainder
/// would be a column nobody wrote.
pub const PITCH: u32 = 256;

/// The multipliers the mixer stirs with.
///
/// Odd, and large enough that the high bits move: an even multiplier throws a bit of state away per
/// factor of two, and the whole job of a hash is not to.
const GOLDEN: u32 = 0x9E37_79B1;
const STIR: u32 = 0x85EB_CA6B;

/// One `u32` mixed on the host, and the reference for everything below.
///
/// **Multiply, shift, add — no exclusive-or**, because the engine declares no opcode for one. The
/// usual `h ^= h >> 16` becomes `h += h >> 16`, which mixes less per round and mixes enough for a
/// world nobody is doing cryptography in. What it is, is *exact*, which is the property this whole
/// directory rests on.
///
/// Written here **and** emitted separately in [`mixed`]. Two spellings of one function on purpose:
/// a reference that shares a line with what it checks agrees with it about the same mistake.
#[must_use]
pub fn mix(value: u32) -> u32 {
    let h = value.wrapping_mul(GOLDEN);
    let h = h.wrapping_add(h >> 16);
    let h = h.wrapping_mul(STIR);
    h.wrapping_add(h >> 13)
}

/// Two coordinates into one number, on the host.
#[must_use]
pub fn mix2(x: u32, y: u32) -> u32 {
    mix(mix(x).wrapping_add(y.wrapping_mul(GOLDEN)))
}

/// The same mixer, emitted.
///
/// Every step is an instruction the lane API offers: `mul`, `add`, `shift_right_logical`. The
/// shift distances and the multipliers are [`mix`]'s, because a mixer that differed by one bit
/// would disagree everywhere at once — which is exactly what the tests would report.
fn mixed<const LANES: u32>(
    lanes: &mut Lanes<'_>,
    value: Vector<U32, LANES>,
) -> Result<Vector<U32, LANES>, LaneError> {
    let golden = lanes.splat_bits::<U32, LANES>(GOLDEN)?;
    let stir = lanes.splat_bits::<U32, LANES>(STIR)?;
    let by16 = lanes.splat_bits::<U32, LANES>(16)?;
    let by13 = lanes.splat_bits::<U32, LANES>(13)?;

    let h = lanes.mul(value, golden)?;
    let folded = lanes.shift_right_logical(h, by16)?;
    let h = lanes.add(h, folded)?;

    let h = lanes.mul(h, stir)?;
    let folded = lanes.shift_right_logical(h, by13)?;
    lanes.add(h, folded)
}

/// One octave of value noise: two coordinates mixed together.
fn octave<const LANES: u32>(
    lanes: &mut Lanes<'_>,
    x: Vector<U32, LANES>,
    y: Vector<U32, LANES>,
) -> Result<Vector<U32, LANES>, LaneError> {
    let golden = lanes.splat_bits::<U32, LANES>(GOLDEN)?;
    let mixed_x = mixed(lanes, x)?;
    let spread_y = lanes.mul(y, golden)?;
    let together = lanes.add(mixed_x, spread_y)?;
    mixed(lanes, together)
}

/// This invocation's own column, as a vector.
///
/// `workgroup_index × WORKGROUP + local_index`, splatted — and the splat is what makes it a *ramp*
/// rather than a constant: at `LANES == subgroup` each lane is a different invocation, so each one
/// splats its own number.
fn column<const LANES: u32>(kernel: &mut Kernel<U32>) -> Result<Vector<U32, LANES>, LaneError> {
    let uint = kernel.index_type();
    let stride = kernel.module().constant_u32(WORKGROUP)?;
    let group = kernel.workgroup_index();
    let local = kernel.local_index();

    let base = kernel.module().i_mul(uint, group, stride)?;
    let index = kernel.module().i_add(uint, base, local)?;
    kernel.lanes()?.splat_id::<U32, LANES>(index)
}

/// A height per column of a `pitch`-wide world, from two octaves of value noise.
///
/// One octave is visibly a grid of independent numbers; two is a landscape. The coarse layer runs
/// at a sixteenth of the resolution and carries the hills, the fine one at a quarter carries the
/// roughness, and they are added eight to one.
///
/// **Eight to one and then a shift, rather than a ninth**, because there is no division. The
/// reference weights it the same way, so the two agree exactly and the picture is the picture on
/// both — an approximation of what a landscape "should" weigh, and an exact agreement about what
/// this one does.
///
/// # Errors
///
/// [`LaneError`] if `LANES` has no mapping onto this subgroup, or the module cannot be built.
pub fn heights<const LANES: u32>(subgroup: u32, pitch: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::grid(subgroup, WORKGROUP, 1, 2))?;
    let x = column::<LANES>(&mut kernel)?;
    let row = kernel.row()?;

    let height = {
        let mut lanes = kernel.lanes()?;
        let y = lanes.splat_id::<U32, LANES>(row)?;

        let by4 = lanes.splat_bits::<U32, LANES>(4)?;
        let by2 = lanes.splat_bits::<U32, LANES>(2)?;
        let by24 = lanes.splat_bits::<U32, LANES>(24)?;
        let eight = lanes.splat_bits::<U32, LANES>(8)?;

        let coarse_x = lanes.shift_right_logical(x, by4)?;
        let coarse_y = lanes.shift_right_logical(y, by4)?;
        let coarse = octave::<LANES>(&mut lanes, coarse_x, coarse_y)?;

        let fine_x = lanes.shift_right_logical(x, by2)?;
        let fine_y = lanes.shift_right_logical(y, by2)?;
        let fine = octave::<LANES>(&mut lanes, fine_x, fine_y)?;

        let weighted = lanes.mul(coarse, eight)?;
        let total = lanes.add(weighted, fine)?;
        lanes.shift_right_logical(total, by24)?
    };

    kernel.store_row(1, pitch, height)?;
    kernel.finish()
}

/// The heights the device should produce, computed on the host.
///
/// **Written from the arithmetic rather than from the kernel.** It re-derives each column from the
/// dispatch's own shape instead of asking the module what it did, which is the only way a reference
/// is worth having.
#[must_use]
pub fn heights_reference(pitch: u32, rows: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity((pitch as usize) * (rows as usize));
    for y in 0..rows {
        for x in 0..pitch {
            let coarse = mix2(x >> 4, y >> 4);
            let fine = mix2(x >> 2, y >> 2);
            out.push(coarse.wrapping_mul(8).wrapping_add(fine) >> 24);
        }
    }
    out
}

/// Layers in a cave column, one bit each in the word that comes back.
pub const LAYERS: u32 = 32;

/// A layer is open where its density exceeds this.
///
/// Near the top of the range, so most of the world is rock and the caves are caves. Chosen by
/// looking at the pictures, which is the one number in this directory that is a matter of taste.
pub const OPEN_ABOVE: u32 = 0xB000_0000;

/// Which of a column's 32 layers are open, one bit each.
///
/// A cave system is a 3D density field with a threshold on it, and a threshold is a *branch* in
/// every tutorial that writes one. `decisions/DR-0003` refuses a per-lane branch, so this is a
/// comparison and a `select`.
///
/// **The bits are added rather than or'd**, and that is not a shortcut: each trip contributes at
/// most one bit and they are all different, so a sum and a bitwise or are the same number. The lane
/// API offers no `or`, and this needed none.
///
/// One rolled loop with two phis — the layer counter and the accumulating word — which is the
/// control-flow shape `runner/src/fuzz` generates and checks against a CPU reference on every push.
///
/// # Errors
///
/// As [`heights`].
pub fn caves<const LANES: u32>(subgroup: u32, pitch: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::grid(subgroup, WORKGROUP, 1, 2))?;
    let x = column::<LANES>(&mut kernel)?;
    let row = kernel.row()?;
    let element = kernel.element();

    let packed = {
        let mut lanes = kernel.lanes()?;
        let y = lanes.splat_id::<U32, LANES>(row)?;
        let seed = octave::<LANES>(&mut lanes, x, y)?;
        let seed = seed.id();
        let zero = lanes.splat_bits::<U32, LANES>(0)?;
        let zero = zero.id();

        // The loop carries `Id`s, so the vector's single strip goes in and comes back out. At
        // `LANES == subgroup` a vector is one element per invocation and has exactly one strip,
        // which is why this file fixes the mapping rather than taking whatever fits.
        let bits = lanes.repeat_rolled(LAYERS, element, zero, |lanes, held, layer| {
            let carried = lanes.from_lane_value::<U32, 1>(held)?;
            let base = lanes.from_lane_value::<U32, 1>(seed)?;
            let depth = lanes.from_lane_value::<U32, 1>(layer)?;

            let golden = lanes.splat_bits::<U32, 1>(GOLDEN)?;
            let spread = lanes.mul(depth, golden)?;
            let stirred = lanes.add(base, spread)?;
            let density = mixed(lanes, stirred)?;

            // Open where the density clears the threshold: a comparison and a select, never a
            // branch. The constant is what decides how much of the world is cave.
            let threshold = lanes.splat_bits::<U32, 1>(OPEN_ABOVE)?;
            let open = lanes.greater_than(density, threshold)?;
            let one = lanes.splat_bits::<U32, 1>(1)?;
            let none = lanes.splat_bits::<U32, 1>(0)?;
            let bit = lanes.select(open, one, none)?;

            let shifted = lanes.shift_left(bit, depth)?;
            Ok(lanes.add(carried, shifted)?.id())
        })?;

        lanes.from_strips::<U32, LANES>(&[bits])?
    };

    kernel.store_row(1, pitch, packed)?;
    kernel.finish()
}

/// The cave words the device should produce.
#[must_use]
pub fn caves_reference(pitch: u32, rows: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity((pitch as usize) * (rows as usize));
    for y in 0..rows {
        for x in 0..pitch {
            let seed = mix2(x, y);
            let mut word = 0_u32;
            for layer in 0..LAYERS {
                let density = mix(seed.wrapping_add(layer.wrapping_mul(GOLDEN)));
                if density > OPEN_ABOVE {
                    word = word.wrapping_add(1 << layer);
                }
            }
            out.push(word);
        }
    }
    out
}

/// How many escape-time iterations the fractal runs.
pub const ORBITS: u32 = 40;

/// The fixed-point scale: sixteen fractional bits.
const ONE: i32 = 1 << 16;

/// Where the fractal's window starts and how far one step moves it, in Q16.16.
/// Chosen for a terminal: 110 columns across the interesting three units of the real axis, and 24
/// rows across two and a half of the imaginary one. A window is a matter of taste and this is the
/// second number in this file that is.
const X_ORIGIN: i32 = -(2 * ONE + ONE / 5);
const X_STEP: i32 = ONE / 36;
const Y_ORIGIN: i32 = -(ONE + ONE / 4);
const Y_STEP: i32 = ONE / 10;

/// The escape radius squared, in Q16.16: four.
const ESCAPE: u32 = 4 * (ONE as u32);

/// An escape-time fractal over a `pitch`-wide grid, in **fixed point**.
///
/// The mathematical pattern of the three, and the one that has to be integer to be checkable at
/// all: a float Mandelbrot compares two roundings and says nothing about the mapping, which is the
/// argument `runner/src/fuzz/domain.rs` makes about its own float domain.
///
/// **Branch-free escape counting.** The textbook loop breaks when the orbit leaves the disc; this
/// one runs every iteration and carries an `alive` flag that a `select` can only ever turn off —
/// so a point that escapes stops counting and never resumes, even though its `z` goes on wrapping.
/// Same answer, no divergence, and `decisions/DR-0003` could not have been satisfied any other way.
///
/// The products are `(a >> 8) × (b >> 8)`, which keeps a Q16.16 multiply inside 32 bits at the cost
/// of the low eight bits of each factor, and the shift is **arithmetic** because the values are
/// signed. The reference does exactly the same thing, so the two agree bit for bit: an
/// approximation of the mathematics and an exact agreement about the arithmetic, which is the only
/// kind this directory can check.
///
/// # Errors
///
/// As [`heights`].
pub fn orbits<const LANES: u32>(subgroup: u32, pitch: u32) -> Result<Vec<u32>, LaneError> {
    let mut kernel = Kernel::<U32>::new(Shape::grid(subgroup, WORKGROUP, 1, 2))?;
    let x = column::<LANES>(&mut kernel)?;
    let row = kernel.row()?;

    let counted = {
        let mut lanes = kernel.lanes()?;
        let y = lanes.splat_id::<U32, LANES>(row)?;

        // The complex point, as Q16.16 read out of a `u32`'s bits. `Vector<U32, _>` carries the
        // words unchanged and the arithmetic below is the signed reading of them.
        let cx = scaled::<LANES>(&mut lanes, x, X_ORIGIN, X_STEP)?;
        let cy = scaled::<LANES>(&mut lanes, y, Y_ORIGIN, Y_STEP)?;

        let mut zx = lanes.splat_bits::<U32, LANES>(0)?;
        let mut zy = lanes.splat_bits::<U32, LANES>(0)?;
        let mut alive = lanes.splat_bits::<U32, LANES>(1)?;
        let mut count = lanes.splat_bits::<U32, LANES>(0)?;

        let escape = lanes.splat_bits::<U32, LANES>(ESCAPE)?;
        let zero = lanes.splat_bits::<U32, LANES>(0)?;
        let one = lanes.splat_bits::<U32, LANES>(1)?;
        let two = lanes.splat_bits::<U32, LANES>(2)?;
        let minus = lanes.splat_bits::<U32, LANES>(u32::MAX)?;

        // **A Rust loop rather than `Lanes::repeat_rolled`**, and the reason is the state: a rolled
        // loop carries one value through its phi and this carries four. Unrolled is what
        // `Lanes::repeat` does anyway, and `caves` above is where the rolled shape is shown.
        for _ in 0..ORBITS {
            let zx2 = squared::<LANES>(&mut lanes, zx, zx)?;
            let zy2 = squared::<LANES>(&mut lanes, zy, zy)?;
            let magnitude = lanes.add(zx2, zy2)?;

            // `alive` can only be turned off: multiplied by 1 while inside and by 0 once out.
            let escaped = lanes.greater_than(magnitude, escape)?;
            let inside = lanes.select(escaped, zero, one)?;
            alive = lanes.mul(alive, inside)?;
            count = lanes.add(count, alive)?;

            let cross = squared::<LANES>(&mut lanes, zx, zy)?;
            let negated = lanes.mul(zy2, minus)?;
            let real = lanes.add(zx2, negated)?;
            let doubled = lanes.mul(cross, two)?;
            zx = lanes.add(real, cx)?;
            zy = lanes.add(doubled, cy)?;
        }

        count
    };

    kernel.store_row(1, pitch, counted)?;
    kernel.finish()
}

/// `origin + index × step`, in fixed point.
fn scaled<const LANES: u32>(
    lanes: &mut Lanes<'_>,
    index: Vector<U32, LANES>,
    origin: i32,
    step: i32,
) -> Result<Vector<U32, LANES>, LaneError> {
    let step = lanes.splat_bits::<U32, LANES>(step as u32)?;
    let origin = lanes.splat_bits::<U32, LANES>(origin as u32)?;
    let moved = lanes.mul(index, step)?;
    lanes.add(moved, origin)
}

/// A Q16.16 product, taken as `(a >> 8) × (b >> 8)`.
///
/// **Arithmetic shifts**, because the operands are signed and a logical one would turn every
/// negative coordinate into an enormous positive one. `Lanes::shift_right_arithmetic` spreads the
/// element's own top bit whatever the type's signedness says, which is SPIR-V's rule and exactly
/// what a signed fixed-point multiply needs from a `Vector<U32, _>`.
fn squared<const LANES: u32>(
    lanes: &mut Lanes<'_>,
    a: Vector<U32, LANES>,
    b: Vector<U32, LANES>,
) -> Result<Vector<U32, LANES>, LaneError> {
    let by8 = lanes.splat_bits::<U32, LANES>(8)?;
    let a = lanes.shift_right_arithmetic(a, by8)?;
    let b = lanes.shift_right_arithmetic(b, by8)?;
    lanes.mul(a, b)
}

/// The escape counts the device should produce.
#[must_use]
pub fn orbits_reference(pitch: u32, rows: u32) -> Vec<u32> {
    let square = |a: i32, b: i32| (a >> 8).wrapping_mul(b >> 8);

    let mut out = Vec::with_capacity((pitch as usize) * (rows as usize));
    for y in 0..rows {
        for x in 0..pitch {
            let cx = (x as i32).wrapping_mul(X_STEP).wrapping_add(X_ORIGIN);
            let cy = (y as i32).wrapping_mul(Y_STEP).wrapping_add(Y_ORIGIN);

            let (mut zx, mut zy) = (0_i32, 0_i32);
            let (mut alive, mut count) = (1_u32, 0_u32);

            for _ in 0..ORBITS {
                let zx2 = square(zx, zx);
                let zy2 = square(zy, zy);
                let magnitude = (zx2 as u32).wrapping_add(zy2 as u32);

                alive *= u32::from(magnitude <= ESCAPE);
                count += alive;

                let cross = square(zx, zy);
                zx = zx2.wrapping_sub(zy2).wrapping_add(cx);
                zy = cross.wrapping_mul(2).wrapping_add(cy);
            }
            out.push(count);
        }
    }
    out
}

/// What ran, or the one reason it did not.
///
/// Four ways of not running and one of having run, which `decisions/DR-0009` argues for and which
/// this directory needs for the reason the last sandbox did: a check that was skipped and looks
/// green is worse than one that failed.
#[derive(Debug)]
pub enum Answer<T> {
    /// The lane API refused to build it — a width with no mapping onto this device.
    Refused(LaneError),
    /// The device does not offer what the module declares.
    Unsupported(Vec<Capability>),
    /// The driver took the module and failed.
    Errored(Error),
    /// It ran, and here is what came back.
    Ran(T),
}

impl<T> Answer<T> {
    /// Why it did not run, phrased for a report. `None` when it ran.
    #[must_use]
    pub fn why(&self) -> Option<String> {
        match self {
            Self::Ran(_) => None,
            Self::Refused(why) => Some(format!("refused: {why}")),
            Self::Unsupported(missing) => Some(format!("unsupported: {missing:?}")),
            Self::Errored(error) => Some(format!("errored: {error}")),
        }
    }
}

/// Build at this device's width, check what the module declares, and generate a `pitch × rows`
/// world.
///
/// The input buffer is never read — every number in the answer comes from the invocation's own
/// coordinates — and it exists because `Gpu::run_grid` sizes the output from it.
pub fn generate<F>(gpu: &Gpu, build: F, pitch: u32, rows: u32) -> Answer<Vec<u32>>
where
    F: FnOnce(u32, u32) -> Result<Vec<u32>, LaneError>,
{
    let spirv = match build(gpu.limits().subgroup_size, pitch) {
        Ok(spirv) => spirv,
        Err(refused) => return Answer::Refused(refused),
    };

    let missing = gpu.limits().unsupported_in(&spirv);
    if !missing.is_empty() {
        return Answer::Unsupported(missing);
    }

    let empty = vec![0_u32; (pitch as usize) * (rows as usize)];
    match gpu.run_grid(&spirv, &empty, Grid::new(pitch / WORKGROUP, rows)) {
        Ok(world) => Answer::Ran(world),
        Err(error) => Answer::Errored(error),
    }
}

/// One entry point per generator, choosing `LANES` from the device's width at runtime.
///
/// **`LANES` is a const generic and a device's width is not**, so the widths have to be listed —
/// `decisions/DR-0002` is why the number cannot simply be passed through, and the arms below are
/// every width this repository has ever run at. One that is not here is *refused by name* rather
/// than guessed at, which is the whole of that record in one match.
macro_rules! at_every_width {
    ($($name:ident from $inner:ident),+ $(,)?) => {
        $(
            /// The generator above, built for whatever width the device reports.
            ///
            /// # Errors
            ///
            /// [`LaneError::NoMapping`] for a width with no arm, otherwise as the generator.
            pub fn $name(subgroup: u32, pitch: u32) -> Result<Vec<u32>, LaneError> {
                match subgroup {
                    4 => $inner::<4>(subgroup, pitch),
                    8 => $inner::<8>(subgroup, pitch),
                    16 => $inner::<16>(subgroup, pitch),
                    32 => $inner::<32>(subgroup, pitch),
                    64 => $inner::<64>(subgroup, pitch),
                    other => Err(LaneError::NoMapping {
                        lanes: other,
                        width: subgroup,
                    }),
                }
            }
        )+
    };
}

at_every_width!(
    landscape from heights,
    caverns from caves,
    fractal from orbits,
);
