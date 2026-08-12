//! The programs the fuzzer generates, and how they become SPIR-V.
//!
//! A program is a straight line of operations over one accumulator, ending in a reduction. That
//! is narrow, and deliberately so: every operation in it has a CPU meaning that can be stated in
//! one line, which is what makes `interpret` short enough to trust as a reference. A reference
//! with bugs of its own proves nothing.

mod emit;

use self::emit::apply;
use super::domain::Domain;
use simdr::kernel::{Kernel, Shape};
use simdr::lanes::{Element, F16, F32, I8, I16, I32, LaneError, U8, U16, U32};

/// One step of a generated program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Add a constant to every element.
    AddConstant(u32),
    /// Multiply every element by a constant.
    MulConstant(u32),
    /// Add the value held `mask` lanes away, exclusive-or.
    ButterflyAdd(u32),
    /// Replace each element by the one `delta` lanes below, where that lane exists.
    ///
    /// Only ever generated with a delta of zero: SPIR-V leaves the out-of-range lanes undefined
    /// and a reference cannot predict undefined.
    ShiftUp(u32),
    /// Keep the element where it exceeds a constant, otherwise substitute that constant.
    ///
    /// The *core* spelling of a lower bound: a comparison and a select. [`Op::MaxConstant`] is the
    /// same arithmetic through one extended instruction, and the pair is worth having for the same
    /// reason [`Op::RepeatAdd`] and [`Op::RolledAdd`] are — one answer, two instruction streams,
    /// and they must agree.
    ClampBelow(u32),
    /// The smaller of each element and a constant, through `OpExtInst` `*Min`.
    ///
    /// Which of `FMin`, `UMin` and `SMin` is reached depends on the domain, and the three agree on
    /// every value below 2³¹ — so what this catches is not the choice between them but whether the
    /// instruction computes a minimum at all. The signed-versus-unsigned confusion needs a value
    /// with its top bit set, which is `runner/tests/extended.rs`.
    MinConstant(u32),
    /// The larger of each element and a constant, through `OpExtInst` `*Max`.
    MaxConstant(u32),
    /// Each element held between two constants, through `OpExtInst` `*Clamp`.
    ///
    /// The bounds are generated in order, because a clamp whose low bound is above its high one is
    /// undefined and a reference cannot predict undefined.
    ClampBoth {
        /// The lower bound.
        low: u32,
        /// The upper bound, never below `low`.
        high: u32,
    },
    /// Add a constant, but only where some element of the subgroup exceeds `when_any_above`.
    ///
    /// A uniform branch: the condition is a vote, so the whole subgroup takes it or none of it
    /// does. The reference can predict that exactly, which is the only reason this is fuzzable —
    /// a per-lane branch would leave a subgroup operation inside it answering for whichever lanes
    /// happened to be running. See `decisions/DR-0003`.
    AddIfAnyAbove {
        /// The threshold every element is compared against.
        when_any_above: u32,
        /// What to add where the vote passes.
        add: u32,
    },
    /// Add a constant `times` times over, as an *unrolled* loop.
    ///
    /// Arithmetically it is one multiplication, and that is the point: the answer is trivial and
    /// the emitted shape is not. `Lanes::repeat` threads a value through a Rust loop, and a
    /// threading bug would show up here as an off-by-one nobody could mistake for rounding.
    RepeatAdd {
        /// How many iterations.
        times: u32,
        /// What each one adds.
        add: u32,
    },
    /// Add a constant on each of `times` trips of a *rolled* loop.
    ///
    /// The four-block shape with its two `OpPhi`s, in every domain. Same arithmetic as
    /// [`Op::RepeatAdd`] and a completely different instruction stream, which is what makes the
    /// pair worth having: they must agree, and only one of them has a back edge.
    RolledAdd {
        /// How many trips.
        times: u32,
        /// What each one adds.
        add: u32,
    },
    /// Add the iteration number on each of `times` trips of a *rolled* loop.
    ///
    /// The counter is a `u32` whatever the domain is, so this also exercises
    /// `Lanes::convert_u32` — and in the float domain a reinterpretation instead of a conversion
    /// would turn iteration 3 into a denormal, which the reference would catch at once.
    ///
    /// The one op here that cannot be written any other way. `Lanes::repeat_rolled` emits the
    /// four-block shape with two `OpPhi`s, and the body — built once — reads the counter phi. The
    /// total added is `times × (times − 1) / 2`, which is wrong for every mistake worth making: a
    /// body handed a copy of the counter adds zero, an off-by-one trip count lands one triangular
    /// number away.
    RolledCounterAdd {
        /// How many trips.
        times: u32,
    },
}

/// How a program's final reduction combines the lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// Sum across the group.
    Sum,
    /// Largest across the group.
    Max,
    /// Smallest across the group.
    Min,
    /// Sum when any element exceeds the threshold, otherwise the largest.
    ///
    /// The only finish that carries a value out of a branch. Both arms end in a subgroup
    /// reduction and exactly one runs, so the answer says which edge the `OpPhi` read from — and a
    /// phi naming the wrong predecessor validates cleanly and computes the wrong thing, which is
    /// the failure no other layer here catches.
    SumOrMax {
        /// The threshold the vote compares against.
        when_any_above: u32,
    },
}

/// A generated program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// Which element type it computes in.
    pub domain: Domain,
    /// The device's subgroup width, which decides the mapping.
    pub subgroup: u32,
    /// Invocations per workgroup.
    pub workgroup: u32,
    /// How many workgroups to dispatch.
    pub groups: u32,
    /// The vector width, as a `LANES` the builder will instantiate.
    pub lanes: u32,
    /// The steps, in order.
    pub steps: Vec<Op>,
    /// How it ends.
    pub finish: Finish,
}

impl Program {
    /// How many workgroups this dispatches.
    #[must_use]
    pub const fn workgroups(&self) -> u32 {
        self.groups
    }

    /// How many elements the input buffer must hold.
    #[must_use]
    pub const fn input_len(&self) -> usize {
        let strips = if self.lanes > self.subgroup {
            self.lanes / self.subgroup
        } else {
            1
        };
        (self.groups * self.workgroup * strips) as usize
    }

    /// Emit this program as SPIR-V.
    ///
    /// # Errors
    ///
    /// [`LaneError`] when the lane count has no mapping onto this subgroup, which is a legitimate
    /// answer rather than a failure.
    pub fn build(&self) -> Result<Vec<u32>, LaneError> {
        match self.domain {
            Domain::Unsigned => self.build_in::<U32>(),
            Domain::Signed => self.build_in::<I32>(),
            Domain::Float => self.build_in::<F32>(),
            Domain::UnsignedByte => self.build_in::<U8>(),
            Domain::Byte => self.build_in::<I8>(),
            Domain::UnsignedShort => self.build_in::<U16>(),
            Domain::Short => self.build_in::<I16>(),
            Domain::Half => self.build_in::<F16>(),
        }
    }

    /// The same, with the element type chosen.
    fn build_in<T: Element>(&self) -> Result<Vec<u32>, LaneError> {
        // `LANES` is a const generic and the generator picks it at runtime, so the widths have to
        // be listed. These are every power of two the mapping can express on a 32- or 64-lane
        // device with `MAX_STRIPS` of headroom.
        match self.lanes {
            1 => self.build_at::<T, 1>(),
            2 => self.build_at::<T, 2>(),
            4 => self.build_at::<T, 4>(),
            8 => self.build_at::<T, 8>(),
            16 => self.build_at::<T, 16>(),
            32 => self.build_at::<T, 32>(),
            64 => self.build_at::<T, 64>(),
            128 => self.build_at::<T, 128>(),
            256 => self.build_at::<T, 256>(),
            other => Err(LaneError::NoMapping {
                lanes: other,
                width: self.subgroup,
            }),
        }
    }

    fn build_at<T: Element, const LANES: u32>(&self) -> Result<Vec<u32>, LaneError> {
        let mut kernel = Kernel::<T>::new(Shape::new(self.subgroup, self.workgroup, 2))?;
        let mut value = kernel.load::<LANES>(0)?;

        for step in &self.steps {
            value = apply::<T, LANES>(&mut kernel.lanes()?, self.domain, value, *step)?;
        }

        let element = kernel.element();
        let total = match self.finish {
            Finish::Sum => kernel.lanes()?.reduce_sum(value)?,
            Finish::Max => kernel.lanes()?.reduce_max(value)?,
            Finish::Min => kernel.lanes()?.reduce_min(value)?,
            Finish::SumOrMax { when_any_above } => {
                let mut lanes = kernel.lanes()?;
                let limit = lanes.splat_bits::<T, LANES>(self.domain.encode(when_any_above))?;
                let above = lanes.greater_than(value, limit)?;
                let vote = lanes.any_uniform(above)?;

                lanes.choose_uniform(
                    vote,
                    element,
                    |lanes| lanes.reduce_sum(value),
                    |lanes| lanes.reduce_max(value),
                )?
            }
        };
        kernel.store_scalar(1, total)?;
        kernel.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_input_length_covers_every_strip() {
        let program = Program {
            domain: Domain::Unsigned,
            subgroup: 32,
            workgroup: 64,
            groups: 2,
            lanes: 128,
            steps: Vec::new(),
            finish: Finish::Sum,
        };

        // Four strips per invocation, 64 invocations per workgroup, two workgroups.
        assert_eq!(program.input_len(), 4 * 64 * 2);
    }

    #[test]
    fn the_two_domains_emit_different_instructions_from_one_program() {
        let base = Program {
            domain: Domain::Unsigned,
            subgroup: 32,
            workgroup: 64,
            groups: 1,
            lanes: 32,
            steps: vec![Op::AddConstant(1)],
            finish: Finish::Sum,
        };
        let floats = Program {
            domain: Domain::Float,
            ..base.clone()
        };

        let integer_words = base.build().expect("built");
        let float_words = floats.build().expect("built");

        assert_ne!(
            integer_words, float_words,
            "the same program in two domains must not produce the same module"
        );
    }
}
