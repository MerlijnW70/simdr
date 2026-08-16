//! What a module's own address arithmetic says each of its buffers needs.
//!
//! [`super`] decides whether a dispatch fits; this reads the module it decides about. The split is
//! the one this project keeps making: a file that walks a word stream and a file that compares two
//! numbers are two jobs, and the walk is the half with something to get wrong.
//!
//! # Per binding, not per module
//!
//! The check used to ask one question of a whole module — *how many elements does an invocation
//! touch* — and take the largest answer it could find. That is right when every buffer is the same
//! size, which is what [`crate::Gpu::run`] allocates, and wrong everywhere else: a reduction reads
//! `Simd<f32, 128>` from binding 0 and writes one scalar to binding 1, so the largest answer is
//! four and applying it to both would demand four times the buffer the output needs.
//!
//! Which buffer an access lands in is in the module — the access chain names the variable and the
//! variable carries a `Binding` decoration — so it is read rather than guessed, the same way the
//! workgroup size is.
//!
//! # What "per invocation" means, and which bindings do not have it
//!
//! An address that depends on [`BuiltIn::LocalInvocationId`] is one every invocation computes
//! differently, so `invocations × elements` of them are touched. An address that does not is
//! something else — `Kernel::store_at` writing one total per *workgroup* is the case this crate
//! actually has — and there is no invocation count to multiply. Those bindings are left out rather
//! than guessed at: over-counting one would refuse a dispatch that is fine, which is the direction
//! this check must never take.
//!
//! # How many elements each invocation touches
//!
//! Not declared anywhere, and it does not need to be. Every access starts from `Kernel::run_start`,
//! which emits `group × (workgroup × strips)` — one `OpIMul` whose left operand is the workgroup
//! index and whose right is a constant. The workgroup size is known from `LocalSize`, so dividing
//! that constant by it gives back the strip count the emitter used.
//!
//! # The constant past the run
//!
//! `Kernel::load_offset` reads `in[i + half]`, and that `half` used to be outside this entirely —
//! the safe direction, since it under-counts, but a hole all the same: a fold whose buffer is
//! exactly `invocations × strips` long reads `half` elements past the end of it and this said the
//! dispatch fit.
//!
//! It needed nothing new to read. The emitter folds the strip's stride and the caller's offset into
//! **one** constant — `Kernel::address` computes `strip × workgroup + offset` at build time — so the
//! constant added to the invocation's own lane already carries it, and the strip term is a number
//! this file has always known:
//!
//! ```text
//! address = group × (workgroup × strips)  +  local + (strip × workgroup + offset)
//!           \_____________ base _______/           \__________ shift __________/
//! ```
//!
//! The largest `shift` on a binding belongs to its last strip, so `shift - (strips - 1) × workgroup`
//! is the caller's `offset` and nothing else. A binding with no offset gives zero, which is the
//! arithmetic agreeing with the answer this file gave before.
//!
//! **`Kernel::load_offset_by` used to stay outside it**, and the reason was true and the
//! conclusion was not: its offset is a specialization constant, a number chosen after the module
//! was built with no literal here to read. So this counted zero for it — and a bound that
//! under-counts is a bound that **lets an overrun through**, which is the opposite of the direction
//! every other unreadable thing here takes.
//!
//! The number is not unknowable, it is known somewhere else. A pipeline is created *with* its
//! specialization, and every caller that bounds a dispatch has that value in scope at the moment it
//! asks. So [`super::Bounds::of`] takes it, [`specialized`] resolves each constant's id to the
//! value the pipeline will carry — the caller's, or the module's own default where the caller sets
//! nothing — and [`open_shift_in`] adds it like any other term.
//!
//! **`OpSpecConstantOp` is still outside, and this time the sentence is the right way round.** A
//! module may derive one constant from another — `offset x 2` — and resolving that would mean an
//! interpreter for constant expressions here. So the derived term is not counted, and the answer
//! stays a **floor**, which is what this file gives for everything it cannot read.
//!
//! Dropping such a binding from the answer entirely was tried and is *worse*, which the test
//! written to prove it worthwhile is what showed: a binding with no entry does not go unjudged, it
//! falls back to the invocation reading — so an address carrying both a folded constant and a
//! derived one would lose the constant it *could* read as well as the one it could not. Counting
//! what is legible and being a floor beats counting nothing and being a cruder floor.
//!
//! # Every condition here is redundant on the modules this crate emits
//!
//! [`row_of`]'s conjunction has four clauses and **no emitted module needs more than one of them**:
//! the only `OpIMul` over `group.y` is the row's own base, and the only term using that base is the
//! row. [`shift_in`]'s two are the same story — of the sums in an address exactly one has a constant
//! on its right, and it is the one the left-hand clause already names. So the mutation gate reported
//! every weakening of both as unguarded, and it was right to.
//!
//! The answer is not to delete the clauses. This reads SPIR-V, not only the SPIR-V it wrote, and
//! each of them rejects a shape a module from somewhere else may have. **It is to write those
//! modules** — four edits to a real one, none of which changes an address:
//!
//! | the edit | what it makes | what it pins |
//! | --- | --- | --- |
//! | every `OpIAdd` copied under a fresh id | two rows | the match must be *unique* |
//! | a sum over the row's base adding something that is not `local.y` | not a row | what the sum is over |
//! | `i_add(index, k)` spliced into an access chain | a constant off the lane | whose sum carries the offset |
//! | `OpExecutionMode` removed | no workgroup size | the divisor is not zero |
//!
//! Three of those four were argued to be **equivalent mutants** before they were written, and the
//! argument was sound about every module this emitter produces. That is a smaller claim than it
//! sounds, and it was wrong all three times.
//!
//! # The walk is deliberately short-sighted
//!
//! It follows `OpIAdd` and `OpIMul` and stops at everything else, because those two are the whole
//! of the address arithmetic `simdr::kernel` emits. A module that computes an address some other
//! way reaches no built-in through this walk, so its bindings go uncounted and unchecked — which is
//! the safe direction, and the same one [`super::fits`] takes for a module it cannot read at all.

use crate::dispatch::Specialization;
use simdr::decode;
use simdr::module::op;
use simdr::spec::{BuiltIn, Decoration};
use std::collections::{BTreeMap, BTreeSet};

/// One step of the address arithmetic, as `simdr::kernel` emits it.
#[derive(Clone, Copy)]
struct Term {
    /// `OpIAdd` or `OpIMul`.
    opcode: u16,
    left: u32,
    right: u32,
}

/// What one binding's addressing asks of the buffer behind it.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Needs {
    /// How many elements each invocation touches — the strip count the emitter used.
    pub(super) per_invocation: u64,
    /// A constant number of elements past the run, from `Kernel::load_offset`.
    ///
    /// Zero for every binding nobody offsets into, which is most of them.
    pub(super) offset: u64,
    /// Elements from one row of this buffer to the next, when the module addresses it as a plane.
    ///
    /// `None` for a linear buffer, which is a different claim from a pitch of zero: the caller
    /// above multiplies by it and a zero would say every row starts at the beginning.
    pub(super) pitch: Option<u64>,
}

/// What each buffer's addressing asks of it, keyed by its binding number.
///
/// A binding whose address does not vary per invocation is absent rather than zero: the two are
/// different claims, and only one of them is safe to multiply by an invocation count.
///
/// Empty when the module declares no workgroup size this can divide by, when it has no
/// `LocalInvocationId` to depend on, or when nothing in it addresses a bound buffer.
pub(super) fn needs(spirv: &[u32], columns: u64, chosen: &Specialization) -> BTreeMap<u32, Needs> {
    let mut wanted: BTreeMap<u32, Needs> = BTreeMap::new();
    if columns == 0 {
        return wanted;
    }

    let Some(local) = component_of(spirv, BuiltIn::LocalInvocationId, 0) else {
        return wanted;
    };
    let group = component_of(spirv, BuiltIn::WorkgroupId, 0);
    let bindings = bindings(spirv);
    let constants = constants(spirv);
    let specialized = specialized(spirv, chosen);
    let terms = terms(spirv);
    let row = row_of(spirv, &terms);

    for (variable, index) in element_accesses(spirv) {
        let Some(&binding) = bindings.get(&variable) else {
            continue;
        };

        let from = reachable(&terms, index);
        if !from.contains(&local) {
            continue;
        }

        // The strip count, or one. `run_start` emits the multiply for every access it computes, so
        // the fallback is for an address that reached the invocation some other way — one element
        // each, as far as anything here can tell.
        let strips = strips_in(&from, &terms, &constants, group, columns).max(1);

        // What is left of this access's folded constant once its own strip is accounted for. On the
        // last strip that is the caller's offset exactly; on an earlier one it is less, or nothing,
        // so the maximum over a binding's accesses is the offset and the others cost nothing.
        // Two halves, because an address can carry both: the constant the emitter folded in, and
        // a specialization constant the pipeline will supply. `load_offset` produces the first and
        // `load_offset_by` the second, and a kernel is free to use one, the other, or neither.
        let offset = shift_in(&from, &terms, &constants, local)
            .saturating_sub((strips - 1).saturating_mul(columns))
            .saturating_add(open_shift_in(&from, &terms, &specialized));

        let pitch = row.and_then(|row| pitch_in(&from, &terms, &constants, row));

        let entry = wanted.entry(binding).or_default();
        entry.per_invocation = entry.per_invocation.max(strips);
        entry.offset = entry.offset.max(offset);
        entry.pitch = entry.pitch.max(pitch);
    }

    wanted
}

/// The id holding this invocation's row, for a module that has one.
///
/// `kernel::binding` computes it two ways and this has to know both, because the shorter one is not
/// a special case of the longer: a workgroup one row deep is `group.y` alone, with no multiply and
/// no local component at all — the `LocalSize` declares y as 1, so every invocation of a workgroup
/// is on that workgroup's row and the arithmetic folds away.
///
/// Deeper than that it is `group.y × rows + local.y`, and the row is the **sum**. Finding the sum
/// rather than the multiply is what keeps `rows` and `pitch` apart: both are constants multiplied
/// into the row chain, and the one this file wants is the one multiplied by the *finished* row.
///
/// **The sum is named by its right operand and not by its shape.** `Kernel::start_of` emits
/// `i_add(i_mul(group.y, pitch), run)`, which is the same shape as the row and is the address the
/// row is *used* to compute — matching on the shape alone found that one instead, reported no pitch
/// at all, and quietly went back to counting invocations. `local.y` is what tells them apart, and a
/// module with no `local.y` in it is the one-row-deep case where the row is `group.y` alone.
///
/// **The match has to be unique, not merely first.** A module with two sums of this shape is one
/// this cannot choose between, and choosing the lower id anyway would be a guess dressed as a fact —
/// the same objection [`super::Bounds::overrun`] states about a binding it cannot read. So two
/// matches give no row, therefore no pitch, therefore the invocation reading: less checking rather
/// than wrong checking, which is the direction this file takes everywhere it cannot see.
fn row_of(spirv: &[u32], terms: &BTreeMap<u32, Term>) -> Option<u32> {
    let group_y = component_of(spirv, BuiltIn::WorkgroupId, 1)?;
    let Some(local_y) = component_of(spirv, BuiltIn::LocalInvocationId, 1) else {
        return Some(group_y);
    };

    let mut sums = terms.iter().filter(|(_, term)| {
        terms.get(&term.left).is_some_and(|left| {
            term.opcode == op::I_ADD
                && term.right == local_y
                && left.opcode == op::I_MUL
                && left.left == group_y
        })
    });

    let (&deep, _) = sums.next()?;
    if sums.next().is_some() {
        return None;
    }
    Some(deep)
}

/// Elements from one row of this buffer to the next, when `row` is multiplied into the address.
///
/// `Kernel::start_of` emits `i_mul(uint, row, pitch)`, so the pitch is the constant beside the row
/// — and a buffer the module addresses linearly has no such multiply, which is the `None` this
/// returns and not a pitch of zero.
///
/// `Kernel::load_row_at` on a row the *caller* computed reaches no pitch through this, because the
/// row it multiplies is not the one above. That under-counts, which is the direction this file
/// takes wherever it cannot see, and the caller of that method already vouches for its own row.
fn pitch_in(
    from: &BTreeSet<u32>,
    terms: &BTreeMap<u32, Term>,
    constants: &BTreeMap<u32, u32>,
    row: u32,
) -> Option<u64> {
    from.iter()
        .filter_map(|id| terms.get(id))
        .filter(|term| term.opcode == op::I_MUL && term.left == row)
        .filter_map(|term| constants.get(&term.right).copied())
        .map(u64::from)
        .max()
}

/// The largest constant added to the invocation's own lane on the way to this address.
///
/// `Kernel::address` emits `i_add(uint, local, shift)` and folds `strip × workgroup + offset` into
/// that one constant, so this is both terms at once and the caller above separates them. Zero when
/// the address is the lane itself, which is what the emitter emits when the fold comes to nothing.
///
/// The lane is the **left** operand, as `run_start`'s workgroup index is: the order is this
/// project's own, and reading it the other way round would find nothing and report zero — the safe
/// direction, and one the offset tests would fail loudly on.
fn shift_in(
    from: &BTreeSet<u32>,
    terms: &BTreeMap<u32, Term>,
    constants: &BTreeMap<u32, u32>,
    local: u32,
) -> u64 {
    from.iter()
        .filter_map(|id| terms.get(id))
        .filter(|term| term.opcode == op::I_ADD && term.left == local)
        .filter_map(|term| constants.get(&term.right).copied())
        .map(u64::from)
        .max()
        .unwrap_or(0)
}

/// The largest specialization constant added into this address, at the value the pipeline will
/// carry.
///
/// `Kernel::load_offset_by` emits `i_add(uint, address, offset)`, so the constant is the right
/// operand — but **both are read**, and that is the one place this file does not lean on operand
/// order. [`shift_in`] and [`strips_in`] use the left operand to *identify* which term they are
/// looking at, and being wrong there finds nothing and counts zero, which is safe. Here the
/// constant's role is the same in either slot: a sum is shifted by it whichever side it sits on,
/// and counting zero because a module wrote the operands the other way round is the permissive
/// direction this whole function exists to close.
///
/// The maximum rather than the sum: `load_offset_by` adds the *same* offset once per strip, so each
/// strip's address carries one copy of it and the binding needs one copy past its run.
fn open_shift_in(
    from: &BTreeSet<u32>,
    terms: &BTreeMap<u32, Term>,
    specialized: &BTreeMap<u32, u32>,
) -> u64 {
    from.iter()
        .filter_map(|id| terms.get(id))
        .filter(|term| term.opcode == op::I_ADD)
        .filter_map(|term| {
            specialized
                .get(&term.right)
                .or_else(|| specialized.get(&term.left))
                .copied()
        })
        .map(u64::from)
        .max()
        .unwrap_or(0)
}

/// Every specialization constant's value, by the id that holds it.
///
/// Both halves of the number: a module declares one with a default and a `SpecId`, and a pipeline
/// may replace the default when it is created. What a dispatch reaches depends on the value the
/// **pipeline** was built with, so the caller's choice wins and the module's default stands where
/// there is none — which is exactly what the driver does with the same two numbers.
///
/// A specialization constant with no `SpecId` cannot be set by anybody and is skipped: its default
/// is its value, and it would be read as a plain constant if anything addressed with it.
fn specialized(spirv: &[u32], chosen: &Specialization) -> BTreeMap<u32, u32> {
    let ids: BTreeMap<u32, u32> = decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::DECORATE)
        .filter_map(|instruction| match instruction.operands() {
            [target, decoration, spec_id] if *decoration == Decoration::SpecId.word() => {
                Some((*target, *spec_id))
            }
            _ => None,
        })
        .collect();

    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::SPEC_CONSTANT)
        .filter_map(|instruction| match instruction.operands() {
            [_type, id, default] => {
                let spec_id = ids.get(id)?;
                Some((*id, chosen.value_of(*spec_id).unwrap_or(*default)))
            }
            _ => None,
        })
        .collect()
}

/// The largest `workgroup × strips` constant multiplied by the workgroup index, divided back down.
///
/// `group` is `None` for a module with no `WorkgroupId`, which is a module of one workgroup and one
/// strip per invocation as far as this can tell.
fn strips_in(
    from: &BTreeSet<u32>,
    terms: &BTreeMap<u32, Term>,
    constants: &BTreeMap<u32, u32>,
    group: Option<u32>,
    workgroup: u64,
) -> u64 {
    let Some(group) = group else {
        return 1;
    };

    from.iter()
        .filter_map(|id| terms.get(id))
        // The workgroup index is the **left** operand: `Kernel::run_start` emits
        // `i_mul(uint, group, run)` and that order is this project's, not the driver's. A module
        // that multiplied the other way round reads as one element per invocation — the safe
        // direction, and a case the strip tests would fail loudly on if the emitter swapped them.
        .filter(|term| term.opcode == op::I_MUL && term.left == group)
        .filter_map(|term| constants.get(&term.right).copied())
        .map(|run| u64::from(run) / workgroup)
        .max()
        .unwrap_or(1)
}

/// Every id the value `from` is computed from, `from` itself included.
///
/// Bounded by the module: each id is expanded once, so a walk cannot loop however the terms are
/// arranged.
fn reachable(terms: &BTreeMap<u32, Term>, from: u32) -> BTreeSet<u32> {
    let mut seen = BTreeSet::new();
    let mut pending = vec![from];

    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Some(term) = terms.get(&id) {
            pending.push(term.left);
            pending.push(term.right);
        }
    }

    seen
}

/// The `OpIAdd` and `OpIMul` in the module, by the id each one defines.
fn terms(spirv: &[u32]) -> BTreeMap<u32, Term> {
    decode::body(spirv)
        .filter(|instruction| {
            instruction.opcode() == op::I_ADD || instruction.opcode() == op::I_MUL
        })
        .filter_map(|instruction| match instruction.operands() {
            [_type, id, left, right] => Some((
                *id,
                Term {
                    opcode: instruction.opcode(),
                    left: *left,
                    right: *right,
                },
            )),
            _ => None,
        })
        .collect()
}

/// Every `OpConstant`'s literal, by id.
fn constants(spirv: &[u32]) -> BTreeMap<u32, u32> {
    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::CONSTANT)
        .filter_map(|instruction| match instruction.operands() {
            [_type, id, literal] => Some((*id, *literal)),
            _ => None,
        })
        .collect()
}

/// Which binding each decorated variable is bound at.
fn bindings(spirv: &[u32]) -> BTreeMap<u32, u32> {
    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::DECORATE)
        .filter_map(|instruction| match instruction.operands() {
            [target, decoration, binding] if *decoration == Decoration::Binding.word() => {
                Some((*target, *binding))
            }
            _ => None,
        })
        .collect()
}

/// Every access chain that reaches one element of a buffer, as the variable and the element index.
///
/// The shape is `simdr::kernel`'s own: the struct member, then the element. A chain of any other
/// length is left alone — this crate emits one shape, and a module that does not is one this
/// cannot make a claim about.
fn element_accesses(spirv: &[u32]) -> Vec<(u32, u32)> {
    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::ACCESS_CHAIN)
        .filter_map(|instruction| match instruction.operands() {
            [_type, _id, base, _member, index] => Some((*base, *index)),
            _ => None,
        })
        .collect()
}

/// One scalar component of a built-in vector — the id the address arithmetic actually uses.
///
/// Traced rather than guessed: the built-in is a three-element vector, loaded once and then
/// extracted from, and it is the *extracted* id that appears in the arithmetic.
///
/// **The component index is matched rather than assumed to come first.** A grid kernel extracts
/// both x and y from the same load — `Kernel::row` is `group.y × rows + local.y` — and taking
/// whichever `OpCompositeExtract` the walk met first would rest on the order `kernel::binding`
/// happens to emit them in. It emits x first today. That is not a thing this file should know, and
/// the row arithmetic below asks for y by name.
fn component_of(spirv: &[u32], built_in: BuiltIn, component: u32) -> Option<u32> {
    let variable = decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::DECORATE)
        .find_map(|instruction| match instruction.operands() {
            [target, decoration, declared]
                if *decoration == Decoration::BuiltIn.word() && *declared == built_in.word() =>
            {
                Some(*target)
            }
            _ => None,
        })?;

    let loaded = decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::LOAD)
        .find_map(|instruction| match instruction.operands() {
            [_type, id, pointer] if *pointer == variable => Some(*id),
            _ => None,
        })?;

    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::COMPOSITE_EXTRACT)
        .find_map(|instruction| match instruction.operands() {
            // The literal after the composite is the component. Zero is x, the axis every linear
            // address is computed on; one is y, which only a grid extracts and which is where a
            // row comes from.
            [_type, id, composite, index] if *composite == loaded && *index == component => {
                Some(*id)
            }
            _ => None,
        })
}
