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
//! # The walk is deliberately short-sighted
//!
//! It follows `OpIAdd` and `OpIMul` and stops at everything else, because those two are the whole
//! of the address arithmetic `simdr::kernel` emits. A module that computes an address some other
//! way reaches no built-in through this walk, so its bindings go uncounted and unchecked — which is
//! the safe direction, and the same one [`super::fits`] takes for a module it cannot read at all.

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

/// How many elements one invocation touches in each buffer, keyed by its binding number.
///
/// A binding whose address does not vary per invocation is absent rather than zero: the two are
/// different claims, and only one of them is safe to multiply by an invocation count.
///
/// Empty when the module declares no workgroup size this can divide by, when it has no
/// `LocalInvocationId` to depend on, or when nothing in it addresses a bound buffer.
pub(super) fn per_invocation(spirv: &[u32], workgroup: u64) -> BTreeMap<u32, u64> {
    let mut wanted = BTreeMap::new();
    if workgroup == 0 {
        return wanted;
    }

    let Some(local) = component_of(spirv, BuiltIn::LocalInvocationId) else {
        return wanted;
    };
    let group = component_of(spirv, BuiltIn::WorkgroupId);
    let bindings = bindings(spirv);
    let constants = constants(spirv);
    let terms = terms(spirv);

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
        let strips = strips_in(&from, &terms, &constants, group, workgroup).max(1);
        let entry = wanted.entry(binding).or_insert(0);
        *entry = (*entry).max(strips);
    }

    wanted
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

/// The scalar **x** component of a built-in vector — the id the address arithmetic actually uses.
///
/// Traced rather than guessed: the built-in is a three-element vector, loaded once and then
/// extracted from, and it is the *extracted* id that appears in the arithmetic.
///
/// **The component index is matched rather than assumed to come first.** A grid kernel extracts
/// both x and y from the same load — `Kernel::row` is `group.y × rows + local.y` — and taking
/// whichever `OpCompositeExtract` the walk met first would rest on the order `kernel::binding`
/// happens to emit them in. It emits x first today. That is not a thing this file should know.
fn component_of(spirv: &[u32], built_in: BuiltIn) -> Option<u32> {
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
            // The literal after the composite is the component. Zero is x, which is the axis every
            // linear address is computed on.
            [_type, id, composite, 0] if *composite == loaded => Some(*id),
            _ => None,
        })
}
