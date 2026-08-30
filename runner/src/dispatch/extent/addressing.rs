//! ```text
//! address = group × (workgroup × strips)  +  local + (strip × workgroup + offset)
//!           \_____________ base _______/           \__________ shift __________/
//! ```

use crate::dispatch::Specialization;
use simdr::decode;
use simdr::module::op;
use simdr::spec::{BuiltIn, Decoration};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
struct Term {
    opcode: u16,
    left: u32,
    right: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Needs {
    pub(super) per_invocation: u64,
    pub(super) offset: u64,
    pub(super) pitch: Option<u64>,
}

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

        let strips = strips_in(&from, &terms, &constants, group, columns).max(1);

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
        .filter(|term| term.opcode == op::I_MUL && term.left == group)
        .filter_map(|term| constants.get(&term.right).copied())
        .map(|run| u64::from(run) / workgroup)
        .max()
        .unwrap_or(1)
}

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

fn constants(spirv: &[u32]) -> BTreeMap<u32, u32> {
    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::CONSTANT)
        .filter_map(|instruction| match instruction.operands() {
            [_type, id, literal] => Some((*id, *literal)),
            _ => None,
        })
        .collect()
}

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

fn element_accesses(spirv: &[u32]) -> Vec<(u32, u32)> {
    decode::body(spirv)
        .filter(|instruction| instruction.opcode() == op::ACCESS_CHAIN)
        .filter_map(|instruction| match instruction.operands() {
            [_type, _id, base, _member, index] => Some((*base, *index)),
            _ => None,
        })
        .collect()
}

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
            [_type, id, composite, index] if *composite == loaded && *index == component => {
                Some(*id)
            }
            _ => None,
        })
}
