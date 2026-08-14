//! Reductions — the operations that cross lanes.
//!
//! Which instruction comes out depends on how `LANES` sits on the subgroup, and that is decided in
//! one place: [`Lanes::mapping`].
//!
//! - As wide as the subgroup: one plain `Reduce`.
//! - Narrower: one `ClusteredReduce` whose cluster size is the vector's own width, so the lanes
//!   that would otherwise idle are running other copies of the same vector.
//! - Wider: fold the strips together inside each lane first — `strips - 1` scalar operations —
//!   then one subgroup instruction over the partials.
//!
//! A **scan** takes the same three mappings and only the first is an instruction: a
//! subgroup-wide vector is one `InclusiveScan`, a wider one is a scan per strip with a running
//! total carried between them, and a narrower one is a ladder — SPIR-V's clustered form is a
//! *reduce*, so there is no clustered scan to ask for.
//!
//! Needs `GroupNonUniform` and `GroupNonUniformArithmetic`, plus `GroupNonUniformClustered`
//! whenever a vector is narrower than the subgroup. The caller declares them; nothing here does.
//! The clustered ladder is the exception in both directions: it declares
//! `GroupNonUniformShuffleRelative` for the shuffles it is built from, and it needs neither of the
//! arithmetic capabilities, because every instruction in it is a scalar one.

use super::{Element, LaneError, Lanes, Mapping, U32, Vector};
use crate::module::{Id, Reduction, op};
use crate::spec::Capability;

impl Lanes<'_> {
    /// The sum of every element, delivered to every lane — `Simd::reduce_sum`.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoMapping`] if `LANES` cannot sit on this subgroup, [`LaneError::Build`] if an
    /// instruction cannot be emitted.
    pub fn reduce_sum<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        self.reduce_with::<T, LANES>(T::GROUP_ADD, T::ADD, vector)
    }

    /// Running totals: each lane receives the sum of itself and every lane before it.
    ///
    /// All three mappings, and only one of them is an instruction. A subgroup-wide vector is one
    /// `InclusiveScan`; a wider one is a scan per strip with a running total carried between them;
    /// a narrower one is a ladder, because SPIR-V's clustered form is a *reduce* and there is no
    /// clustered scan to ask for.
    ///
    /// # Errors
    ///
    /// [`LaneError::NoMapping`] if `LANES` cannot sit on this subgroup, [`LaneError::Build`] if an
    /// instruction cannot be emitted.
    pub fn prefix_sum<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Reduction::InclusiveScan, vector)
    }

    /// The same, with each lane's own element left out: lane 0 receives the additive identity.
    ///
    /// **The form a multi-block scan needs, and the reason it is a separate instruction rather than
    /// a subtraction.** Block `b` of a long scan owes the total of every block before it and not
    /// its own, which is an exclusive scan of the block totals. Computing it as `inclusive - own`
    /// costs an operation and, in floating point, is not the same number — subtracting a large
    /// running total back off itself loses precisely the low bits the scan just accumulated.
    ///
    /// SPIR-V has the operation, so this asks for it. `GroupOperation::ExclusiveScan` was in
    /// `spec::group` from the beginning and nothing had ever emitted one.
    ///
    /// # Errors
    ///
    /// As [`Lanes::prefix_sum`].
    pub fn prefix_sum_exclusive<T: Element, const LANES: u32>(
        &mut self,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        self.scan_with::<T, LANES>(Reduction::ExclusiveScan, vector)
    }

    /// Both scans, at all three mappings — they differ in the group operation they name, and in
    /// the clustered case in which of two values each lane keeps.
    ///
    /// It used to take the caller's name as well, to tell a refused caller which scan it had asked
    /// for. No mapping is refused any more, so there is nothing left to name.
    fn scan_with<T: Element, const LANES: u32>(
        &mut self,
        reduction: Reduction,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        match self.mapping::<LANES>()? {
            Mapping::WholeSubgroup => {}
            // **A ladder, because there is no clustered scan to ask for.** `log2(size)` steps of
            // shuffle, compare and select, with the mask keeping each cluster's running total
            // inside itself — see [`Lanes::scan_clusters`].
            Mapping::Clusters { size } => {
                let exclusive = matches!(reduction, Reduction::ExclusiveScan);
                return self.scan_clusters::<T, LANES>(size, exclusive, vector);
            }
            // **Built now, and it is the running total that makes it work.** Lane `l` holds the
            // elements at `l`, `l + width`, `l + 2·width`, so strip `s` is vector positions
            // `s·width ..` — every element of strip `s - 1` comes before every element of strip
            // `s`. Each strip is therefore scanned on its own and raised by the total of the
            // strips below it, which is `strips` scans, `strips - 1` reduces and the adds.
            Mapping::Strips { .. } => {
                return self.scan_strips::<T, LANES>(reduction, vector);
            }
        }

        let element = self.type_of::<T>()?;
        let scope = self.scope();
        self.module()
            .require_capability(Capability::GroupNonUniform)?;
        self.module()
            .require_capability(Capability::GroupNonUniformArithmetic)?;

        let id =
            self.module()
                .subgroup_reduce(T::GROUP_ADD, element, scope, reduction, vector.id())?;
        self.from_lane_value(id)
    }

    /// A scan of a vector wider than the subgroup, one strip at a time.
    ///
    /// The order is the one [`crate::lanes::vector`] documents: lane `l` holds `l`, `l + width`,
    /// `l + 2·width`, so vector position `j` is strip `j / width` of lane `j % width` and the
    /// strips are consecutive runs of the vector. A prefix over position `j` is then everything in
    /// the strips below `j`'s, plus the scan within it.
    ///
    /// **The carry is a `Reduce`, not the last lane of the scan.** An exclusive scan hands no lane
    /// the strip's whole total — leaving it out is what makes it exclusive — so reading the carry
    /// off the scan would be short by one lane's element in exactly the form where it matters.
    ///
    /// The last strip needs no total, and does not compute one: nothing comes after it.
    fn scan_strips<T: Element, const LANES: u32>(
        &mut self,
        reduction: Reduction,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        let element = self.type_of::<T>()?;
        let scope = self.scope();
        self.module()
            .require_capability(Capability::GroupNonUniform)?;
        self.module()
            .require_capability(Capability::GroupNonUniformArithmetic)?;

        let strips = vector.strips().to_vec();
        let last = strips.len().saturating_sub(1);
        let mut scanned = Vec::with_capacity(strips.len());
        let mut carried: Option<Id> = None;

        for (index, &strip) in strips.iter().enumerate() {
            let within =
                self.module()
                    .subgroup_reduce(T::GROUP_ADD, element, scope, reduction, strip)?;
            scanned.push(match carried {
                None => within,
                Some(carry) => self.module().binary(T::ADD, element, within, carry)?,
            });

            if index == last {
                continue;
            }
            let total = self.module().subgroup_reduce(
                T::GROUP_ADD,
                element,
                scope,
                Reduction::Reduce,
                strip,
            )?;
            carried = Some(match carried {
                None => total,
                Some(carry) => self.module().binary(T::ADD, element, carry, total)?,
            });
        }

        self.from_strips(&scanned)
    }

    /// A scan of a vector narrower than the subgroup, within each cluster.
    ///
    /// **The one mapping SPIR-V has no instruction for.** There is a `ClusteredReduce` and no
    /// clustered scan, so this is a Hillis-Steele ladder: `log2(size)` steps, each adding the
    /// element `distance` lanes below and keeping it only where that neighbour belongs to the same
    /// cluster.
    ///
    /// ```text
    ///   for distance in 1, 2, 4, … < size:
    ///       value += (lane % size) > distance - 1  ?  the value `distance` lanes below  :  nothing
    /// ```
    ///
    /// Both arms are computed and one is discarded, which is what makes it branch-free *and* safe:
    /// a shuffle leaves the bottom `distance` lanes of the subgroup undefined, and the mask is what
    /// stops either that or a neighbouring cluster's total reaching the answer.
    ///
    /// **Exact, and that is why it is a ladder rather than a subtraction.** The cheap alternative
    /// is one subgroup-wide scan minus each cluster's starting offset — three instructions instead
    /// of `3 · log2(size)` — and over floats it takes a large running total back off itself and
    /// loses precisely the low bits the scan just accumulated. The same reason
    /// [`Lanes::prefix_sum_exclusive`] exists rather than a subtraction.
    ///
    /// The exclusive form is the inclusive one shifted a lane up, with the identity where the
    /// cluster begins — a shuffle and a select, and no arithmetic that could round.
    fn scan_clusters<T: Element, const LANES: u32>(
        &mut self,
        size: u32,
        exclusive: bool,
        vector: Vector<T, LANES>,
    ) -> Result<Vector<T, LANES>, LaneError> {
        // **A one-lane cluster is answered before anything is emitted.** Its inclusive prefix is
        // the element itself and its exclusive one is the identity, and everything below —
        // the built-in, the mask, the shuffle, the select — would compute exactly that at runtime
        // and say something else about the kernel. `Simd<T, 1>` is a mapping this API accepts, so
        // it is a case rather than an impossibility.
        if size == 1 {
            return if exclusive {
                self.splat_bits::<T, LANES>(0)
            } else {
                Ok(vector)
            };
        }

        // Where this invocation sits inside its cluster. `size` divides the width and both are
        // powers of two, so the remainder is a mask — and it is the position within the *cluster*
        // rather than within the subgroup, because that is what decides whether the neighbour
        // `distance` below is a neighbour at all.
        let lane = self.lane_index()?;
        let uint = self.type_of::<U32>()?;
        let wrap = self.module().constant_u32(size.saturating_sub(1))?;
        let within = self.module().binary(op::BITWISE_AND, uint, lane, wrap)?;

        let mut value = vector;
        let mut distance = 1;
        while distance < size {
            let below = self.shift_up_across_clusters(value, distance)?;
            let raised = self.add(value, below)?;
            let inside = self.beyond::<LANES>(within, distance.saturating_sub(1))?;
            value = self.select(inside, raised, value)?;
            distance = distance.saturating_mul(2);
        }

        if !exclusive {
            return Ok(value);
        }

        let shifted = self.shift_up_across_clusters(value, 1)?;
        // Zero in every element type this crate has: `0.0`, `0`, and a half of the same bits.
        let identity = self.splat_bits::<T, LANES>(0)?;
        let inside = self.beyond::<LANES>(within, 0)?;
        self.select(inside, shifted, identity)
    }

    /// Whether this lane's position within its cluster is above `edge`, as a per-element mask.
    ///
    /// `> edge` rather than `>= edge + 1`: unsigned greater-than is the comparison the lane API
    /// has, and over integers the two say the same thing.
    fn beyond<const LANES: u32>(
        &mut self,
        within: Id,
        edge: u32,
    ) -> Result<crate::lanes::Predicate<LANES>, LaneError> {
        let position = self.from_lane_value::<U32, LANES>(within)?;
        let edge = self.splat_bits::<U32, LANES>(edge)?;
        self.greater_than(position, edge)
    }

    /// Fold the strips inside each lane with `local`, then reduce across the subgroup with
    /// `group`.
    fn reduce_with<T: Element, const LANES: u32>(
        &mut self,
        group: u16,
        local: u16,
        vector: Vector<T, LANES>,
    ) -> Result<Id, LaneError> {
        let element = self.type_of::<T>()?;
        let scope = self.scope();
        let (reduction, partial) = self.fold_strips::<T, LANES>(local, vector)?;

        Ok(self
            .module()
            .subgroup_reduce(group, element, scope, reduction, partial)?)
    }

    /// Combine a vector's strips into one per-lane value, and say which reduction shape the
    /// subgroup step then needs.
    fn fold_strips<T: Element, const LANES: u32>(
        &mut self,
        local: u16,
        vector: Vector<T, LANES>,
    ) -> Result<(Reduction, Id), LaneError> {
        let element = self.type_of::<T>()?;
        let reduction = self.reduction::<LANES>()?;

        let mut partial = vector
            .strips()
            .first()
            .copied()
            .ok_or(LaneError::no_strips())?;
        for &next in vector.strips().iter().skip(1) {
            partial = self.module().binary(local, element, partial, next)?;
        }

        Ok((reduction, partial))
    }

    /// The reduction shape this vector's mapping implies, declaring what it needs on the way.
    ///
    /// A strip-mined vector reduces over the *whole* subgroup once its strips are folded — the
    /// strips are within a lane, so they never needed a cluster. Only the clustered case asks for
    /// `GroupNonUniformClustered`, which is why a kernel that never uses one stays runnable on a
    /// device that does not offer it.
    pub(super) fn reduction<const LANES: u32>(&mut self) -> Result<Reduction, LaneError> {
        let mapping = self.mapping::<LANES>()?;

        self.module()
            .require_capability(Capability::GroupNonUniform)?;
        self.module()
            .require_capability(Capability::GroupNonUniformArithmetic)?;

        match mapping {
            Mapping::WholeSubgroup | Mapping::Strips { .. } => Ok(Reduction::Reduce),
            Mapping::Clusters { size } => {
                self.module()
                    .require_capability(Capability::GroupNonUniformClustered)?;
                let size = self.module().constant_u32(size)?;
                Ok(Reduction::Clustered { size })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // A test may panic — that is how it reports.
    #![allow(clippy::expect_used, clippy::indexing_slicing)]

    use super::*;
    use crate::decode;
    use crate::lanes::{F32, U32};
    use crate::module::{Module, Version, op};
    use crate::spec::GroupOperation;

    fn count(words: &[u32], opcode: u16) -> usize {
        decode::body(words)
            .filter(|instruction| instruction.opcode() == opcode)
            .count()
    }

    /// The operands of the group reduction in a module built at `width` for `LANES`.
    fn reduce_operands<const LANES: u32>(width: u32) -> Vec<u32> {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, width).expect("built");
        let value = lanes
            .splat_bits::<F32, LANES>(1.0_f32.to_bits())
            .expect("splat");
        lanes.reduce_sum(value).expect("reduced");

        let words = module.finish();
        decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .expect("emitted")
            .operands()
            .to_vec()
    }

    #[test]
    fn a_full_width_vector_reduces_over_the_whole_subgroup() {
        let operands = reduce_operands::<32>(32);

        assert_eq!(operands[3], GroupOperation::Reduce.word());
        assert_eq!(operands.len(), 5, "a plain reduce carries no cluster size");
    }

    #[test]
    fn a_narrow_vector_reduces_in_clusters_of_its_own_width() {
        let operands = reduce_operands::<8>(32);

        assert_eq!(operands[3], GroupOperation::ClusteredReduce.word());
        assert_eq!(operands.len(), 6);
    }

    #[test]
    fn a_strip_mined_vector_folds_locally_then_reduces_over_the_whole_subgroup() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 128>(1.0_f32.to_bits())
            .expect("splat");

        lanes.reduce_sum(value).expect("reduced");

        let words = module.finish();

        // Four elements per lane: three scalar adds to fold them, then one subgroup instruction.
        assert_eq!(count(&words, op::F_ADD), 3);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_ADD), 1);

        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .expect("emitted")
            .operands()
            .to_vec();
        assert_eq!(
            operands[3],
            GroupOperation::Reduce.word(),
            "the strips are within a lane, so the subgroup step never needed a cluster"
        );
    }

    #[test]
    fn the_cluster_size_that_is_emitted_is_the_vectors_own_width() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 4>(1.0_f32.to_bits())
            .expect("splat");
        lanes.reduce_sum(value).expect("reduced");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .expect("emitted")
            .operands()
            .to_vec();

        let cluster_id = operands[5];
        let size = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CONSTANT)
            .find(|instruction| instruction.operands().get(1) == Some(&cluster_id))
            .and_then(|instruction| instruction.operands().get(2).copied())
            .expect("the cluster size is a declared constant");

        assert_eq!(size, 4);
    }

    #[test]
    fn one_lane_count_reduces_three_different_ways_across_two_devices() {
        // DR-0002, read off the emitted instruction rather than the mapping.
        assert_eq!(reduce_operands::<32>(32)[3], GroupOperation::Reduce.word());
        assert_eq!(
            reduce_operands::<32>(64)[3],
            GroupOperation::ClusteredReduce.word()
        );
        assert_eq!(
            reduce_operands::<64>(32)[3],
            GroupOperation::Reduce.word(),
            "two strips folded, then the whole subgroup"
        );
    }

    #[test]
    fn a_lane_count_with_no_mapping_is_refused() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");

        assert_eq!(
            lanes.splat_bits::<F32, 12>(0).err(),
            Some(LaneError::NoMapping {
                lanes: 12,
                width: 32
            })
        );
    }

    #[test]
    fn integers_reduce_with_their_own_opcode() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<U32, 32>(1).expect("splat");

        lanes.reduce_sum(value).expect("reduced");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_I_ADD), 1);
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_F_ADD), 0);
    }

    #[test]
    fn a_prefix_sum_scans_when_the_vector_is_the_whole_subgroup() {
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.prefix_sum(value).expect("scanned");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(operands[3], GroupOperation::InclusiveScan.word());
    }

    #[test]
    fn an_exclusive_prefix_sum_names_the_other_group_operation() {
        // The whole difference between the two scans is this one literal. Both emit the same
        // opcode with the same operands otherwise, so a version that ignored the argument and
        // always scanned inclusively would look right everywhere except here — and would give a
        // multi-block scan every block its own total twice over.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes
            .splat_bits::<F32, 32>(1.0_f32.to_bits())
            .expect("splat");

        lanes.prefix_sum_exclusive(value).expect("scanned");

        let words = module.finish();
        let operands = decode::body(&words)
            .find(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .expect("emitted")
            .operands()
            .to_vec();

        assert_eq!(operands[3], GroupOperation::ExclusiveScan.word());
        assert_ne!(operands[3], GroupOperation::InclusiveScan.word());
    }

    #[test]
    fn the_two_scans_differ_in_nothing_but_that_literal() {
        // They share a builder, and this is what says the sharing did not quietly make them the
        // same instruction — or leave one of them declaring a capability the other does not.
        let scan = |exclusive: bool| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            let value = lanes
                .splat_bits::<F32, 32>(1.0_f32.to_bits())
                .expect("splat");
            if exclusive {
                lanes.prefix_sum_exclusive(value).expect("scanned");
            } else {
                lanes.prefix_sum(value).expect("scanned");
            }
            module.finish()
        };

        let inclusive = scan(false);
        let exclusive = scan(true);

        assert_eq!(inclusive.len(), exclusive.len(), "same instruction count");
        assert_eq!(
            decode::opcodes(&inclusive),
            decode::opcodes(&exclusive),
            "same instructions, in the same order"
        );
        assert_ne!(inclusive, exclusive, "and not the same words");
    }

    #[test]
    fn the_two_scans_accept_exactly_the_same_shapes() {
        // Both go through one builder, so a caller must not find that one accepts a shape the
        // other refuses. Every mapping is accepted now; what differs is what each one costs.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let narrow = lanes.splat_bits::<F32, 8>(0).expect("splat");
        let whole = lanes.splat_bits::<F32, 32>(0).expect("splat");
        let wide = lanes.splat_bits::<F32, 128>(0).expect("splat");

        // Written out rather than looped: `LANES` is part of the type, so a clustered vector, a
        // whole-subgroup one and a strip-mined one cannot share an array.
        assert!(lanes.prefix_sum(narrow).is_ok());
        assert!(lanes.prefix_sum_exclusive(narrow).is_ok());
        assert!(lanes.prefix_sum(whole).is_ok());
        assert!(lanes.prefix_sum_exclusive(whole).is_ok());
        assert!(lanes.prefix_sum(wide).is_ok());
        assert!(lanes.prefix_sum_exclusive(wide).is_ok());
    }

    #[test]
    fn a_clustered_scan_is_a_ladder_of_one_step_per_doubling() {
        // `log2(size)` steps, each a shuffle, a comparison and a select. A loop bound of `<=`
        // rather than `<` is invisible in the *answer* — the extra step asks whether a lane's
        // position exceeds `size - 1` and no position does — and is three instructions the module
        // should not contain, so the count is what says so.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<F32, 8>(0).expect("splat");

        lanes.prefix_sum(value).expect("scanned");

        let words = module.finish();
        assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE_UP), 3);
        assert_eq!(count(&words, op::SELECT), 3);
        assert_eq!(
            count(&words, op::GROUP_NON_UNIFORM_F_ADD),
            0,
            "there is no clustered scan instruction to have emitted"
        );
    }

    #[test]
    fn a_cluster_of_one_lane_scans_nothing_and_emits_nothing() {
        // The degenerate end, and it is answered before a single instruction is emitted. A
        // one-lane cluster's inclusive prefix is the element itself and its exclusive one is the
        // identity — and the ladder would *compute* exactly that: the mask is `lane & 0`, so the
        // shuffle's result is selected away in every lane. Right answer, and a module describing
        // work the kernel does not do.
        //
        // Counted as instructions rather than as opcodes, because what the ladder would leave
        // behind is a built-in variable, a load, a mask, a comparison and a select — and only the
        // last two are opcodes this file otherwise counts.
        for exclusive in [false, true] {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            let value = lanes.splat_bits::<F32, 1>(0).expect("splat");

            let scanned = if exclusive {
                lanes.prefix_sum_exclusive(value).expect("scanned")
            } else {
                lanes.prefix_sum(value).expect("scanned")
            };
            if !exclusive {
                assert_eq!(scanned.id(), value.id(), "the same value, untouched");
            }

            let words = module.finish();
            assert_eq!(count(&words, op::SELECT), 0);
            assert_eq!(count(&words, op::GROUP_NON_UNIFORM_SHUFFLE_UP), 0);
            assert_eq!(count(&words, op::BITWISE_AND), 0, "no mask was needed");
            assert_eq!(
                count(&words, op::VARIABLE),
                0,
                "and `SubgroupLocalInvocationId` was never asked for"
            );
        }
    }

    #[test]
    fn the_exclusive_clustered_scan_shifts_rather_than_subtracting() {
        // One shuffle and one select more than the inclusive form, and no arithmetic at all beyond
        // the ladder's adds. Subtracting each lane's own element is the cheap alternative and the
        // wrong one: over floats it takes a large running total back off itself and loses the low
        // bits the scan just accumulated.
        let ladder = |exclusive: bool| {
            let mut module = Module::new(Version::V1_3);
            let mut lanes = Lanes::new(&mut module, 32).expect("built");
            let value = lanes.splat_bits::<F32, 8>(0).expect("splat");
            if exclusive {
                lanes.prefix_sum_exclusive(value).expect("scanned");
            } else {
                lanes.prefix_sum(value).expect("scanned");
            }
            module.finish()
        };

        let inclusive = ladder(false);
        let exclusive = ladder(true);

        assert_eq!(
            count(&exclusive, op::GROUP_NON_UNIFORM_SHUFFLE_UP),
            count(&inclusive, op::GROUP_NON_UNIFORM_SHUFFLE_UP) + 1
        );
        assert_eq!(
            count(&exclusive, op::SELECT),
            count(&inclusive, op::SELECT) + 1
        );
        // Nothing is subtracted, and `module::op` has no subtraction in it to have been used —
        // so this counts the arithmetic that *is* there and finds the exclusive form adding none
        // of its own.
        assert_eq!(
            count(&exclusive, op::F_ADD),
            count(&inclusive, op::F_ADD),
            "the exclusive form adds nothing the inclusive one does not"
        );
    }

    #[test]
    fn the_clustered_ladder_masks_with_the_lane_the_specification_defines() {
        // `SubgroupLocalInvocationId` and not an index into the workgroup. The two agree on every
        // implementation here, and they agree because subgroups happen to be cut from consecutive
        // local indices — which Vulkan promises only for a pipeline that asked for full subgroups.
        use crate::spec::{BuiltIn, Decoration};

        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<F32, 8>(0).expect("splat");
        lanes.prefix_sum(value).expect("scanned");

        let words = module.finish();
        let built_ins: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::DECORATE)
            .filter_map(|instruction| match instruction.operands() {
                [_target, decoration, built_in] if *decoration == Decoration::BuiltIn.word() => {
                    Some(*built_in)
                }
                _ => None,
            })
            .collect();

        assert_eq!(
            built_ins,
            vec![BuiltIn::SubgroupLocalInvocationId.word()],
            "one built-in, and it is the lane's own position"
        );
        assert_eq!(count(&words, op::BITWISE_AND), 1, "masked into its cluster");
    }

    #[test]
    fn a_clustered_scan_declares_the_shuffles_it_uses_and_no_arithmetic_it_does_not() {
        // Every instruction in the ladder is scalar or a shuffle, so neither group-arithmetic
        // capability belongs in the module — and `GroupNonUniformClustered` least of all, since
        // the mapping is clustered and the instruction is not. A surplus capability is worse than
        // noise: it makes the module refuse to run on a device that would have run it.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let value = lanes.splat_bits::<F32, 8>(0).expect("splat");

        lanes.prefix_sum(value).expect("scanned");

        let words = module.finish();
        let declared: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::CAPABILITY)
            .filter_map(|instruction| instruction.operands().first().copied())
            .collect();

        assert!(declared.contains(&Capability::GroupNonUniform.word()));
        assert!(declared.contains(&Capability::GroupNonUniformShuffleRelative.word()));
        assert!(!declared.contains(&Capability::GroupNonUniformArithmetic.word()));
        assert!(!declared.contains(&Capability::GroupNonUniformClustered.word()));
    }

    #[test]
    fn a_strip_mined_scan_is_one_scan_per_strip_and_one_reduce_fewer() {
        // Four strips: four scans, and three totals — the last strip needs no carry because
        // nothing comes after it. A version that reduced four times would emit an instruction
        // whose result goes nowhere; one that reduced twice would leave the last strip short.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let wide = lanes.splat_bits::<F32, 128>(0).expect("splat");

        let scanned = lanes.prefix_sum(wide).expect("scanned");
        assert_eq!(scanned.strip_count(), 4);

        let words = module.finish();
        let operations: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .filter_map(|instruction| instruction.operands().get(3).copied())
            .collect();

        assert_eq!(
            operations
                .iter()
                .filter(|&&op| op == GroupOperation::InclusiveScan.word())
                .count(),
            4,
            "one scan per strip"
        );
        assert_eq!(
            operations
                .iter()
                .filter(|&&op| op == GroupOperation::Reduce.word())
                .count(),
            3,
            "a carry for every strip but the last"
        );
    }

    #[test]
    fn the_strip_mined_exclusive_scan_carries_the_same_way() {
        // The carry is a `Reduce` in both, and that is the point: an exclusive scan hands no lane
        // the strip's whole total, so taking the carry from the scan would be short by one lane's
        // element — invisible in the inclusive form and wrong in this one.
        let mut module = Module::new(Version::V1_3);
        let mut lanes = Lanes::new(&mut module, 32).expect("built");
        let wide = lanes.splat_bits::<F32, 64>(0).expect("splat");

        lanes.prefix_sum_exclusive(wide).expect("scanned");

        let words = module.finish();
        let operations: Vec<u32> = decode::body(&words)
            .filter(|instruction| instruction.opcode() == op::GROUP_NON_UNIFORM_F_ADD)
            .filter_map(|instruction| instruction.operands().get(3).copied())
            .collect();

        assert_eq!(
            operations,
            vec![
                GroupOperation::ExclusiveScan.word(),
                GroupOperation::Reduce.word(),
                GroupOperation::ExclusiveScan.word(),
            ],
            "scan, carry, scan — and the last strip takes no carry"
        );
    }
}
