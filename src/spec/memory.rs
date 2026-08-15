//! Where values live, and what is attached to them.

use crate::encode::Word;

/// Which memory a pointer points into (`OpTypePointer`, `OpVariable`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// Read-only input to the stage.
    Input,
    /// Memory shared by every invocation in a workgroup — the fallback when a shuffle cannot
    /// express a lane exchange.
    Workgroup,
    /// Invocation-private, living as long as the module.
    Private,
    /// A function's locals.
    Function,
    /// A buffer the host bound: where a kernel's operands arrive and its results go.
    StorageBuffer,
}

impl StorageClass {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::Input => 1,
            Self::Workgroup => 4,
            Self::Private => 6,
            Self::Function => 7,
            Self::StorageBuffer => 12,
        }
    }
}

/// What a barrier orders, as a bit mask (`OpControlBarrier`, `OpMemoryBarrier`).
///
/// The ordering half and the storage half are separate flags and both are needed: saying *when*
/// without saying *what memory* orders nothing a driver has to respect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemorySemantics {
    /// No ordering at all — SPIR-V spells it `Relaxed`.
    ///
    /// **Legal on an atomic and refused on a barrier.**
    /// `VUID-StandaloneSpirv-MemorySemantics-10869` forbids it on `OpMemoryBarrier`: a barrier that
    /// orders nothing is an invalid module rather than a cheap one. See
    /// [`crate::module::Module::memory_barrier`], which is where that was found — by pointing the
    /// validator at an operation nothing had ever called.
    None,
    /// Everything before is visible to everything after, for workgroup memory.
    ///
    /// `AcquireRelease | WorkgroupMemory`, which is what a GLSL `barrier()` emits and what a
    /// shared-memory handover needs: writes before the barrier must be readable after it.
    AcquireReleaseWorkgroup,
    /// The same for buffer memory: `AcquireRelease | UniformMemory`.
    ///
    /// What an atomic needs when it publishes something *other* than itself — a counter whose
    /// value another invocation uses to decide where to read. An atomic that is only ever summed
    /// up after the dispatch needs none of it, and [`MemorySemantics::None`] is the honest mask
    /// for **an atomic** doing that: ordering nothing is cheaper than ordering nothing while
    /// saying otherwise.
    ///
    /// It is also the mask a barrier may not have, which the sentence above did not say for as
    /// long as nothing had validated a barrier. See [`MemorySemantics::None`].
    AcquireReleaseBuffer,
}

impl MemorySemantics {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::None => 0,
            // AcquireRelease is 0x8 and WorkgroupMemory is 0x100.
            Self::AcquireReleaseWorkgroup => 0x108,
            // The same 0x8, with UniformMemory's 0x40 — which is the flag that covers a storage
            // buffer, despite the name reading as though it would not.
            Self::AcquireReleaseBuffer => 0x48,
        }
    }
}

/// A property attached to an id or a struct member (`OpDecorate`, `OpMemberDecorate`).
///
/// Several carry a literal operand of their own — `Binding 0`, `Offset 4` — which is why
/// [`crate::module::Module::decorate`] takes a slice of extra operands rather than just this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Decoration {
    /// This struct is a shader-interface block. Required on the struct behind a `StorageBuffer`
    /// variable, and it is what makes `Offset` on its members mean anything.
    Block,
    /// The pre-1.3 spelling of `Block`, paired with the `Uniform` storage class. Named so the
    /// distinction exists; nothing here emits it.
    BufferBlock,
    /// Bytes between consecutive elements of an array. Required on a runtime array in a buffer.
    ArrayStride,
    /// Which specialization id a constant answers to.
    ///
    /// Without it a specialization constant is a constant with a strange opcode: nothing can
    /// replace it, and a `VkSpecializationInfo` naming an id no constant carries is ignored rather
    /// than refused.
    SpecId,
    /// A struct member's byte offset from the start of the struct.
    Offset,
    /// Which binding within its descriptor set a variable occupies.
    Binding,
    /// Which descriptor set a variable belongs to.
    DescriptorSet,
    /// This variable *is* a built-in, named by the [`BuiltIn`] operand that follows.
    BuiltIn,
    /// Nothing writes through this.
    NonWritable,
    /// Nothing reads through this.
    NonReadable,
}

impl Decoration {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::SpecId => 1,
            Self::Block => 2,
            Self::BufferBlock => 3,
            Self::ArrayStride => 6,
            Self::BuiltIn => 11,
            Self::NonWritable => 24,
            Self::NonReadable => 25,
            Self::Binding => 33,
            Self::DescriptorSet => 34,
            Self::Offset => 35,
        }
    }
}

/// A value the implementation supplies rather than the shader computing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltIn {
    /// Which workgroup this invocation is in. A three-component vector of `u32`.
    WorkgroupId,
    /// Where this invocation sits inside its workgroup. A three-component vector of `u32`.
    LocalInvocationId,
    /// Where this invocation sits across the whole dispatch. A three-component vector of `u32`.
    GlobalInvocationId,
    /// How many invocations a subgroup holds: 32 on NVIDIA, 32 or 64 on AMD.
    ///
    /// This is the number a `Simd<T, N>` has to agree with, and it is a *runtime* value rather
    /// than a compile-time one — which is most of the difficulty of the lane mapping, and why
    /// `decisions/DR-0002` exists.
    SubgroupSize,
    /// How many subgroups make up the workgroup.
    NumSubgroups,
    /// Which lane this invocation is within its subgroup.
    SubgroupLocalInvocationId,
}

impl BuiltIn {
    /// The word this encodes to.
    #[must_use]
    pub const fn word(self) -> Word {
        match self {
            Self::WorkgroupId => 26,
            Self::LocalInvocationId => 27,
            Self::GlobalInvocationId => 28,
            Self::SubgroupSize => 36,
            Self::NumSubgroups => 38,
            Self::SubgroupLocalInvocationId => 41,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_storage_class_matches_the_khronos_grammar() {
        assert_eq!(StorageClass::Input.word(), 1);
        assert_eq!(StorageClass::Workgroup.word(), 4);
        // Read off `spirv-as` on 2026-08-12: a barrier written with these numbers assembles and
        // `spirv-val --target-env vulkan1.1` accepts it. DR-0001.
        assert_eq!(MemorySemantics::None.word(), 0);
        assert_eq!(MemorySemantics::AcquireReleaseWorkgroup.word(), 264);
        // AcquireRelease 0x8 | UniformMemory 0x40. `UniformMemory` is the flag that covers a
        // storage buffer, which the name does not suggest and the grammar does say.
        assert_eq!(MemorySemantics::AcquireReleaseBuffer.word(), 72);
        assert_eq!(StorageClass::Private.word(), 6);
        assert_eq!(StorageClass::Function.word(), 7);
        assert_eq!(StorageClass::StorageBuffer.word(), 12);
    }

    #[test]
    fn every_decoration_matches_the_khronos_grammar() {
        assert_eq!(Decoration::SpecId.word(), 1);
        assert_eq!(Decoration::Block.word(), 2);
        assert_eq!(Decoration::BufferBlock.word(), 3);
        assert_eq!(Decoration::ArrayStride.word(), 6);
        assert_eq!(Decoration::BuiltIn.word(), 11);
        assert_eq!(Decoration::NonWritable.word(), 24);
        assert_eq!(Decoration::NonReadable.word(), 25);
        assert_eq!(Decoration::Binding.word(), 33);
        assert_eq!(Decoration::DescriptorSet.word(), 34);
        assert_eq!(Decoration::Offset.word(), 35);
    }

    #[test]
    fn every_builtin_matches_the_khronos_grammar() {
        assert_eq!(BuiltIn::WorkgroupId.word(), 26);
        assert_eq!(BuiltIn::LocalInvocationId.word(), 27);
        assert_eq!(BuiltIn::GlobalInvocationId.word(), 28);
        assert_eq!(BuiltIn::SubgroupSize.word(), 36);
        assert_eq!(BuiltIn::NumSubgroups.word(), 38);
        assert_eq!(BuiltIn::SubgroupLocalInvocationId.word(), 41);
    }
}
