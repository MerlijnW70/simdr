//! Opcode numbers, from the SPIR-V instruction table (§3.52).
//!
//! Only the ones something in this crate emits are listed, and each was read out of Khronos'
//! grammar before it was written down — see `decisions/DR-0001`. A number invented from memory is
//! a module that assembles cleanly and means something else.

/// `OpName` — a debug name for an id. Ignored by hardware, invaluable in a disassembly.
pub const NAME: u16 = 5;
/// `OpExtension` — declares a SPIR-V extension the module uses.
///
/// Not the same thing as an extended instruction set: this one changes what the *core* language
/// allows, and it carries no id because nothing refers back to it.
pub const EXTENSION: u16 = 10;
/// `OpExtInstImport` — names an extended instruction set and yields an id for it.
pub const EXT_INST_IMPORT: u16 = 11;
/// `OpExtInst` — calls an instruction from an imported set.
///
/// The instruction number it carries is a literal in *that set's* numbering, which is why
/// [`crate::spec::Glsl`] is a separate list read from a separate grammar.
pub const EXT_INST: u16 = 12;
/// `OpMemoryModel` — the addressing and memory model, exactly one per module.
pub const MEMORY_MODEL: u16 = 14;
/// `OpEntryPoint` — names a function as an entry point and lists its interface.
pub const ENTRY_POINT: u16 = 15;
/// `OpExecutionMode` — a mode an entry point executes under, such as its workgroup size.
pub const EXECUTION_MODE: u16 = 16;
/// `OpCapability` — declares a capability the module needs.
pub const CAPABILITY: u16 = 17;
/// `OpTypeVoid` — the type of a function that returns nothing.
pub const TYPE_VOID: u16 = 19;
/// `OpTypeArray` — an element type and a constant length.
pub const TYPE_ARRAY: u16 = 28;
/// `OpTypeRuntimeArray` — an element type, with the length coming from the bound buffer.
pub const TYPE_RUNTIME_ARRAY: u16 = 29;
/// `OpTypeStruct` — its member types, in order.
pub const TYPE_STRUCT: u16 = 30;
/// `OpTypeBool` — the type a comparison produces.
pub const TYPE_BOOL: u16 = 20;
/// `OpTypeInt` — width in bits, plus whether it is signed.
pub const TYPE_INT: u16 = 21;
/// `OpTypeFloat` — width in bits.
pub const TYPE_FLOAT: u16 = 22;
/// `OpTypeVector` — a component type and a count.
pub const TYPE_VECTOR: u16 = 23;
/// `OpTypePointer` — a storage class and a pointee type.
pub const TYPE_POINTER: u16 = 32;
/// `OpTypeFunction` — a return type and its parameter types.
pub const TYPE_FUNCTION: u16 = 33;
/// `OpConstantTrue` — the boolean true.
pub const CONSTANT_TRUE: u16 = 41;
/// `OpConstantFalse` — the boolean false.
pub const CONSTANT_FALSE: u16 = 42;
/// `OpConstant` — a scalar constant, carrying its value as literal words.
pub const CONSTANT: u16 = 43;
/// `OpSpecConstantTrue` — a boolean specialization constant defaulting to true.
pub const SPEC_CONSTANT_TRUE: u16 = 48;
/// `OpSpecConstantFalse` — and to false.
pub const SPEC_CONSTANT_FALSE: u16 = 49;
/// `OpSpecConstant` — a scalar constant a pipeline may replace, carrying its default.
pub const SPEC_CONSTANT: u16 = 50;
/// `OpSpecConstantOp` — a constant computed from other constants at pipeline creation.
///
/// Carries an ordinary opcode as a *literal* operand, which is the one instruction in this list
/// whose second word is an opcode rather than an id.
pub const SPEC_CONSTANT_OP: u16 = 52;
/// `OpFunction` — opens a function definition.
pub const FUNCTION: u16 = 54;
/// `OpFunctionEnd` — closes one.
pub const FUNCTION_END: u16 = 56;
/// `OpVariable` — declares storage and yields a pointer to it.
pub const VARIABLE: u16 = 59;
/// `OpLoad` — read through a pointer.
pub const LOAD: u16 = 61;
/// `OpStore` — write through a pointer.
pub const STORE: u16 = 62;
/// `OpAccessChain` — walk into an aggregate and yield a pointer to the part.
pub const ACCESS_CHAIN: u16 = 65;
/// `OpDecorate` — attach a property to an id.
pub const DECORATE: u16 = 71;
/// `OpMemberDecorate` — attach a property to one member of a struct.
pub const MEMBER_DECORATE: u16 = 72;
/// `OpCompositeExtract` — pull a component out of a composite by constant index.
pub const COMPOSITE_EXTRACT: u16 = 81;
/// `OpCopyObject` — the same value under a new id.
pub const COPY_OBJECT: u16 = 83;
/// `OpConvertSToF` — a signed integer's numeric value, as a float.
///
/// What a dot product's result needs on the way into float arithmetic: a *conversion*, so −7
/// becomes −7.0 rather than the float whose bits are `0xfffffff9`.
pub const CONVERT_S_TO_F: u16 = 111;
/// `OpConvertUToF` — an unsigned integer's numeric value, as a float.
///
/// A *conversion*, not a reinterpretation: 7u32 becomes 7.0f32 rather than a denormal.
pub const CONVERT_U_TO_F: u16 = 112;
/// `OpUConvert` — an integer's value at a different width, truncating or zero-extending.
///
/// **The result type's signedness must be 0.** So this cannot produce an `i8`, even though
/// narrowing to one is the same truncation — that case is [`S_CONVERT`].
pub const U_CONVERT: u16 = 113;
/// `OpSConvert` — an integer's value at a different width, truncating or sign-extending.
pub const S_CONVERT: u16 = 114;
/// `OpBitcast` — the same bits under a different type of the same width.
///
/// The right instruction for `u32` to `i32`: at equal widths there is no numeric conversion to
/// make, and the two differ only in how the bits are read.
pub const BITCAST: u16 = 124;
/// `OpIAdd` — integer addition.
pub const I_ADD: u16 = 128;
/// `OpFAdd` — floating-point addition.
pub const F_ADD: u16 = 129;
/// `OpFNegate` — a float with its sign flipped.
///
/// One instruction rather than a multiply by −1.0, and not the same thing: negating flips the sign
/// bit and leaves everything else, including on a zero and a NaN, where a multiply is arithmetic
/// the implementation may contract or reassociate.
pub const F_NEGATE: u16 = 127;
/// `OpISub` — integer subtraction.
pub const I_SUB: u16 = 130;
/// `OpFSub` — floating-point subtraction.
pub const F_SUB: u16 = 131;
/// `OpIMul` — integer multiplication.
pub const I_MUL: u16 = 132;
/// `OpFMul` — floating-point multiplication.
pub const F_MUL: u16 = 133;
/// `OpUDiv` — unsigned integer division.
///
/// Read from the grammar rather than guessed, twice: writing the multi-pass reduction on 2026-08-11
/// this was 152 from memory, and the probe said 134. `decisions/DR-0001` is that story.
pub const U_DIV: u16 = 134;
/// `OpFDiv` — floating-point division.
///
/// Vulkan does not require it to be correctly rounded and implementations differ in the last place,
/// which is worth knowing before a kernel and a CPU reference are asked to agree exactly.
pub const F_DIV: u16 = 136;
/// `OpShiftRightLogical` — shift right, filling with zeros.
pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
/// `OpShiftRightArithmetic` — shift right, filling with copies of the sign bit.
///
/// The difference from [`SHIFT_RIGHT_LOGICAL`] is invisible for values with the top bit clear,
/// which is every value a small test uses.
pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
/// `OpShiftLeftLogical` — shift left.
pub const SHIFT_LEFT_LOGICAL: u16 = 196;
/// `OpBitwiseAnd` — bitwise and of two integers.
///
/// Read out of Khronos' own assembler rather than a table: `spirv-as` was given a module
/// containing `OpBitwiseAnd` and the emitted word carried 199. DR-0001 says the number comes from
/// the authority and not from memory, and the authority answers questions as well as publishing
/// them — the grammar JSON is not installed here and the tool that consumes it is.
pub const BITWISE_AND: u16 = 199;
/// `OpBitwiseOr` — bitwise or.
///
/// **197, below the and.** The bitwise instructions run *downwards* in the grammar — or, xor, and
/// — so the number next to `BITWISE_AND` is not the one this needs. Read out of `spirv-as`, the
/// way `decisions/DR-0001` says to.
pub const BITWISE_OR: u16 = 197;
/// `OpLogicalOr` — boolean or.
pub const LOGICAL_OR: u16 = 166;
/// `OpLogicalAnd` — boolean and.
pub const LOGICAL_AND: u16 = 167;
/// `OpSelect` — pick between two values per component.
pub const SELECT: u16 = 169;
/// `OpIEqual` — integer equality.
pub const I_EQUAL: u16 = 170;
/// `OpUGreaterThan` — unsigned integer `>`.
pub const U_GREATER_THAN: u16 = 172;
/// `OpULessThan` — unsigned integer `<`.
pub const U_LESS_THAN: u16 = 176;
/// `OpSGreaterThan` — signed integer `>`.
pub const S_GREATER_THAN: u16 = 173;
/// `OpFOrdGreaterThan` — ordered floating-point comparison, yielding a bool.
pub const F_ORD_GREATER_THAN: u16 = 186;
/// `OpFOrdEqual` — ordered floating-point equality: false if either operand is NaN.
///
/// 180, and not the 170 the integer form uses nor the 186 beside it — read out of `spirv-as` the
/// way `decisions/DR-0001` says to, because the comparisons are *not* consecutive in the grammar
/// and a number remembered from the neighbourhood would have assembled into something else.
pub const F_ORD_EQUAL: u16 = 180;
/// `OpGroupNonUniformElect` — true in exactly one lane of the group.
pub const GROUP_NON_UNIFORM_ELECT: u16 = 333;
/// `OpGroupNonUniformAll` — true when the predicate holds in every active lane.
pub const GROUP_NON_UNIFORM_ALL: u16 = 334;
/// `OpGroupNonUniformAny` — true when the predicate holds in any active lane.
pub const GROUP_NON_UNIFORM_ANY: u16 = 335;
/// `OpGroupNonUniformAllEqual` — true when every active lane holds the same value.
pub const GROUP_NON_UNIFORM_ALL_EQUAL: u16 = 336;
/// `OpGroupNonUniformBroadcast` — one named lane's value, to every lane.
pub const GROUP_NON_UNIFORM_BROADCAST: u16 = 337;
/// `OpGroupNonUniformBroadcastFirst` — the lowest active lane's value, to every lane.
pub const GROUP_NON_UNIFORM_BROADCAST_FIRST: u16 = 338;
/// `OpGroupNonUniformBallot` — the predicate's value in every lane, as a bitmask.
pub const GROUP_NON_UNIFORM_BALLOT: u16 = 339;
/// `OpGroupNonUniformShuffle` — read another lane's value by its index.
pub const GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;
/// `OpGroupNonUniformShuffleXor` — read the lane whose index is ours XOR a mask. The butterfly.
pub const GROUP_NON_UNIFORM_SHUFFLE_XOR: u16 = 346;
/// `OpGroupNonUniformShuffleUp` — read the lane `delta` below ours.
pub const GROUP_NON_UNIFORM_SHUFFLE_UP: u16 = 347;
/// `OpGroupNonUniformShuffleDown` — read the lane `delta` above ours.
pub const GROUP_NON_UNIFORM_SHUFFLE_DOWN: u16 = 348;
/// `OpGroupNonUniformIAdd` — integer add across the group.
pub const GROUP_NON_UNIFORM_I_ADD: u16 = 349;
/// `OpGroupNonUniformFAdd` — floating-point add across the group.
pub const GROUP_NON_UNIFORM_F_ADD: u16 = 350;
/// `OpGroupNonUniformSMin` — signed minimum across the group.
pub const GROUP_NON_UNIFORM_S_MIN: u16 = 353;
/// `OpGroupNonUniformUMin` — unsigned minimum across the group.
pub const GROUP_NON_UNIFORM_U_MIN: u16 = 354;
/// `OpGroupNonUniformFMin` — floating-point minimum across the group.
pub const GROUP_NON_UNIFORM_F_MIN: u16 = 355;
/// `OpGroupNonUniformSMax` — signed maximum across the group.
pub const GROUP_NON_UNIFORM_S_MAX: u16 = 356;
/// `OpGroupNonUniformUMax` — unsigned maximum across the group.
pub const GROUP_NON_UNIFORM_U_MAX: u16 = 357;
/// `OpGroupNonUniformFMax` — floating-point maximum across the group.
pub const GROUP_NON_UNIFORM_F_MAX: u16 = 358;
/// `OpSDot` — four signed 8-bit products summed into one 32-bit result.
///
/// Takes an optional `PackedVectorFormat`. Left off, the operands are integer *vectors* rather
/// than packed scalars — so the operand's absence is a different instruction, not a default.
pub const S_DOT: u16 = 4450;
/// `OpUDot` — the same over unsigned components.
pub const U_DOT: u16 = 4451;
/// `OpSUDot` — signed on the left, unsigned on the right. Not symmetric.
pub const SU_DOT: u16 = 4452;
/// `OpSDotAccSat` — [`S_DOT`] plus an accumulator, saturating rather than wrapping.
pub const S_DOT_ACC_SAT: u16 = 4453;
/// `OpAtomicLoad` — read a location without another invocation's write landing in the middle.
pub const ATOMIC_LOAD: u16 = 227;
/// `OpAtomicStore` — the same for a write. Produces no id.
pub const ATOMIC_STORE: u16 = 228;
/// `OpAtomicExchange` — write, and yield what was there.
pub const ATOMIC_EXCHANGE: u16 = 229;
/// `OpAtomicIIncrement` — add one, and yield what was there. Takes no value operand.
pub const ATOMIC_I_INCREMENT: u16 = 232;
/// `OpAtomicIAdd` — add, and yield what was there.
pub const ATOMIC_I_ADD: u16 = 234;
/// `OpPhi` — a value that depends on which block control arrived from.
pub const PHI: u16 = 245;
/// `OpControlBarrier` — every invocation in the scope waits here.
///
/// Takes *two* scopes: which invocations synchronise, and which memory the accompanying semantics
/// apply to. They are usually the same and the specification does not make them so.
pub const CONTROL_BARRIER: u16 = 224;
/// `OpMemoryBarrier` — orders memory accesses without making anyone wait.
pub const MEMORY_BARRIER: u16 = 225;
/// `OpLoopMerge` — declares where a loop's back edge and exit go.
pub const LOOP_MERGE: u16 = 246;
/// `OpSelectionMerge` — declares where the arms of a selection rejoin.
pub const SELECTION_MERGE: u16 = 247;
/// `OpLabel` — opens a block; every block starts with one.
pub const LABEL: u16 = 248;
/// `OpBranch` — an unconditional jump.
pub const BRANCH: u16 = 249;
/// `OpBranchConditional` — a two-way jump on a boolean.
pub const BRANCH_CONDITIONAL: u16 = 250;
/// `OpReturn` — returns from a function whose return type is void.
pub const RETURN: u16 = 253;
