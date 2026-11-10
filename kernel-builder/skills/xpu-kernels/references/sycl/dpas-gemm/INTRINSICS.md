# SPIRV Intrinsic Declarations for DPAS Kernels

Paste all of these `extern SYCL_EXTERNAL` declarations into every kernel file,
before `int main()`. They must be at file scope.

For what the register arrays actually contain once loaded — which lane holds which
matrix element — see [REGISTER_LAYOUTS.md](REGISTER_LAYOUTS.md). That can be
ignored while every operand comes from a block load; it cannot once you build one
yourself.

Two size constraints apply to every load, store and prefetch below:

- `memoryWidth ≥ 64` bytes — pad the allocation if the matrix is narrower.
- `BlockWidth × BlockCount × ElementSize ≤ 64` bytes **per call**. Exceeding it
  fails at backend link time with `undefined reference to
  '__internal_intel_sub_group_2d_block_read_16b_8r16x4c_cache_controls'`, where
  `16x4c` encodes the `BlockWidth × BlockCount` requested. Split into several calls
  with a smaller `BlockCount`.

```cpp
// ── 2D Block Load (plain, row-major A) ───────────────────────────────────────
// ElementSize : sizeof(data_type)  e.g. 1 for int8, 2 for bf16
// BlockWidth  : number of columns to load per subgroup lane (in elements)
// BlockHeight : number of rows to load
// BlockCount  : how many (BlockWidth × BlockHeight) tiles to load back-to-back
// srcBasePointer : base of the 2-D buffer in global memory
// memoryWidth : total row width in BYTES  (must be ≥64 and a multiple of 4)
// memoryHeight: total number of rows (in elements, NOT bytes)
// memoryPitch : row stride in BYTES (= memoryWidth for contiguous allocation)
// coordinate  : {x = col-ELEMENT-offset, y = row-element-offset} of the top-left corner
//   PITFALL: x is NOT in bytes — it is always in elements for all element sizes.
//   Alignment: x must be multiple of 4 for 8-bit, multiple of 2 for 16-bit, any for 32-bit.
// dstPointer  : register array to fill
extern SYCL_EXTERNAL void __spirv_Subgroup2DBlockLoadINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void* srcBasePointer,
    int memoryWidth, int memoryHeight, int memoryPitch,
    block_coord coordinate, void* dstPointer);

// ── 2D Block Load + VNNI Transform (row-major B → DPAS-ready) ────────────────
// Converts a row-major B block into VNNI packed format in registers.
// Valid for ElementSize = 1 (int8) or 2 (bf16/fp16).
// coordinate.x = col-ELEMENT-offset into B (advances along N; PITFALL: not bytes)
// coordinate.y = row-element-offset into B (advances along K)
// PITFALL: memoryWidth must be ≥64 bytes. If N < 32 fp16 (or < 64 int8) allocate
//   a padded buffer with stride ≥64 bytes and only fill the real columns.
extern SYCL_EXTERNAL void __spirv_Subgroup2DBlockLoadTransformINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void* srcBasePointer,
    int memoryWidth, int memoryHeight, int memoryPitch,
    block_coord coordinate, void* dstPointer);

// ── 2D Block Load + Transpose (col-major B → DPAS-ready) ─────────────────────
// Reads a col-major B block and produces VNNI format in registers.
// ALWAYS pass ElementSize = sizeof(int32_t) regardless of actual data type.
// PITFALL: because ElementSize is 4, BlockWidth and coordinate.x are counted in
//   DWORDS, not in elements of the data type. For fp16/bf16 one unit is two
//   elements, so a 16-deep K-slice means BlockWidth = 8 and x advances by 8.
//   Advancing by 16 overshoots by 2x and silently reads the wrong K data.
// coordinate.x = col offset into B in DWORDS (advances along K)
// coordinate.y = row-element-offset into B (advances along N)
// memoryWidth / memoryPitch use the col-major matrix dimensions:
//   memoryWidth = mat_k * sizeof(data_type_b)
//   memoryHeight = mat_n
extern SYCL_EXTERNAL void __spirv_Subgroup2DBlockLoadTransposeINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void* srcBasePointer,
    int memoryWidth, int memoryHeight, int memoryPitch,
    block_coord coordinate, void* dstPointer);

// ── 2D Block Store ────────────────────────────────────────────────────────────
// Stores a register tile back to global memory (row-major).
// ElementSize : sizeof(data_type_c)
// BlockWidth  : 16 (one DPAS output column per lane)
// BlockHeight : 8  (one DPAS output row batch)
// BlockCount  : 1  (store one 8×16 tile per call)
// srcPointer  : register array to read from (rC[a][b])
// coordinate  : {x = col-ELEMENT-offset, y = row-element-offset} in C
extern SYCL_EXTERNAL void __spirv_Subgroup2DBlockStoreINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    void* srcPointer,
    const void* dstBasePointer,
    int memoryWidth, int memoryHeight, int memoryPitch,
    block_coord coordinate);

// ── 2D Block Prefetch (non-blocking, fills L1 & L2 only) ─────────────────────
// Same parameter layout as BlockLoadINTEL but does not return data.
// Cannot be used for 2-stage SLM prefetch.
extern SYCL_EXTERNAL void __spirv_Subgroup2DBlockPrefetchINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void* srcBasePointer,
    int memoryWidth, int memoryHeight, int memoryPitch,
    block_coord coordinate);

// ── DPAS: int8 × int8 → int32 ────────────────────────────────────────────────
// K_Dim    : 32  (elements per lane in the K dimension for int8)
// Matrix_A : short8_dpas  (reinterpret_cast from int8 reg slice)
// Matrix_B : int8_dpas    (reinterpret_cast from int8 reg slice, VNNI format)
// Matrix_C : int8_dpas    (accumulator in, also int32 data in 8×int vector)
// Operands : 0x33
extern SYCL_EXTERNAL int8_dpas __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int K_Dim, short8_dpas Matrix_A, int8_dpas Matrix_B,
    int8_dpas Matrix_C, int Operands);

// ── DPAS: 16-bit × 16-bit → float32  (covers BOTH bf16 and fp16) ─────────────
// K_Dim    : 16  (elements per lane in the K dimension)
// Operands : 0x3000  bf16  (MatrixAPackedBFloat16 | MatrixBPackedBFloat16)
//            0xC00   fp16  (MatrixAPackedFloat16  | MatrixBPackedFloat16)
// Declare this ONCE. bf16 and fp16 share one C++ signature - the element type is
// carried only by the runtime Operands argument, so nothing diagnoses a mismatch
// and using 0x3000 on fp16 data compiles cleanly and returns garbage.
// Operands is a bitmask; see OPERAND_FLAGS.md for the individual bits.
extern SYCL_EXTERNAL float8_dpas __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int K_Dim, short8_dpas Matrix_A, int8_dpas Matrix_B,
    float8_dpas Matrix_C, int Operands);
```

## Quick parameter reference

### BlockLoadINTEL / PrefetchINTEL for matrix A (row-major, int8)
```cpp
__spirv_Subgroup2DBlockLoadINTEL(
    /*ElementSize*/ 1,
    /*BlockWidth*/  32,       // k_dim elements per lane = sg_k for int8
    /*BlockHeight*/ 8,        // rows per DPAS call
    /*BlockCount*/  sg_a_count, // or 1 for simple kernels
    pA.get(),
    /*memoryWidth*/ mat_k * sizeof(int8_t),
    /*memoryHeight*/mat_m,
    /*memoryPitch*/ mat_k * sizeof(int8_t),
    load_coord_a, reg_A);
```

### BlockLoadTransformINTEL for matrix B (row-major, int8)
```cpp
__spirv_Subgroup2DBlockLoadTransformINTEL(
    /*ElementSize*/ 1,
    /*BlockWidth*/  16,         // sg_n / sg_b_count columns
    /*BlockHeight*/ 32,         // sg_k rows
    /*BlockCount*/  sg_b_count * sg_k_count,
    pB.get(),
    /*memoryWidth*/ mat_n * sizeof(int8_t),
    /*memoryHeight*/mat_k,
    /*memoryPitch*/ mat_n * sizeof(int8_t),
    load_coord_b, reg_B);
```

### BlockLoadTransposeINTEL for matrix B (col-major, int8)
```cpp
__spirv_Subgroup2DBlockLoadTransposeINTEL(
    /*ElementSize*/ sizeof(int32_t),  // always int32 for Transpose
    /*BlockWidth*/  8,                // K-depth per tile (in 4-byte units)
    /*BlockHeight*/ 16,               // N-cols per tile
    /*BlockCount*/  1,
    pB.get(),
    /*memoryWidth*/ mat_k * sizeof(int8_t),
    /*memoryHeight*/mat_n,
    /*memoryPitch*/ mat_k * sizeof(int8_t),
    load_coord_b, reg_B[k][b]);
```

### BlockStoreINTEL for matrix C (row-major, int32 or float32)
```cpp
__spirv_Subgroup2DBlockStoreINTEL(
    /*ElementSize*/ sizeof(int32_t),
    /*BlockWidth*/  16,
    /*BlockHeight*/ 8,
    /*BlockCount*/  1,
    reg_C[a][b],
    pC.get(),
    /*memoryWidth*/ mat_n * sizeof(int32_t),
    /*memoryHeight*/mat_m,
    /*memoryPitch*/ mat_n * sizeof(int32_t),
    store_coord_c);
```

## Spec links
- 2D Block I/O: https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_2d_block_io.asciidoc
- MMA: https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_subgroup_matrix_multiply_accumulate.asciidoc
- OpenCL parameter table: https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_2d_block_io.html
