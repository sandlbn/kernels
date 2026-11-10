# Correctness Debugging Guide

When a DPAS GEMM kernel produces wrong results, work through this checklist in
order. Each item lists the symptom, cause, and fix.

---

## 1. Wrong B load intrinsic for the matrix layout

**Symptom**: all output values wrong; errors not localised to a region.

| B storage  | Correct intrinsic                              |
|-----------|------------------------------------------------|
| Row-major (K×N stride=N) | `__spirv_Subgroup2DBlockLoadTransformINTEL` |
| Col-major (N×K stride=K) | `__spirv_Subgroup2DBlockLoadTransposeINTEL` |

Fix: match intrinsic to layout. Also verify the CPU reference loop:
- Row-major B: `gold += A[i*K+k] * B[j + k*N]`
- Col-major B: `gold += A[i*K+k] * B[j*K + k]`

---

## 2. Swapped or wrong block_coord x/y

**Symptom**: systematic block-level misalignment; diagonal bands of correct values.

`block_coord = {x, y}`. Both coordinates are in units of the **`ElementSize`
argument you passed** — never in bytes:

- Load / Transform / Store / Prefetch pass `ElementSize = sizeof(T)`, so x and y
  are in elements of `T`.
- `BlockLoadTranspose` must pass `ElementSize = sizeof(int32_t)`, so its x is in
  **dwords** — for fp16 or bf16, one unit is *two* elements.

> **PITFALL**: the x coordinate is NEVER in bytes. Advancing `ca.x` by
> `DPAS_SG_K * sizeof(half)` (bytes) instead of `DPAS_SG_K` (elements) causes
> every other K-slice to be skipped, producing completely wrong results.

> **PITFALL**: the mirror-image mistake on Transpose. Advancing `cb.x` by 16 for a
> 16-deep fp16 K-slice overshoots by 2×, because the unit is a dword — the correct
> advance is 8. Measured effect: max abs error ~2.0, with no diagnostic of any kind.

Alignment requirements (enforced by hardware):
- 8-bit data: x must be multiple of 4 elements (= 4 bytes aligned)
- 16-bit data: x must be multiple of 2 elements (= 4 bytes aligned)
- 32-bit data: x must be multiple of 1 element

Common mistakes:
```cpp
// WRONG: advancing by bytes instead of elements (2x overshoot for fp16)
ca.x += DPAS_SG_K * sizeof(half);   // skips every other K-slice!

// CORRECT:
ca.x += DPAS_SG_K;                  // element advance

// WRONG: x/y swapped for B Transform
block_coord cb{0, sg_startx};       // Transform needs {sg_startx (elem), 0}

// CORRECT:
block_coord cb{(int32_t)sg_startx, 0};
```

---

## 3. Wrong k_dim_length passed to DPAS

**Symptom**: output values scaled or completely wrong.

```cpp
// WRONG (always 32):
__spirv_SubgroupMatrixMultiplyAccumulateINTEL(32, rA, rB, rC, 0x3000); // bf16

// CORRECT:
__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16, rA, rB, rC, 0x3000); // bf16 needs 16
__spirv_SubgroupMatrixMultiplyAccumulateINTEL(32, rA, rB, rC, 0x33);   // int8 needs 32
```

---

## 4. Wrong Operands flag

**Symptom**: NaN, garbage, or systematically scaled output.

| Types         | Flag     | Bit meaning                                       |
|---------------|----------|---------------------------------------------------|
| int8→int32   | `0x33`   | MatrixAPackedInt8 \| MatrixBPackedInt8             |
| bf16→float32 | `0x3000` | MatrixAPackedBFloat16 \| MatrixBPackedBFloat16     |
| fp16→float32 | `0xC00`  | MatrixAPackedFloat16 \| MatrixBPackedFloat16       |

> **PITFALL**: fp16 and bf16 share the same C++ signature but have **different**
> Operands flags. `0x3000` is bf16; `0xC00` is fp16. Using `0x3000` for fp16
> produces completely wrong results.

---

## 5. Mismatched register array dimensions

**Symptom**: partially correct output; errors appear in a regular tiled pattern.

The `rA`, `rB`, `rC` dimensions must be consistent with BlockWidth, BlockHeight,
BlockCount, and SG_SIZE:

```
rA[sg_k_count][sg_a_count][ 8 * k_dim / SG_SIZE * sizeof(A) ]   bytes
rB[sg_b_count][sg_k_count][16 * k_dim / SG_SIZE * sizeof(B) ]   bytes  (Transform)
rB[sg_k_count][sg_b_count][16 * k_dim / SG_SIZE * sizeof(B) ]   bytes  (Transpose)
rC[sg_a_count][sg_b_count][ 8 * 16   / SG_SIZE * sizeof(C) ]   bytes
```

Also verify the reinterpret_cast pointer arithmetic in the DPAS call matches
these dimensions exactly.

---

## 6. Wrong coordinate advancement direction

**Symptom**: first K-slice correct, subsequent slices wrong.

| Matrix & layout       | Advance per k-iteration          |
|-----------------------|----------------------------------|
| A (row-major)         | `ca.x += sg_k`  (elements)       |
| B (Transform, row-major) | `cb.y += sg_k`  (elements)   |
| B (Transpose, col-major) | `cb.x += sg_k / (4 / sizeof(data_type_b))`  (**dwords**) |

The Transpose row is the one people get wrong. Its `ElementSize` is forced to 4,
so x counts dwords, not elements of the data type:

| data type | K-slice depth | correct `cb.x` advance |
|-----------|---------------|------------------------|
| fp16 / bf16 | 16 | **8** |
| fp16 / bf16 | 32 | **16** |
| int8        | 32 | **8** |

Advancing by the element count instead (16 for a 16-deep fp16 slice) overshoots by
2× and silently reads the wrong K data — max abs error ~2.0 in a measured case.

For the store, advance `cc.x += 16` per b-tile, then reset and advance `cc.y += 8` per a-tile.

---

## 7. Wrong memoryWidth / memoryHeight / memoryPitch

**Symptom**: off-by-sizeof errors; output rows progressively shifted.

| Parameter     | Value                                     |
|---------------|-------------------------------------------|
| memoryWidth   | `num_cols * sizeof(element)` (bytes)      |
| memoryHeight  | `num_rows` (elements, NOT bytes)          |
| memoryPitch   | `num_cols * sizeof(element)` (bytes, same as width for contiguous) |

```cpp
// WRONG: passed num_cols instead of bytes
__spirv_Subgroup2DBlockLoadINTEL(1, 32, 8, 1, pA.get(),
    mat_k,           // should be mat_k * sizeof(int8_t)
    mat_m,
    mat_k,           // should be mat_k * sizeof(int8_t)
    ca, rA);

// CORRECT:
__spirv_Subgroup2DBlockLoadINTEL(1, 32, 8, 1, pA.get(),
    mat_k * sizeof(data_type_a),
    mat_m,
    mat_k * sizeof(data_type_a),
    ca, rA);
```

---

## 8. Device code running on host (missing `#ifdef __SYCL_DEVICE_ONLY__`)

**Symptom**: crashes or garbage even for small inputs; no errors on GPU validation.

Wrap all register variables, loads, DPAS calls, and stores inside:

```cpp
#ifdef __SYCL_DEVICE_ONLY__
    // ... everything ...
#endif
```

The host compiler does not understand the SPIRV intrinsics and will either crash
or produce undefined behaviour if they execute on the host.

**But do not put `address_space_cast` inside the guard.** Those calls produce the
pointers the lambda captures; guarded out on the host pass, the lambda captures
nothing and the two passes disagree on its size. The error is a ~20-line
`static_assert` from `handler.hpp` — *"Unexpected kernel lambda size ... In many
cases the difference is related to capturing constexpr variables"* — which points
at constexpr capture and not at the actual cause.

---

## 9. memoryWidth too narrow (< 64 bytes)

**Symptom**: wrong results or undefined behaviour; no compile error.

All 2D block intrinsics require `memoryWidth >= 64 bytes`. If the actual matrix
column count gives fewer than 64 bytes (e.g. a 16-column fp16 B tile = 32 bytes),
you must **pad the allocation**:

```cpp
// WRONG: 16 fp16 cols = 32 bytes < 64 minimum
half* memB = malloc_shared<half>(MAT_N * MAT_K, q);       // MAT_N = 16
const int b_mem_w = MAT_N * sizeof(half);                  // = 32 bytes  WRONG

// CORRECT: pad row to 32 fp16 = 64 bytes
static constexpr int B_STRIDE = 32;                        // >= 32 for fp16
half* memB = malloc_shared<half>(B_STRIDE * MAT_K, q);
const int b_mem_w = B_STRIDE * sizeof(half);               // = 64 bytes  OK
// Also update gold computation: memB[j + k * B_STRIDE]
```

The `BlockWidth` of the load can still be 16 (reading only the real columns);
only the memory allocation and memoryWidth/memoryPitch need the extra padding.

There is a **maximum** as well, and it is per-load rather than per-allocation:
`BlockWidth × BlockCount × ElementSize ≤ 64` bytes. Exceeding it is not a runtime
error but a backend link failure naming an internal builtin, e.g. `undefined
reference to '__internal_intel_sub_group_2d_block_read_16b_8r16x4c_cache_controls'`
— the `16x4c` in the symbol is the `BlockWidth × BlockCount` you asked for. Split
the load into several calls with smaller `BlockCount`.

---

## 10. Register spills

**Symptom**: correct results but poor performance; build log contains
`compiled SIMD16 allocated 128 regs and spilled around N`. Also visible in an ISA
dump as **two** `_simd16_*.asm` files for one kernel (IGC's retry manager
recompiles after a spill).

In order of preference:

1. Add `-ze-opt-large-register-file` to the backend options (128 → 256 GRFs).
2. Shrink the tile — accumulators dominate, at 512 real bytes per 8×16 fp32 tile.
3. Restructure so short-lived operands do not all coexist: build the B operand
   for one k-slice inside the DPAS loop rather than pre-loading every slice.

Do the arithmetic with [TILING.md](TILING.md)'s corrected formula before
concluding the tile "obviously fits" — per-lane byte counts understate real usage
by the subgroup width.

---

## 11. Wrong register layout when an operand is built rather than loaded

**Symptom**: a plain GEMM is correct but a chained or fused kernel (attention,
epilogue feeding a second DPAS) is wrong in a structured way — rows or columns
permuted, or one 16-wide block of the output correct and the rest garbage.

The 2D block loads produce the correct per-lane layout for you. Code that
*constructs* an operand — from an accumulator, from a reduction, from SLM — must
reproduce that layout by hand. See
[REGISTER_LAYOUTS.md](REGISTER_LAYOUTS.md), which also gives two probes for
determining a layout empirically instead of reasoning about it. The most common
errors are:

- Converting an accumulator to an A operand numerically instead of bit-casting:
  `rA[m] = (int16_t)rC[m]` truncates to an integer. `short8_dpas` is a vector of
  `int16_t`, so the half must be `sycl::bit_cast`.
- Reducing over the wrong axis. In an accumulator, a fixed row lives at one vector
  index across all 16 lanes, so per-row reductions are `reduce_over_group` calls —
  one per row — not loops over the array.
- Assuming a chained accumulator needs shuffles or an SLM round-trip. When the
  second GEMM contracts over the first's N dimension, it does not: the transfer is
  elementwise.

---

## Quick diagnostic sequence

For a plain GEMM:

1. Test with a tiny problem (M=8, N=16, K=32) — all values in one DPAS tile.
2. Print `sg_startx`, `sg_starty` for one work-item to verify tiling.
3. Comment out prefetch; confirm correctness without it.
4. Re-enable prefetch; if now wrong, check prefetch coordinate initialization.
5. Set all B values to 1 and verify `C[i][j] == sum of A row i`.
6. Set all A values to 1 and verify `C[i][j] == sum of B col j` (row-major) or row j (col-major).

For a fused or chained kernel, the steps above will not localise the bug. Instead:

7. Set B to the 16×16 **identity** and store the accumulator. DPAS is now a copy,
   so the stored tile *is* your A operand — any layout error is directly visible.
8. Fill an operand with position-encoding values (`A[m][k] = m*100 + k`) and repeat
   step 7 to read off exactly which source element landed where.
9. Bypass one stage at a time: replace the softmax (or other epilogue) with the
   identity and check that the two GEMMs alone compose correctly, then reinstate it.
10. Verify with inputs that make the intermediate *non-uniform*. A softmax fed
    small logits is nearly uniform, and a nearly uniform intermediate hides
    permutation errors and missing max-subtraction almost perfectly — a broken
    kernel can agree with a reference to 1e-5. Amplify the inputs until the
    intermediate is strongly peaked, then compare.
