# Raw DPAS/SPIR-V intrinsics GEMM (Intel Xe)

Write, debug, and tune high-performance GEMM and attention kernels for Intel GPUs
using DPAS (Dot Product Accumulate Systolic) instructions directly via SYCL and
SPIR-V intrinsics, for the case where no SYCL-TLA collective fits (see
[../sycl-implementation.md](../sycl-implementation.md), "Choosing a DSL") — chaining
one DPAS result into another (attention), reducing across a DPAS accumulator, or
choosing tile sizes for a given device. Prefer a SYCL-TLA collective
([../tla-gemm/README.md](../tla-gemm/README.md), [../tla-cute-xe/README.md](../tla-cute-xe/README.md))
when one fits — it encodes the legal atom/copy/layout combinations for you.

Requires `icpx` (Intel DPC++ compiler) and an Intel GPU (Xe2/Xe3 or later).
Compiler flag `--spirv-ext=+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate`
must be supported.

## Overview

A DPAS GEMM kernel computes **C = A × B** where each of the 16-wide subgroups
executes a sequence of `dpas` instructions to accumulate an 8×16 (rows×cols)
output tile. The SPIRV intrinsics below expose those instructions directly from
SYCL device code.

See [SPIRV intrinsic declarations](INTRINSICS.md) for the complete
extern declarations to paste into every kernel file.

See [Register layouts](REGISTER_LAYOUTS.md) for what each operand
actually holds per lane. **Read this before writing anything that is not a plain
GEMM** — chaining one DPAS into another (attention), reducing across an
accumulator (softmax, row-max), or building an operand in registers rather than
loading it. The 2D block loads hide these layouts; the moment you build an
operand yourself they are the entire problem.

See [Operand flags](OPERAND_FLAGS.md) for the `Operands` bitmask —
the per-operand type bits the `0x33` / `0x3000` / `0xC00` constants are built
from, plus the int4, tf32 and bf16-output bits not covered below.

See [Tiling & performance guide](TILING.md) for GRF budgets, tile
size selection, spill remedies, and prefetch tuning.

See [Correctness debugging guide](DEBUGGING.md) for a systematic
checklist when results are wrong.

---

## Compilation command

```bash
icpx -fsycl \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend "-device cri -options -ze-opt-large-register-file" \
  -Xspirv-translator "--spirv-ext=+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
  your_kernel.cpp
```

`-ze-opt-large-register-file` doubles the per-thread GRF allocation, 128 → 256
GRFs. Needed for 32×64 or larger tiles, and it is also **the first thing to try
when the backend reports spills** (`compiled SIMD16 allocated 128 regs and
spilled around N`). Omit it for small tiles on bandwidth-bound workloads, since
it halves the number of threads per EU.

Replace `-device cri` with the architecture you are actually targeting — `lnl`
for Lunar Lake, `bmg` for Battlemage. `sycl-ls` names the device; running
`ocloc ids <arch>` confirms the mapping. Building for the wrong architecture
either fails in `ocloc` or silently produces a binary the runtime will not use.

This AOT form (`-fsycl-targets=spir64_gen`) is also what makes the generated ISA
inspectable — see "Confirming DPAS reached the ISA" below. A JIT build produces
no `.asm` at compile time.

---

## Required boilerplate (paste at top of every kernel)

```cpp
#include <sycl/sycl.hpp>
using namespace sycl;

#define SG_SIZE 16

typedef int32_t block_coord __attribute__((ext_vector_type(2)));
using int8_dpas   = int     __attribute__((ext_vector_type(8)));
using float8_dpas = float   __attribute__((ext_vector_type(8)));
using short8_dpas = int16_t __attribute__((ext_vector_type(8)));

static inline uint32_t ceil_div_int(int a, int b) { return (a+b-1)/b; }
```

Paste the extern SPIRV declarations from
[INTRINSICS.md](INTRINSICS.md) immediately after.

---

## DPAS data types and operand flags

| A type  | B type  | Acc type | k_dim | Operands flag |
|---------|---------|----------|-------|---------------|
| int8    | int8    | int32    | 32    | `0x33`        |
| bf16    | bf16    | float32  | 16    | `0x3000`      |
| fp16    | fp16    | float32  | 16    | `0xC00`       |

`k_dim` is the first argument to `__spirv_SubgroupMatrixMultiplyAccumulateINTEL`.

> **PITFALL**: fp16 uses `0xC00`, NOT `0x3000` (that is bf16). Same C++ overload signature.

The flag is a bitmask, and these three values are just OR-ed combinations of
per-operand type bits — `0xC00` is `MatrixAPackedFloat16 | MatrixBPackedFloat16`.
See [OPERAND_FLAGS.md](OPERAND_FLAGS.md) for the full
bitfield, which is also where to look for int4, tf32 or a bf16 accumulator.
Derive the value from those bits instead of copying a constant; the element type
is not part of the C++ signature, so a wrong flag is never diagnosed.

**B must be in VNNI format** — use the load intrinsics below; never pass raw
row-major B data directly to DPAS.

---

## Choosing the B load intrinsic

| B storage layout | Intrinsic                                      |
|-----------------|------------------------------------------------|
| Row-major (K×N) | `__spirv_Subgroup2DBlockLoadTransformINTEL`    |
| Col-major (N×K) | `__spirv_Subgroup2DBlockLoadTransposeINTEL`    |

Transform works only for 1- and 2-byte element types.
Transpose works for all types; always pass `sizeof(int32_t)` as ElementSize.

**"Col-major" is a statement about the operand, not about the allocation.** A
matrix stored row-major as `N×K` *is* column-major when used as a `K×N` operand.
This is why `B = M^T` for a row-major `M` needs Transpose and no explicit
transposition pass: attention's `S = Q·Kᵀ` with `K` stored row-major `[keys][D]`
is exactly this case.

Because Transpose forces `ElementSize = 4`, its `BlockWidth` and its
`coordinate.x` are counted in **dwords**, not in elements of the underlying type.
For fp16 that means one unit is two halves — so a K-slice of 16 halves advances
`x` by **8**, not 16. Getting this wrong produces plausible-looking but entirely
wrong numbers (measured: max abs error ~2.0).

There is also a hard per-load size limit that no error message explains:

> **`BlockWidth × BlockCount × ElementSize ≤ 64` bytes per load.** Exceed it and
> the backend fails at link time with `undefined reference to
> '__internal_intel_sub_group_2d_block_read_16b_8r16x4c_cache_controls'` or
> similar. Split the load into several calls with a smaller `BlockCount`.

---

## Tiling constants

```cpp
// One DPAS call produces an 8×16 output tile per subgroup.
// Larger tiles amortise load cost over more DPAS calls.
static constexpr size_t sg_m = 32;  // rows  per subgroup (multiple of 8)
static constexpr size_t sg_n = 64;  // cols  per subgroup (multiple of 16)
static constexpr size_t sg_k = 32;  // K-depth per loop iteration (32 for int8, 16·N for bf16)
static constexpr size_t wg_m = 256; // rows  per work-group (multiple of sg_m)
static constexpr size_t wg_n = 256; // cols  per work-group (multiple of sg_n)

static constexpr size_t sg_a_count = sg_m / 8;
static constexpr size_t sg_b_count = sg_n / 16;
static constexpr size_t sg_k_count = sg_k / (32 / sizeof(data_type_a));
```

See [TILING.md](TILING.md) for GRF budget maths and
device-specific limits.

---

## ND-range setup (host side)

```cpp
static constexpr uint32_t lx = wg_n / sg_n, ly = wg_m / sg_m;
range<2> local_range  = {ly, lx * SG_SIZE};
range<2> global_range = local_range *
    range<2>{ceil_div_int(mat_m, wg_m), ceil_div_int(mat_n, wg_n)};

queue q{gpu_selector_v, property::queue::enable_profiling{}};
q.submit([&](auto& h) {
    h.parallel_for(nd_range<2>(global_range, local_range),
        [=](nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            // kernel body
        });
}).wait();
```

---

## Kernel body template

All device-only code must be inside `#ifdef __SYCL_DEVICE_ONLY__` — but note
where the guard *starts*. The `address_space_cast` calls stay **outside** it, as
below. If you move them inside, the host-side lambda captures nothing and you get
a 20-line `static_assert` from `handler.hpp` reading *"Unexpected kernel lambda
size ... In many cases the difference is related to capturing constexpr
variables"*, which misdiagnoses the cause entirely.

```cpp
// Inside the parallel_for lambda:
auto pA = address_space_cast<access::address_space::global_space,
                              access::decorated::no>(memA);
auto pB = address_space_cast<access::address_space::global_space,
                              access::decorated::no>(memB);
auto pC = address_space_cast<access::address_space::global_space,
                              access::decorated::no>(memC);

const auto sg_startx = (item.get_global_id(1) / SG_SIZE) * sg_n;
const auto sg_starty =  item.get_global_id(0)             * sg_m;

#ifdef __SYCL_DEVICE_ONLY__
// Coordinates: {x = col-ELEMENT-offset, y = row-element-offset}
// PITFALL: x is in ELEMENTS (not bytes) for all intrinsics. Advancing by bytes
// skips every other K-slice and produces completely wrong results.
block_coord ca{0,                  (int32_t)sg_starty};
block_coord cb{(int32_t)sg_startx, 0};               // row-major B (Transform)
// block_coord cb{0, (int32_t)sg_startx};             // col-major B (Transpose)
block_coord cc{(int32_t)sg_startx, (int32_t)sg_starty};

data_type_a rA[sg_k_count][sg_a_count][8  * k_dim / SG_SIZE];
data_type_b rB[sg_b_count][sg_k_count][16 * k_dim / SG_SIZE]; // Transform
// data_type_b rB[sg_k_count][sg_b_count][16 * k_dim / SG_SIZE]; // Transpose
data_type_c rC[sg_a_count][sg_b_count][(8 * 16) / SG_SIZE] = {{{0}}};

#pragma nounroll
for (int i = 0; i < (int)(mat_k / sg_k); ++i) {
    // Load A (row-major)
    __spirv_Subgroup2DBlockLoadINTEL(sizeof(data_type_a),
        k_dim, sg_a_count*8, sg_k_count,
        pA.get(), mat_k*sizeof(data_type_a), mat_m, mat_k*sizeof(data_type_a),
        ca, rA);
    ca.x += sg_k;

    // Load B with VNNI transform (row-major B)
    __spirv_Subgroup2DBlockLoadTransformINTEL(sizeof(data_type_b),
        sg_b_count*16, sg_k, 1,
        pB.get(), mat_n*sizeof(data_type_b), mat_k, mat_n*sizeof(data_type_b),
        cb, rB);
    cb.y += sg_k;

    asm volatile("" ::: "memory");   // see rule 7 - "fence_sw" with an empty
                                     // clobber list does not order anything

    #pragma unroll
    for (int a = 0; a < (int)sg_a_count; ++a)
    #pragma unroll
    for (int b = 0; b < (int)sg_b_count; ++b)
    #pragma unroll
    for (int k = 0; k < (int)sg_k_count; ++k)
        *reinterpret_cast<AccVec*>(&rC[a][b]) =
            __spirv_SubgroupMatrixMultiplyAccumulateINTEL(k_dim,
                *reinterpret_cast<short8_dpas*>(&rA[k][a]),
                *reinterpret_cast<int8_dpas*> (&rB[b][k]),
                *reinterpret_cast<AccVec*>    (&rC[a][b]),
                OPERANDS_FLAG);
}

// Store C
#pragma unroll
for (int a = 0; a < (int)sg_a_count; ++a) {
    #pragma unroll
    for (int b = 0; b < (int)sg_b_count; ++b) {
        __spirv_Subgroup2DBlockStoreINTEL(sizeof(data_type_c),
            16, 8, 1, rC[a][b],
            pC.get(), mat_n*sizeof(data_type_c), mat_m, mat_n*sizeof(data_type_c), cc);
        cc.x += 16;
    }
    cc.x -= sg_b_count * 16;
    cc.y += 8;
}
#endif
```

Where `AccVec` is `int8_dpas` for int32 output or `float8_dpas` for float32
output, and `OPERANDS_FLAG` is `0x33` (int8), `0x3000` (bf16), or `0xC00` (fp16)
— see the table above; fp16 and bf16 are **not** interchangeable here.

Two things in this template are expedient rather than correct, and both have
bitten real kernels:

- **The `reinterpret_cast` punning is undefined behaviour.** `rA`/`rB`/`rC` are
  declared as `half`/`float` arrays and read as `int16_t`/`int`/`float` vectors,
  which violates strict aliasing; the optimiser is entitled to reorder or discard
  the stores that fill them. Prefer `__builtin_memcpy` into a local vector, which
  compiles to the same registers with no aliasing assumption:
  ```cpp
  short8_dpas va; int8_dpas vb; AccVec vc;
  __builtin_memcpy(&va, &rA[k][a], sizeof(va));
  __builtin_memcpy(&vb, &rB[b][k], sizeof(vb));
  __builtin_memcpy(&vc, &rC[a][b], sizeof(vc));
  vc = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(k_dim, va, vb, vc, OPERANDS_FLAG);
  __builtin_memcpy(&rC[a][b], &vc, sizeof(vc));
  ```
- **`asm volatile("fence_sw":::)` has an empty clobber list, so it orders
  nothing.** Without a `"memory"` clobber the optimiser may freely move loads and
  stores across it — which is the opposite of the intent. Use
  `asm volatile("" ::: "memory")` if you want a barrier the compiler must honour.

A third trap, if you declare an operand buffer *inside* an unrolled loop: clang
hoists it to a **single** entry-block alloca that every unrolled copy shares, so N
block loads all target the same storage and N-1 results are lost. Declare
per-iteration arrays (`half rK[N][K]`) instead of one buffer reused inside
`#pragma unroll`.

---

## Essential rules

1. **Always wrap device code in `#ifdef __SYCL_DEVICE_ONLY__`** — but keep the
   `address_space_cast` calls outside the guard, or the kernel will not compile.
2. **Always require sub-group size 16.** `[[sycl::reqd_sub_group_size(16)]]` on
   the lambda, *or* `sub_group_size<16>` in the launch property list — **not
   both.** Mixing the two makes the compiler warn *"kernel has both attribute
   'reqd_sub_group_size' and kernel properties; conflicting properties are
   ignored"* and silently drop one, leaving the SIMD width to a driver heuristic.
   If you pass any property list (e.g. `grf_size`), use the property form for the
   sub-group size too. Every layout in this skill depends on the width being 16,
   so verify rather than assume: query
   `kernel_device_specific::max_sub_group_size` on the built bundle and fail
   loudly if it is not 16.
3. **`block_coord` is `{x, y}`, and both are in units of `ElementSize` — never
   bytes.** For Load / Transform / Store that means elements of the data type.
   For Transpose, `ElementSize` is forced to 4, so x is in **dwords** — for fp16,
   one unit is two halves. (`memoryWidth`/`memoryPitch` *are* in bytes; the
   coordinates are not.)
4. **Advance A by `ca.x += sg_k` each iteration** (column moves right in K).
5. **Advance B (Transform) by `cb.y += sg_k` each iteration** (row moves down in K).
6. **Advance B (Transpose) by `cb.x += sg_k / (4 / sizeof(data_type_b))`** —
   dwords, so `sg_k / 2` for fp16/bf16 and `sg_k / 4` for int8.
7. **A compiler barrier between loads and DPAS needs a `"memory"` clobber** to do
   anything: `asm volatile("" ::: "memory")`. The `asm volatile("fence_sw":::)`
   found in older code has an empty clobber list and does not order memory
   operations.
8. **memoryWidth and memoryPitch are in bytes; memoryHeight is in elements.**
9. **`BlockWidth × BlockCount × ElementSize ≤ 64` bytes per load**, and
   `memoryWidth ≥ 64` bytes (pad the allocation if the real matrix is narrower).

---

## Runnable examples (`assets/`)

Reference implementations live under `assets/` (compile lines at the top of each `.cpp`),
built and run via `assets/build.sh`.

Naming: `gemm_<A>_<B>_<C>_sg<M>x<N>x<K>_b_<transform|transpose>[_prefetch].cpp`
— dtypes, subgroup tile (`sg_m`×`sg_n`×`sg_k`), row-major B load
(`b_transform`) vs col-major B (`b_transpose`), optional `_prefetch` suffix.

| Example | Role |
|---------|------|
| `gemm_i8_i8_i32_sg8x16x32_b_transform` | Smallest int8 tile, row-major B |
| `gemm_i8_i8_i32_sg8x16x32_b_transpose` | Same tile, col-major B |
| `gemm_i8_i8_i32_sg32x64x32_b_*` | Large int8 tile + prefetch in mainloop |
| `gemm_bf16_bf16_f32_sg32x64x32_b_transform_prefetch` | Large BF16 tile, pipelined prefetch. Also the **FP16** example: bf16 and fp16 are both 2-byte types, so only the type alias and the operand flag differ — the two spots are marked `FP16:` in the file |

---

## Confirming DPAS reached the ISA

A kernel that quietly compiles to scalar FMAs can still be numerically correct,
so verify the instructions rather than inferring them from the result. The ISA is
generated by `ocloc` at **build** time for an AOT build, so dump it there:

```bash
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=$PWD/isa_dump \
  icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device lnl" \
  -Xspirv-translator "--spirv-ext=+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
  -o kernel kernel.cpp

grep -c dpas isa_dump/*_simd16_*.asm
```

The AOT flags are load-bearing: without `-fsycl-targets=spir64_gen` and a
`-device`, no `ocloc` runs and no `.asm` appears. The directory is created for you.

Reading the output:

- Expect one `.asm` per kernel entry point per SIMD variant. Which variants
  appear is build-dependent — with `-ze-opt-large-register-file` you typically
  get only SIMD16, without it you may also get a SIMD32 variant containing zero
  `dpas`. Grep the entry that corresponds to your kernel and its required width.
- **Two `_simd16_` files for one kernel means it spilled** and IGC's retry
  manager recompiled it. Check the build log for the spill count.
- Confirm the count against the DPAS calls your tiling implies
  (`sg_a_count × sg_b_count × sg_k_count × k-iterations`, times the number of
  chained GEMMs). A count that is a clean fraction of the expectation usually
  means a loop collapsed or a tile was optimised away.
- The operand types are visible in the mnemonic, e.g.
  `dpas.8x8 (16|M0) r10:f r20:f r30:hf r40:hf` — `:hf` confirms 16-bit input and
  `:f` an fp32 accumulator.
