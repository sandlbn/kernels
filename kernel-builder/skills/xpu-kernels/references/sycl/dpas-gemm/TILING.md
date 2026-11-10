# Tiling & Performance Guide

## Hardware limits summary

| Device family | GRF registers | GRF mode     | Max threads/EU | GRF/thread |
|---------------|---------------|--------------|----------------|------------|
| 1024-GRF      | 1024          | Small        | 64             | 8 KB       |
| 1024-GRF      | 1024          | Large        | 32             | 16 KB      |
| 1024-GRF      | 1024          | SLarge       | 16             | 32 KB      |
| 2048-GRF(CRI) | 2048          | Small        | 64             | 16 KB      |
| 2048-GRF(CRI) | 2048          | Large        | 32             | 32 KB      |

Enable Large GRF: add `-ze-opt-large-register-file` to the backend options.

L1/SLM per XeCore: 192–512 KB (512 KB on CRI). L1 is shared across all
subgroups in the work-group.

## Register budget calculation

> **The most common mistake here** is comparing **per-lane** bytes against a
> **per-thread** budget. `float rC[4][4][8]` is written per work item, but registers
> are allocated per *thread*, and one thread is the whole 16-wide subgroup. Every
> operand costs its per-lane size **× SG_SIZE**. Drop the ×16 and a design that
> really needs ~224 of 256 GRFs looks like it uses 6% of the budget.

For a tile of `sg_m × sg_n` output and `sg_k` K-depth:

```
sg_a_count = sg_m / 8
sg_b_count = sg_n / 16
sg_k_count = sg_k / (32 / sizeof(data_type_a))   # int8→2, bf16→1

# per-lane bytes
A_lane = sg_a_count * sg_k_count * 8  * k_dim / 16 * sizeof(A)
B_lane = sg_b_count * sg_k_count * 16 * k_dim / 16 * sizeof(B)
C_lane = sg_a_count * sg_b_count * 8  * 16    / 16 * sizeof(C)

# real cost in the thread's register file
Total_bytes = (A_lane + B_lane + C_lane) * SG_SIZE
Total_GRFs  = Total_bytes / 64          # 64-byte GRFs on Xe2/Xe3
```

Compare `Total_GRFs` against **128** (default) or **256** with
`-ze-opt-large-register-file`. Leave headroom — the compiler also needs registers
for addresses, loop state and the epilogue, so budget for ~80% of the limit.

Per-tile costs worth memorising:

| Operand tile | per lane | per thread | GRFs |
|---|---|---|---|
| A, fp16, 8×16 (`short8_dpas`) | 16 B | 256 B | 4 |
| B, fp16, 16×16 (`int8_dpas`) | 32 B | 512 B | 8 |
| C, fp32, 8×16 (`float8_dpas`) | 32 B | 512 B | **8** |

Accumulators dominate. Each 8×16 fp32 output tile is 8 GRFs, so holding an 8×256
accumulator — what a full-row attention softmax needs — costs 128 GRFs, half of
Large GRF mode before any operand is loaded.

### Example: int8, 32×64 tile, sg_k=64
```
sg_a_count = 4, sg_b_count = 4, sg_k_count = 2, k_dim = 32
A_lane = 4 * 2 * 8  * 32/16 * 1 = 128  bytes
B_lane = 4 * 2 * 16 * 32/16 * 1 = 256  bytes
C_lane = 4 * 4 * 8  * 16/16 * 4 = 512  bytes
per lane   =   896 bytes
per thread = 14336 bytes = 224 GRFs
→ requires Large GRF (256); spills badly in the 128-GRF default
```

### Example: bf16, 32×64 tile, sg_k=32
```
sg_a_count = 4, sg_b_count = 4, sg_k_count = 2, k_dim = 16
A_lane = 4 * 2 * 8  * 16/16 * 2 = 128  bytes
B_lane = 4 * 2 * 16 * 16/16 * 2 = 256  bytes
C_lane = 4 * 4 * 8  * 16/16 * 4 = 512  bytes
per lane   =   896 bytes
per thread = 14336 bytes = 224 GRFs
→ same conclusion: Large GRF required, with no room to grow the tile
```

Earlier versions of this file read the 896-byte per-lane figure as the total and
concluded these tiles "fit in 16 KB easily" and "can go larger". They sit at ~88%
of Large GRF. If the backend reports `compiled SIMD16 allocated 128 regs and
spilled around N`, redo this calculation before changing anything else.

## Tile size recommendations

Start with **sg_m=32, sg_n=64** for Large GRF mode. This gives 16 DPAS calls
per K-iteration (4 A-tiles × 4 B-tiles), which is enough to hide load latency.

| Scenario                          | sg_m | sg_n | wg_m | wg_n | prefetch_distance |
|-----------------------------------|------|------|------|------|-------------------|
| int8, CRI, Large GRF (baseline)   | 32   | 64   | 256  | 256  | 2                 |
| bf16, CRI, Large GRF              | 32   | 64   | 256  | 256  | 4                 |
| int8, small K (< 512)             | 8    | 16   | 32   | 128  | 0                 |
| int8, bandwidth-bound device      | 16   | 32   | 128  | 256  | 2                 |

Rules of thumb:
- **DPAS-to-load ratio** = `sg_a_count * sg_b_count * sg_k_count`; aim for ≥ 8.
- Increasing `sg_k` improves ratio but costs more A/B registers.
- Increasing `sg_m` or `sg_n` increases C registers (most expensive at 4 bytes each).
- Work-group tile should give ≥ 32 subgroups per WG for good occupancy
  (`wg_m/sg_m * wg_n/sg_n ≥ 32`).

## Prefetch

Non-blocking prefetch is **essential** to saturate the systolic pipeline.

```cpp
static constexpr int prefetch_distance = 2; // iterations ahead to prefetch

// Warmup: issue prefetch_distance batches before the main K-loop
block_coord pca{0, (int32_t)sg_starty};
block_coord pcb{(int32_t)sg_startx, 0};  // row-major B

for (int i = 0; i < prefetch_distance; ++i) {
    __spirv_Subgroup2DBlockPrefetchINTEL(sizeof(data_type_a),
        /*BlockWidth*/ 32, /*BlockHeight*/ 32, /*BlockCount*/ 2,
        pA.get(), mat_k*sizeof(data_type_a), mat_m, mat_k*sizeof(data_type_a), pca);
    pca.x += sg_k;

    __spirv_Subgroup2DBlockPrefetchINTEL(sizeof(data_type_b),
        /*BlockWidth*/ 32, /*BlockHeight*/ 32, /*BlockCount*/ 2,
        pB.get(), mat_n*sizeof(data_type_b), mat_k, mat_n*sizeof(data_type_b), pcb);
    pcb.y += sg_k;
}

// Inside K-loop: prefetch ahead, then fence, then load + DPAS
for (int i = 0; i < mat_k/sg_k; ++i) {
    __spirv_Subgroup2DBlockPrefetchINTEL(..., pca);  pca.x += sg_k;
    __spirv_Subgroup2DBlockPrefetchINTEL(..., pcb);  pcb.y += sg_k;
    asm volatile("" ::: "memory");   // needs the "memory" clobber to order anything
    // ... loads ...
    asm volatile("" ::: "memory");
    // ... DPAS ...
}
```

**L1 budget check**: `prefetch_distance × (A_tile_bytes + B_tile_bytes) < L1_size`.
For a 32×64 int8 tile with sg_k=64, each iteration prefetches ~2 KB.
With prefetch_distance=2 and 32 subgroups/WG: 2 × 2KB × 32 = 128 KB < 512 KB. ✓

## Cross-row-block prefetch (low-occupancy / decode kernels)

When a kernel processes data in row-blocks (e.g. SDPA decode: 8 K-rows per call)
and has too few threads to hide latency through occupancy, prefetch the **next
row-block's** A data at the very start of the current call. This gives the memory
subsystem a full block's worth of DPAS time to warm the cache.

```cpp
// At the top of the function/loop-body, before the K-loop warmup:
static constexpr int SG_K_COUNT = HEAD_SIZE / DPAS_SG_K;
if (k_row0 + 8 < k_mem_height) {          // guard: next block exists
    block_coord pnext{0, static_cast<int32_t>(k_row0 + 8)};
#pragma unroll
    for (int i = 0; i < SG_K_COUNT; ++i) {
        __spirv_Subgroup2DBlockPrefetchINTEL(sizeof(half), DPAS_K_DIM, 8, 1,
            pA.get(), a_mem_w, k_mem_height, a_mem_pitch, pnext);
        pnext.x += DPAS_SG_K;
    }
}
// ... then do the warmup + K-loop for the CURRENT block as normal
```

This is complementary to within-loop prefetch (distance=2): together they hide
both intra-block and inter-block memory latency.

## Cooperative prefetch (Xe3+)

Each subgroup prefetches a different region so the WG collectively warms L1
with no duplicated cache lines:

```cpp
auto sg    = item.get_sub_group();
int  sg_id = sg.get_group_id();  // 0 .. (wg_m/sg_m * wg_n/sg_n - 1)
// Use sg_id to offset prefetch_coord within the WG tile
int pfa_y = 256 * (sg_starty / 256) + 8 * sg_id;
int pfb_x = 256 * (sg_startx / 256) + 128 * (sg_id % 2);
int pfb_y = 8   * (sg_id / 2);
block_coord pca{0, pfa_y};
block_coord pcb{pfb_x, pfb_y};
```
