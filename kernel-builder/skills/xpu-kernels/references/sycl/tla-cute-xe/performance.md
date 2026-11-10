# Intel Xe CuTe Performance Tuning reference

Companion to `sycl-tla-cute-xe`. Reflects the Xe CuTe performance model in the
SYCL*TLA headers.

## What limits an Xe GEMM
- **Bandwidth vs compute.** Bandwidth-bound → increase K (amortize 2D-load issue
  overhead). Compute-bound → increase M/N (more XMX work per iteration).
- **GRF pressure.** 256 registers/thread. Larger tiles / more pipeline stages
  consume more GRF; exceeding the budget spills to SLM and erases the benefit.
- **Subgroup utilization.** 16-wide subgroups, up to 32 per work-group. Tile and
  subgroup layout decide how busy each subgroup stays.
- **Prefetch depth.** Prefetching future K-blocks hides global-memory latency;
  too many stages waste GRF, too few leave XMX idle.
- **Alignment.** 2D block hardware has strict alignment/size limits; violations
  silently produce wrong results.

## Tuning knobs (in priority order)

### 1. Subgroup size — verify first
- Must be `sub_group_size<16>` in kernel properties.
- `TiledMMAHelper` subgroup layout must be consistent with 16 lanes.
- Mismatch = silent correctness failure (kernel runs, results wrong).

### 2. Tile size — most impactful
Known-good starting points:
| Type | Tile (M×N×K) | Subgroup layout | Note |
|------|--------------|-----------------|------|
| BF16/FP16 | `Shape<_256,_256,_32>` | `Shape<_8,_4,_1>` (stride `_4,_1,_0`) | 32 subgroups |
| TF32 | `Shape<_256,_256,_16>` | `Shape<_8,_4,_1>` | DPAS K=8 → 2 iters |

| Knob | ↑ effect | Watch |
|------|----------|-------|
| M, N | more compute/WG, better XMX util | GRF spill; fewer concurrent WGs |
| K | fewer 2D-load issues/K-loop | GRF spill; K multiple of DPAS K |
| subgroup count | better occupancy | smaller per-SG tile; diminishing past 32 |

### 3. Pipeline stages (prefetch depth)
Default 3 for plain GEMM, 2 for grouped, 1 for FA-decode. Warmup loop fills the
first `Stages` slots before the first compute iteration; guard prefetch with
`if (prefetch_k < k_tile_count)`. Each stage keeps one more K-block's copy
fragments live in GRF — going 2→3 can help latency hiding but may spill.

### 4. Prefetch strategy
```cpp
using Prefetch = XE_LOAD_2D<16,8,16>::PREFETCH;   // → XE_PREFETCH_2D<16,8,16>
auto prefetch_a = make_block_2d_prefetch(copy_a);
prefetch(prefetch_a, pAgA(_,_,_,prefetch_k));     // no destination — L1/L2 hint, no regs
```

### 5. SLM — usually skip
Standard Xe GEMM streams Global→GRF, no SLM. Use SLM only when **all** hold:
(1) register pressure already high, (2) the loop double/multi-buffers SLM so loads
overlap compute, (3) subgroups sharing a buffer have low producer-consumer
dependency. Also needed if a tile exceeds the max 2D-block load size or multiple
subgroups must share a broadcast tile. Reference: `xe_gemm_slm.cpp`.

### 6. Reorder
`reorder(src, dst)` shuffles copy-fragment layout → MMA-fragment layout
(subgroup-scope register op). If layouts match, the compiler elides it entirely;
otherwise it is a necessary shuffle (cheaper than SLM round-trip). Impl:
`<cute/arch/reorder_xe.hpp>`.

## Hardware constraints (compile-time / optional runtime assert)

### 2D block op limits (bytes unless noted)
| Constraint | Value | Applies to |
|------------|-------|------------|
| Base pointer alignment | 64 B | all 2D ops |
| Pitch (stride) alignment | ≥ 16 B preferred (runtime assert requires ≥ 4 B) | all 2D ops |
| Width alignment | 4 B | all 2D ops |
| Max height (loads/prefetch) | 32 rows | LOAD_2D / _VNNI / _TRANSPOSE / PREFETCH |
| **Max height (stores)** | **8 rows** | XE_STORE_2D |
| Max total width | 64 B (`Bits×Width ≤ 512`) | all 2D ops |
| Block count | 1, 2, or 4 (not 3) | LOAD_2D / _VNNI |
| Bits per element | 8/16/32/64 | all 2D ops |
| Max dimension | 2^24 | all 2D ops |

### Compute
| Constraint | Value |
|------------|-------|
| Subgroup size | 16 (always) |
| GRF/thread | 256 registers |
| DPAS M | 1–8 |
| DPAS N | 16 (fixed) |
| DPAS K | `256 / max(bits(A),bits(B))` |

Enable runtime alignment checks (debug builds) by adding the define to your own
compile command:
```bash
icpx ... -DCUTE_ENABLE_XE_BLOCK_2D_ASSERT=1 ...
```

## Tuning workflow
1. Start from the known-good config above (`Shape<_256,_256,_32>`, `Stages=2`).
2. Profile: Intel PTI for GPU (`unitrace --device-timing`) or VTune GPU Hotspots
   → determine bandwidth- vs compute-bound.
3. Adjust tile size accordingly; watch for GRF spill (compiler `-v` / PTI regs).
4. Try `Stages=3`; revert if spilling increases.
5. Verify alignment with `-DCUTE_ENABLE_XE_BLOCK_2D_ASSERT=1`; pad/adjust strides.
6. Re-confirm `sub_group_size<16>`.

## Runtime env for representative numbers
```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=100000000000
export IGC_VISAOptions="-perfmodel"
```
