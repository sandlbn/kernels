# Hand-written CuTe kernels on Intel Xe (header library)

Write hand-tuned CuTe device kernels for Intel Xe GPUs using SYCL*TLA (CUTLASS-SYCL)
as a header library, at the level of atoms, tensors, and the K-loop — selecting MMA
atoms (`XE_DPAS_TT`), 2D block copy/store/prefetch atoms, building a `TiledMMA` via
`TiledMMAHelper`, writing the prefetch+load+reorder+gemm mainloop, and tuning tile
size, pipeline depth, GRF pressure, and 2D-block alignment. For ready-made GEMM via
the CUTLASS collective API instead, see [../tla-gemm/README.md](../tla-gemm/README.md).

Requires the sycl-tla headers, the Intel oneAPI DPC++ compiler (`icpx`, 2025.1+),
and an Intel Xe GPU (Arc/BMG or PVC). Linux only.

Use SYCL*TLA's CuTe layer (`<cute/...>`) to write a custom GEMM-style device
kernel directly from atoms and primitives, launched from your own `sycl::queue`.
Prefer the ready-made collective GEMM ([../tla-gemm/README.md](../tla-gemm/README.md)) unless you need
atom-level control (custom fusions, unusual tiling, research kernels).

## Build setup (same headers, same flags as any sycl-tla consumer)

```bash
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -std=c++17 -fsycl-targets=spir64 \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -I$SYCL_TLA/include -I$SYCL_TLA/tools/util/include \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier \
  my_kernel.cpp -o my_kernel
```

`-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier` is **required** — the Xe
mainloop uses split barriers. AOT: replace `spir64` with `intel_gpu_bmg_g21`
(BMG) or `intel_gpu_pvc` (PVC). Full build/CMake details are in the
[../tla-gemm/integration.md](../tla-gemm/integration.md).

## Mental model (CUDA → Intel)

| CUDA | Xe / SYCL*TLA | Note |
|------|---------------|------|
| warp (32) | sub_group (**16** lanes) | fixed 16-wide; never change |
| threadblock | work-group | one output tile |
| shared mem | SLM (`sycl::local_accessor`) | usually skipped on Xe |
| registers | GRF (**256** regs/thread) | spills to SLM if exceeded |
| Tensor Core | XMX | runs DPAS |
| `mma.sync` | `XE_DPAS_TT` atom | one DPAS instruction |
| `cp.async`/TMA | 2D block load (`XE_LOAD_2D*`) | global → GRF |
| Sm80/Sm90 tag | `cutlass::arch::Xe12` (PVC) / `Xe20` (BMG) | |

Data path (no SLM): `Global ─prefetch→ L1 ─2D block load→ GRF ─reorder→ MMA frag
─DPAS→ accumulator ─2D block store→ Global`.

## Building blocks (headers + signatures)

**MMA atom** — `<cute/arch/mma_xe.hpp>`
```cpp
template <int M, class TypeD, class TypeA, class TypeB = TypeA, class TypeC = TypeD>
struct XE_DPAS_TT;     // M∈1..8, N=16 fixed, K = 256/max(bits(A),bits(B))
using Atom = cute::MMA_Atom<cute::XE_DPAS_TT<8, float, cute::bfloat16_t>>;
```
Type aliases in `cute::dpas_type::` (`f`,`tf32`,`bf`,`hf`,`u8`,`s8`,`u4`,`s4`,`d`,
`ud`). Full supported type-combination table: `atoms.md`.

**2D copy atoms** — `<cute/arch/copy_xe_2d.hpp>`
```cpp
template <int Bits,int H,int W,int BW=W> struct XE_LOAD_2D;          // row-major load (A)
template <int Bits,int H,int W,int BW=W> struct XE_LOAD_2D_VNNI;     // VNNI load (B), Bits∈{8,16}
template <int Bits,int H,int W> struct XE_LOAD_2D_TRANSPOSE;         // transposed load, Bits∈{32,64}
template <int Bits,int H,int W> struct XE_PREFETCH_2D;               // prefetch (no regs)
template <int Bits,int H,int W> struct XE_STORE_2D;                  // store (D), H ≤ 8
// each load exposes ::PREFETCH → XE_PREFETCH_2D<...>
```
`Bits` = element bit-width (BF16/FP16→16, FP32→32, INT8→8). Prefer the factory
functions below; they auto-select the right atom.

**TiledMMA** — `<cute/atom/mma_atom.hpp>`
```cpp
using WGTile   = cute::Shape<cute::_256, cute::_256, cute::_32>;          // M,N,K
using SGLayout = cute::Layout<cute::Shape<cute::_8, cute::_4, cute::_1>,  // 8×4 subgroups, n-major
                              cute::Stride<cute::_4, cute::_1, cute::_0>>;
using MMA = typename cute::TiledMMAHelper<
    cute::MMA_Atom<cute::XE_DPAS_TT<8, float, cute::bfloat16_t>>,
    cute::Layout<WGTile>, SGLayout>::TiledMMA;
```

## Kernel skeleton (the proven structure)

```cpp
// 1. Auto-build copy + prefetch atoms from the global tensors
auto copy_a = make_block_2d_copy_A(mma, A);     // selects XE_LOAD_2D / _TRANSPOSE
auto copy_b = make_block_2d_copy_B(mma, B);     // selects XE_LOAD_2D_VNNI
auto copy_c = make_block_2d_copy_D(mma, C);     // selects XE_STORE_2D
auto prefetch_a = make_block_2d_prefetch(copy_a);
auto prefetch_b = make_block_2d_prefetch(copy_b);

// 2. Tile global → work-group, then slice to this work-item
auto wg_tile = mma.tile_mnk();
Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m, _)); // (BLK_M,BLK_K,k)
Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n, _)); // (BLK_N,BLK_K,k)
auto thr_mma    = mma.get_slice(local_id);
auto thr_copy_a = copy_a.get_slice(local_id);

// 3. Allocate fragments — copy layout and MMA layout differ
auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));     // MMA layout
auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));
auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));  // copy layout
auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));
Tensor tCrC = partition_fragment_C(mma, select<0,1>(wg_tile));

// 4. Prefetch warmup (guarded for short K), then zero accumulators
clear(tCrC);
CUTE_UNROLL
for (; k_prefetch < Stages && k_prefetch < k_tile_count; ++k_prefetch) {
  prefetch(prefetch_a, pAgA(_,_,_,k_prefetch));
  prefetch(prefetch_b, pBgB(_,_,_,k_prefetch));
}

// 5. Main K-loop: barrier → load → prefetch → reorder → DPAS → barrier
for (int k = 0; k < k_tile_count; ++k, ++k_prefetch) {
  barrier_arrive(barrier_scope);
  copy(copy_a, tAgA(_,_,_,k), tArA);
  copy(copy_b, tBgB(_,_,_,k), tBrB);
  if (k_prefetch < k_tile_count) {
    prefetch(prefetch_a, pAgA(_,_,_,k_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_prefetch));
  }
  reorder(tArA, tCrA);          // copy layout → MMA layout (elided if layouts match)
  reorder(tBrB, tCrB);
  gemm(mma, tCrA, tCrB, tCrC);  // XMX DPAS accumulate
  barrier_wait(barrier_scope);
}

// 6. Store
copy(copy_c, tCrC, tCgC);       // GRF → global via XE_STORE_2D
```

`barrier_scope = 2` covers the work-group. Selection logic:
`<cute/atom/copy_traits_xe_2d.hpp>`. Split barrier:
`<cute/util/xe_split_barrier.hpp>`. Reorder: `<cute/arch/reorder_xe.hpp>`.

## Launch from your own queue (required Xe properties)

```cpp
namespace syclex  = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

sycl::queue q{sycl::gpu_selector_v};
sycl::range<2> local  = {size(mma), 1};        // e.g. 32 subgroups × 16 = 512
sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                         local[1] * ceil_div(shape<0>(A), get<0>(mma.tile_mnk()))};

syclex::properties kernel_props {
    syclex::sub_group_size<16>,    // REQUIRED — Xe DPAS is always 16-wide
    intelex::grf_size<256>         // large-GRF mode (16 KB per thread)
};

q.parallel_for<class MyKernel>(sycl::nd_range<2>(global, local), kernel_props,
    [=](auto) { gemm_device(A, B, C, mma); });
q.wait();
```

A, B, C here are CuTe `Tensor`s wrapping your USM device pointers
(`sycl::malloc_device`). In pure-CuTe GEMM, **B is represented/tiled as `(N, K)`** —
construct or transpose B accordingly (otherwise indexing is wrong).

## Best-known methods & tuning

- **`sub_group_size<16>` is mandatory.** A mismatch produces silently wrong
  results with no error — verify this first when debugging correctness.
- **Start from the known-good config:** `Shape<_256,_256,_32>` + `<_8,_4,_1>`
  subgroup layout + `Stages=2/3` for BF16. Tune from there.
- **`reorder` is free when copy and MMA layouts match** (compiler elides it);
  otherwise it's a register shuffle (still cheaper than an SLM round-trip).
- **Tile size:** ↑M/N for more XMX work (compute-bound); ↑K to amortize 2D-load
  issue overhead (bandwidth-bound). Watch GRF spill either way.
- **Pipeline stages:** more stages hide latency but keep more copy fragments live
  in GRF. Try 3; revert to 2 if spilling.
- **Skip SLM** unless register pressure is high AND you double-buffer AND
  subgroups share tiles with low dependency — otherwise barriers negate it.
- **2D block alignment is strict:** base ptr 64 B, pitch ≥16 B preferred (runtime
  assert requires ≥4 B), width 4 B; max load height 32, max store height 8, total
  width ≤ 64 B (`Bits×Width ≤ 512`), block count ∈ {1,2,4}. Violations silently
  corrupt. Debug with `-DCUTE_ENABLE_XE_BLOCK_2D_ASSERT=1`. `sycl::malloc_device`
  satisfies base-pointer alignment for contiguous matrices.
- **Profile** with Intel PTI for GPU (`unitrace --device-timing`) or VTune to
  decide bandwidth- vs compute-bound before changing knobs.

## References (load on demand)

- `atoms.md` — every atom's params, constraints, full type-combination
  matrix, copy-variant selection (LD_N/LD_V/LD_T/ST_N), VNNI/packed data layout.
- `performance.md` — tuning workflow, GRF/stage/tile tradeoffs,
  alignment & hardware limits, profiling, performance env vars.
