# SYCL*TLA GEMM — API reference (consumer)

Detail for using the GEMM API from your own application. Pairs with the
`sycl-tla-gemm` SKILL.

## Includes

```cpp
// Core (always):
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"            // explicit mainloop
#include "cutlass/gemm/collective/collective_builder.hpp"        // builder mainloop
#include "cutlass/epilogue/collective/collective_builder.hpp"    // builder epilogue
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include <cute/tensor.hpp>
// Optional helpers (tools/util/include):
#include "cutlass/util/packed_stride.hpp"                        // make_cute_packed_stride
#include "cutlass/util/device_memory.h"                          // cutlass::DeviceAllocation (optional)
using namespace cute;
```

## Data types and accumulators

| Inputs A/B | DPAS atom `XE_DPAS_TT<8,D,A,B>` | Accumulator | K (DPAS) |
|------------|--------------------------------|-------------|----------|
| BF16 `cutlass::bfloat16_t` | `<8, float, bfloat16_t>` | `float` | 16 |
| FP16 `cutlass::half_t`     | `<8, float, half_t>`     | `float` | 16 |
| TF32 `cutlass::tfloat32_t` | `<8, float, tfloat32_t>` | `float` | 8 (use K=16 tile) |
| INT8 `int8_t`/`uint8_t`    | `<8, int32_t, int8_t>`   | `int32_t` | 32 |
| INT4 `cutlass::int4_t`     | `<8, int32_t, int4_t>`   | `int32_t` | 64 |
| FP8 E4M3/E5M2              | specialized | `float` | — |

`K = 256 / max(bits(A), bits(B))`. DPAS N is fixed at 16.

## Arch tags & dispatch policies

- Builder arch tag: `cutlass::arch::IntelXe`; op class `cutlass::arch::OpClassTensorOp`.
- Explicit arch tags: `cutlass::arch::Xe20` (BMG), `cutlass::arch::Xe12` (PVC).
- Mainloop dispatch (new atoms): `cutlass::gemm::MainloopXeL1Staged<Stages>`.
  Legacy: `cutlass::gemm::MainloopIntelXeXMX16`. Grouped variants:
  `MainloopXeL1StagedGroup`, `MainloopIntelXeXMX16Group`.
- Epilogue dispatch: `cutlass::epilogue::IntelXeGeneric`.
- `Stages` (prefetch depth): 3 typical plain GEMM, 2 grouped, 1 FA-decode. With the
  builder use `StageCountAuto`.

## Explicit CollectiveMma + TiledMMAHelper

Use when you must pin the MMA atom, copy atoms, tile shape, or stage count.

```cpp
using TileShape = Shape<_256, _256, _32>;
using TiledMma  = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
    Layout<TileShape>,
    Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>>::TiledMMA;   // 32 subgroups, n-major

constexpr int PipelineStages = 2;
using GEMMDispatch     = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
using EpilogueDispatch = cutlass::epilogue::IntelXeGeneric;

// Leave copy atoms void => mainloop auto-selects XE_LOAD_2D / _VNNI / _TRANSPOSE.
using GmemTiledCopyA = void;   // or XE_LOAD_2D<16,32,32>
using GmemTiledCopyB = void;   // or XE_LOAD_2D_VNNI<16,32,32>

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    float, float, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatch, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatch, TileShape, void,
    float, cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
    float, cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>,
    FusionCallbacks, void, void>;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatch, TileShape,
    cutlass::bfloat16_t, cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>,
    cutlass::bfloat16_t, cutlass::gemm::TagToStrideB_t<cutlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,
    GmemTiledCopyB, void, void, cute::identity>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

Copy-atom selection rules (when overriding `GmemTiledCopy*`):
A row-major → `XE_LOAD_2D`; A col-major → `XE_LOAD_2D_TRANSPOSE`; B row-major
(16/8-bit) → `XE_LOAD_2D_VNNI`; B col-major → `XE_LOAD_2D_TRANSPOSE`; D →
`XE_STORE_2D` (row-major, height ≤ 8). Prefer `void` and let the mainloop pick.

## Arguments and launching with your own queue

```cpp
typename Gemm::GemmKernel::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},                                    // runtime problem shape
    {ptrA, strideA, ptrB, strideB},                  // mainloop args
    {{alpha, beta}, ptrC, strideC, ptrD, strideD},   // epilogue args (varies by fusion op)
    hw_info};

Gemm gemm_op;
size_t ws_size = Gemm::get_workspace_size(args);     // often 0 on Xe
uint8_t* ws = sycl::malloc_device<uint8_t>(std::max<size_t>(ws_size,1), q);
if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) { /* handle */ }
gemm_op.initialize(args, ws, &q);    // &q optional; default = compat in-order queue
gemm_op.run(&q);
q.wait();
```

- `GemmUniversalMode::kGemm` for plain/batched; grouped GEMM uses
  `kGrouped`/`kArray` with array arguments (see grouped examples).
- Strides via `cutlass::make_cute_packed_stride(StrideX{}, make_shape(...))`:
  A `(M,K,L)`, B `(N,K,L)`, C/D `(M,N,L)`.
- `hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);`

## Variants

- **Batched:** set `L>1`; strides already carry `L`. One call computes all batches.
- **Grouped (per-group shapes):** array of problem sizes + array of pointers;
  dispatch `MainloopXeL1StagedGroup`. Use `PipelineStages=2`.
- **Stream-K** (better balance for skinny/odd shapes): use a Stream-K tile
  scheduler in the kernel layer (`cutlass::gemm::StreamKScheduler`).
- **FP8 / mixed-dtype / dequant:** specialized mainloops
  (`MainloopIntelXeXMX16MixedPrecision`, `...FP8Scaling`, `...GroupFP8`); supply
  scale / zero-point tensors. Copy a matching upstream example rather than
  adapting the plain builder.

## Optional: verifying results

You can verify on-device with the util reference GEMM (needs oneMKL at build):

```cpp
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
// cutlass::reference::device::GemmComplex({M,N,K}, alpha, refA, kNone, refB, kNone,
//     beta, refC, refD, ElementAcc(0), L, M*K, K*N, M*N, M*N);
// bool ok = cutlass::reference::device::BlockCompareEqual(refD, D, M*N*L);
```

Or just check a known case on the host (e.g. all-ones A,B ⇒ each D element = K),
as in `assets/minimal_gemm.cpp` — no extra dependencies.

## Memory: USM vs DeviceAllocation

- **Plain SYCL (recommended for your app):** `sycl::malloc_device<T>(n, q)` +
  `sycl::free`. Fully under your control, integrates with the rest of your code.
- **Convenience helper:** `cutlass::DeviceAllocation<T>` (from
  `cutlass/util/device_memory.h`) manages a device buffer with RAII and works with
  the default queue. Handy for quick tests; not required.
