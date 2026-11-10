# Using SYCL*TLA as a header library in your own app

Use SYCL*TLA (CUTLASS-SYCL) as a header-only C++ library to add high-performance
GEMM and fused-epilogue kernels to a SYCL application for Intel Xe GPUs — composing
a GEMM type from data types + tile shape + mainloop + epilogue, then launching it
from your own `sycl::queue` on USM device pointers. For hand-written CuTe device
kernels and atom-level tuning instead, see [../tla-cute-xe/README.md](../tla-cute-xe/README.md).

Requires the sycl-tla headers, the Intel oneAPI DPC++ compiler (`icpx`, 2025.1+),
and an Intel Xe GPU (Arc/BMG or PVC) at runtime. Linux only.

SYCL*TLA is a header-only CUTLASS fork that provides GEMM and fused-epilogue
kernels for Intel Xe GPUs. You **compose a GEMM type** from data types + a tile
shape + a mainloop + an epilogue, then **launch it from your own `sycl::queue`**
on USM device pointers. Nothing in your project needs to live inside the sycl-tla
repo — you just add its `include/` directories and a few compiler flags.

`assets/minimal_gemm.cpp` + `assets/CMakeLists.txt` are a complete, runnable
starting point (validated end-to-end on Intel Arc). Copy them into your project.

## 1. Integrate into your build (do this first)

You only need headers. Required pieces (validated, oneAPI 2025.2):

- **Compiler:** `icpx` (Intel oneAPI DPC++), C++17.
- **Include dirs:** `<sycl-tla>/include` (core) and `<sycl-tla>/tools/util/include`
  (optional helpers: `make_cute_packed_stride`, reference kernels, `DeviceAllocation`).
- **Defines:** `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET`.
- **SYCL flags:** `-fsycl -fsycl-targets=<target>` where `<target>` is
  `spir64` (JIT, simplest), `intel_gpu_bmg_g21` (BMG AOT), or `intel_gpu_pvc` (PVC AOT).
- **Link flag (REQUIRED):** `-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier`.
  Without it you get `RequiresExtension: ... SPV_INTEL_split_barrier` at link time —
  the Xe mainloop uses split barriers. Needed even on 2025.2.

Quick one-liner build:

```bash
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -std=c++17 -fsycl-targets=spir64 \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -I$SYCL_TLA/include -I$SYCL_TLA/tools/util/include \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier \
  my_gemm.cpp -o my_gemm
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./my_gemm
```

CMake (FetchContent or a checkout path) and full flag rationale, AOT device
flags, and performance env vars are in **integration.md**.

## 2. Compose the GEMM type (CollectiveBuilder — recommended)

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"   // make_cute_packed_stride (tools/util/include)
#include <cute/tensor.hpp>
using namespace cute;

using ElementA = cutlass::bfloat16_t, ElementB = cutlass::bfloat16_t;
using ElementC = float, ElementAcc = float, ElementComp = float;
using LayoutA = cutlass::layout::RowMajor;   // keep D row-major for Xe epilogues
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignA = sizeof(ElementA), AlignB = sizeof(ElementB);
constexpr int AlignC = sizeof(ElementAcc), AlignD = sizeof(ElementC);

using TileShape = Shape<_256, _256, _32>;    // proven BF16 starting point

using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA,  ElementB, LayoutB, AlignB,  ElementAcc,
    TileShape, Shape<_1,_1,_1>,                 // ClusterShape always 1,1,1 on Xe
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<  // D = alpha*acc + beta*C
    ElementC, ElementComp, ElementAcc, ElementAcc,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    TileShape, Shape<_1,_1,_1>,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementComp, ElementAcc,
    ElementAcc, LayoutC, AlignC,  ElementC, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto, EpilogueOp>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

To pin atoms/tiles/stages yourself, use the explicit `CollectiveMma` +
`TiledMMAHelper` form in **api_reference.md**.

## 3. Launch from your own queue and USM pointers

```cpp
sycl::queue q{sycl::gpu_selector_v};
int M=..., N=..., K=..., L=1;                 // L = batch count
auto* A = sycl::malloc_device<ElementA>(std::size_t(M)*K*L, q);  // fill via q.copy/q.memcpy
auto* B = sycl::malloc_device<ElementB>(std::size_t(N)*K*L, q);
auto* C = sycl::malloc_device<ElementC>(std::size_t(M)*N*L, q);
auto* D = sycl::malloc_device<ElementC>(std::size_t(M)*N*L, q);

using StrideA = typename Gemm::GemmKernel::StrideA; /* B,C,D analogous */
auto sA = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
auto sB = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L)); // note (N,K,L)
auto sC = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, L));
auto sD = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, L));

cutlass::KernelHardwareInfo hw_info;
hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

typename Gemm::GemmKernel::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, L},
    {A, sA, B, sB},
    {{/*alpha*/1.0f, /*beta*/0.0f}, C, sC, D, sD},
    hw_info};

Gemm gemm_op;
auto* ws = sycl::malloc_device<uint8_t>(std::max<size_t>(Gemm::get_workspace_size(args),1), q);
if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) { /* unsupported shape */ }
gemm_op.initialize(args, ws, &q);   // pass &q to use YOUR queue
gemm_op.run(&q);                    // omit &q to use the default in-order queue
q.wait();
```

Key points:
- **`initialize(args, workspace, &q)` and `run(&q)`** take your `sycl::queue*`.
  Omit the pointer to use SYCL*TLA's default in-order `compat` queue.
- **Workspace** is device USM; most Xe GEMMs report size 0 (allocate ≥1 byte).
- **Always call `can_implement(args)`** before initialize/run.
- **Strides carry batch (`L`)**; B uses `make_shape(N,K,L)`.

## 4. Data types & tiles (cheat sheet)

| Inputs A/B | MMA element types | Accumulator | Tile (start) |
|------------|-------------------|-------------|--------------|
| BF16 / FP16 | `bfloat16_t` / `half_t` | `float` | `Shape<_256,_256,_32>` |
| TF32 | `tfloat32_t` | `float` | `Shape<_256,_256,_16>` |
| INT8 | `int8_t`/`uint8_t` | `int32_t` | `Shape<_256,_256,_32>` |
| FP8 (E4M3/E5M2), mixed/quantized | specialized | `float`/`int32_t` | use specialized recipe |

FP8 and mixed-dtype (dequant scale/zero-point) use specialized mainloops — see
**api_reference.md**; don't assume the plain BF16 builder generalizes.

## 5. Epilogue fusion

Swap `EpilogueOp` for a fused op (activation, bias, aux store, softmax, split-K).
Example — ReLU on top of the linear combination:

```cpp
using EpilogueOp = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::ReLu, ElementC, ElementComp, ElementAcc, ElementAcc,
    cutlass::FloatRoundStyle::round_to_nearest>;
```

Full list of fusion ops + their runtime argument fields: **epilogues.md**.

## Best-known methods / pitfalls

- **Add the `SPV_INTEL_split_barrier` link flag** — the #1 first-build failure.
- **`-fsycl-targets` must match your GPU for AOT** (`intel_gpu_bmg_g21` vs
  `intel_gpu_pvc`); use `spir64` for portable JIT.
- **Keep output D row-major / N-major** — Xe epilogues assert on this; `ColumnMajor`
  D can fail to compile.
- **Subgroup size is always 16** on Xe and is baked into the kernel; don't override
  it. A mismatch would silently produce wrong results.
- **USM must be device-accessible**: `sycl::malloc_device` (or shared). The base
  pointers should be 64-byte aligned and row pitch ≥16-byte aligned for the 2D
  block hardware — `malloc_device` satisfies alignment for contiguous matrices.
- **Use `make_cute_packed_stride`** to build strides; don't hand-roll them.
- **Set performance env vars** (`-ze-opt-large-register-file`, 256-GRF) for real
  benchmarking — see integration.md.

## References (load on demand)

- `integration.md` — CMake + FetchContent, icpx flags, AOT targets,
  required defines, runtime/perf env vars, troubleshooting.
- `api_reference.md` — explicit `CollectiveMma`/`TiledMMAHelper`,
  dispatch policies, batched/grouped/Stream-K/FP8/mixed-dtype, Arguments layout,
  optional verification with util reference kernels.
- `epilogues.md` — all fusion ops with usage + runtime args.
- `assets/minimal_gemm.cpp`, `assets/CMakeLists.txt` — runnable starting point.
