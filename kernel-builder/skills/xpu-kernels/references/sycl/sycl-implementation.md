# SYCL and SYCL-TLA Implementation Reference (Intel XPU)

Templates and conventions for writing SYCL kernels — plain SYCL or the
[SYCL-TLA](https://github.com/intel/sycl-tla) (CUTLASS-for-SYCL, CuTe) library — for Intel
Xe GPUs, in two forms:

1. **Trial form** — a self-contained Python file that JIT-compiles SYCL with
   `torch.utils.cpp_extension.load_inline` and exposes a KernelBench-style `Model`. This is
   what `scripts/validate_sycl.py`, `scripts/benchmark.py` and `scripts/trial_manager.py`
   consume during the optimization loop.
2. **Shipping form** — a kernel-builder project (`language = "cpp-xpu"` in `build.toml`,
   `TORCH_LIBRARY_EXPAND` bindings). This is what gets built, tested and uploaded to the Hub.

The device code is the same in both forms; only the host wrapper differs. Port the winning
trial to the shipping form at the end (see [Porting a trial to kernel-builder](#porting-a-trial-to-kernel-builder)).

Companion knowledge base:

| File | Contents |
|------|----------|
| [sycl_tla_patterns.yaml](sycl_tla_patterns.yaml) | Constraints and patterns for SYCL and SYCL-TLA: subgroup/MMA matching, SLM budget, reductions, online softmax, flash-attention recurrence, epilogue fusion, quantization |
| [cutlass_sycl_framework.yaml](cutlass_sycl_framework.yaml) | SYCL-TLA tile configurations, dispatch policies, tile schedulers, epilogue fusion catalog, fused-kernel patterns (dual GEMM, FA2, MoE grouped GEMM) |
| [resource_feedback.yaml](resource_feedback.yaml) | How to read unitrace kernel properties (SIMD, SLM, private/spill memory, register file) |
| [compute_basic.yaml](compute_basic.yaml) | unitrace ComputeBasic / EU-stall collection and interpretation |
| [../common/correctness.yaml](../common/correctness.yaml) | DSL-independent correctness rules (shared with Triton) |
| [../common/algorithmic_patterns.yaml](../common/algorithmic_patterns.yaml) | DSL-independent algorithmic rewrites |

---

## Choosing between plain SYCL and SYCL-TLA

| Workload | Start with |
|----------|------------|
| Elementwise, activation, normalization, reductions, softmax, RoPE, gather/scatter | Plain SYCL (`nd_range` + subgroup/work-group collectives) |
| GEMM, batched/grouped GEMM, GEMM + epilogue, MoE GEMM, quantized (W4A16/W8A16) GEMM | SYCL-TLA collectives (`CollectiveMma` + Xe epilogue) |
| Attention (prefill/decode, paged, chunked) | SYCL-TLA flash-attention collectives, or Triton if the shape set is broad |
| Anything where XMX/DPAS is needed but SYCL-TLA has no fitting collective | `sycl::ext::oneapi::experimental::matrix::joint_matrix` (verify the supported combination first — `sycl_joint_matrix_gemm`) |

Do not hand-roll a DPAS GEMM in plain SYCL when a SYCL-TLA collective fits; the library
encodes the legal atom/copy/layout combinations (`sycl_dpas_tile_shapes_per_generation`).

---

## Hard rules

- **Use PyTorch's queue.** Submit to `c10::xpu::getCurrentXPUStream().queue()`. Never create
  a new `sycl::queue` per call and never `.wait()` inside the op — both break stream
  ordering with surrounding PyTorch ops and serialize the host.
- **Unique kernel names per implementation.** Baseline and trial extensions live in the same
  process. Put every named kernel type (and file-scope helpers) in a namespace unique to
  the trial (`namespace rmsnorm_t3 { ... }`) and change it for every new trial. A unique
  `load_inline` module name alone does **not** isolate device kernels.
- **Pin the AOT target from the queried device.** Run `python scripts/xpu_device_info.py`
  and set `TORCH_XPU_ARCH_LIST` to the `torch.xpu.get_arch_list()` entry that matches the
  present device, before `load_inline`. The default compiles for every architecture the
  PyTorch build knows, which is slow and can fail. If no entry can be matched with
  certainty, leave it unset (JIT) and record that in the trial strategy.
- **Fix the subgroup size explicitly** with `[[sycl::reqd_sub_group_size(N)]]` whenever the
  code relies on it (subgroup collectives, shuffles, MMA atoms). Choose N from what the
  selected matrix atom / API requires (`sycl_dpas_requires_simd16`), and check that N is in
  the device's `sub_group_sizes` (`scripts/xpu_device_info.py`).
- **Accumulate in fp32** for fp16/bf16 inputs unless the reference contract says otherwise
  (`sycl_fp32_accumulator_required`).
- **Check inputs on the host:** device (`is_xpu()`), dtype, contiguity/strides, shapes —
  with `TORCH_CHECK`, before launch.
- **Respect reduction scope:** `reduce_over_group(sub_group, ...)` only combines one
  subgroup. A row owned by a work-group needs a work-group collective
  (`sycl_use_subgroup_reduction_not_slm`).
- **Budget SLM:** `stages * (TILE_M*TILE_K*sizeof(A) + TILE_K*TILE_N*sizeof(B))` plus padding
  and any other `local_accessor`, against the device's per-work-group limit
  (`sycl_slm_size_limits`).
- **Shipping form only:** no pybind11, no `setup.py`, no `load_inline`. Bind with
  `TORCH_LIBRARY_EXPAND` and let kernel-builder build the extension.

---

## Trial form: `load_inline` wrapper

Each trial file is self-contained: copy `get_inputs()`, `get_init_inputs()` and dimension
constants from the PyTorch reference into it (trials run in isolation, without access to
sibling files). `load_inline(..., sycl_sources=...)` requires a PyTorch build with XPU
support (2.8+) and the oneAPI DPC++ compiler (`icpx`) on `PATH`.

```python
"""RMSNorm in SYCL — trial t0."""

import os

# AOT target: the torch.xpu.get_arch_list() entry matching this device, as reported by
# `python scripts/xpu_device_info.py`. Replace the placeholder before running.
ARCH = "<arch from scripts/xpu_device_info.py>"
os.environ.setdefault("TORCH_XPU_ARCH_LIST", ARCH)

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CPP = r"""
#include <torch/extension.h>
torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps);
"""

_SYCL = r"""
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace rmsnorm_t0 {  // unique per trial: change for every new implementation

template <typename T> class RmsNormKernel;  // kernel name type

template <typename T, int WG, int SG>
void launch(T* out, const T* x, const T* w, int64_t rows, int64_t cols, float eps,
            sycl::queue& q) {
  q.parallel_for<RmsNormKernel<T>>(
      sycl::nd_range<1>(sycl::range<1>(rows * WG), sycl::range<1>(WG)),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int64_t row = it.get_group(0);
        const int lid = static_cast<int>(it.get_local_id(0));
        const T* xr = x + row * cols;
        T* orow = out + row * cols;

        float acc = 0.f;                               // fp32 accumulation
        for (int64_t c = lid; c < cols; c += WG) {
          const float v = static_cast<float>(xr[c]);
          acc += v * v;
        }
        // The whole work-group owns the row -> work-group collective, not sub-group.
        const float total = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        const float inv = sycl::rsqrt(total / static_cast<float>(cols) + eps);

        for (int64_t c = lid; c < cols; c += WG)
          orow[c] = static_cast<T>(static_cast<float>(xr[c]) * inv * static_cast<float>(w[c]));
      });
}

}  // namespace rmsnorm_t0

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_xpu() && weight.is_xpu(), "inputs must be XPU tensors");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "dtype mismatch");
  const int64_t cols = x.size(-1);
  TORCH_CHECK(weight.numel() == cols, "weight must match the last dim");
  const int64_t rows = x.numel() / cols;

  auto out = torch::empty_like(x);
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();  // PyTorch's queue
  // SG must be one of the device's sub_group_sizes (scripts/xpu_device_info.py);
  // WG must be a multiple of SG and <= max_work_group_size. Tune both.
  constexpr int WG = 256, SG = 16;

  switch (x.scalar_type()) {
    case at::kBFloat16: {
      using T = sycl::ext::oneapi::bfloat16;
      rmsnorm_t0::launch<T, WG, SG>(reinterpret_cast<T*>(out.data_ptr()),
                                    reinterpret_cast<const T*>(x.data_ptr()),
                                    reinterpret_cast<const T*>(weight.data_ptr()),
                                    rows, cols, static_cast<float>(eps), q);
      break;
    }
    case at::kHalf: {
      using T = sycl::half;
      rmsnorm_t0::launch<T, WG, SG>(reinterpret_cast<T*>(out.data_ptr()),
                                    reinterpret_cast<const T*>(x.data_ptr()),
                                    reinterpret_cast<const T*>(weight.data_ptr()),
                                    rows, cols, static_cast<float>(eps), q);
      break;
    }
    case at::kFloat:
      rmsnorm_t0::launch<float, WG, SG>(out.data_ptr<float>(), x.data_ptr<float>(),
                                        weight.data_ptr<float>(), rows, cols,
                                        static_cast<float>(eps), q);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype ", x.scalar_type());
  }
  return out;  // no q.wait(): stream ordering is preserved by PyTorch's queue
}
"""

_mod = load_inline(
    name="rmsnorm_t0_ext",           # unique per trial
    cpp_sources=_CPP,
    sycl_sources=_SYCL,
    functions=["rmsnorm_forward"],
    extra_sycl_cflags=["-O3"],
    verbose=False,
)

HIDDEN = 4096
EPS = 1e-6


class Model(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN, eps: float = EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return _mod.rmsnorm_forward(x, self.weight.to(x.dtype), self.eps)


def get_inputs():
    return [torch.randn(16, 2048, HIDDEN, dtype=torch.bfloat16)]


def get_init_inputs():
    return [HIDDEN, EPS]
```

Notes:

- The `.to(x.dtype)` in `forward` is a per-call conversion; in a tuned trial cache the
  converted weight once (see `mem_wrapper_one_time_device_moves` in
  [../memory_patterns.yaml](../memory_patterns.yaml)).
- For small `cols`, one subgroup per row (`reduce_over_group(it.get_sub_group(), ...)`, with
  `WG == SG` or several rows per work-group) avoids the work-group barrier — but only if the
  subgroup owns the entire row.
- Vectorize loads (`sycl::vec<T, N>` / `sycl::marray`) only after checking alignment, stride
  and tail handling (`sycl_vectorized_loads`).
- Header-only libraries (SYCL-TLA) are added with `extra_include_paths=[...]` **and** an
  `-I` in `extra_sycl_cflags`: the host pass sees only `extra_include_paths`/`extra_cflags`,
  the device pass only `extra_sycl_cflags`.

---

## SYCL-TLA (CUTLASS for SYCL) GEMM skeleton

SYCL-TLA kernels are assembled from CuTe types. The full, buildable example in this
repository is
[`examples/kernels/cutlass-gemm/gemm_sycl.cpp`](https://github.com/huggingface/kernels/blob/main/examples/kernels/cutlass-gemm/gemm_sycl.cpp)
(BF16 x BF16 -> FP32 GEMM with `depends = ["torch", "sycl_tla"]`). The type stack:

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include <cute/tensor.hpp>
using namespace cute;

// 1. Work-group tile (M, N, K) and subgroup layout over a DPAS MMA atom.
using TileShape = Shape<_256, _256, _32>;
using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,          // fp32 += bf16 * bf16
    Layout<TileShape>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;   // 8x4 = 32 subgroups

// 2. Global->register 2D block copies (N = normal, V = VNNI, T = transposed).
using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

// 3. Mainloop dispatch policy (pipeline stages).
constexpr int PipelineStages = 2;
using GEMMDispatchPolicy     = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

// 4. Epilogue fusion: D = alpha * acc + beta * C (swap for bias/activation callbacks).
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    float /*ElementOutput*/, float /*Compute*/, float /*Accum*/, float /*Source*/,
    cutlass::FloatRoundStyle::round_to_nearest>;
using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

// 5. Collectives -> kernel -> device adapter.
//    CollectiveMma<GEMMDispatchPolicy, TileShape, ElementA, StrideA, ElementB, StrideB,
//                  TiledMma, GmemTiledCopyA, ..., GmemTiledCopyB, ...>
//    GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>
//    GemmUniversalAdapter<GemmKernel>
```

Tuning knobs, in the order to try them (details and measured examples in
[cutlass_sycl_framework.yaml](cutlass_sycl_framework.yaml)):

1. **TileShape + subgroup layout** — start from `cutlass_gemm_tile_example` and search around it for
   the actual shapes (e.g. smaller M tiles for skinny / decode shapes); INT4 dequant needs much
   smaller tiles (`cutlass_mixed_dtype_tiles`). Every work-group tile must be a multiple
   of the MMA atom and the subgroup layout must divide it.
2. **Dispatch policy** — `MainloopIntelXeXMX16<Stages>` (general) vs `MainloopXeL1Staged`
   (`cutlass_mainloop_xe_l1_staged`); grouped GEMM uses `MainloopXeL1StagedGroup`.
3. **Pipeline stages / prefetch distance** — trade overlap against registers and SLM.
4. **Tile scheduler** — `PersistentScheduler` (default), `StreamKScheduler` for irregular
   shapes / few tiles, `GroupScheduler` for grouped/MoE GEMM.
5. **Epilogue fusion** — bias / activation / aux output / softmax / top-k / split-K callbacks
   from the epilogue catalog; check every template argument against the installed
   revision (`cutlass_epilogue_*` patterns in [sycl_tla_patterns.yaml](sycl_tla_patterns.yaml)).

Integration notes:

- Always check `gemm_op.can_implement(arguments)` and fail with `TORCH_CHECK` instead of
  launching an unsupported configuration.
- Allocate the workspace through PyTorch (`torch::empty({ws}, x.options().dtype(at::kByte))`)
  rather than a library allocator in the hot path, and pass its pointer to `initialize`.
- Make sure the GEMM runs on PyTorch's current XPU queue; check how the installed SYCL-TLA
  revision's adapter selects its queue, and avoid trailing `wait()` calls in the op.
- SYCL-TLA versions follow the DPC++/oneAPI version that the Torch variant is built with
  (kernel-builder selects a matching `sycl_tla`; see `nix-builder/pkgs/xpu-packages/sycl-tla.nix`
  in the kernels repo). Do not hard-code types that only exist in a newer revision.

---

## Porting a trial to kernel-builder

Scaffold an XPU project and move the device code from the winning trial into it:

```bash
kernel-builder init --name my-org/my-kernel --backends xpu
```

```toml
# build.toml
[general]
name = "my-kernel"          # dash-separated
version = 1
license = "Apache-2.0"
backends = ["xpu"]          # add "cuda", "rocm", ... if other backends are provided

[general.hub]
repo-id = "my-org/my-kernel"

[torch]
src = [
    "torch-ext/torch_binding.cpp",
    "torch-ext/torch_binding.h",
]

[kernel.rmsnorm_xpu]
language = "cpp-xpu"
depends = ["torch"]         # add "sycl_tla" for SYCL-TLA/CuTe kernels
src = ["rmsnorm_xpu/rmsnorm.cpp"]
# sycl-flags = ["..."]     # only if really needed
```

```cpp
// torch-ext/torch_binding.cpp
#include <torch/library.h>
#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("rmsnorm(Tensor! out, Tensor x, Tensor weight, float eps) -> ()");
  ops.impl("rmsnorm", torch::kXPU, &rmsnorm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```

Porting checklist:

- [ ] Device code copied verbatim from the winning trial; only the host wrapper changes.
- [ ] Op signature follows kernel-builder conventions (`Tensor! out` first, returns `()`),
      with a thin Python wrapper in `torch-ext/<name>/__init__.py` that allocates `out`.
- [ ] Trial-specific namespace renamed to a stable one.
- [ ] No pybind11, `load_inline` or `TORCH_XPU_ARCH_LIST` handling left in shipped code.
- [ ] Tests in `tests/` compare against the PyTorch reference on `xpu` for every supported
      dtype and shape class.
- [ ] Build with `nix run .#build-and-copy -L` (or `kernel-builder build`), then
      `nix run .#ci-test`. Mixed-backend kernels can be guarded with `#if defined(XPU_KERNEL)`
      in the binding (see `examples/kernels/cutlass-gemm/torch-ext/torch_binding.cpp`).

---

## Profiling SYCL kernels

Prefer **unitrace** (Intel PTI) and fall back to VTune (`scripts/xpu_profiler.py`). Run
profiles one at a time — there is a single XPU.

```bash
# Discover what this GPU/driver supports
unitrace --metric-list

# Device timing + ComputeBasic hardware counters, driving the trial through the benchmark
# harness (--baseline-us skips re-timing the baseline)
ZE_FLAT_DEVICE_HIERARCHY=FLAT unitrace --device-timing --metric-sampling --group ComputeBasic \
    -o profile_t3/trace \
    python scripts/benchmark.py <pytorch_file> t3.py --baseline-us <cached>

# Separate run only to test a stall hypothesis
ZE_FLAT_DEVICE_HIERARCHY=FLAT unitrace --device-timing --stall-sampling \
    -o profile_t3_stall/trace \
    python scripts/benchmark.py <pytorch_file> t3.py --baseline-us <cached>
```

Timings printed by an instrumented run are **not** results; record only the numbers from an
uninstrumented `scripts/benchmark.py` run. Select the trial's kernels in the trace by their
exact (namespaced) kernel names.

Interpretation rules (from [resource_feedback.yaml](resource_feedback.yaml) and
[compute_basic.yaml](compute_basic.yaml)):

- Kernel properties (SIMD, SLM per work-group, private/spill memory per thread, register
  file size) are **driver-reported allocations**, not traffic or time.
- Match a property record to the timed kernel by its **exact** name; keep unknown fields
  unknown.
- Report DRAM bytes and GB/s together; a higher GB/s at the same bytes means faster
  delivery, not less traffic.
- High `SbidStall` points at dependency waits; it does not by itself prove DRAM saturation.
  High occupancy does not imply efficient execution.
- Reason as **observation → hypothesis → one change → expected counter change → benchmark
  result**, and keep the profile directories of parent and child trials for comparison.
