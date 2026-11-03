---
name: xpu-kernels
description: "Provides guidance for writing, optimizing, and benchmarking Triton, SYCL, and SYCL-TLA (CUTLASS for SYCL) kernels for Intel XPU GPUs in the Hugging Face kernels ecosystem, without assuming a specific GPU model (the device is queried through PyTorch). Includes a branching trial-loop workflow (analyze, validate, benchmark, profile, finalize), an Intel XPU knowledge base (tensor descriptors, GRF mode, tile swizzling, DPAS/XMX tiling, SLM, sub-group collectives, epilogue fusion), unitrace/VTune profiling guidance, and porting to kernel-builder (cpp-xpu) projects."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "kernel type: gemm, attention, flash-attention, rmsnorm, softmax, reduction, fused-epilogue, moe, triton, sycl, sycl-tla, cutlass, optimize, benchmark, profile, tensor-descriptors"
---

# XPU Kernels for Intel GPUs (Triton, SYCL, SYCL-TLA)

This skill provides patterns and a repeatable optimization workflow for developing fast,
numerically equivalent kernels for Intel XPU GPUs. Kernels are written in one of three ways:

- **Triton** with the [Intel XPU backend](https://github.com/intel/intel-xpu-backend-for-triton) — fastest iteration; the default for fused elementwise/reduction/GEMM-epilogue/attention work.
- **SYCL** (DPC++) — explicit control over work-groups, sub-groups, SLM and vectorization.
- **SYCL-TLA** ([CUTLASS for SYCL](https://github.com/intel/sycl-tla), CuTe) — library collectives for DPAS/XMX GEMM, grouped/MoE GEMM, flash attention and epilogue fusion.

The methodology and knowledge base are adapted from [Xe-Forge](https://github.com/IntelLabs/Xe-Forge) (see [Acknowledgments](#acknowledgments)).

## Key Concepts (read before the Quick Start)

- **Trial** — one candidate implementation (`t<N>.py`), saved into a branching tree by `scripts/trial_manager.py`. Every trial is validated, benchmarked and recorded.
- **Baseline** — the PyTorch reference (`*_pytorch.py`, KernelBench-style `Model` + `get_inputs()` + `get_init_inputs()`) or an existing kernel (`--triton-baseline`). Every trial must match it numerically.
- **Trial form vs. shipping form** — trials are self-contained Python files (Triton, or SYCL compiled with `load_inline`) so a single harness can measure them; the winner is then ported to a kernel-builder project (`torch-noarch` for Triton, `cpp-xpu` for SYCL/SYCL-TLA).
- **Single XPU** — GPU work (benchmark, profiling) runs strictly one at a time.
- **No hardware assumptions** — the skill does not assume a GPU model. Query the device (`scripts/xpu_device_info.py`) and derive every hardware-dependent choice from what it reports; measure everything else.

## Quick Start

```bash
# 0. Query the device, pick the DSL (see "Choosing a DSL"), read scripts/config.yaml
python scripts/xpu_device_info.py

# 1. Analyze the PyTorch baseline
python scripts/analyze_kernel.py my_op_pytorch.py

# 2. Initialize trial tracking
python scripts/trial_manager.py init my_op my_op_pytorch.py

# 3. Validate a generated trial (no GPU needed)
python scripts/validate_triton.py t0.py        # Triton
python scripts/validate_sycl.py   t0.py        # SYCL / SYCL-TLA (load_inline trial)

# 4. Save, then benchmark correctness + performance
python scripts/trial_manager.py save my_op t0.py --strategy "baseline port"
python scripts/benchmark.py my_op_pytorch.py t0.py

# 5. Profile (optional, one job at a time)
python scripts/xpu_profiler.py t0.py           # VTune
# or unitrace for SYCL — see references/sycl/sycl-implementation.md#profiling-sycl-kernels

# 6. Finalize the best trial
python scripts/trial_manager.py finalize my_op my_op_optimized.py
```

## Device Discovery

The skill makes **no claims about specific GPUs**. Before writing a kernel, query the device
that is actually present:

```bash
python scripts/xpu_device_info.py          # or --json
```

```python
import torch
props = torch.xpu.get_device_properties(torch.xpu.current_device())
print(torch.xpu.get_device_name(), props)  # name, gpu_subslice_count, sub_group_sizes, has_fp64, ...
print(torch.xpu.get_arch_list())           # AOT targets this PyTorch build supports
```

Derive hardware-dependent choices from the result, never from a GPU name or a remembered number:

| Decision | Source |
|----------|--------|
| Persistent / Stream-K program count | `gpu_subslice_count` (`xpu_hardware_subslice_query`) |
| SYCL sub-group size, Triton warp size candidates | `sub_group_sizes` |
| Work-group size limits | `max_work_group_size`, `max_num_sub_groups` |
| SLM staging budget | per-work-group local memory limit reported by the device / runtime |
| Matrix (DPAS/XMX) and 2D block I/O paths | `has_subgroup_matrix_multiply_accumulate`, `has_subgroup_2d_block_io` |
| fp64 / bf16 / 64-bit atomics | `has_fp64`, `has_bfloat16_conversions`, `has_atomic64` |
| `TORCH_XPU_ARCH_LIST` for SYCL trials | the `torch.xpu.get_arch_list()` entry matching this device |
| Peak bandwidth / FLOPS, roofline | **not reported** — measure on the device; never assume |

Properties a given PyTorch build does not expose are **unknown**, not defaults.

## When This Skill Applies

Use this skill when:
- Writing or optimizing **Triton, SYCL, or SYCL-TLA** kernels for **Intel XPU**
- Converting PyTorch operations (GEMM, fused epilogues, normalization, softmax, attention, MoE) into fast XPU kernels
- Running the **trial-loop optimization** (analyze → validate → benchmark → profile → finalize)
- Profiling XPU kernels with **unitrace** or **VTune** and turning counters into the next experiment
- Packaging an XPU kernel with **kernel-builder** and publishing it to the Hugging Face Hub

## Choosing a DSL

| Workload | Recommended | Why |
|----------|-------------|-----|
| Elementwise / activation / normalization / softmax / reductions | **Triton** (or plain SYCL) | Simple to write; Triton autotune covers the shape space |
| GEMM + light epilogue, fused KernelBench-style ops | **Triton** first | Tensor descriptors + `grf_mode` + swizzling reach the roofline on many shapes |
| Large BF16/FP16 GEMM, grouped/MoE GEMM, W4A16/W8A16 GEMM | **SYCL-TLA** | Library DPAS mainloops, 2D block copies, Stream-K / group schedulers |
| Flash attention (prefill/decode/paged) | **Triton** or **SYCL-TLA** | Triton for broad shape sets; SYCL-TLA FA2 collectives for fixed head dims |
| Explicit sub-group / SLM control, existing SYCL code base | **SYCL** | Direct `nd_range`, sub-group collectives, `local_accessor` |

You can mix: e.g. a SYCL-TLA GEMM followed by a Triton reduction. Keep each trial a single
self-contained file.

## Optimization Workflow

### Configuration — read `scripts/config.yaml` first

- **`max_trials`** — hard cap on optimization trials; run all of them
- **`vtune_enabled`** / **`vtune_bin`** — VTune profiling (`VTUNE_BIN` env var also works)
- **`unitrace_enabled`** / **`unitrace_bin`** — unitrace profiling (preferred for SYCL when available)

### Rules — never violate

1. **Only create kernel files** — trial files `t<N>.py` and the final output file.
2. **Never create** benchmark scripts, test scripts or helper utilities.
3. **Never write custom timing code** — `scripts/benchmark.py` is the only source of a timing. A number obtained any other way (including from an instrumented profiling run) is not comparable and must not be recorded.
4. A **correctness failure rejects the candidate**, not the session: use the diagnostic, fix the trial, and re-run the mandatory tools. Never time or rank a failing candidate. For **infrastructure failures** (driver, compiler, harness), stop and report the error — do not work around it.
5. Trials are **self-contained**: helpers inline; copy `get_inputs()`, `get_init_inputs()` and constants from the reference instead of importing them.
6. **Run all `max_trials`**. Do not stop on a plateau — try a structurally different approach. The only valid early stop is speedup > 5x.
7. **SYCL trials:** submit to PyTorch's queue, namespace every kernel name per trial, and pin `TORCH_XPU_ARCH_LIST` (see [sycl-implementation.md](references/sycl/sycl-implementation.md#hard-rules)).

### Mandatory tools

**Single-XPU serialization**: `benchmark.py`, `xpu_profiler.py` and unitrace runs must execute one at a time. CPU-only tools (`analyze_kernel.py`, `validate_*.py`, `trial_manager.py`) can run in parallel.

| Tool | Command | Purpose |
|------|---------|---------|
| **Device info** | `python scripts/xpu_device_info.py [--json]` | Name, properties, AOT targets of the present XPU |
| **Analyze** | `python scripts/analyze_kernel.py <pytorch_file>` | Operations, shapes, fusion opportunities |
| **Validate (Triton)** | `python scripts/validate_triton.py <file>` | Syntax + XPU Triton constraint checks |
| **Validate (SYCL)** | `python scripts/validate_sycl.py <trial.py \| kernel.cpp \| dir>` | Queue, sub-group, kernel-name, binding checks |
| **Benchmark** | `python scripts/benchmark.py <baseline> <trial> [--triton-baseline] [--baseline-us <cached>]` | Correctness + performance via ai-bench (any DSL exposing `Model`) |
| **Profile** | `python scripts/xpu_profiler.py <file>` | VTune GPU counters + KB-grounded recommendations |
| **Init / save / result** | `python scripts/trial_manager.py {init,save,result} ...` | Trial tree management |
| **Status / best / baseline-us** | `python scripts/trial_manager.py {status,best,baseline-us} <kernel_name>` | Inspect the tree, cached baseline time |
| **Finalize** | `python scripts/trial_manager.py finalize <kernel_name> <output_file>` | Copy best trial to the output |

Full arguments: [workflow_details.md](references/workflow_details.md).

### Steps

#### Step 1: Analyze
- Read the baseline; identify shapes, dtypes, operations, fusion opportunities, and whether the problem is compute- or memory-bound (FLOPs vs. bytes).
- PyTorch baseline: run `analyze_kernel.py`. Existing-kernel baseline (`--triton-baseline`): read it directly.
- Read the knowledge base for the chosen DSL (tables below). Always start with the correctness files.

#### Step 2: Initialize
```bash
python scripts/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]
```

#### Step 3: Trial loop (all `max_trials`)
1. **Write** — start from a template ([implementation_reference.md](references/implementation_reference.md) for Triton, [sycl-implementation.md](references/sycl/sycl-implementation.md) for SYCL/SYCL-TLA) or modify the parent trial.
2. **Validate** — `validate_triton.py` / `validate_sycl.py`; fix until passing (does not count as a trial).
3. **Save** — `trial_manager.py save <kernel_name> t<N>.py --parent <id> --strategy "..."` (omit `--parent` for t0).
4. **Benchmark** (every trial):
   - t0: `benchmark.py <baseline> t0.py` (measures both).
   - t1+: `benchmark.py <baseline> t<N>.py --baseline-us $(python scripts/trial_manager.py baseline-us <kernel_name>)`.
   - After finalize: rerun without `--baseline-us` for the final comparison.
5. **Record** — `trial_manager.py result <kernel_name> <id> --validation pass --correctness <pass|fail> --speedup <x> --baseline_us <us> --triton_us <us>`.
6. **Profile** after t1 (if enabled) and again on a plateau. Reason as **observation → hypothesis → one change → expected counter change → benchmark result**; cite the counter and the profile directory.
7. **Decide**:
   - Speedup > 5x → finalize.
   - Improved → continue on this branch; apply the next optimization level ([optimization_levels.yaml](references/optimization_levels.yaml)).
   - Regressed → branch from the best trial with a different strategy.
   - Correctness failed → fix on the same branch.
   - Plateau → structurally different approach (algorithm, tiling, fusion boundary, or DSL). See [optimization_strategies.md](references/optimization_strategies.md).

#### Step 4: Finalize and ship
```bash
python scripts/trial_manager.py finalize <kernel_name> <output_file>
```
Then port to kernel-builder ([Building and Shipping](#building-and-shipping)).

## Reference Docs

### Read first (all DSLs)

| Doc | Contents |
|-----|----------|
| [common/correctness.yaml](references/common/correctness.yaml) | DSL-independent correctness: harness ownership, dtype/rounding contracts |
| [common/algorithmic_patterns.yaml](references/common/algorithmic_patterns.yaml) | DSL-independent algebraic rewrites |
| [workflow_details.md](references/workflow_details.md) | Detailed trial loop, tool arguments, profiler output |
| [kernelbench-classification.md](references/kernelbench-classification.md) | KernelBench operator taxonomy |

### Triton

| Doc | Contents |
|-----|----------|
| [correctness.yaml](references/correctness.yaml) | Triton-on-XPU hard constraints |
| [xpu_optimizations.yaml](references/xpu_optimizations.yaml) | Tensor descriptors, GRF mode, warps, swizzling, spill-aware pruning, exp2 tricks |
| [autotuning_patterns.yaml](references/autotuning_patterns.yaml) | Autotune config spaces, keys, pruning |
| [memory_patterns.yaml](references/memory_patterns.yaml) | Descriptor rules, liveness, pre-packing, atomics |
| [fusion_patterns.yaml](references/fusion_patterns.yaml) | When to fuse vs. split |
| [dtype_optimizations.yaml](references/dtype_optimizations.yaml) | bf16/fp16 I/O, fp32 accumulation, fp64 avoidance |
| [algorithmic_patterns.yaml](references/algorithmic_patterns.yaml) | Triton-specific algorithmic rewrites |
| [persistent_kernel_patterns.yaml](references/persistent_kernel_patterns.yaml) | Persistent kernels, Stream-K |
| [optimization_levels.yaml](references/optimization_levels.yaml) | Progressive L1→L4 checklist + "try harder" tree |
| [optimization_strategies.md](references/optimization_strategies.md) | Strategy reference and checklist |
| [implementation_reference.md](references/implementation_reference.md) | Templates, Model wrapper pattern |

### SYCL / SYCL-TLA

| Doc | Contents |
|-----|----------|
| [sycl/sycl-implementation.md](references/sycl/sycl-implementation.md) | Trial (`load_inline`) and kernel-builder templates, SYCL-TLA GEMM skeleton, unitrace profiling |
| [sycl/sycl_tla_patterns.yaml](references/sycl/sycl_tla_patterns.yaml) | Constraints/patterns: sub-group + MMA, SLM, reductions, online softmax, FA recurrence, epilogues, quantization |
| [sycl/cutlass_sycl_framework.yaml](references/sycl/cutlass_sycl_framework.yaml) | SYCL-TLA tile configs, dispatch policies, schedulers, epilogue catalog, fused kernels |
| [sycl/resource_feedback.yaml](references/sycl/resource_feedback.yaml) | Reading kernel properties (SIMD, SLM, spill, GRF) |
| [sycl/compute_basic.yaml](references/sycl/compute_basic.yaml) | unitrace ComputeBasic / EU-stall collection and interpretation |

## Core Triton Patterns

### Tensor descriptors (required; block pointers are deprecated)

```python
desc = tl.make_tensor_descriptor(
    base=a_ptr, shape=[M, K], strides=[stride_am, stride_ak], block_shape=[BLOCK_M, BLOCK_K],
)
a = desc.load([pid_m * BLOCK_M, k])   # element offsets; out-of-bounds is handled by the descriptor
```

- `tl.make_block_ptr` / `tl.advance` are deprecated — do not use them in new kernels.
- Descriptor `.load()` takes no `boundary_check`/`mask`.
- 2D block I/O needs the contiguous dimension and row stride to be multiples of 16 elements and a 16-byte-aligned base; otherwise loads silently fall back to gathers (`descriptor_2d_block_io_alignment`).

### GRF mode (tune, do not assume)

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "grf_mode": "256"}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "grf_mode": "256"}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "grf_mode": "128"}, num_warps=4,  num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

Put both GRF modes in the autotune space (`grf_mode_both_variants_in_autotune`) and sweep `num_warps` — never hard-code 32 warps.

### Register-spill-aware autotune pruning

Prune configs likely to spill even in large-GRF mode with an `early_config_prune` estimator:

```text
regs ~= (accumulators * BM * BN + stages * (BM * BK + weight_tiles * BN * BK)) / (num_warps * warp_size)
```

Apply only on XPU, count accumulators/weight tiles explicitly (gate/up fused kernels often have 2), keep a never-empty fallback, and treat it as a coarse filter. Pattern: `xpu_register_spill_autotune_pruning` in [xpu_optimizations.yaml](references/xpu_optimizations.yaml).

### Tile swizzling (1D grid)

```python
grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
group_id = pid // (GROUP_SIZE_M * num_pid_n)
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % (GROUP_SIZE_M * num_pid_n)) // group_size_m
```

### bf16/fp16 I/O with fp32 accumulation

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    acc = tl.dot(a_desc.load([off_m, k]), b_desc.load([k, off_n]), acc)
c_desc.store([off_m, off_n], acc.to(tl.bfloat16))
```

Pre-pack weights (transpose/cast) once, not per forward (`xpu_pack_weight_transpose_once`, `xpu_cache_packed_transposes`). Use `exp2`-based sigmoid/tanh/softmax scaling on XPU.

## Core SYCL / SYCL-TLA Patterns

Full templates: [sycl-implementation.md](references/sycl/sycl-implementation.md).

### Launch on PyTorch's queue with a pinned sub-group size

```cpp
namespace my_op_t3 {               // unique per trial
template <typename T> class MyKernel;
}

sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
q.parallel_for<my_op_t3::MyKernel<T>>(
    sycl::nd_range<1>(rows * WG, WG),
    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {   // SG from sub_group_sizes
      // fp32 accumulation; work-group collective when the work-group owns the row
      float s = sycl::reduce_over_group(it.get_group(), partial, sycl::plus<float>());
      ...
    });
// no q.wait()
```

### SYCL-TLA GEMM building blocks

```cpp
using TileShape = Shape<_256, _256, _32>;                                  // work-group tile
using TiledMma  = typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                    Layout<TileShape>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;                               // 2D block loads
using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;
using DispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<2>;             // or MainloopXeL1Staged
// CollectiveMma -> GemmUniversal -> GemmUniversalAdapter; check can_implement() first
```

Tune in this order: TileShape/sub-group layout → dispatch policy → pipeline stages/prefetch → tile scheduler (Persistent / Stream-K / Group) → epilogue fusion. Starting configs and their caveats: [cutlass_sycl_framework.yaml](references/sycl/cutlass_sycl_framework.yaml).

### Things that change numerics — keep the reference contract

- Activation identity and approximation mode (exact vs. tanh GELU; SiLU ≠ GELU) — `sycl_epilogue_sigmoid_gelu_silu`
- Quantization: scale/zero-point semantics, rounding then saturation before integer conversion — `sycl_quantized_gemm_epilogue`
- Online softmax / flash attention: rescale both running sum **and** output when the max changes; handle fully masked rows — `sycl_flash_attention_tiling`
- Split-K / independent accumulators reorder fp additions — validate tolerances

## Critical Constraints

**Triton**
- No default values for `@triton.autotune` meta-parameters in the kernel signature
- 1D grid whenever GROUP_SIZE_M swizzling is used
- Tensor descriptors only; no block pointers; no `boundary_check` on descriptor loads
- Cast batch/offset indices to `int64` before stride multiplication on large tensors
- Pre-zero outputs used with atomic accumulation
- Never autotune `BLOCK_D` (the reduction/head dimension) — use `triton.next_power_of_2(D)`

**SYCL / SYCL-TLA**
- PyTorch's queue only; no per-call `sycl::queue`, no `wait()` in the op
- Unique namespace for kernel names per trial; `TORCH_XPU_ARCH_LIST` set from the queried device
- `[[sycl::reqd_sub_group_size(N)]]` matching the MMA atom / collectives used
- Sub-group reductions only when one sub-group owns the whole reduction
- SLM budget = stages × tile bytes + padding + other local allocations ≤ per-work-group limit
- SYCL-TLA: verify template arguments against the installed revision; `can_implement()` before `run()`

**All**
- `Model` must mirror the reference's `__init__` parameters so weights can be copied; match `get_inputs()` / `get_init_inputs()`
- No device→host syncs (`.item()`, `.cpu()`) in `forward()`

> Full lists: [correctness.yaml](references/correctness.yaml), [common/correctness.yaml](references/common/correctness.yaml), [sycl_tla_patterns.yaml](references/sycl/sycl_tla_patterns.yaml) (`constraints:`)

## Performance Profiling

| Tool | Use for | How |
|------|---------|-----|
| **unitrace** (Intel PTI) | Device timing, ComputeBasic counters, EU stall sampling, kernel properties (SIMD, SLM, spill, GRF) | `unitrace --device-timing --metric-sampling --group ComputeBasic -o <dir>/trace python scripts/benchmark.py ...` |
| **VTune** | XVE active/stalled/idle, occupancy limiters, cache hierarchy, KB-grounded recommendations | `python scripts/xpu_profiler.py <trial>` |

- Run `unitrace --metric-list` first; available groups depend on device, driver, metrics library and permissions. Missing counters are **unknown**, not zero.
- Profiler timings are never results — only uninstrumented `benchmark.py` numbers are recorded.
- Kernel properties are driver-reported allocations; spill memory size is not spill traffic. High occupancy does not imply efficiency; high `SbidStall` does not by itself prove DRAM saturation.

Details: [compute_basic.yaml](references/sycl/compute_basic.yaml), [resource_feedback.yaml](references/sycl/resource_feedback.yaml), [workflow_details.md](references/workflow_details.md#profiling).

## Building and Shipping

Scaffold with kernel-builder and move the winning trial's code into it:

```bash
kernel-builder init --name my-org/my-kernel --backends xpu
```

| Trial DSL | `build.toml` | Notes |
|-----------|--------------|-------|
| Triton | `[torch-noarch]` section, Python package in `torch-ext/<name>/` | See `examples/kernels/relu-triton` |
| SYCL | `[kernel.<name>]` with `language = "cpp-xpu"`, `depends = ["torch"]` | See `examples/kernels/relu` (`relu_xpu/`) |
| SYCL-TLA | `language = "cpp-xpu"`, `depends = ["torch", "sycl_tla"]` | See `examples/kernels/cutlass-gemm` |

- Bind C++ ops with `TORCH_LIBRARY_EXPAND` + `ops.impl(..., torch::kXPU, &fn)`; no pybind11, no `setup.py`, no `load_inline` in shipped code.
- Extra SYCL compiler flags go in `sycl-flags` — only when really needed.
- Build: `nix run .#build-and-copy -L` (or `kernel-builder build`); test: `nix run .#ci-test`; publish: `kernel-builder build-and-upload`.

Porting checklist and binding template: [sycl-implementation.md](references/sycl/sycl-implementation.md#porting-a-trial-to-kernel-builder). Hub loading and library integration: [huggingface-kernels-integration.md](references/huggingface-kernels-integration.md).

## Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Autotuned `BLOCK_D` | Large numerical mismatch vs. reference | Fix `BLOCK_D = triton.next_power_of_2(D)` |
| Python `min`/`max` in kernel | Compile/runtime error | `tl.minimum()` / `tl.maximum()` |
| Descriptor loads slow | No `ttig.block_io` in TTGIR | Align base to 16 B and contiguous dim/stride to 16 elements |
| Autotune takes forever | Many configs spill and recompile | Spill-aware `early_config_prune` |
| SYCL trial slower than expected | JIT from generic SPIR-V | Set `TORCH_XPU_ARCH_LIST` to this device's entry from `xpu_device_info.py` before `load_inline` |
| Wrong results only when baseline and trial run together | Kernel name collision across extensions | Unique namespace per trial |
| Out-of-order / stale results in SYCL op | Private `sycl::queue` | Use `c10::xpu::getCurrentXPUStream().queue()` |
| Partial-row results in SYCL reduction | Sub-group reduce on a work-group-owned row | Work-group collective, or one sub-group per row |
| SYCL-TLA build errors after toolchain bump | Template signature changed | Check declarations in the installed sycl-tla revision |
| No ComputeBasic counters | Metrics library / permissions | `unitrace --metric-list`; ask an admin about `observation_paranoid` — never change it silently |

## Project Structure

```
xpu-kernels/
├── SKILL.md                         # This file
├── README.md
├── manifest.txt                     # Files installed by `kernel-builder skills add`
├── scripts/
│   ├── analyze_kernel.py            # PyTorch → operations, shapes, fusion opportunities
│   ├── validate_triton.py           # Triton static checks
│   ├── validate_sycl.py             # SYCL / SYCL-TLA static checks
│   ├── benchmark.py                 # Correctness + performance via ai-bench
│   ├── benchmark_kernels.py         # Micro-benchmarks: RMSNorm, RoPE 3D, GEGLU, AdaLN (Triton)
│   ├── trial_manager.py             # Branching trial tree
│   ├── xpu_profiler.py              # VTune counters + recommendations
│   ├── xpu_device_info.py           # Query the present XPU through PyTorch
│   ├── config.py / config.yaml      # Session configuration
│   ├── huggingface_kernels_example.py
│   ├── transformers_injection_example.py
│   └── requirements.txt
└── references/
    ├── common/                      # DSL-independent correctness + algorithmic patterns
    ├── sycl/                        # SYCL / SYCL-TLA knowledge base + implementation guide
    ├── *.yaml, *.md                 # Triton-on-XPU knowledge base
    ├── workflow_details.md
    ├── huggingface-kernels-integration.md
    └── kernelbench-classification.md
```

## See Also

- [triton-kernels](../triton-kernels/SKILL.md) — portable Triton patterns (NVIDIA/AMD)
- [cuda-kernels](../cuda-kernels/SKILL.md), [rocm-kernels](../rocm-kernels/SKILL.md), [cpu-kernels](../cpu-kernels/SKILL.md) — other backends
- [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton)
- [SYCL-TLA (CUTLASS for SYCL)](https://github.com/intel/sycl-tla)
- [oneAPI DPC++ / SYCL](https://github.com/intel/llvm) and the [SYCL 2020 specification](https://registry.khronos.org/SYCL/)
- [unitrace (Intel PTI)](https://github.com/intel/pti-gpu/tree/master/tools/unitrace)
- [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) — production SYCL/SYCL-TLA kernels referenced by the knowledge base
- [AI-Bench](https://github.com/libxsmm/AI-bench) — benchmark harness used by `scripts/benchmark.py`
- [Xe-Forge](https://github.com/IntelLabs/Xe-Forge) — fully automated multi-stage optimizer (DSPy and Claude Code engines, tile search, kernel-repo mode)

## Acknowledgments

The methodology, scripts and knowledge base of this skill were contributed by Intel AI researchers from the IntelLabs team behind [Xe-Forge](https://github.com/IntelLabs/Xe-Forge) and are kept in sync with its `knowledge_base/` (Triton, common and SYCL sections).
