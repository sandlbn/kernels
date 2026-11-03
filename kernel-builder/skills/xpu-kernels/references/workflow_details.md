# Detailed Workflow Reference

## Analysis Phase

When given a PyTorch kernel (typically `*_pytorch.py`, but can be any user-specified path):

> **Note**: Existing naive Triton baselines (e.g. KernelBench `*_triton.py` ports) are **unoptimized**. Do NOT treat them as examples of good Triton code. Use `references/implementation_reference.md` instead.

0. **Query the device**: `python scripts/xpu_device_info.py`. Use its properties (sub-group sizes, work-group limits, sub-slice count, `has_*` flags, AOT targets) for every hardware-dependent decision; the skill assumes no specific GPU model.

0. **Choose the DSL** (Triton, SYCL or SYCL-TLA) using the "Choosing a DSL" table in `SKILL.md`. The trial loop, tools and decision tree are the same for all three; only the templates, validator and knowledge-base files differ.

1. **Parse the PyTorch code** to identify:
   - Input/output shapes and dtypes
   - Mathematical operations (matmul, activations, reductions)
   - Operation fusion opportunities
   - Memory access patterns

2. **Consult the knowledge base** (`references/` directory):
   - All DSLs: `common/correctness.yaml`, `common/algorithmic_patterns.yaml`
   - Triton:
     - `correctness.yaml`: Critical Triton-on-XPU constraints
     - `xpu_optimizations.yaml`: XPU-specific patterns (tensor descriptors, GRF mode, warp count, tile swizzling, spill-aware pruning)
     - `autotuning_patterns.yaml`: Autotune config spaces and pruning
     - `fusion_patterns.yaml`: When to fuse operations
     - `memory_patterns.yaml`: Memory access best practices
     - `dtype_optimizations.yaml`: Data type choices
   - SYCL / SYCL-TLA:
     - `sycl/sycl-implementation.md`: Templates (trial + kernel-builder), SYCL-TLA GEMM skeleton, profiling
     - `sycl/sycl_tla_patterns.yaml`: Constraints and patterns (sub-group/MMA, SLM, reductions, attention, epilogues)
     - `sycl/cutlass_sycl_framework.yaml`: SYCL-TLA tiles, dispatch policies, schedulers, epilogue catalog
     - `sycl/resource_feedback.yaml`, `sycl/compute_basic.yaml`: Interpreting unitrace output

3. **Use the skills** to help:
   - `python scripts/analyze_kernel.py <pytorch_file>` - Extract operation structure

## Design Phase

1. **Identify the kernel type**: Pure GEMM, GEMM + epilogue, GEMM + reduction, complex fusion
2. **Select optimization strategies** from KB (memory, tiling, parallelism, fusion, dtypes)
3. **Apply critical constraints**:
   - Triton (from `references/correctness.yaml` and `references/xpu_optimizations.yaml`):
     - NO default values for `@triton.autotune` meta-parameters in kernel signature
     - Use 1D grid when applying tile swizzling (GROUP_SIZE_M)
     - Use tensor descriptors; `tl.make_block_ptr` / `tl.advance` are deprecated
     - Descriptor `.load()` takes no `boundary_check`; plain `tl.load` boundary checks use dimension indices (0, 1), not booleans
     - Cast batch indices to int64 before stride multiplication
     - Pre-zero output buffers when using atomic accumulation
   - SYCL / SYCL-TLA (from `references/sycl/sycl-implementation.md` and `references/sycl/sycl_tla_patterns.yaml`):
     - Submit to `c10::xpu::getCurrentXPUStream().queue()`; no per-call queue, no `wait()` in the op
     - Unique namespace for kernel names per trial; `TORCH_XPU_ARCH_LIST` pinned before `load_inline`
     - `[[sycl::reqd_sub_group_size(N)]]` matching the MMA atom / collectives
     - Reduction scope: sub-group reductions only when one sub-group owns the reduction
     - SLM budget within the per-work-group limit
   - All: Model class must be compatible with ai-bench (standard `nn.Module` mirroring the reference's parameters)

## Trial Loop Detail

For each trial:

### a. Implement / Modify Kernel
Start from a template (`references/implementation_reference.md`) or modify the previous trial's code. See `references/implementation_reference.md`.

### b. Validate Syntax
```bash
python scripts/validate_triton.py <trial_file>   # Triton
python scripts/validate_sycl.py <trial_file>     # SYCL / SYCL-TLA (load_inline trial)
```
If validation fails, fix and retry - doesn't count as a new trial. Note: `<trial_file>` should be `t<trial_id>.py`.

### c. Save Trial
```bash
python scripts/trial_manager.py save <kernel_name> <trial_file> --parent <parent_id> --strategy "description"
```
For the first trial, omit `--parent`.

### d. Benchmark
```bash
# Trial t0 — measures both baseline and triton:
python scripts/benchmark.py <baseline_file> <trial_file> [--triton-baseline] [--spec <spec.yaml>]

# Trials t1+ — use cached baseline to save time:
python scripts/trial_manager.py baseline-us <kernel_name>   # get cached value
python scripts/benchmark.py <baseline_file> <trial_file> [--triton-baseline] [--spec <spec.yaml>] --baseline-us <cached_value>

# After finalize — re-run without --baseline-us for final accurate comparison
```

### e. Record Results
```bash
python scripts/trial_manager.py result <kernel_name> <trial_id> \
    --validation pass --correctness <pass|fail> --speedup <float> \
    --baseline_us <float> --triton_us <float>
```

### f. Decision Tree

| Condition                                  | Action                                                                                                                                                |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Speedup > 5x**                           | Stop - excellent result (the only valid early stop)                                                                                                   |
| **Speedup improved**                       | Continue on this branch, try next optimization level                                                                                                  |
| **Speedup regressed**                      | Branch back to best trial, try a different strategy                                                                                                   |
| **Correctness failed**                     | Fix code on same branch                                                                                                                               |
| **After t1 (if profiling enabled)**        | Profile: unitrace (if `unitrace_enabled`, preferred for SYCL) or `python scripts/xpu_profiler.py <trial_file>` (if `vtune_enabled`) — mandatory first profile |
| **Speedup plateaued after 2+ more trials** | Profile again; try a fundamentally different approach (including switching DSL, e.g. Triton → SYCL-TLA for GEMM-heavy work)                          |
| **Plateau / diminishing returns**          | Do NOT stop. Try a fundamentally different approach (different algorithm, tiling, fusion strategy). LLM sampling can discover new ideas at any point. |
| **Max trials reached**                     | Stop — must run all `max_trials` from `config.yaml`                                                                                                   |

### g. Check Status
```bash
python scripts/trial_manager.py status <kernel_name>
python scripts/trial_manager.py best <kernel_name>
```

## Trial Manager Commands Reference
```bash
python scripts/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]
python scripts/trial_manager.py save <kernel_name> <file> [--parent <parent_id>] [--strategy "..."]
python scripts/trial_manager.py result <kernel_name> <trial_id> [--validation pass] [--correctness pass] [--speedup 3.2] [--baseline_us 150.0] [--triton_us 47.0]
python scripts/trial_manager.py status <kernel_name>
python scripts/trial_manager.py best <kernel_name>
python scripts/trial_manager.py baseline-us <kernel_name>
python scripts/trial_manager.py finalize <kernel_name> <output_file>
```

## Benchmarking Details

`scripts/benchmark.py` uses [ai-bench](https://github.com/libxsmm/AI-bench) (installed via `scripts/requirements.txt`) for both correctness and performance. It imports any Python file that defines `Model`, `get_inputs()` and `get_init_inputs()`, so it measures Triton trials and SYCL/SYCL-TLA `load_inline` trials the same way (the "Triton" labels in its output refer to the optimized candidate, whatever its DSL).

1. **Correctness** - Compares outputs between the baseline and the candidate
   - Uses `check_correctness()` with per-variant tolerances from YAML spec (defaults: rtol=1e-2, atol=1e-5)
   - Syncs model weights via `copy_model_weights()`
   - Falls back to direct module loading when no spec file is available

2. **Performance** - Benchmarks both implementations on XPU hardware
   - Reads the YAML spec file: `--spec <spec.yaml>`, otherwise auto-detected next to the candidate and then next to the baseline (`<stem>.yaml` / `<stem>_pytorch.yaml` after stripping `_triton`/`_optimized`/`_opt`/`_pytorch`) — so `my_op_pytorch.py` + `my_op.yaml` side by side works for `t<N>.py` trials
   - Reports speedup metrics (candidate vs baseline) per spec variant

**Both checks must pass** for the kernel to be considered complete.

**Setup**: `pip install -r scripts/requirements.txt` (or the equivalent in your environment).

## Profiling

Two profilers are supported; run either strictly one job at a time.

- **unitrace** (Intel PTI; enable with `unitrace_enabled` in `scripts/config.yaml`) — device timing, ComputeBasic counters, EU stall sampling and driver-reported kernel properties. Preferred for SYCL / SYCL-TLA. Commands and interpretation rules: `references/sycl/sycl-implementation.md#profiling-sycl-kernels`, `references/sycl/compute_basic.yaml`, `references/sycl/resource_feedback.yaml`. The same rules apply to Triton kernels profiled with unitrace.
- **VTune** via `scripts/xpu_profiler.py` (enable with `vtune_enabled`) — described below.

Use profiler output to choose the next experiment, never as a timing result.

### VTune (`scripts/xpu_profiler.py`)

```bash
python scripts/xpu_profiler.py <trial_file> [--warmup 5] [--iters 20]
```

Runs Intel VTune `gpu-offload` collection to capture both Level Zero API tasks and OA (Observation Architecture) hardware counters, then maps bottlenecks to KB optimization patterns.

**Prerequisite**: OA counters require `observation_paranoid=0` (`/proc/sys/dev/xe/observation_paranoid`, or `perf_stream_paranoid` on i915). This is a system setting: ask the developer or an administrator to change it — never change it silently.

### When to Profile
- **MANDATORY** after the first benchmarked trial (t1) — always run at least once per session
- Run again if speedup plateaus after 2+ additional trials
- You're unsure which optimization level to try next

### What It Reports
1. **Platform info**: GPU name, XVE count, max frequency
2. **Host tasks**: CPU-side overhead (JIT compilation, data copies, synchronization)
3. **GPU computing tasks table** (per-kernel): Time, instance count, XVE Active/Stalled/Idle %, occupancy %, memory bandwidth read/write
4. **Primary kernel detail**: Full OA hardware counter breakdown including:
   - XVE execution: Active/Stalled/Idle percentages
   - Occupancy limiters: Work Size Limit, SLM Use Limit, Barriers Use Limit (tells WHY occupancy is low)
   - Memory bandwidth: Read/Write GB/s
   - Cache hierarchy: L3 Busy/Stalled %, L3 Miss Ratio, LSC Miss Ratio, LSC→L3 Miss Ratio
   - Register spill size, SLM bank conflicts, TLB misses
5. **Optimization recommendations**: Each grounded in a specific KB pattern:
   - XVE Stalled > Active → memory bound → `references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern)` + `references/optimization_levels.yaml (level_2_bandwidth_reduction)`
   - Low occupancy + Work Size limiter → grid too small → `references/xpu_optimizations.yaml (xpu_tile_swizzling)`
   - Low occupancy + SLM limiter → tile too large → `references/xpu_optimizations.yaml (xpu_grf_mode)`
   - High L3 Miss → poor reuse → `references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern, xpu_tile_swizzling)`
   - Register spill > 0 → reduce liveness → `references/memory_patterns.yaml (reduce_liveness_sink_load_and_prefetch)`
   - Overhead kernels dominate → pre-pack to bf16 → `references/optimization_levels.yaml (level_2_bandwidth_reduction)`
   - Host time >> GPU time → sync in hot path → `references/memory_patterns.yaml (no_device_to_host_scalar_sync)`

### How to Use the Output
The profiler prints specific recommendations with references:
```
>> XVE Stalled (<S>%) > Active (<A>%): memory/dependency bound.
   Use tensor descriptors for better address codegen, pre-pack to bf16 to halve bandwidth.
   Reference: references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern, xpu_tile_swizzling) +
              references/optimization_levels.yaml (level_2_bandwidth_reduction)
```
Read the referenced file and apply the suggested pattern in your next trial.

## Validation Details

`scripts/validate_triton.py` checks:
- Syntax correctness
- Autotune config issues (no default params)
- Grid/swizzling consistency
- boundary_check format, deprecated block pointers
- Data type usage

`scripts/validate_sycl.py` checks (trial `.py`, `.cpp`/`.hpp` sources or a kernel directory):
- `load_inline` uses `sycl_sources=`, `TORCH_XPU_ARCH_LIST` pinned, trial is self-contained, `Model`/`get_inputs`/`get_init_inputs` present
- No private `sycl::queue` / host `wait()` in the op; PyTorch's XPU stream used
- Sub-group size pinned when sub-group operations or MMA atoms are used; reduction-scope and SLM reminders
- Kernel names namespaced in trials; `TORCH_CHECK` input validation present
- Shipping form: no pybind11, `TORCH_LIBRARY_EXPAND` registers a `torch::kXPU` impl; SYCL-TLA `can_implement()` checked

## Project Structure

See the Project Structure section in `SKILL.md`.
