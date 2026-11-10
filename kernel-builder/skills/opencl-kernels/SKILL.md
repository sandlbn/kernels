---
name: opencl-kernels
description: "Provides guidance for writing, debugging, and optimizing OpenCL C kernels for Intel GPUs (Xe architecture). Covers GEMM via DPAS/XMX subgroup matrix builtins, work-group/sub-group synchronization (split work-group barriers), and Xe device topology/roofline for sizing kernels. Reference-only — no PyTorch backend, so no trial-loop automation: pairs with manual compilation (icpx/ocloc) and clinfo-based device queries."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "kernel type: gemm, dpas, barrier, workgroup-sync, architecture, roofline, tiling"
---

# OpenCL C Kernels for Intel GPUs

This skill provides patterns and guidance for writing and optimizing OpenCL C kernels
targeting Intel Xe GPUs (Lunar Lake, Battlemage, Panther Lake and later). Unlike the other
backends in this repo, OpenCL has no PyTorch extension path — there is no `load_inline`
equivalent, so this skill is reference-only: read the relevant doc, write the kernel, build
it with `icpx`/`ocloc`, and validate/benchmark it with your own host code or `clinfo`-driven
checks. There is no `scripts/` trial-loop automation here.

## Choosing a topic

| Task | Start with |
|------|------------|
| Writing or optimizing a GEMM / matmul kernel (DPAS, tiling, quantized `B`) | [references/gemm.md](references/gemm.md) |
| Looking up a specific DPAS builtin signature, operand packing, or the DPASW split variant | [references/gemm-dpas-reference.md](references/gemm-dpas-reference.md) |
| Synchronizing a work-group with latency-hiding (`intel_work_group_barrier_arrive`/`_wait`) | [references/split-workgroup-barrier.md](references/split-workgroup-barrier.md) |
| Sizing work-groups, SLM, subgroup width, or estimating a roofline ceiling before writing a kernel | [references/architecture.md](references/architecture.md) |
| Full Khronos spec wording for the split-barrier extension (SPIR-V details, memory ordering) | [references/split-workgroup-barrier-extension.md](references/split-workgroup-barrier-extension.md) |
| Spec links for the DPAS extension, IGC assembly docs, OpenCL C language spec | [references/gemm-reference.md](references/gemm-reference.md) |

## Workflow

1. **Query the device first.** Device topology, SLM size, subgroup widths, and extension
   support vary by SKU — read them with `clinfo` or `CL_DEVICE_*` queries rather than
   hardcoding numbers from [references/architecture.md](references/architecture.md), which
   only gives reference figures to plan around.
2. **Get a correct scalar baseline** before reaching for DPAS. It is the oracle for every
   later optimization and the fallback for devices/dtypes DPAS doesn't cover.
3. **Pick the compute path** (DPAS/XMX vs scalar FMA) based on shape and dtype — see
   "Choosing the compute path" in [references/gemm.md](references/gemm.md).
4. **Implement and verify** against the scalar baseline on edge shapes, not just one size.
5. **Measure** with OpenCL profiling events and compare against the roofline from
   [references/architecture.md](references/architecture.md).
6. **Only then** consider synchronization optimizations like split work-group barriers
   ([references/split-workgroup-barrier.md](references/split-workgroup-barrier.md)) — they
   recover latency already lost at a barrier, so profile the barrier stall first.

## Project Structure

```
opencl-kernels/
├── SKILL.md                                      # This file (skill definition + workflow)
├── manifest.txt                                  # Files included in this skill
│
└── references/                                   # Knowledge base
    ├── gemm.md                                   # GEMM contract, DPAS essentials, tiling, quantized B
    ├── gemm-dpas-reference.md                    # Full DPAS builtin signatures, operand layouts, VNNI packing
    ├── gemm-reference.md                         # Spec links (DPAS extension, IGC assembly docs, OpenCL C)
    ├── architecture.md                           # Xe2 device topology, SLM, subgroup sizing, roofline
    ├── split-workgroup-barrier.md                # intel_work_group_barrier_arrive/_wait usage and pitfalls
    └── split-workgroup-barrier-extension.md      # Full Khronos spec wording, SPIR-V details
```

## See Also

- [xpu-kernels skill](../xpu-kernels/SKILL.md) — Triton, SYCL, and SYCL-TLA kernels for Intel
  XPU via the PyTorch `load_inline` trial-loop workflow. Use that skill instead when a
  PyTorch backend is available; OpenCL is the path when it isn't.
- [Khronos OpenCL registry](https://registry.khronos.org/OpenCL/) — index of extension specs.
- [kernel-builder Documentation](https://github.com/huggingface/kernels/tree/main/docs)
