# XPU Kernels Skill

Guidance, tools and a knowledge base for writing and optimizing **Triton**, **SYCL** and
**SYCL-TLA** (CUTLASS for SYCL) kernels on Intel XPU GPUs, and for
shipping them with kernel-builder. Start with [SKILL.md](SKILL.md).

The methodology, scripts and knowledge base are adapted from
[Xe-Forge](https://github.com/IntelLabs/Xe-Forge), an LLM-driven multi-stage kernel
optimizer for Intel GPUs. The knowledge base in `references/` mirrors Xe-Forge's
`knowledge_base/` (`triton/xpu` → `references/`, `common` → `references/common/`,
`sycl/xpu` → `references/sycl/`), edited to refer to this skill's tools.
Last synced with Xe-Forge commit `8f84e39`.

## Full Xe-Forge

For the fully automated pipeline (DSPy and Claude Code engines, CUTLASS SYCL tile search,
optimizing kernels inside an existing repository, framework capture):

```bash
git clone https://github.com/IntelLabs/Xe-Forge
cd Xe-Forge
uv sync --extra intel
```

## Prerequisites

- Python 3.10+
- PyTorch with XPU support (2.8+ for SYCL `load_inline` trials)
- Triton: the [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton) matching your PyTorch
- SYCL / SYCL-TLA: the oneAPI DPC++ compiler (`icpx`) environment sourced; [SYCL-TLA](https://github.com/intel/sycl-tla) headers for CuTe/CUTLASS kernels
- An Intel XPU visible to PyTorch (`python scripts/xpu_device_info.py` reports what is present; the skill assumes no specific model)
- Optional profilers: [unitrace](https://github.com/intel/pti-gpu/tree/master/tools/unitrace) and/or Intel VTune Profiler 2025+ (toggle in `scripts/config.yaml`)

## Install Dependencies

```bash
pip install -r scripts/requirements.txt
```
