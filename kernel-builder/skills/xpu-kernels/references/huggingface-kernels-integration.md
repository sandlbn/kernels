# HuggingFace Kernels Integration Guide (XPU)

Complete guide for using and publishing kernels with the HuggingFace Kernels library (`get_kernel`) on Intel XPU.

> **Quick Start:** See [huggingface_kernels_example.py](../scripts/huggingface_kernels_example.py) for a minimal working example.

## Overview

The [HuggingFace Kernels](https://huggingface.co/docs/kernels/en/index) library enables dynamic loading of pre-compiled kernels from the Hugging Face Hub. This eliminates the need for local compilation and ensures compatibility across different Python, PyTorch, and backend versions.

**Key Benefits:**
- **No local compilation** — download pre-built binaries
- **Version management** — load specific kernel versions
- **Multi-version support** — multiple versions coexist in one Python process
- **Automatic compatibility** — matches your PyTorch configuration

**XPU Note:** Not all Hub kernels have XPU builds. Triton-based kernels (e.g., `triton-layer-norm`) are more likely to work on XPU than CUDA C kernels. Always check with `has_kernel()` first.

## Installation

```bash
pip install kernels torch numpy
```

Requirements:
- PyTorch >= 2.5 (XPU build)
- Intel XPU GPU
- Python 3.8+

## Core API

### get_kernel

Download and load a kernel from the Hub:

```python
from kernels import get_kernel

kernel = get_kernel("kernels-community/triton-layer-norm")

# With specific version
kernel = get_kernel("kernels-community/triton-layer-norm", version=1)

# With specific revision
kernel = get_kernel("kernels-community/flash-attn", revision="v2.0.0")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | str | required | Hub repository (e.g., "kernels-community/activation") |
| `revision` | str | "main" | Branch, tag, or commit hash |
| `version` | int/str | None | Kernel version number (mutually exclusive with `revision`) |

**Returns:** `ModuleType` — the imported kernel module

### has_kernel

Check if a kernel build exists for your environment:

```python
from kernels import has_kernel

if has_kernel("kernels-community/triton-layer-norm"):
    kernel = get_kernel("kernels-community/triton-layer-norm")
else:
    print("No compatible build for this XPU/PyTorch version")
```

### get_local_kernel

Load a kernel from a local path (useful during development):

```python
from kernels import get_local_kernel

kernel = get_local_kernel("/path/to/my-kernel")
```

### load_kernel & get_locked_kernel

For reproducible, offline-capable deployments using lockfiles:

```python
from kernels import load_kernel, get_locked_kernel

kernel = load_kernel("lockfile.json")
kernel = get_locked_kernel("kernels-community/activation", lockfile="kernel.lock")
```

## Usage Examples

### 1. RMSNorm Kernel from Hub

**Note:** The actual function name may vary by kernel version. Use `dir(kernel)` to inspect, and check for `rms_norm_fn`, `rms_norm`, or `rmsnorm`.

```python
import torch
from kernels import get_kernel, has_kernel

repo_id = "kernels-community/triton-layer-norm"

if has_kernel(repo_id):
    layer_norm = get_kernel(repo_id)

    # Inspect available functions
    print([f for f in dir(layer_norm) if not f.startswith('_')])
    # e.g. ['layer_norm', 'layer_norm_fn', 'rms_norm_fn', ...]

    x = torch.randn(2, 1024, 2048, dtype=torch.bfloat16, device="xpu")
    weight = torch.ones(2048, dtype=torch.bfloat16, device="xpu")

    # Use the actual function name (rms_norm_fn in current version)
    out = layer_norm.rms_norm_fn(x, weight, eps=1e-6)
    print(f"Output shape: {out.shape}")
else:
    print("No XPU-compatible build available")
```

### 2. Integration with Transformers Models

```python
import torch
from kernels import get_kernel, has_kernel

repo_id = "kernels-community/triton-layer-norm"

if has_kernel(repo_id):
    rmsnorm_kernel = get_kernel(repo_id)

    def patch_rmsnorm_with_hub_kernel(model):
        """Patch model's RMSNorm to use Hub kernel."""
        patched = 0
        for name, module in model.named_modules():
            if 'RMSNorm' in type(module).__name__:
                eps = getattr(module, 'variance_epsilon', None) or getattr(module, 'eps', 1e-6)

                def make_forward(mod, epsilon):
                    def forward(hidden_states):
                        return rmsnorm_kernel.rms_norm(hidden_states, mod.weight, eps=epsilon)
                    return forward

                module.forward = make_forward(module, eps)
                patched += 1
        return patched
```

### 3. Integration with Diffusers Pipelines

```python
import torch
from diffusers import LTXPipeline
from kernels import get_kernel, has_kernel

if has_kernel("kernels-community/triton-layer-norm"):
    rmsnorm_kernel = get_kernel("kernels-community/triton-layer-norm")

    def patch_rmsnorm(model):
        for name, module in model.named_modules():
            if type(module).__name__ == 'RMSNorm':
                eps = getattr(module, 'eps', 1e-6)
                has_weight = hasattr(module, 'weight') and module.weight is not None

                if has_weight:
                    def make_forward(mod, epsilon):
                        def forward(x):
                            return rmsnorm_kernel.rms_norm(x, mod.weight, eps=epsilon)
                        return forward
                    module.forward = make_forward(module, eps)

    pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
    pipe.to("xpu")
    patch_rmsnorm(pipe.transformer)
```

### 4. Benchmark Hub Kernel vs PyTorch

```python
import time
import torch
from kernels import get_kernel

kernel = get_kernel("kernels-community/triton-layer-norm")

sizes = [(2, 1024, 2048), (4, 4096, 4096)]
for shape in sizes:
    x = torch.randn(shape, dtype=torch.bfloat16, device="xpu")
    w = torch.ones(shape[-1], dtype=torch.bfloat16, device="xpu")

    for _ in range(10):
        kernel.rms_norm(x, w, eps=1e-6)
        variance = x.pow(2).mean(-1, keepdim=True)
        _ = x * torch.rsqrt(variance + 1e-6) * w
    torch.xpu.synchronize()

    iters = 100
    start = time.perf_counter()
    for _ in range(iters):
        kernel.rms_norm(x, w, eps=1e-6)
    torch.xpu.synchronize()
    hub_ms = (time.perf_counter() - start) / iters * 1000

    start = time.perf_counter()
    for _ in range(iters):
        variance = x.pow(2).mean(-1, keepdim=True)
        _ = x * torch.rsqrt(variance + 1e-6) * w
    torch.xpu.synchronize()
    pt_ms = (time.perf_counter() - start) / iters * 1000

    print(f"Shape {shape}: Hub={hub_ms:.3f}ms, PyTorch={pt_ms:.3f}ms, Speedup={pt_ms/hub_ms:.2f}x")
```

## XPU-Specific Notes

### Kernel Compatibility

Not all Hub kernels have XPU builds:

| Kernel Type | XPU Support | Notes |
|-------------|:----------:|-------|
| Triton-based (e.g., `triton-layer-norm`) | Likely | Triton compiles via Intel XPU backend |
| SYCL / SYCL-TLA (`cpp-xpu`) | Yes | Built per XPU Torch variant by kernel-builder |
| CUDA C-based (e.g., `flash-attn`) | Check | Needs explicit XPU (SYCL) build |
| Custom CUDA ops | Unlikely | CUDA-only unless ported |

**Always check availability first:**
```python
from kernels import has_kernel

if has_kernel("kernels-community/triton-layer-norm"):
    print("XPU build available")
else:
    print("No XPU build — use local Triton kernel instead")
```

### Fallback Strategy

When a Hub kernel is not available for XPU, fall back to the local Triton implementation:

```python
from kernels import has_kernel, get_kernel

def get_rmsnorm_function():
    """Get best available RMSNorm implementation."""
    if has_kernel("kernels-community/triton-layer-norm"):
        kernel = get_kernel("kernels-community/triton-layer-norm")
        return lambda x, w, eps: kernel.rms_norm(x, w, eps=eps)
    else:
        from your_local_kernels import triton_rmsnorm
        return triton_rmsnorm
```

### Environment Check

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
print(f"GPU: {torch.xpu.get_device_name()}")
```

## Publishing Kernels to Hub

Scaffold with `kernel-builder init --name your-org/your-kernel --backends xpu`, then replace
the stub sources. The layout depends on the kernel's language.

### Triton kernels (`torch-noarch`)

Triton kernels are JIT-compiled, so they ship as a Python package without an AOT build
section (see `examples/kernels/relu-triton` in the kernels repository):

```
my-triton-kernel/
├── build.toml
├── tests/
└── torch-ext/
    └── my_triton_kernel/
        ├── __init__.py      # public API
        └── op.py            # @triton.jit kernels + launchers
```

```toml
[general]
name = "my-triton-kernel"     # dash-separated
version = 1
license = "Apache-2.0"
backends = ["cuda", "rocm", "xpu"]

[general.hub]
repo-id = "your-org/my-triton-kernel"

[torch-noarch]
# pyext = ["py", "json"]     # e.g. to ship Triton autotune configs
```

### SYCL and SYCL-TLA kernels (`cpp-xpu`)

AOT-compiled for every XPU Torch variant (see `examples/kernels/relu` and
`examples/kernels/cutlass-gemm`):

```
my-sycl-kernel/
├── build.toml
├── rmsnorm_xpu/
│   └── rmsnorm.cpp          # SYCL / SYCL-TLA device + host code
├── tests/
└── torch-ext/
    ├── torch_binding.cpp    # TORCH_LIBRARY_EXPAND registration (torch::kXPU)
    ├── torch_binding.h
    └── my_sycl_kernel/
        └── __init__.py
```

```toml
[general]
name = "my-sycl-kernel"
version = 1
license = "Apache-2.0"
backends = ["xpu"]

[general.hub]
repo-id = "your-org/my-sycl-kernel"

[torch]
src = ["torch-ext/torch_binding.cpp", "torch-ext/torch_binding.h"]

[kernel.rmsnorm_xpu]
language = "cpp-xpu"
depends = ["torch"]          # add "sycl_tla" for SYCL-TLA / CuTe kernels
src = ["rmsnorm_xpu/rmsnorm.cpp"]
```

Binding template and porting checklist: `references/sycl/sycl-implementation.md`.

### Build and Publish

```bash
# The kernel directory must be a git repository (Nix only sees tracked files)
nix run .#build-and-copy -L        # or: kernel-builder build
nix run .#ci-test                  # run the kernel's tests for the built variants

# Build and upload to the Hub repo set by [general.hub] repo-id
kernel-builder build-and-upload
```

Uploads go to a **kernel**-type Hub repository; the owning user/org needs kernel-creation access.

### Others Load It

```python
from kernels import get_kernel

rmsnorm = get_kernel("your-org/your-kernel")
```

## Available Community Kernels

Popular kernels from `kernels-community`:

| Kernel | Description | XPU? |
|--------|-------------|:----:|
| `triton-layer-norm` | LayerNorm, RMSNorm | Likely |
| `activation` | GELU, SiLU, etc. | Check |
| `flash-attn` | Flash Attention 2 | Check |
| `quantization` | INT8/INT4 ops | Check |

Browse all kernels: https://huggingface.co/kernels-community

## Caching and Offline Usage

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"

# Will only use cached kernels
kernel = get_kernel("kernels-community/triton-layer-norm")
```

## Best Practices

1. **Always check availability** — `has_kernel()` before `get_kernel()`
2. **Pin versions** — `get_kernel(repo, version=1)` for reproducibility
3. **Have a fallback** — local Triton kernel or PyTorch path when a Hub build is unavailable
4. **Use lockfiles in production** — `load_kernel("kernel.lock")`
5. **Test on your GPU** — verify correctness after loading

## See Also

- [HuggingFace Kernels Documentation](https://huggingface.co/docs/kernels/en/index)
- [HuggingFace Kernels GitHub](https://github.com/huggingface/kernels)
- [Kernel Builder Documentation](https://github.com/huggingface/kernel-builder)
- [Community Kernels](https://huggingface.co/kernels-community)
- [Blog: Learn the Kernel Hub in 5 Minutes](https://huggingface.co/blog/hello-hf-kernels)
