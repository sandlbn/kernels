#!/usr/bin/env python3
"""
Query the Intel XPU that is actually present, through PyTorch.

The skill makes no assumptions about the GPU model: run this first and size kernels
(grids, persistent program counts, work-group / sub-group sizes, SLM use, dtypes) and
pick the AOT target from what it reports.

Usage:
    python scripts/xpu_device_info.py            # human-readable
    python scripts/xpu_device_info.py --json     # machine-readable
    python scripts/xpu_device_info.py --device 1
"""

import argparse
import json
import os
import sys

# Properties that kernel decisions depend on, when this PyTorch build exposes them.
# Anything else the build reports is printed under "all properties".
KEY_PROPERTIES = [
    ("name", "Device name"),
    ("platform_name", "Platform"),
    ("driver_version", "Driver version"),
    ("version", "Device/runtime version"),
    ("architecture", "Architecture id (sycl::ext::oneapi::experimental::architecture)"),
    ("device_id", "PCI device id"),
    ("total_memory", "Global memory (bytes)"),
    ("max_compute_units", "Compute units"),
    ("gpu_eu_count", "EU / XVE count"),
    ("gpu_subslice_count", "Sub-slice (Xe-core) count  -> persistent program count"),
    ("gpu_eu_simd_width", "EU SIMD width"),
    ("gpu_eu_count_per_subslice", "EUs per sub-slice"),
    ("gpu_hw_threads_per_eu", "HW threads per EU"),
    ("max_work_group_size", "Max work-group size"),
    ("max_num_sub_groups", "Max sub-groups per work-group"),
    ("sub_group_sizes", "Supported sub-group sizes"),
    ("local_mem_size", "Local memory (SLM) per work-group (bytes)"),
    ("has_fp16", "fp16"),
    ("has_fp64", "fp64"),
    ("has_bfloat16_conversions", "bf16 conversions"),
    ("has_atomic64", "64-bit atomics"),
    ("has_subgroup_matrix_multiply_accumulate", "Sub-group matrix MMA (DPAS/XMX)"),
    ("has_subgroup_matrix_multiply_accumulate_tensor_float32", "Sub-group matrix MMA (tf32)"),
    ("has_subgroup_2d_block_io", "Sub-group 2D block I/O"),
]


def _jsonable(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def query(device_index: int | None) -> dict:
    try:
        import torch
    except ImportError:
        return {"error": "PyTorch is not installed."}

    info: dict = {"torch_version": torch.__version__}
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        info["error"] = "torch.xpu is not available (no XPU build of PyTorch, or no device/driver)."
        return info

    info["device_count"] = torch.xpu.device_count()
    idx = torch.xpu.current_device() if device_index is None else device_index
    info["device_index"] = idx

    if hasattr(torch.xpu, "get_device_name"):
        info["device_name"] = torch.xpu.get_device_name(idx)
    if hasattr(torch.xpu, "get_device_capability"):
        try:
            info["capability"] = _jsonable(torch.xpu.get_device_capability(idx))
        except Exception as e:  # capability is optional
            info["capability"] = f"unavailable: {e}"

    props = {}
    if hasattr(torch.xpu, "get_device_properties"):
        p = torch.xpu.get_device_properties(idx)
        for attr in dir(p):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(p, attr)
            except Exception:
                continue
            if callable(value):
                continue
            props[attr] = _jsonable(value)
    info["properties"] = props

    # AOT targets this PyTorch build was compiled for. TORCH_XPU_ARCH_LIST for SYCL
    # load_inline trials should name the entry that corresponds to this device.
    if hasattr(torch.xpu, "get_arch_list"):
        try:
            info["torch_arch_list"] = list(torch.xpu.get_arch_list())
        except Exception as e:
            info["torch_arch_list"] = f"unavailable: {e}"

    info["env"] = {
        k: os.environ[k]
        for k in ("TORCH_XPU_ARCH_LIST", "ZE_AFFINITY_MASK", "ZE_FLAT_DEVICE_HIERARCHY", "ONEAPI_DEVICE_SELECTOR")
        if k in os.environ
    }
    return info


def print_human(info: dict) -> None:
    print(f"\n{'=' * 70}\nXPU device information (from PyTorch)\n{'=' * 70}")
    print(f"  torch: {info.get('torch_version', 'unknown')}")
    if "error" in info:
        print(f"\n  ERROR: {info['error']}")
        return

    print(f"  devices: {info['device_count']} (showing index {info['device_index']})")
    print(f"  name:    {info.get('device_name', 'unknown')}")
    if "capability" in info:
        print(f"  capability: {info['capability']}")

    props = info.get("properties", {})
    print("\nKey properties (unknown = not exposed by this PyTorch build):")
    for key, label in KEY_PROPERTIES:
        value = props.get(key, "unknown")
        print(f"  {label:<58} {value}")

    other = {k: v for k, v in props.items() if k not in {k for k, _ in KEY_PROPERTIES}}
    if other:
        print("\nAll other properties:")
        for k in sorted(other):
            print(f"  {k:<58} {other[k]}")

    arch_list = info.get("torch_arch_list")
    print("\nAOT targets in this PyTorch build (torch.xpu.get_arch_list()):")
    print(f"  {arch_list if arch_list else 'unknown'}")
    print(
        "  For SYCL load_inline trials, set TORCH_XPU_ARCH_LIST to the entry that matches this\n"
        "  device (confirm with `sycl-ls --verbose` or the driver documentation). If no entry\n"
        "  can be matched with certainty, leave it unset and record that the trial was JIT-compiled."
    )

    if info.get("env"):
        print("\nRelevant environment:")
        for k, v in info["env"].items():
            print(f"  {k}={v}")

    print(
        "\nUse these values instead of assumed hardware numbers: e.g. persistent kernels launch\n"
        "gpu_subslice_count-based program counts, sub-group sizes must be in sub_group_sizes,\n"
        "SLM staging must fit local_mem_size, and fp64 / matrix / 2D-block paths require the\n"
        "corresponding has_* flag. Peak bandwidth / FLOPS are not reported here: measure them."
    )


def main():
    parser = argparse.ArgumentParser(description="Query the Intel XPU through PyTorch")
    parser.add_argument("--device", type=int, default=None, help="XPU device index (default: current)")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()

    info = query(args.device)
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        print_human(info)
    sys.exit(1 if "error" in info else 0)


if __name__ == "__main__":
    main()
