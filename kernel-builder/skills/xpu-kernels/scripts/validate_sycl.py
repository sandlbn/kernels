#!/usr/bin/env python3
"""
Static checks for SYCL / SYCL-TLA kernels targeting Intel XPU (no GPU needed).

Accepts either form described in references/sycl/sycl-implementation.md:
  - a trial file (.py) that JIT-compiles SYCL via torch.utils.cpp_extension.load_inline
    and exposes a KernelBench-style Model, or
  - kernel-builder sources (.cpp/.hpp/.h), or a directory containing them.

Usage:
    python scripts/validate_sycl.py <trial.py | kernel.cpp | kernel_dir>
"""

import ast
import re
import sys
from pathlib import Path
from typing import List

CPP_SUFFIXES = {".cpp", ".cc", ".cxx", ".hpp", ".h", ".hh"}


class ValidationError:
    def __init__(self, level: str, message: str, line_num: int = None, file: str = None):
        self.level = level  # 'ERROR', 'WARNING', 'INFO'
        self.message = message
        self.line_num = line_num
        self.file = file

    def __str__(self):
        prefix = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}[self.level]
        loc = ""
        if self.file and self.line_num:
            loc = f" ({self.file}:{self.line_num})"
        elif self.line_num:
            loc = f" (line {self.line_num})"
        return f"{prefix} {self.level}: {self.message}{loc}"


def _first_line(lines: List[str], pattern: str) -> int | None:
    rx = re.compile(pattern)
    for i, line in enumerate(lines):
        if rx.search(line):
            return i + 1
    return None


def _strip_cpp_comments(source: str) -> str:
    """Blank out // and /* */ comments while keeping line numbers stable."""
    source = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def check_device_code(source: str, name: str, trial_form: bool) -> List[ValidationError]:
    """Checks that apply to SYCL device/host C++ code, wherever it lives."""
    errors: List[ValidationError] = []
    code = _strip_cpp_comments(source)
    lines = code.split("\n")

    def add(level, msg, pattern=None, line=None):
        if line is None and pattern is not None:
            line = _first_line(lines, pattern)
        errors.append(ValidationError(level, msg, line, name))

    uses_sycl = bool(re.search(r"#include\s*<sycl/sycl\.hpp>|sycl::", code))
    uses_tla = bool(re.search(r"#include\s*[<\"](cute|cutlass)/", code))
    if not uses_sycl and not uses_tla:
        return errors  # e.g. a pure binding header; nothing SYCL-specific to check

    # 1. Queue: must be PyTorch's current XPU stream (references/sycl/sycl-implementation.md)
    if re.search(r"\bsycl::queue\s+\w+\s*(;|\{|\()|\bsycl::queue\s*\(|\bsycl::queue\s*\{", code):
        add(
            "ERROR",
            "A new sycl::queue is constructed. Submit to PyTorch's queue instead: "
            "c10::xpu::getCurrentXPUStream().queue(). A private queue breaks stream ordering "
            "with surrounding PyTorch ops.",
            r"\bsycl::queue\s+\w+\s*(;|\{|\()|\bsycl::queue\s*[\(\{]",
        )
    elif uses_sycl and "getCurrentXPUStream" not in code and not uses_tla:
        add(
            "WARNING",
            "No c10::xpu::getCurrentXPUStream() found. Make sure kernels are submitted to "
            "PyTorch's current XPU queue.",
        )

    # 2. Host synchronization inside the op
    if re.search(r"\.wait\s*\(\s*\)|\.wait_and_throw\s*\(|\bcompat::wait\s*\(|syclcompat::wait\s*\(", code):
        add(
            "WARNING",
            "Host-side wait() found. Ops should not block the host; PyTorch's queue keeps "
            "ordering. Remove it unless the result is consumed on the host.",
            r"\.wait\s*\(\s*\)|\.wait_and_throw\s*\(|compat::wait\s*\(",
        )

    # 3. Sub-group-dependent code must pin the sub-group size
    subgroup_dependent = re.search(
        r"get_sub_group\s*\(|sub_group\b|select_from_group|shift_group_|permute_group|"
        r"joint_matrix|MMA_Atom|XE_\d+x\d+x\d+_",
        code,
    )
    if subgroup_dependent and "reqd_sub_group_size" not in code and not uses_tla:
        add(
            "WARNING",
            "Sub-group operations / matrix atoms used without [[sycl::reqd_sub_group_size(N)]]. "
            "Pin the sub-group size the code relies on (see sycl_dpas_requires_simd16).",
            r"get_sub_group\s*\(|joint_matrix|select_from_group|shift_group_",
        )

    # 4. Reduction scope reminder
    if re.search(r"reduce_over_group\s*\(\s*[\w.]*get_sub_group\s*\(", code):
        add(
            "INFO",
            "Sub-group reduce_over_group only combines one sub-group. Make sure a single "
            "sub-group owns the entire reduction (sycl_use_subgroup_reduction_not_slm).",
            r"reduce_over_group\s*\(\s*[\w.]*get_sub_group",
        )

    # 5. Kernel-name isolation for trials (baseline and trials share one process)
    if trial_form and re.search(r"parallel_for\s*<|single_task\s*<", code) and not re.search(
        r"\bnamespace\s+\w+\s*\{", code
    ):
        add(
            "WARNING",
            "Named SYCL kernels are not inside a namespace. Put kernel name types in a "
            "namespace unique to this trial; a unique load_inline module name alone does not "
            "isolate device kernels.",
            r"parallel_for\s*<|single_task\s*<",
        )

    # 6. SLM budget reminder
    if "local_accessor" in code or "group_local_memory" in code:
        add(
            "INFO",
            "Local memory (SLM) used. Check stages * tile bytes + padding against the "
            "per-work-group limit (sycl_slm_size_limits).",
            r"local_accessor|group_local_memory",
        )

    # 7. Host-side input checks
    if "TORCH_CHECK" not in code and re.search(r"torch::Tensor|at::Tensor", code):
        add(
            "WARNING",
            "No TORCH_CHECK found. Check device (is_xpu()), dtype, contiguity and shapes "
            "on the host before launching.",
        )

    # 8. SYCL-TLA: guard unsupported configurations
    if uses_tla and "GemmUniversalAdapter" in code and "can_implement" not in code:
        add(
            "WARNING",
            "GemmUniversalAdapter used without can_implement(). Check it and fail with "
            "TORCH_CHECK instead of launching an unsupported configuration.",
            r"GemmUniversalAdapter",
        )

    # 9. fp16/bf16 accumulation reminder for DPAS atoms
    if re.search(r"XE_\d+x\d+x\d+_F16|XE_\d+x\d+x\d+_BF16", code):
        add(
            "INFO",
            "MMA atom accumulates in reduced precision. Confirm the reference contract allows "
            "it (sycl_fp32_accumulator_required).",
            r"XE_\d+x\d+x\d+_(F16|BF16)",
        )

    return errors


def check_trial_python(source: str, name: str) -> List[ValidationError]:
    """Checks for the load_inline trial form."""
    errors: List[ValidationError] = []
    lines = source.split("\n")

    try:
        ast.parse(source)
    except SyntaxError as e:
        errors.append(ValidationError("ERROR", f"Python syntax error: {e.msg}", e.lineno, name))
        return errors

    if "load_inline" not in source:
        errors.append(
            ValidationError(
                "ERROR",
                "No torch.utils.cpp_extension.load_inline call found. SYCL trial files compile "
                "their device code with load_inline(..., sycl_sources=...).",
                None,
                name,
            )
        )
    elif "sycl_sources" not in source:
        errors.append(
            ValidationError(
                "ERROR",
                "load_inline is called without sycl_sources=. Device code passed via "
                "cpp_sources is compiled by the host compiler, not as SYCL.",
                _first_line(lines, r"load_inline\s*\("),
                name,
            )
        )

    if "<arch from scripts/xpu_device_info.py>" in source:
        errors.append(
            ValidationError(
                "ERROR",
                "TORCH_XPU_ARCH_LIST placeholder not filled in. Use the entry reported by "
                "scripts/xpu_device_info.py for this device.",
                _first_line(lines, r"<arch from scripts/xpu_device_info.py>"),
                name,
            )
        )
    if "TORCH_XPU_ARCH_LIST" not in source:
        errors.append(
            ValidationError(
                "WARNING",
                "TORCH_XPU_ARCH_LIST is not set before load_inline. Set it to this device's "
                "entry from scripts/xpu_device_info.py; the default AOT-compiles for every "
                "architecture the PyTorch build knows.",
                None,
                name,
            )
        )

    if re.search(r"\bfrom\s+\.|\bimport\s+\w+_pytorch\b|\bfrom\s+\w+_pytorch\s+import", source):
        errors.append(
            ValidationError(
                "ERROR",
                "Trial imports from a sibling file. Trials must be self-contained: copy "
                "get_inputs(), get_init_inputs() and constants into this file.",
                _first_line(lines, r"\bfrom\s+\.|_pytorch\b"),
                name,
            )
        )

    if "class Model" not in source:
        errors.append(
            ValidationError(
                "ERROR", "No `class Model` found (required by scripts/benchmark.py).", None, name
            )
        )
    else:
        for fn in ("get_inputs", "get_init_inputs"):
            if f"def {fn}" not in source:
                errors.append(
                    ValidationError(
                        "WARNING",
                        f"Model class found but no {fn}() function. This is required by the "
                        "benchmark harness (ai-bench).",
                        None,
                        name,
                    )
                )

    for i, line in enumerate(lines):
        if ".item()" in line or ".cpu()" in line:
            if "def forward" in "\n".join(lines[max(0, i - 20) : i]):
                errors.append(
                    ValidationError(
                        "WARNING",
                        "Device-to-host sync (.item()/.cpu()) inside forward().",
                        i + 1,
                        name,
                    )
                )
                break

    errors.extend(check_device_code(source, name, trial_form=True))
    return errors


def check_shipping_sources(files: List[Path]) -> List[ValidationError]:
    """Checks for kernel-builder (shipping-form) C++ sources."""
    errors: List[ValidationError] = []
    for f in files:
        source = f.read_text()
        lines = _strip_cpp_comments(source).split("\n")
        if re.search(r"PYBIND11_MODULE|#include\s*<pybind11", source):
            errors.append(
                ValidationError(
                    "ERROR",
                    "pybind11 used. kernel-builder kernels bind ops with TORCH_LIBRARY_EXPAND "
                    "in torch-ext/torch_binding.cpp.",
                    _first_line(lines, r"PYBIND11_MODULE|pybind11"),
                    f.name,
                )
            )
        if (
            re.search(r"^\s*TORCH_LIBRARY_EXPAND\s*\(", source, re.M)
            and "kXPU" not in source
        ):
            errors.append(
                ValidationError(
                    "WARNING",
                    "TORCH_LIBRARY_EXPAND without a torch::kXPU impl. Register the XPU "
                    "implementation (guard with #if defined(XPU_KERNEL) in mixed-backend kernels).",
                    _first_line(lines, r"TORCH_LIBRARY_EXPAND"),
                    f.name,
                )
            )
        errors.extend(check_device_code(source, f.name, trial_form=False))
    return errors


def validate(path: Path) -> List[ValidationError]:
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix in CPP_SUFFIXES)
        if not files:
            return [ValidationError("ERROR", f"No C++/SYCL sources found under {path}")]
        errors = check_shipping_sources(files)
    elif path.suffix == ".py":
        errors = check_trial_python(path.read_text(), path.name)
    elif path.suffix in CPP_SUFFIXES:
        errors = check_shipping_sources([path])
    else:
        return [ValidationError("ERROR", f"Unsupported file type: {path.suffix}")]

    if not any(e.level in ("ERROR", "WARNING") for e in errors):
        errors.append(ValidationError("INFO", "No critical issues found! ✓"))
    return errors


def print_validation_results(errors: List[ValidationError], path: Path) -> int:
    print(f"\n{'=' * 70}")
    print(f"SYCL validation: {path.name}")
    print(f"{'=' * 70}\n")

    for level, title in (("ERROR", "ERRORS (must fix):"), ("WARNING", "WARNINGS (should review):"), ("INFO", "INFO:")):
        items = [e for e in errors if e.level == level]
        if items:
            print(title)
            for e in items:
                print(f"  {e}")
            print()

    n_err = sum(e.level == "ERROR" for e in errors)
    n_warn = sum(e.level == "WARNING" for e in errors)
    if n_err:
        print(f"Status: ❌ FAILED ({n_err} errors)")
        return 1
    if n_warn:
        print(f"Status: ⚠️  PASSED with warnings ({n_warn} warnings)")
    else:
        print("Status: ✅ PASSED")
    print("Note: static checks only — compile and run scripts/benchmark.py for correctness.")
    return 0


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_sycl.py <trial.py | kernel.cpp | kernel_dir>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: not found: {path}")
        sys.exit(1)
    sys.exit(print_validation_results(validate(path), path))


if __name__ == "__main__":
    main()
