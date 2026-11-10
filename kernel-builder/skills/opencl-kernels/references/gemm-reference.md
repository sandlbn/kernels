# Reference links

OpenCL-only sources for the extensions and instructions used by the GEMM kernels in this skill.

- [cl_intel_subgroup_matrix_multiply_accumulate](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html)
  — normative definition of the DPAS builtins, their `M`/`N`/`K` constraints, and operand layouts.
- [IGC DPAS instruction documentation](https://github.com/intel/intel-graphics-compiler/blob/6ecd10bc2d721e79d8efec39c62b740bded8aa95/documentation/visa/instructions/DPAS.md)
  — the underlying vISA instruction the builtins lower to; useful when reading generated assembly.
- [OpenCL C specification](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html)
  — vector types, `as_type()` reinterpretation rules, and address-space semantics.
- [Khronos OpenCL registry](https://registry.khronos.org/OpenCL/) — index of all extension specs,
  including the `cl_intel_subgroups*` family.
