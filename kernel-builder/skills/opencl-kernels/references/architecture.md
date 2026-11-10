# Intel Xe architecture for kernel developers

Use this skill to pick realistic launch parameters and roofline ceilings. Always confirm the exact SKU with `clinfo` or vendor docs before committing to numbers.

## Common Xe2 parameters (reference SKUs)

| SKU / codename | Xe cores | EUs / Xe core | Threads / EU | Subgroup sizes | Registers / EU |
|----------------|----------|---------------|--------------|----------------|----------------|
| Lunar Lake (LNL) | 8 | 8 | 8 | 16 or 32 | 256 × 256 B |
| Battlemage B580 (BMG) | 20 | 8 | 8 | 16 or 32 | 256 × 256 B |
| Panther Lake (PTL) | 12 | 8 | 8 | 16 or 32 | 256 × 256 B |


Shared local memory is a query, not a constant: read `CL_DEVICE_LOCAL_MEM_SIZE`. Xe2 parts report
128 KiB per work-group (verified: 131072 bytes on Arc B580). The 32 KiB figure quoted in older Xe
material does not apply to Xe2, so sizing SLM tiles against it leaves most of the capacity unused.

Typical OpenCL build flags for compute kernels on Intel GPU:

- `-cl-std=CL3.0 -cl-mad-enable`
- Enable DPAS when using matrix MAD builtins: `cl_intel_subgroup_matrix_multiply_accumulate`
- Enable subgroup block reads when needed: `cl_intel_subgroups`, `cl_intel_subgroups_short`

## How to use in kernel work

1. **Subgroup size**: Match `intel_sub_group_*` and DPAS `N` to the required subgroup size (8 or 16). Use `intel_reqd_sub_group_size(N)` when the kernel depends on a fixed width.
2. **Occupancy**: Register spill (`SPILL` in cliloader) usually hurts more than micro-optimizing FLOPs on memory-bound kernels.
3. **SLM**: Budget against the queried `CL_DEVICE_LOCAL_MEM_SIZE` — tiling must account for double-buffering and bank conflicts.
4. **Bandwidth**: Treat DRAM peak as a config input (e.g. ~100 GB/s class for LNL iGPU, ~456 GB/s for B580). Measure, do not assume.

## Matrix engine (XMX / DPAS)

Xe cores include a systolic matrix engine, exposed to OpenCL through the
`cl_intel_subgroup_matrix_multiply_accumulate` extension. Treat its presence as a device property
and query it rather than inferring it from the SKU name:

```c
// CL_DEVICE_EXTENSIONS must contain:
//   cl_intel_subgroup_matrix_multiply_accumulate
// The split (DPASW) variant is a separate extension string and is not on every device.
```

Two hardware-side consequences for kernel sizing:

- The instruction's `N` equals the subgroup size (8 or 16), so the matrix engine constrains the
  launch geometry, not just the arithmetic. The per-data-type `M`/`N`/`K` values are instruction
  properties and live in [gemm.md](gemm.md).
- Matrix-engine peak is far above the fp32 FMA peak, which makes a DPAS kernel compute-bound at a
  much higher arithmetic intensity than a scalar one. Do not take a marketing TFLOPS number as the
  target: measure a micro-benchmark that keeps the matrix engine saturated on the device in front of
  you, and treat that measurement as the practical ceiling.

## Roofline quick check

For a kernel with arithmetic intensity `AI` (FLOP/byte):

```
attainable = min(peak_compute, AI × peak_bandwidth)
efficiency = achieved / attainable
```

Classify before optimizing:

- **Compute-bound** (GEMM, large DPAS tiles): chase XMX/DPAS utilization and tile sizes.
- **Memory-bound** (GEMV, decode, attention with small batch): chase coalesced/block reads, fusion, and fewer bytes moved.

## Additional resources

- DPAS builtins, operand layouts, and GEMM tiling: [gemm.md](gemm.md)
