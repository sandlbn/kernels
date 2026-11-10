# GEMM kernels in OpenCL

This skill covers OpenCL C kernels only.

Device topology, SLM capacity, bandwidth figures, and roofline ceilings are deliberately not
listed here. Get them from [architecture.md](architecture.md) or by
querying the device, so that a kernel written with this skill is not tied to one SKU.

## Workflow

1. **Read what already exists.** If there is a current GEMM kernel, read it first and record its
   tiling, subgroup size, and data layout. An existing kernel encodes constraints (layouts,
   alignment guarantees) that are expensive to rediscover.
2. **Write down the contract** before touching a kernel (see below).
3. **Get a correct scalar baseline** and keep it. It is the oracle for every later optimization.
4. **Pick the compute path**: DPAS/XMX or scalar FMA (see "Choosing the compute path").
5. **Implement** the tiled kernel.
6. **Verify correctness** against the baseline on the edge shapes, not just one size.
7. **Measure** and compare with the roofline; iterate while the gap is large.
8. **Document** what changed and what it bought.

## The GEMM contract

GEMM computes `C = A * B + D`, where `A` is `M×K`, `B` is `K×N`, `C` is `M×N`, and the bias `D` is
optional. Performance is decided almost entirely by memory access patterns, data types, and
tiling — not by the arithmetic itself.

Pin down every row before writing code, because each one changes the kernel structure:

| Item | Why it matters |
|------|----------------|
| Shapes `M`, `N`, `K` and which are dynamic | `M=1` is a different kernel (GEMV, memory-bound) |
| `A` type and layout | Decides how the DPAS `a` operand is gathered |
| `B` type and layout | Decides whether `B` needs repacking and whether dequantization is in the loop |
| `C`/`D` type | Accumulate in fp32 even when the output is fp16 |
| Alignment / padding guarantees | Decides whether block reads are legal and whether bounds checks are needed |
| Performance target | A concrete number, e.g. 80% of the roofline attainable value |

A typical LLM-inference contract is: `A` fp16, `B` fp16 or group/channel-quantized int4/int8,
accumulate fp32, dimensions padded to a multiple of 16. Confirm it rather than assuming it.

## Choosing the compute path

Large-`M` GEMM is compute-bound, so the goal is to keep the matrix engine busy and hide memory
latency behind it. Small-`M` GEMM (`M` of 1 to a few rows) is memory-bound and dominated by the
cost of streaming `B`; DPAS cannot help much there because most of the `M×N` result tile is wasted.

| Situation | Path |
|-----------|------|
| `M` large enough to fill a DPAS tile (>= 8), fp16/bf16/int8 data | DPAS builtins |
| `M` very small (decode, GEMV-like) | Scalar/vector FMA, focus on bytes moved |
| Data type unsupported by DPAS, or extension missing | Scalar FMA with `-cl-mad-enable` |

Always query `CL_DEVICE_EXTENSIONS` for `cl_intel_subgroup_matrix_multiply_accumulate` and provide
a scalar fallback; calling the builtins on a device without the extension is undefined.

## DPAS essentials

DPAS (also marketed as XMX) exposes the systolic matrix engine through *subgroup-collective*
builtins from `cl_intel_subgroup_matrix_multiply_accumulate`. One call computes
`result[M][N] = a[M][K] * b[K][N] + acc[M][N]` for the whole subgroup at once, so every work item
in the subgroup must reach the call.

```c
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
```

| Parameter | Meaning | Allowed values |
|-----------|---------|----------------|
| `M` | Result rows per call; also the accumulator vector width | 1, 2, 4, 8 |
| `N` | Result columns; **must equal the subgroup size** | 8 or 16 |
| `K` | Reduction depth per call; fixed by the data type | 16 (fp16/bf16), 32 (int8), 64 (int4) |

Because `N` is the subgroup size, annotate the kernel with
`__attribute__((intel_reqd_sub_group_size(N)))` and choose the signature to match.

The two most used builtins:

```c
// fp16 x fp16 -> fp32, K=16, subgroup size 16, M=8
float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc);

// int8 x int8 -> int32, K=32, subgroup size 16, M=8
int8   intel_sub_group_i8_i8_matrix_mad_k32(short8 a, int8 b, int8 acc);
```

**The argument order is `(a, b, acc)`** — the accumulator comes last. Passing it first does not
compile, and it is the single most common mistake when writing these kernels.

Operand ownership within the subgroup is what makes or breaks the kernel:

- **`a`** — each work item holds `M` rows and contributes `K/N` elements of each row, packed into
  one 32-bit `int` (subgroup size 8) or one 16-bit `short` (subgroup size 16) per row. The vector
  width of `a` is therefore `M`.
- **`b`** — each work item holds exactly **one column** of the `K×N` tile: `K` elements, always
  256 bits total, hence `int8` for every data type.
- **`acc`** — each work item holds one column of `M` accumulator values; the result has the same
  layout, so it feeds straight back into the next `K` step.

fp16 and bf16 operands are passed through *signed* types (`short`/`int`), so reinterpret with
`as_short()` / `as_int()` rather than converting.

### A minimal correct fp16 DPAS GEMM

This kernel is verified against a reference matmul and takes plain row-major `A` and `B`. Each
subgroup produces one 8×16 tile of `C`. Launch it with global size `(N, M/8)` and local size
`(16, 1)`, with `M`, `N`, `K` multiples of 8, 16, and 16 respectively.

```c
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define SG 16  // subgroup size == DPAS N
#define MT 8   // DPAS M
#define KT 16  // DPAS K for fp16

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_dpas_f16(__global const half *A,  // [M][K] row-major
                            __global const half *B,  // [K][N] row-major
                            __global float *C,       // [M][N] row-major
                            const int M, const int N, const int K)
{
    const int n0 = get_group_id(0) * SG;
    const int m0 = get_group_id(1) * MT;
    const int lid = get_sub_group_local_id();

    float8 acc = (float8)(0.0f);

    for (int k0 = 0; k0 < K; k0 += KT) {
        // a: this work item owns element k0+lid of each of the MT rows
        short8 a;
        for (int m = 0; m < MT; ++m)
            ((short *)&a)[m] = as_short(A[(m0 + m) * K + k0 + lid]);

        // b: this work item owns column n0+lid, KT deep, K-adjacent pairs per int
        int8 b;
        for (int j = 0; j < 8; ++j) {
            half2 pair = (half2)(B[(k0 + 2 * j) * N + n0 + lid],
                                 B[(k0 + 2 * j + 1) * N + n0 + lid]);
            ((int *)&b)[j] = as_int(pair);
        }

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);
    }

    for (int m = 0; m < MT; ++m)
        C[(m0 + m) * N + n0 + lid] = ((float *)&acc)[m];
}
```

Treat this as the correctness starting point, not the fast version: it loads `B` one element at a
time. The fast path stores `B` in the packed layout that lets a subgroup block read fill `b`
directly, which is covered with working code in [gemm-dpas-reference.md](gemm-dpas-reference.md)
together with the full signature list, int8/int4 packing, and the split (DPASW) variant.

## Tiling and memory

Once DPAS is in place, the kernel is limited by how fast operands reach the registers.

- **Reuse the `a` operand.** Give each work item several `N` tiles (several accumulators) that
  share one loaded `a`. This is the cheapest large win: it multiplies arithmetic intensity without
  extra `A` traffic.
- **Stage tiles in SLM** when the same `A` or `B` tile is needed by several subgroups in the
  work-group, and size tiles against `CL_DEVICE_LOCAL_MEM_SIZE` instead of a hardcoded number.
  Budget for double buffering, and pad SLM rows to avoid bank conflicts.
- **Double buffer** the SLM (or register) tiles so the next `K` step is being loaded while the
  current one is fed to DPAS.
- **Use subgroup block reads** (`intel_sub_group_block_read*`) for contiguous operand tiles rather
  than per-work-item scalar loads. Respect the alignment rules: 4-byte for block reads, 16-byte for
  block writes.
- **Keep accesses coalesced.** Adjacent work items should touch adjacent addresses; a transposed or
  strided access on the fast-moving dimension costs more than any arithmetic saving.
- **Unroll the `K` loop** (`#pragma unroll`) so independent DPAS calls overlap, but watch for
  register spilling — spills usually cost more than the unroll gains. Check the build log or an
  offline compile for spill messages, and shrink the tile if they appear.
- **Minimize divergence.** Handle ragged edges with separate tail kernels or padded buffers instead
  of per-element bounds checks inside the hot loop.

## Quantized B operands

Weight-quantized GEMM (int4/int8 `B` with fp16 `A`) is common and changes the balance: unpacking
and dequantizing happens in the inner loop, so it can become the bottleneck instead of the MADs.

- Unpack int4 with bitwise ops on wide types (process 8 values per `uint`), never with per-element
  shifts driven by a data-dependent branch.
- Hoist scales and zero points out of the `K` loop; apply them once per group boundary, not per
  element.
- Prefer accumulating in the integer domain with `int8` DPAS (or `dot()`-style products) and
  dequantizing the accumulator afterwards, when the quantization scheme allows it.
- Note the group size relative to the DPAS `K` step: if a quantization group is smaller than `K`,
  the accumulator has to be drained and rescaled at group boundaries.

## Verification

- Compare against an independent reference (host matmul in fp64 or fp32) using a relative
  tolerance; fp16 inputs with fp32 accumulation typically land within ~1e-2 relative error, so an
  exact match is the wrong criterion. Integer paths must match exactly.
- Test the shapes that break kernels: `K` not a multiple of the DPAS `K`, `M`/`N` smaller than one
  tile, non-square, and the largest supported size.
- Include values that expose accumulation error (large `K`, mixed magnitudes), not just small
  random data.

## Measurement and reporting

- Warm up, then take the median of several timed runs using OpenCL profiling events.
- Report achieved GFLOP/s using `2*M*N*K` FLOPs, and state it as a fraction of the attainable
  roofline value `min(peak_compute, AI × peak_bandwidth)`, taking the ceilings from
  [architecture.md](architecture.md).
- Sweep several shapes and data types; a kernel tuned for one shape often regresses on others.

Document the optimizations applied, before-and-after numbers, and the resulting roofline
percentage, so the next change starts from evidence.

## Build options

Start with `-cl-std=CL3.0 -cl-mad-enable`. Add `-D` JIT constants for tile sizes and shapes so the
compiler can fully unroll the inner loops, and keep the generated source reproducible for
debugging.

## Additional resources

- Full DPAS reference with packed-`B` layouts, all signatures, and validated examples:
  [gemm-dpas-reference.md](gemm-dpas-reference.md)
- Specification links: [gemm-reference.md](gemm-reference.md)
- Device parameters and roofline ceilings: [architecture.md](architecture.md)

Keep explanations conversational. For complex concepts, use multiple analogies.
