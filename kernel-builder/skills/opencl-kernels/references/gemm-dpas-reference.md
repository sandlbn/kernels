# DPAS reference for OpenCL GEMM

DPAS (Dot Product Accumulate Systolic, marketed as XMX) is reached from OpenCL C through the
subgroup-collective builtins of the `cl_intel_subgroup_matrix_multiply_accumulate` extension. One
call computes, for the whole subgroup:

```
result[M][N] = a[M][K] * b[K][N] + acc[M][N]
```

This document covers only the OpenCL C interface. For device topology and peak-throughput figures
see [architecture.md](architecture.md).

## 1. Extensions and subgroup size

```c
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable      // block_read_us
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
```

Check `CL_DEVICE_EXTENSIONS` at runtime before selecting a DPAS kernel. Calling the builtins on a
device without the extension, or with a subgroup size that does not match the signature, is
undefined behaviour.

Annotate every DPAS kernel with the subgroup size it was written for:

```c
__attribute__((intel_reqd_sub_group_size(16)))
```

## 2. Dimensions

| Parameter | Meaning | Allowed values |
|-----------|---------|----------------|
| `M` | Result rows per call; vector width of `a`, `acc` and the result | 1, 2, 4, 8 |
| `N` | Result columns; equal to the subgroup size | 8 or 16 |
| `K` | Reduction depth per call; encoded in the builtin name | 16, 32 or 64 by type |

`N` is not a free parameter: it *is* the subgroup size, and which set of signatures exists depends
on the device's minimum subgroup size. `K` is fixed per data type so that one column of `b` is
always exactly 256 bits.

## 3. Data-type variants

| Builtin | `a` type | `b` type | Accumulator / result | `K` |
|---------|----------|----------|----------------------|-----|
| `intel_sub_group_f16_f16_matrix_mad_k16` | fp16 | fp16 | fp32, or fp16 when `N`=16 | 16 |
| `intel_sub_group_bf16_bf16_matrix_mad_k16` | bf16 | bf16 | fp32, or bf16 (`short`) | 16 |
| `intel_sub_group_i8_i8_matrix_mad_k32` | int8 | int8 | int32 | 32 |
| `intel_sub_group_i8_u8_matrix_mad_k32` | int8 | uint8 | int32 | 32 |
| `intel_sub_group_u8_i8_matrix_mad_k32` | uint8 | int8 | int32 | 32 |
| `intel_sub_group_u8_u8_matrix_mad_k32` | uint8 | uint8 | int32 | 32 |
| `intel_sub_group_i4_i4_matrix_mad_k64` | int4 | int4 | int32 | 64 |

fp16 and bf16 values travel through *signed* integer types. Reinterpret with `as_short()` /
`as_int()`; do not convert.

## 4. Signatures

The accumulator is always the **last** argument. The order is `(a, b, acc)`; passing `acc` first
fails to compile.

### fp16, subgroup size 16 (`N` = 16)

```c
float  intel_sub_group_f16_f16_matrix_mad_k16(short  a, int8 b, float  acc);  // M=1
float2 intel_sub_group_f16_f16_matrix_mad_k16(short2 a, int8 b, float2 acc);  // M=2
float4 intel_sub_group_f16_f16_matrix_mad_k16(short4 a, int8 b, float4 acc);  // M=4
float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc);  // M=8
```

The same shapes exist with `half` accumulators (`half`, `half2`, `half4`, `half8`). Prefer the fp32
accumulator: fp16 accumulation loses accuracy quickly as `K` grows.

### fp16, subgroup size 8 (`N` = 8)

```c
float  intel_sub_group_f16_f16_matrix_mad_k16(int  a, int8 b, float  acc);  // M=1
float2 intel_sub_group_f16_f16_matrix_mad_k16(int2 a, int8 b, float2 acc);  // M=2
float4 intel_sub_group_f16_f16_matrix_mad_k16(int4 a, int8 b, float4 acc);  // M=4
float8 intel_sub_group_f16_f16_matrix_mad_k16(int8 a, int8 b, float8 acc);  // M=8
```

### int8

```c
// subgroup size 16
int8 intel_sub_group_i8_i8_matrix_mad_k32(short8 a, int8 b, int8 acc);  // M=8
// subgroup size 8
int8 intel_sub_group_i8_i8_matrix_mad_k32(int8   a, int8 b, int8 acc);  // M=8
```

`M` = 1, 2 and 4 exist for int8 as well, with `short`/`short2`/`short4` (or `int`/`int2`/`int4`)
for `a` and `int`/`int2`/`int4` for the accumulator.

## 5. Operand layout

This is where DPAS kernels usually go wrong. The builtin is a subgroup collective, so each work
item holds a *slice* of each matrix, and the slice is defined by the work item's
`get_sub_group_local_id()` (`lid` below).

### `a` — `M×K`, distributed along `K`

Each work item owns all `M` rows but only `K/N` elements of each row, and those elements are
*consecutive along `K`*. They are packed into one component of `a` per row:

| Subgroup size | Elements per row per work item | Component type |
|---------------|-------------------------------|----------------|
| 16 | `K/16` | `short` (16 bits) |
| 8 | `K/8` | `int` (32 bits) |

So for fp16 with subgroup size 16, `K/N` = 1 and work item `lid` supplies `A[m][k0 + lid]`. For
int8 with subgroup size 16, `K/N` = 2 and it supplies `A[m][k0 + 2*lid]` and `A[m][k0 + 2*lid + 1]`,
low element in the low byte.

### `b` — `K×N`, distributed along `N`

Each work item owns exactly **one column** of the tile: all `K` elements of column `n0 + lid`.
That is 256 bits for every data type, which is why `b` is always `int8`. Within one component
(`uint`), elements are packed in increasing `K` order, lowest element in the least-significant bits.

### `acc` and the result — `M×N`, distributed along `N`

Each work item holds the `M` accumulator values of its own column. The result has the identical
layout, so it feeds directly into the next `K` iteration without any shuffling.

## 6. Packed (VNNI) layout for `B`

A row-major `B[K][N]` is the wrong layout for a subgroup block read: a block read hands work item
`lid` the value at `p[lid]`, but `b` needs `K` values *down a column*. Reading `B` row-major
therefore requires one strided load per element, as in the baseline kernel in
[gemm.md](gemm.md).

The fast path pre-packs `B` once so that a block read fills `b` directly. Pack `P = 32/bits`
elements that are adjacent along `K` — the same column — into one `uint`:

| `B` element type | `P` (elements per `uint`) | Packed shape |
|------------------|---------------------------|--------------|
| fp16 / bf16 | 2 | `uint[K/2][N]` |
| int8 | 4 | `uint[K/4][N]` |
| int4 | 8 | `uint[K/8][N]` |

Formally, with `Bp` of shape `uint[K/P][N]`:

```
Bp[kp][n] holds B[P*kp + 0][n] .. B[P*kp + P-1][n],
with B[P*kp + r][n] at bit offset r * (32/P)
```

A one-off transform kernel for fp16:

```c
__kernel void pack_b_vnni_f16(__global const half *B,  // [K][N] row-major
                              __global uint *Bp,       // [K/2][N]
                              const int K, const int N)
{
    const int n = get_global_id(0);   // 0 .. N-1
    const int kp = get_global_id(1);  // 0 .. K/2-1
    const ushort lo = as_ushort(B[(2 * kp) * N + n]);
    const ushort hi = as_ushort(B[(2 * kp + 1) * N + n]);
    Bp[kp * N + n] = (uint)lo | ((uint)hi << 16);
}
```

Because the packed row stride is `N` uints, fill `b` with eight single-`uint` block reads, one per
packed row, rather than one `block_read8`:

```c
int8 b;
for (int j = 0; j < 8; ++j)
    ((int *)&b)[j] = as_int(intel_sub_group_block_read(Bp + (k0 / P + j) * N + n0));
```

A single `intel_sub_group_block_read8(p)` only works when the packed tile is contiguous with a row
stride of exactly the subgroup size, since it returns component `j` of work item `lid` from
`p[j * sg_size + lid]`. Use it when `B` is stored tile-by-tile; use the eight separate reads when
the packed weights keep a full-width `N` stride.

## 7. fp16 GEMM with packed `B`

Each subgroup computes one 8×16 tile of `C`. Launch with global size `(N, M/8)` and local size
`(16, 1)`.

```c
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define SG 16
#define MT 8
#define KT 16

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_dpas_f16_packed(__global const half *A,  // [M][K] row-major
                                   __global const uint *Bp, // [K/2][N] VNNI packed
                                   __global float *C,       // [M][N] row-major
                                   const int M, const int N, const int K)
{
    const int n0 = get_group_id(0) * SG;
    const int m0 = get_group_id(1) * MT;
    const int lid = get_sub_group_local_id();

    float8 acc = (float8)(0.0f);

    for (int k0 = 0; k0 < K; k0 += KT) {
        short8 a;
        for (int m = 0; m < MT; ++m)
            ((short *)&a)[m] = as_short(A[(m0 + m) * K + k0 + lid]);

        int8 b;
        for (int j = 0; j < 8; ++j)
            ((int *)&b)[j] = as_int(intel_sub_group_block_read(Bp + (k0 / 2 + j) * N + n0));

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);
    }

    for (int m = 0; m < MT; ++m)
        C[(m0 + m) * N + n0 + lid] = ((float *)&acc)[m];
}
```

## 8. Reusing `a` across several accumulators

The `a` operand is the expensive one to gather, so amortize it: hold several `N` tiles per subgroup
and issue one DPAS per tile from the same `a`. This raises arithmetic intensity without extra `A`
traffic and is normally the first optimization after a correct kernel exists. Launch with global
size `(N/NB, M/8)` and local size `(16, 1)`.

```c
#define NB 2  // N tiles per subgroup -> each subgroup produces MT x (NB*SG)

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_dpas_f16_nb(__global const half *A,
                               __global const uint *Bp,
                               __global float *C,
                               const int M, const int N, const int K)
{
    const int n0 = get_group_id(0) * (SG * NB);
    const int m0 = get_group_id(1) * MT;
    const int lid = get_sub_group_local_id();

    float8 acc[NB];
    for (int t = 0; t < NB; ++t)
        acc[t] = (float8)(0.0f);

    for (int k0 = 0; k0 < K; k0 += KT) {
        short8 a;
        for (int m = 0; m < MT; ++m)
            ((short *)&a)[m] = as_short(A[(m0 + m) * K + k0 + lid]);

        for (int t = 0; t < NB; ++t) {
            int8 b;
            for (int j = 0; j < 8; ++j)
                ((int *)&b)[j] =
                    as_int(intel_sub_group_block_read(Bp + (k0 / 2 + j) * N + n0 + t * SG));
            acc[t] = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc[t]);
        }
    }

    for (int t = 0; t < NB; ++t)
        for (int m = 0; m < MT; ++m)
            C[(m0 + m) * N + n0 + t * SG + lid] = ((float *)&acc[t])[m];
}
```

Increase `NB` (and optionally the number of `M` tiles) until the build log reports register
spilling, then step back one notch.

## 9. int8 GEMM

`K` is 32, so `a` carries two int8 values per row per work item, and packed `B` is `uint[K/4][N]`.

```c
#define SG 16
#define MT 8
#define KT 32

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_dpas_i8(__global const char *A,  // [M][K] row-major, signed int8
                           __global const uint *Bp, // [K/4][N] VNNI packed
                           __global int *C,         // [M][N] row-major, int32
                           const int M, const int N, const int K)
{
    const int n0 = get_group_id(0) * SG;
    const int m0 = get_group_id(1) * MT;
    const int lid = get_sub_group_local_id();

    int8 acc = (int8)(0);

    for (int k0 = 0; k0 < K; k0 += KT) {
        short8 a;
        for (int m = 0; m < MT; ++m) {
            char2 v = (char2)(A[(m0 + m) * K + k0 + 2 * lid],
                              A[(m0 + m) * K + k0 + 2 * lid + 1]);
            ((short *)&a)[m] = as_short(v);
        }

        int8 b;
        for (int j = 0; j < 8; ++j)
            ((int *)&b)[j] = as_int(intel_sub_group_block_read(Bp + (k0 / 4 + j) * N + n0));

        acc = intel_sub_group_i8_i8_matrix_mad_k32(a, b, acc);
    }

    for (int m = 0; m < MT; ++m)
        C[(m0 + m) * N + n0 + lid] = ((int *)&acc)[m];
}
```

## 10. Split DPAS (DPASW)

`cl_intel_subgroup_split_matrix_multiply_accumulate` adds variants where two neighbouring subgroups
each supply half of the `a` rows and share `b`, which reduces register-file bandwidth pressure:

```c
float8 intel_sub_group_f16_f16_split_matrix_mad_k16(int4 a, int8 b, float8 acc);  // M=8
int8   intel_sub_group_i8_i8_split_matrix_mad_k32(int4 a, int8 b, int8 acc);      // M=8
```

These require subgroup size 8 and are not present on every device — query the extension string
separately from the non-split extension.

## 11. Loading rules

- `intel_sub_group_block_read`/`_read8` take a `uint` pointer that must be 4-byte aligned; block
  writes require 16-byte alignment.
- `intel_sub_group_block_read_us*` reads 16-bit elements and is convenient for fp16 tiles that are
  not pre-packed into `uint`.
- For `a`, either let each work item load its own slice by stride (as above) or broadcast one work
  item's value with `sub_group_broadcast` when the same row slice is needed by everyone.
- Do not put a block read behind divergent control flow: every work item in the subgroup must
  execute it.

## 12. Checklist

1. `intel_reqd_sub_group_size(N)` is present and `N` matches the signature and the launch geometry.
2. Argument order is `(a, b, acc)`.
3. `a` component count equals `M`; `b` is `int8`; accumulator width equals `M`.
4. fp16/bf16 go through `as_short()` / `as_int()`, not conversions.
5. Every work item in the subgroup reaches the builtin — no divergent DPAS.
6. Packed `B` uses `P = 32/bits` elements per `uint`, adjacent along `K`, same column.
7. Accumulate in fp32 (or int32) and convert only when storing.
8. Extension presence is queried at runtime and a scalar fallback exists.
9. The `K` loop feeds the previous result back in as `acc`.
10. Build log checked for register spilling.

## 13. Specification links

- [cl_intel_subgroup_matrix_multiply_accumulate](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html)
- [cl_intel_subgroup_split_matrix_multiply_accumulate](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_split_matrix_multiply_accumulate.html)
- [cl_intel_subgroups](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html)
- [cl_intel_subgroups_short](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups_short.html)

The fp16 and int8 kernels above were checked against a reference matmul on an Intel Arc B580
(subgroup size 16). The subgroup-size-8 signatures come from the extension specification and are
selected by the device's minimum subgroup size.
