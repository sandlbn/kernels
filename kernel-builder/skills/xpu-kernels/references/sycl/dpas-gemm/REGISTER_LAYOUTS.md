# DPAS Register Layouts

Read this before writing any kernel that does something to DPAS operands other
than load them from memory with a 2D block intrinsic — in particular any kernel
that **feeds one DPAS result into another DPAS** (attention, chained GEMMs,
fused epilogues) or **reduces across an accumulator** (softmax, layernorm,
row-max, row-sum).

The 2D block load intrinsics hide these layouts, which is why the GEMM examples
in `assets/` never mention them. The moment you build an operand yourself, they
are the whole problem.

---

## The mental model

`short8_dpas` / `int8_dpas` / `float8_dpas` are declared per work-item, but DPAS
is a **subgroup-collective** instruction. One `__spirv_SubgroupMatrixMultiplyAccumulateINTEL`
call consumes the operand registers of all 16 lanes together and produces one
matrix tile. So an operand has two indices:

- the **vector element index** inside the per-lane array (`0..7`), and
- the **lane index** within the subgroup (`0..15`).

Physically the registers are SIMD-width-major (structure-of-arrays): vector
element `i` of all 16 lanes is contiguous in the register file, then element
`i+1` follows. That is why the element index maps to the *row* of the tile and
the lane maps to the *column-ish* dimension — never the other way round.

---

## FP16 / BF16 layouts (`K_Dim = 16`, tile M=8, N=16, K=16)

Both fp16 and bf16 use identical register layouts; only the Operands flag
differs (`0xC00` vs `0x3000`).

### A operand — `short8_dpas`, 8 halves per lane

```
lane  = k    (the contracted index, 0..15)
elem  = m    (the row, 0..7)

rA[m]  ==  A[m][lane]
```

Each lane owns one **column** of the 8×16 A tile.

### B operand — `int8_dpas`, 8 dwords = 16 halves per lane (VNNI)

```
lane   = n   (the output column, 0..15)
dword j holds two halves packed along k:
    low  16 bits = B[2*j    ][lane]
    high 16 bits = B[2*j + 1][lane]
```

Each lane owns one **column** of the 16×16 B tile, with k-pairs packed into
dwords. This pairing is what "VNNI format" means, and it is what
`LoadTransformINTEL` and `LoadTransposeINTEL` produce for you.

### C / accumulator — `float8_dpas`, 8 floats per lane

```
lane  = n    (the output column, 0..15)
elem  = m    (the row, 0..7)

rC[m]  ==  C[m][lane]
```

Each lane owns one **column** of the 8×16 output tile.

> **INT8 (`K_Dim = 32`) is not covered here.** The packing differs (16 int8 per
> lane in A, four-way dword packing in B) and is not documented in this skill.
> If you need it, verify empirically with the identity probe below rather than
> assuming it generalises from the 16-bit case.

---

## The chaining rule: accumulator → next A operand

Compare the two layouts above:

| | element index | lane index |
|---|---|---|
| C accumulator of GEMM 1 | row `m` | column `n` |
| A operand of GEMM 2 | row `m` | contracted index `k` |

They are **the same shape**. So if GEMM 2 contracts over the dimension that was
GEMM 1's output columns — i.e. `n` of GEMM 1 *is* `k` of GEMM 2 — then feeding
the first result into the second is a **pure elementwise convert with no
cross-lane data movement at all**: no shuffles, no permutes, no SLM round-trip,
no barriers.

```cpp
// rC: float8_dpas from GEMM 1  (8 floats/lane, elem = row, lane = n)
// rA: short8_dpas for GEMM 2   (8 halves/lane, elem = row, lane = k)
short8_dpas rA;
#pragma unroll
for (int m = 0; m < 8; ++m)
    rA[m] = sycl::bit_cast<int16_t>(static_cast<sycl::half>(rC[m]));
```

`short8_dpas` is a vector of `int16_t`, so the half must be bit-cast, not
converted numerically — `rA[m] = (int16_t)rC[m]` silently truncates to an
integer and gives garbage.

**Preconditions.** This only holds when:

1. GEMM 2's `K_Dim` is 16 and GEMM 1's tile N is 16 — both are the lane
   dimension, so they must be the same width;
2. the quantity in the lane really is the same in both (GEMM 1's output column
   index and GEMM 2's contracted index refer to the same thing);
3. the second GEMM's A is the *first* operand. Chaining into the B operand is
   **not** elementwise — B needs VNNI k-pair packing along the vector index,
   which the accumulator does not have, so that direction does require data
   movement.

### Worked example: attention

```
S = Q @ K^T   tile: 8 queries × 16 keys, lane = key   (accumulator)
P = softmax(S)                                        (elementwise, see below)
O = P @ V     A operand: 8 queries × 16 keys, lane = key
```

`n` of the first GEMM and `k` of the second are both the key index, so each
16-key accumulator tile of `S` becomes one A operand tile of `P` by the loop
above. `V` is the B operand and comes from memory via `LoadTransformINTEL` as
usual. Head dim `D` splits into `D/16` output tiles, each accumulating over all
key tiles.

---

## Reductions across an accumulator (softmax, row-max, row-sum)

For a fixed row `m`, the 16 columns of the tile live in **16 different lanes at
the same vector element index**. So a per-row reduction over the output-column
dimension is a *sub-group* reduction, not a loop over the array:

```cpp
auto sg = item.get_sub_group();
float row_max[8];
#pragma unroll
for (int m = 0; m < 8; ++m) {
    float local = rC[0][m];                       // this lane's column
    for (int t = 1; t < n_tiles; ++t)             // more column-tiles, same row
        local = sycl::fmax(local, rC[t][m]);
    row_max[m] = sycl::reduce_over_group(sg, local, sycl::maximum<float>());
}
```

Two consequences that are easy to get wrong:

- You need **8 separate reductions**, one per row, because each lane holds 8
  different rows. A single `reduce_over_group` over the whole array collapses
  rows together.
- The result is already **broadcast to every lane** — no explicit broadcast or
  SLM staging is needed afterwards.

Conversely, a reduction along the *row* dimension (`m`) is a plain loop over the
vector index with no communication at all.

---

## Verifying a layout empirically instead of guessing

If you are unsure of a layout, do not reason about it and hope. Two probes:

**Identity probe (reveals the A layout).** Set B to the 16×16 identity, then
`C[m][n] = A[m][n]`. Run one DPAS, store the accumulator with
`BlockStoreINTEL`, and compare against the A tile you believe you built. Any
transposition or lane rotation shows up immediately, and the store path is
already known-good from the GEMM examples.

**Index probe (reveals any layout).** Fill the operand with a value that encodes
its position — e.g. `A[m][k] = m * 100 + k` — and multiply by the identity. The
stored tile tells you exactly which source element landed where.

Both are cheap, run in one kernel launch, and turn a layout bug from a
multi-hour debugging session into a single build.

---

## Register cost

Per-lane sizes are **not** what the register allocator sees. Each of these
occupies its per-lane size × the subgroup width:

| Operand | per lane | per thread (SIMD16) | GRFs (64 B) |
|---|---|---|---|
| `short8_dpas` (A, fp16) | 16 B | 256 B | 4 |
| `int8_dpas` (B, VNNI) | 32 B | 512 B | 8 |
| `float8_dpas` (C, fp32) | 32 B | 512 B | 8 |

See [TILING.md](TILING.md) for the full budget calculation and what to do when
the compiler reports spills.
