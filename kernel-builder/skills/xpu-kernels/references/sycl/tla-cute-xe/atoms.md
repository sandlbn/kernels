# Intel Xe CuTe Atom reference

Detailed companion to `sycl-tla-cute-xe`. Reflects the Xe atom headers:
`<cute/arch/mma_xe.hpp>`, `<cute/arch/copy_xe_2d.hpp>`,
`<cute/atom/copy_traits_xe_2d.hpp>`.

## XE_DPAS_TT — MMA atom

```cpp
template <int M, class TypeD, class TypeA, class TypeB = TypeA, class TypeC = TypeD>
struct XE_DPAS_TT;
```
- `M` = output rows, hardware supports **1–8**.
- N is **fixed 16** for all CUTLASS-supported Xe GPUs.
- `K = 256 / max(sizeof_bits(TypeA), sizeof_bits(TypeB))`
  (BF16→16, TF32→8, INT8→32, INT4→64).
- Computes `D[M×16] += A[M×K] × B[K×16]`.

Type aliases (`cute::dpas_type::`): `f`=float, `tf32`=tfloat32_t, `bf`=bfloat16_t,
`hf`=half_t, `ud`=uint32_t, `d`=int32_t, `u8`=uint8_t, `s8`=int8_t, `u4`=uint4_t,
`s4`=int4_t. You may also pass the underlying C++ types directly.

### Supported (TypeD, TypeA, TypeB, TypeC) combinations
Floating point:
- TF32×TF32→FP32: `<M, f, tf32, tf32, f>`
- BF16×BF16→FP32: `<M, f, bf, bf, f>` (also `bf` accum-in/out variants)
- FP16×FP16→FP32: `<M, f, hf, hf, f>` (also `hf` accum-in/out variants)

Integer (accumulate to S32/U32):
- U8×U8→U32 `<M, ud, u8, u8, ud>`; U8×U8→S32 `<M, d, u8, u8, d>`
- Any S8/U8 × S8/U8 → S32 `<M, d, {s8,u8}, {s8,u8}, d>`
- 8-bit × 4-bit and 4-bit × 8-bit and 4-bit × 4-bit combinations → S32/U32
  (e.g. `<M, d, s8, s4, d>`, `<M, d, u4, u4, d>`).

Wrap before use: `using Atom = cute::MMA_Atom<cute::XE_DPAS_TT<8, float, cute::bfloat16_t>>;`

## 2D block copy atoms

```cpp
template <int Bits, int Height, int Width, int BlockWidth = Width> struct XE_LOAD_2D;
template <int Bits, int Height, int Width, int BlockWidth = Width> struct XE_LOAD_2D_VNNI;
template <int Bits, int Height, int Width> struct XE_LOAD_2D_TRANSPOSE;
template <int Bits, int Height, int Width> struct XE_PREFETCH_2D;
template <int Bits, int Height, int Width> struct XE_STORE_2D;
```
- `Bits` = element bit-width of the memory op (8/16/32/64), **not** the C++ type —
  one atom covers any two types of the same width.
- `Height` = rows (stride dimension), `Width` = contiguous columns.
- `BlockWidth` sub-tiles the width into register blocks (default = Width).
- `XE_LOAD_2D::PREFETCH` aliases `XE_PREFETCH_2D<Bits,Height,Width>`.

### Per-atom constraints
| Atom | Bits | Extra |
|------|------|-------|
| `XE_LOAD_2D` | 8/16/32/64 | row-major load (A) |
| `XE_LOAD_2D_VNNI` | **8 or 16 only** | VNNI-packed load for B (BF16/FP16/INT8) |
| `XE_LOAD_2D_TRANSPOSE` | **32 or 64 only** | Width ≤ 8; 64-bit ⇒ Height=8, Width<4 |
| `XE_PREFETCH_2D` | 8/16/32/64 | no register output |
| `XE_STORE_2D` | 8/16/32/64 | **Height ≤ 8** (loads allow ≤ 32) |

### Copy-variant selection (which atom for which operand)
| Operand | Memory layout | Atom |
|---------|---------------|------|
| A | row-major | `XE_LOAD_2D` (LD_N) |
| A | column-major | `XE_LOAD_2D_TRANSPOSE` (LD_T) |
| B | row-major (16/8-bit) | `XE_LOAD_2D_VNNI` (LD_V) |
| B | row-major (32-bit) | `XE_LOAD_2D` (LD_N) — LD_V unavailable for U32 |
| B | column-major | `XE_LOAD_2D_TRANSPOSE` (LD_T) |
| D (output) | row-major | `XE_STORE_2D` (ST_N) |
| A (U8/U4 MMA) | — | packed load (`Packed_U8`/`Packed_U4`) |

In a kernel, prefer the factories `make_block_2d_copy_A/B/D(mma, tensor)` and
`make_block_2d_prefetch(copy)` — they apply this table automatically based on
element type, layout, and MMA requirements.

## Data distribution (subgroup = 16 work-items)

Each 2D copy is a subgroup-cooperative operation. For `XE_2D_U16x32x16_LD_N`
(32 rows × 16 cols): work-item *i* gets column *i* (32 elements). Doubling width
to ×32: each work-item gets two columns 16 apart (cols *i* and *i*+16).

**VNNI** packs elements from multiple rows of one column into 32-bit values for
DPAS consumption — it only changes storage format, not values. Used for B.

**Packed loads** (`XE_2D_Packed_U8x8x32_LD_N`) give each work-item adjacent
columns — required when feeding U8/U4 A-operands to integer DPAS.

## Legacy naming (still in code, being migrated)

Legacy per-shape atoms (`XE_2D_U16x32x16_LD_N`, `XE_8x16x16_F32BF16BF16F32_TT`,
etc.) live in `copy_xe_legacy_U*.hpp` / `mma_xe_legacy.hpp`. New code should use
`XE_DPAS_TT` and the parameterized `XE_LOAD_2D*` / `XE_STORE_2D` API. Dispatch
policy for legacy atoms: `cutlass::gemm::MainloopIntelXeXMX16`; for new atoms:
`cutlass::gemm::MainloopXeL1Staged<Stages>`.
