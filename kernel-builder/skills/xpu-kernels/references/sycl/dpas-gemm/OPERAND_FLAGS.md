# DPAS `Operands` bitfield

The last argument to `__spirv_SubgroupMatrixMultiplyAccumulateINTEL` is a
bitmask, not an opaque constant. The element type of every operand is carried
*only* here — it is not part of the C++ signature, so nothing diagnoses a wrong
value. A wrong flag compiles cleanly and returns garbage.

Derive the flag from this table rather than copying a constant.

| Bit | Name | Meaning |
|---|---|---|
| `0x1`    | `MatrixASignedComponentsINTEL`  | components of A are signed |
| `0x2`    | `MatrixBSignedComponentsINTEL`  | components of B are signed |
| `0x4`    | `MatrixCBFloat16INTEL`          | C (accumulator in) is bf16 data |
| `0x8`    | `MatrixResultBFloat16INTEL`     | result is written as bf16 data |
| `0x10`   | `MatrixAPackedInt8INTEL`        | A is packed 8-bit integer |
| `0x20`   | `MatrixBPackedInt8INTEL`        | B is packed 8-bit integer |
| `0x40`   | `MatrixAPackedInt4INTEL`        | A is packed 4-bit integer |
| `0x80`   | `MatrixBPackedInt4INTEL`        | B is packed 4-bit integer |
| `0x100`  | `MatrixATF32INTEL`              | A is tf32 |
| `0x200`  | `MatrixBTF32INTEL`              | B is tf32 |
| `0x400`  | `MatrixAPackedFloat16INTEL`     | A is packed fp16 |
| `0x800`  | `MatrixBPackedFloat16INTEL`     | B is packed fp16 |
| `0x1000` | `MatrixAPackedBFloat16INTEL`    | A is packed bf16 |
| `0x2000` | `MatrixBPackedBFloat16INTEL`    | B is packed bf16 |

Note the pattern: A and B each get their own bit, and for a symmetric GEMM they
are always set together. The flags used elsewhere in this skill decompose as:

| Flag | Decomposition |
|---|---|
| `0x33`   | `0x1 \| 0x2 \| 0x10 \| 0x20` — A signed, B signed, A packed int8, B packed int8 |
| `0x3000` | `0x1000 \| 0x2000` — A packed bf16, B packed bf16 |
| `0xC00`  | `0x400 \| 0x800` — A packed fp16, B packed fp16 |

Nothing sets a signed bit for the float types — sign is part of the float format.

> **The fp16/bf16 confusion, mechanically.** `0xC00` and `0x3000` are different
> bit positions of the same field and share one C++ overload, so passing the
> bf16 flag with fp16 register data is invisible to the compiler: the systolic
> array reinterprets your half bit patterns as bfloat16. bf16 has 8 exponent
> bits where fp16 has 5, so the results come out with wrong magnitudes and wrong
> signs in a pattern that reads like a numerical bug rather than a type error.
> If a 16-bit-input kernel produces plausible-looking but wrong numbers, check
> this flag before anything else.

The `int4`, `tf32` and bf16-accumulator/bf16-output rows are unused by this
skill and unverified here. If you need them, confirm the register layout on
hardware with a small probe (see [REGISTER_LAYOUTS.md](REGISTER_LAYOUTS.md))
rather than assuming it follows from the 16-bit case.

Authoritative source: the operands table in
[SPV_INTEL_subgroup_matrix_multiply_accumulate](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_subgroup_matrix_multiply_accumulate.asciidoc).
