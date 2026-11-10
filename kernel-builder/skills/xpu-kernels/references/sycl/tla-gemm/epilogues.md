# SYCL*TLA Epilogue Fusion reference (consumer)

The epilogue computes the final `D` after the GEMM accumulate. Choose a fusion op
(`cutlass::epilogue::fusion::*`) and pass it as the last template argument of the
epilogue `CollectiveBuilder` (or wire it through `FusionCallbacks` in the explicit
path). Switching the epilogue does not change how you launch the kernel — only the
epilogue runtime arguments differ.

Header: `#include "cutlass/epilogue/fusion/xe_callbacks.hpp"`. Activations live in
`cutlass::epilogue::thread::` (`ReLu`, `GELU`, `SiLu`, `Sigmoid`, `Tanh`,
`Identity`, ...).

Common params: `<..., ElementOutput, ElementCompute, ElementAccumulator,
ElementScalar, FloatRoundStyle>`. `ElementCompute`/`ElementScalar` are usually
`float`; round style `cutlass::FloatRoundStyle::round_to_nearest`.

## Linear combination — `D = alpha*acc + beta*C`
```cpp
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementCompute, ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;
```
Runtime epilogue args: `{alpha, beta}` (plus `C, strideC, D, strideD`).

## + Elementwise activation — `D = act(alpha*acc + beta*C)`
```cpp
using EpilogueOp = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::ReLu,   // or GELU, SiLu, Sigmoid, Tanh, ...
    ElementOutput, ElementCompute, ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;
```

## + Per-row / per-column bias — `D = act(alpha*acc + beta*C + bias)`
```cpp
using ElementBias = ElementOutput;
using EpilogueOp = cutlass::epilogue::fusion::LinCombPerRowBias<   // or LinCombPerColBias
    ElementOutput, ElementCompute, ElementBias, ElementAccumulator,
    ElementAccumulator, 128 / cutlass::sizeof_bits_v<ElementBias>,
    cutlass::FloatRoundStyle::round_to_nearest>;
```
Runtime args add a `ptr_bias` (length M for per-row, N for per-col).

## + Bias + activation + auxiliary store
Stores the pre-activation tensor alongside `D`:
```cpp
using EpilogueOp = cutlass::epilogue::fusion::LinCombPerRowBiasEltActAux<
    LayoutC, cutlass::epilogue::thread::ReLu, ElementOutput,
    ElementCompute, ElementAccumulator, ElementAccumulator>;
// Explicit path: FusionCallbacks takes an extra R2G copy-atom argument for the aux store.
```
Runtime args add `ptr_aux` + its stride.

## De-quant / dual activation
```cpp
using EpilogueOp = cutlass::epilogue::fusion::LinCombDeEltAct<
    LayoutC, ActivationFn, ElementOutput, ElementCompute>;
```

## Row softmax (e.g. attention-style outputs)
```cpp
using EpilogueOp = cutlass::epilogue::fusion::LinCombSoftmaxRow<
    ElementOutput, ElementCompute, XE_2D_U32x8x16_ST_N,
    ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;
```
A top-k + softmax variant also exists.

## Split-K (K-dimension splitting with partial accumulation)
```cpp
using EpilogueOp = cutlass::epilogue::fusion::LinCombSplitK<
    ElementOutput, ElementCompute, XE_2D_U32x8x16_ST_N,
    ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;
```

## Wiring (explicit, non-builder path)
```cpp
using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    cutlass::epilogue::IntelXeGeneric, EpilogueOp, TileShape,
    decltype(tile_shape(TiledMma()))   /*, CopyOpR2G for aux ops */>;

using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::IntelXeGeneric, TileShape, void,
    ElementAccumulator, cutlass::gemm::TagToStrideC_t<LayoutC>,
    ElementOutput,      cutlass::gemm::TagToStrideC_t<LayoutD>,
    FusionCallbacks, void, void>;
```
With the `CollectiveBuilder` epilogue path you skip `FusionCallbacks` and pass
`EpilogueOp` directly as the final template argument (see SKILL §5).

## Passing fusion arguments at runtime
The epilogue field of the kernel `Arguments` mirrors the fusion op: plain linear
combination is `{alpha, beta}`; bias/aux ops add pointers (`ptr_bias`, `ptr_aux`,
strides). The exact nested-struct field order depends on the fusion op — when in
doubt, check the matching `examples/05_bmg_gemm_with_epilogues/*` source for the
precise argument initialization.

> **Xe constraint:** keep output `D` row-major / N-major; the Xe epilogue asserts
> on the supported layout.
