// Minimal standalone SYCL*TLA consumer: C = alpha*(A*B) + beta*C on Intel GPU.
// Uses the user's own sycl::queue and USM device memory. No repo example helpers.
//
// Build (JIT, simplest — validated on Intel Arc, oneAPI 2025.2):
//   icpx -fsycl -std=c++17 -fsycl-targets=spir64 \
//     -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
//     -I<sycl-tla>/include -I<sycl-tla>/tools/util/include \
//     -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier \
//     minimal_gemm.cpp -o gemm
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./gemm
// For AOT, replace spir64 with intel_gpu_bmg_g21 (BMG) or intel_gpu_pvc (PVC).
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>

using namespace cute;

int main() {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementC = float;
  using ElementAcc = float;
  using ElementComp = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  constexpr int AlignA = sizeof(ElementA);
  constexpr int AlignB = sizeof(ElementB);
  constexpr int AlignC = sizeof(ElementAcc);
  constexpr int AlignD = sizeof(ElementC);

  using TileShape = Shape<_256, _256, _32>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAcc,
      TileShape, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementC, ElementComp, ElementAcc, ElementAcc,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      TileShape, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementComp,
      ElementAcc,
      ElementAcc, LayoutC, AlignC,
      ElementC,   LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  int M = 256, N = 256, K = 256, L = 1;
  sycl::queue q{sycl::gpu_selector_v};

  auto* A = sycl::malloc_device<ElementA>(std::size_t(M)*K*L, q);
  auto* B = sycl::malloc_device<ElementB>(std::size_t(K)*N*L, q);
  auto* C = sycl::malloc_device<ElementC>(std::size_t(M)*N*L, q);
  auto* D = sycl::malloc_device<ElementC>(std::size_t(M)*N*L, q);

  std::vector<ElementA> hA(std::size_t(M)*K, ElementA(1.0f));
  std::vector<ElementB> hB(std::size_t(K)*N, ElementB(1.0f));
  std::vector<ElementC> hC(std::size_t(M)*N, ElementC(0.0f));
  q.copy(hA.data(), A, hA.size()).wait();
  q.copy(hB.data(), B, hB.size()).wait();
  q.copy(hC.data(), C, hC.size()).wait();

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  auto sA = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto sB = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto sC = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, L));
  auto sD = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, L));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, L},
      {A, sA, B, sB},
      {{1.0f, 0.0f}, C, sC, D, sD},
      hw_info};

  Gemm gemm_op;
  if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
    std::cerr << "cannot implement\n"; return 1;
  }
  std::vector<uint8_t> ws(Gemm::get_workspace_size(args));
  auto* ws_dev = sycl::malloc_device<uint8_t>(ws.size() ? ws.size() : 1, q);
  if (gemm_op.initialize(args, ws_dev, &q) != cutlass::Status::kSuccess) return 2;
  if (gemm_op.run(&q) != cutlass::Status::kSuccess) return 3;
  q.wait();

  q.copy(D, hC.data(), hC.size()).wait();
  std::cout << "D[0]=" << hC[0] << " (expect " << (float)K << ")\n";
  bool ok = std::abs(hC[0] - (float)K) < 1.0f;
  std::cout << (ok ? "PASSED" : "FAILED") << "\n";
  sycl::free(A,q); sycl::free(B,q); sycl::free(C,q); sycl::free(D,q); sycl::free(ws_dev,q);
  return ok ? 0 : 4;
}
