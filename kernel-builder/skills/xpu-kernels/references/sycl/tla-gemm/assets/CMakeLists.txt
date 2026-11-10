cmake_minimum_required(VERSION 3.22)
project(sycltla_app CXX)

# Point this at your sycl-tla checkout (header-only).
set(SYCL_TLA_DIR "" CACHE PATH "Path to sycl-tla repository root")
# Intel GPU AOT target: intel_gpu_bmg_g21 (BMG) / intel_gpu_pvc (PVC) / spir64 (JIT).
set(SYCL_TARGET "spir64" CACHE STRING "DPC++ SYCL target")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(app gemm.cpp)

target_include_directories(app PRIVATE
  ${SYCL_TLA_DIR}/include
  ${SYCL_TLA_DIR}/tools/util/include)

target_compile_definitions(app PRIVATE CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)

target_compile_options(app PRIVATE -fsycl -fsycl-targets=${SYCL_TARGET})
target_link_options(app PRIVATE
  -fsycl -fsycl-targets=${SYCL_TARGET}
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier)
