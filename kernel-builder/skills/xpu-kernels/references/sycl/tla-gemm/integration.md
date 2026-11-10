# SYCL*TLA integration reference (header-only consumer)

How to build your own application against the SYCL*TLA headers. All flags below
were validated end-to-end (compile + run) on Intel Arc with oneAPI 2025.2.

## What you need

- **Intel oneAPI DPC++** (`icpx` / `icx`), version **2025.1+**. Source it:
  `source /opt/intel/oneapi/setvars.sh`.
- The **sycl-tla checkout** (headers only — nothing is compiled from the repo).
- An **Intel Xe GPU** at runtime (Arc/BMG or Data Center Max/PVC).

No libraries to link for the kernels themselves. oneMKL is only needed if you use
the repo's *reference* GEMM for verification.

## Include directories

| Path | Contents | Needed? |
|------|----------|---------|
| `<sycl-tla>/include` | `cutlass/` + `cute/` core headers (the kernels) | **Yes** |
| `<sycl-tla>/tools/util/include` | `cutlass/util/*` helpers: `make_cute_packed_stride`, `DeviceAllocation`, reference kernels, `GPU_Clock`, command-line | Optional but recommended |

## Required preprocessor defines

```
-DCUTLASS_ENABLE_SYCL     # turns on the SYCL backend
-DSYCL_INTEL_TARGET       # selects Intel Xe code paths
```

## Required compiler / linker flags

```
-fsycl
-fsycl-targets=<target>                          # compile + link
-std=c++17
-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier   # link (REQUIRED)
```

`<target>` options:
- `spir64` — JIT (SPIR-V); portable, compiles fastest, runs on any installed Xe
  GPU. **Best default.**
- `intel_gpu_bmg_g21` — AOT for BMG / Arc B-series.
- `intel_gpu_pvc` — AOT for PVC / Data Center Max.

> **The split-barrier extension is not optional.** Omitting it produces a link
> error `RequiresExtension: Feature requires the following SPIR-V extension:
> SPV_INTEL_split_barrier`. This is required even on oneAPI 2025.2 because the Xe
> mainloop uses split barriers.

## Validated `icpx` one-liner

```bash
source /opt/intel/oneapi/setvars.sh
SYCL_TLA=/path/to/sycl-tla
icpx -fsycl -std=c++17 -fsycl-targets=spir64 \
  -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
  -I$SYCL_TLA/include -I$SYCL_TLA/tools/util/include \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier \
  my_gemm.cpp -o my_gemm
```

For AOT, also pass the backend device, e.g.:
```bash
-fsycl-targets=intel_gpu_bmg_g21
# (icpx maps this to the right backend device automatically; for the generic
#  spir64 path use JIT. For multi-GPU AOT you can add
#  -Xsycl-target-backend "-device bmg-g21")
```

## CMake recipe (validated)

`assets/CMakeLists.txt` is a copy of this. Point `SYCL_TLA_DIR` at your checkout.

```cmake
cmake_minimum_required(VERSION 3.22)
project(my_app CXX)

set(SYCL_TLA_DIR "" CACHE PATH "Path to sycl-tla repository root")
set(SYCL_TARGET "spir64" CACHE STRING "DPC++ target: spir64 | intel_gpu_bmg_g21 | intel_gpu_pvc")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(app my_gemm.cpp)
target_include_directories(app PRIVATE
  ${SYCL_TLA_DIR}/include
  ${SYCL_TLA_DIR}/tools/util/include)
target_compile_definitions(app PRIVATE CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
target_compile_options(app PRIVATE -fsycl -fsycl-targets=${SYCL_TARGET})
target_link_options(app PRIVATE
  -fsycl -fsycl-targets=${SYCL_TARGET}
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier)
```

Configure with the Intel compiler:
```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DSYCL_TLA_DIR=/path/to/sycl-tla
cmake --build build
```

### Pulling sycl-tla via FetchContent

Since you only need headers, you don't need to build the repo — just fetch the
sources and reference the include dirs:

```cmake
include(FetchContent)
FetchContent_Declare(sycl_tla
  GIT_REPOSITORY https://github.com/intel/sycl-tla.git
  GIT_TAG main)        # pin a release tag in production
FetchContent_MakeAvailable(sycl_tla)
# Then use ${sycl_tla_SOURCE_DIR}/include and ${sycl_tla_SOURCE_DIR}/tools/util/include
target_include_directories(app PRIVATE
  ${sycl_tla_SOURCE_DIR}/include
  ${sycl_tla_SOURCE_DIR}/tools/util/include)
```

(Adding it as a git submodule and pointing `SYCL_TLA_DIR` at it works equally well
and avoids network access at configure time.)

## Runtime environment

Select the GPU and (for representative performance) enable large-GRF mode:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# Performance (optional, for benchmarking):
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=100000000000
```

## Debugging 2D-block alignment

Add `-DCUTE_ENABLE_XE_BLOCK_2D_ASSERT=1` to enable runtime asserts that check base
pointer (64 B), pitch (≥16 B preferred; ≥4 B required), and width alignment before
each 2D block op. Use it while bringing up new matrix layouts; alignment
violations otherwise silently corrupt results.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `RequiresExtension: ... SPV_INTEL_split_barrier` at link | Add `-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier`. |
| `No kernel named ...` / no device found at runtime | Set `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`; confirm with `sycl-ls`. |
| Wrong results, no error | A non-row-major D, or feeding non-device pointers; verify USM is `malloc_device`/shared and D is RowMajor. |
| `can_implement` returns error | Unsupported problem shape/alignment for the chosen tile/types; adjust sizes or types. |
| Huge template error dumps | Add `-ftemplate-backtrace-limit=0` for readable diagnostics. |
| Slow build | Use `-fsycl-targets=spir64` (JIT) during development; AOT only for release. |
