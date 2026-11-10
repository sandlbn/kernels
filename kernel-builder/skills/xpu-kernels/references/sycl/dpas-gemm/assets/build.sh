#!/usr/bin/env bash
# Compile and run DPAS example programs in this directory.
#
# Prereqs: Intel oneAPI (icpx), Intel GPU (Level Zero).
#
# Usage:
#   ./build.sh [all|compile|run] [basename...]
#   ./build.sh compile gemm_i8_i8_i32_sg8x16x32_b_transform
#
# Uses JIT -fsycl-targets=spir64 and the SPIRV extensions from
# ../README.md.
# 32×64 int8 examples may need AOT + -ze-opt-large-register-file on some setups;
# see the AOT icpx line at the top of those .cpp files.
# gemm_bf16_bf16_f32_sg32x64x32_b_transform_prefetch is CRI / Xe3p only (see its header).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT}/build"
SPIRV_EXT='+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate'
CXXFLAGS=(
  -fsycl -std=c++17 -fsycl-targets=spir64
  -Xspirv-translator "--spirv-ext=${SPIRV_EXT}"
)

EXAMPLES=(
  gemm_i8_i8_i32_sg8x16x32_b_transform
  gemm_i8_i8_i32_sg8x16x32_b_transpose
  gemm_i8_i8_i32_sg32x64x32_b_transform
  gemm_i8_i8_i32_sg32x64x32_b_transpose
  gemm_bf16_bf16_f32_sg32x64x32_b_transform_prefetch
)

compile_one() {
  local base="$1"
  local src="${ROOT}/${base}.cpp"
  [[ -f "$src" ]] || { echo "missing ${src}" >&2; exit 1; }
  echo "compile ${base}.cpp"
  mkdir -p "$OUT"
  icpx "${CXXFLAGS[@]}" -o "${OUT}/${base}" "$src"
}

run_one() {
  local base="$1"
  echo "run ${base}"
  ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}" \
    "${OUT}/${base}"
}

usage() {
  echo "usage: $0 [all|compile|run] [${EXAMPLES[*]}]" >&2
}

if ((${#@} == 0)); then
  set -- all
fi

if [[ "$1" == -h || "$1" == --help ]]; then
  usage
  exit 0
fi

action=all
targets=()
if [[ "$1" == all || "$1" == compile || "$1" == run ]]; then
  action="$1"
  shift
fi
if ((${#@} > 0)); then
  targets=("$@")
else
  targets=("${EXAMPLES[@]}")
fi

for base in "${targets[@]}"; do
  case "$action" in
    compile) compile_one "$base" ;;
    run) run_one "$base" ;;
    all)
      compile_one "$base"
      run_one "$base"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done
