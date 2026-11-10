// Requires CRI / Xe3p devices (large GRF). The 128-wide 2D-block prefetch path is not
// supported on earlier Intel GPUs (e.g. Battlemage); JIT ./build.sh may fail at
// runtime there. On Xe3p hardware, prefer AOT below (ocloc device e.g.
// ptl-h or lnl, not battlemage targets).
// Build/run (JIT, Xe3p only): ./build.sh gemm_bf16_bf16_f32_sg32x64x32_b_transform_prefetch
// AOT (large GRF): icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device ptl-h -options -ze-opt-large-register-file" -Xspirv-translator "--spirv-ext=+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" gemm_bf16_bf16_f32_sg32x64x32_b_transform_prefetch.cpp

#include <sycl/sycl.hpp>
#include <iostream>
#include <random>
#include <vector>

#define SG_SIZE 16

using namespace sycl;

typedef int32_t block_coord __attribute__((ext_vector_type(2)));

using int8_dpas = int __attribute__((ext_vector_type(8)));
using float8_dpas = float __attribute__((ext_vector_type(8)));
using short8_dpas = int16_t __attribute__((ext_vector_type(8)));

static inline uint32_t ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

extern SYCL_EXTERNAL void __spirv_Subgroup2DBlockLoadINTEL(
    int ElementSize,
    int BlockWidth,
    int BlockHeight,
    int BlockCount,
    const void* srcBasePointer,
    int memoryWidth,
    int memoryHeight,
    int memoryPitch,
    block_coord coordinate,
    void *dstPointer);

    extern SYCL_EXTERNAL void __spirv_Subgroup2DBlockLoadTransformINTEL(
    int ElementSize,
    int BlockWidth,
    int BlockHeight,
    int BlockCount,
    const void* srcBasePointer,
    int memoryWidth,
    int memoryHeight,
    int memoryPitch,
    block_coord coordinate,
    void *dstPointer);

extern SYCL_EXTERNAL void __spirv_Subgroup2DBlockLoadTransposeINTEL(
    int ElementSize,
    int BlockWidth,
    int BlockHeight,
    int BlockCount,
    const void* srcBasePointer,
    int memoryWidth,
    int memoryHeight,
    int memoryPitch,
    block_coord coordinate,
    void *dstPointer);

extern SYCL_EXTERNAL void __spirv_Subgroup2DBlockStoreINTEL(
    int ElementSize,
    int BlockWidth,
    int BlockHeight,
    int BlockCount,
    void* srcPointer,
    const void* dstBasePointer,
    int memoryWidth,
    int memoryHeight,
    int memoryPitch,
    block_coord coordinate);

    extern SYCL_EXTERNAL void __spirv_Subgroup2DBlockPrefetchINTEL(
    int ElementSize,
    int BlockWidth,
    int BlockHeight,
    int BlockCount,
    const void* srcBasePointer,
    int memoryWidth,
    int memoryHeight,
    int memoryPitch,
    block_coord coordinate);

extern SYCL_EXTERNAL int8_dpas __spirv_SubgroupMatrixMultiplyAccumulateINTEL (
    int K_Dim,
    short8_dpas Matrix_A,
    int8_dpas Matrix_B,
    int8_dpas Matrix_C,
    int Operands
);

extern SYCL_EXTERNAL float8_dpas __spirv_SubgroupMatrixMultiplyAccumulateINTEL (
    int K_Dim,
    short8_dpas Matrix_A,
    int8_dpas Matrix_B,
    float8_dpas Matrix_C,
    int Operands
);

int main()
{

    // For FP16: swap this for `using bf16 = sycl::half;` (and the operand flag in
    // the DPAS call below). Nothing else in the file changes.
    using bf16 = sycl::ext::oneapi::bfloat16;

    static constexpr size_t mat_m = 4*256;
    static constexpr size_t mat_n = 8*256;
    static constexpr size_t mat_k = 1024;
    static constexpr size_t wg_m = 256;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 32;
    static constexpr size_t sg_n = 64;
    static constexpr size_t sg_k = 32;
    static constexpr size_t sg_a_count = sg_m / 8;
    static constexpr size_t sg_b_count = sg_n / 16;
    using data_type_a = bf16;
    using data_type_b = bf16;
    using data_type_c = float;
    static constexpr size_t sg_k_count = sg_k /(32 / sizeof(data_type_a));
    static constexpr size_t prefetch_distance = 4;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-1.0f, 1.0f);

    queue gpu_sycl_queue{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    std::cout << "Device: " << gpu_sycl_queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    std::cout << "Allowed by device subgroup sizes: ";
    for (const int &s : gpu_sycl_queue.get_device().get_info<sycl::info::device::sub_group_sizes>())
    {
        std::cout << s << " ";
    }
    std::cout << std::endl;
    std::cout << "Subgroup size in use: " << SG_SIZE << std::endl;

    data_type_a *memA = malloc_shared<data_type_a>(mat_m * mat_k, gpu_sycl_queue);
    data_type_b *memB = malloc_shared<data_type_b>(mat_n * mat_k, gpu_sycl_queue);
    data_type_c *memC = malloc_shared<data_type_c>(mat_m * mat_n, gpu_sycl_queue);
    for (int i = 0; i < mat_m * mat_k; ++i)
    {
        memA[i] = distrib(gen);
    }
    for (int i = 0; i < mat_n * mat_k; ++i)
    {
        memB[i] = distrib(gen);
    }
    static constexpr uint32_t local_range_x = wg_n / sg_n;
    static constexpr uint32_t local_range_y = wg_m / sg_m;
    range<2> local_range = {local_range_y, local_range_x * SG_SIZE};
    range<2> pre_global_range = {ceil_div_int(mat_m, wg_m), ceil_div_int(mat_n, wg_n)};
    range<2> global_range = local_range * pre_global_range;
    std::cout << "Global Range: (" << pre_global_range[0] << ", " << pre_global_range[1] << ")" << std::endl;
    std::cout << "Local Range: (" << local_range[0] << ", " << local_range[1] / SG_SIZE << ")" << std::endl;

    auto ev = gpu_sycl_queue.submit([&](auto &h) {
        h.parallel_for(sycl::nd_range<2>(global_range, local_range),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            
            auto pA = address_space_cast<sycl::access::address_space::global_space,
                                sycl::access::decorated::no>(memA);
            auto pB = address_space_cast<sycl::access::address_space::global_space,
                                sycl::access::decorated::no>(memB);
            auto pC = address_space_cast<sycl::access::address_space::global_space,
                                sycl::access::decorated::no>(memC);
            
            const auto global_idx = item.get_global_id(1);
            const auto global_idy = item.get_global_id(0);
            
            const auto sg_startx = (global_idx / SG_SIZE) * sg_n;
            const auto sg_starty = global_idy * sg_m;

            auto sg = item.get_sub_group();
            int sg_group_id = sg.get_group_id(); // 0..31
            int prefetch_a_y_shift = 256 * (sg_starty / 256) + 8 * sg_group_id;
            int prefetch_b_x_shift = 256 * (sg_startx / 256) + 128 * (sg_group_id % 2);
            int prefetch_b_y_shift = 8 * (sg_group_id / 2);


#ifdef __SYCL_DEVICE_ONLY__

            block_coord prefetch_coord_a = (block_coord) {0, prefetch_a_y_shift};
            block_coord prefetch_coord_b = (block_coord) {prefetch_b_x_shift, prefetch_b_y_shift};
            block_coord pf_a = prefetch_coord_a;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                __spirv_Subgroup2DBlockPrefetchINTEL(sizeof(data_type_a),
                                                    16,
                                                    8,
                                                    2,
                                                    pA.get(),
                                                    mat_k * sizeof(data_type_a),
                                                    mat_m,
                                                    mat_k * sizeof(data_type_a),
                                                    pf_a);
                pf_a.x += 32;
            }
            prefetch_coord_a.x += 4 * sg_k;

            block_coord pf_b = prefetch_coord_b;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                __spirv_Subgroup2DBlockPrefetchINTEL(sizeof(data_type_b),
                                                    16,
                                                    8,
                                                    2,
                                                    pB.get(),
                                                    mat_n * sizeof(data_type_b),
                                                    mat_k,
                                                    mat_n * sizeof(data_type_b),
                                                    pf_b);
                pf_b.x += 32;
            }
            prefetch_coord_b.y += 4 * sg_k;

            block_coord load_coord_a = (block_coord) {0, static_cast<int32_t>(sg_starty)};
            block_coord load_coord_b = (block_coord) {static_cast<int32_t>(sg_startx), 0};
            block_coord store_coord_c = (block_coord) {static_cast<int32_t>(sg_startx), static_cast<int32_t>(sg_starty)};

            asm volatile("fence_sw":::);

            data_type_a reg_A[sg_k_count][sg_a_count][ 8 * 16 / SG_SIZE];
            data_type_b reg_B[sg_b_count][sg_k_count][16 * 16 / SG_SIZE];
            data_type_c reg_C[sg_a_count][sg_b_count][ (8 * 16) / SG_SIZE];

#pragma unroll
            for (int i = 0; i < sg_a_count; ++i) {
#pragma unroll
                for (int j = 0; j < sg_b_count; ++j) {
#pragma unroll
                    for (int k = 0; k < (8 * 16) / SG_SIZE; ++k) {
                        reg_C[i][j][k] = 0;
                    }
                }
            }

#pragma nounroll
            for (int i = 0; i < mat_k/sg_k; ++i) {
                if ((i % 4) == 0) {
                    block_coord pf_a = prefetch_coord_a;
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        __spirv_Subgroup2DBlockPrefetchINTEL(sizeof(data_type_a),
                                                        16,
                                                        8,
                                                        2,
                                                        pA.get(),
                                                        mat_k * sizeof(data_type_a),
                                                        mat_m,
                                                        mat_k * sizeof(data_type_a),
                                                        pf_a);
                        pf_a.x += 32;
                    }
                    prefetch_coord_a.x += 4 * sg_k;

                    block_coord pf_b = prefetch_coord_b;
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        __spirv_Subgroup2DBlockPrefetchINTEL(sizeof(data_type_b),
                                                        16,
                                                        8,
                                                        2,
                                                        pB.get(),
                                                        mat_n * sizeof(data_type_b),
                                                        mat_k,
                                                        mat_n * sizeof(data_type_b),
                                                        pf_b);
                        pf_b.x += 32;
                    }
                    prefetch_coord_b.y += 4 * sg_k;

                }

                asm volatile("fence_sw":::);

                __spirv_Subgroup2DBlockLoadINTEL(sizeof(data_type_a),
                                                 16,
                                                 32,
                                                 2,
                                                 pA.get(),
                                                 mat_k * sizeof(data_type_a),
                                                 mat_m,
                                                 mat_k * sizeof(data_type_a),
                                                 load_coord_a,
                                                 reg_A);
                load_coord_a.x += sg_k;

                __spirv_Subgroup2DBlockLoadTransformINTEL(sizeof(data_type_b),
                                                 16,
                                                 32,
                                                 2,
                                                 pB.get(),
                                                 mat_n * sizeof(data_type_b),
                                                 mat_k,
                                                 mat_n * sizeof(data_type_b),
                                                 load_coord_b,
                                                 reg_B[0]);
                load_coord_b.x += 32;
                __spirv_Subgroup2DBlockLoadTransformINTEL(sizeof(data_type_b),
                                                 16,
                                                 32,
                                                 2,
                                                 pB.get(),
                                                 mat_n * sizeof(data_type_b),
                                                 mat_k,
                                                 mat_n * sizeof(data_type_b),
                                                 load_coord_b,
                                                 reg_B[2]);
                load_coord_b.x -= 32;
                load_coord_b.y += sg_k;


                asm volatile("fence_sw":::);

#pragma unroll
                for (int a = 0; a < sg_a_count; ++a) {
#pragma unroll
                    for (int b = 0; b < sg_b_count; ++b) {
#pragma unroll
                        for (int k = 0; k < sg_k_count; ++k) {

                            *reinterpret_cast<float8_dpas*>(&reg_C[a][b]) = __spirv_SubgroupMatrixMultiplyAccumulateINTEL (
                                16,
                                *reinterpret_cast<short8_dpas*>(&reg_A[k][a]),
                                *reinterpret_cast<int8_dpas*>(&reg_B[b][k]),
                                *reinterpret_cast<float8_dpas*>(&reg_C[a][b]),
                                0x3000  // FP16: 0xC00. Must match the type alias above.
                            );
                        }
                    }
                }
                asm volatile("fence_sw":::);
            }
#pragma unroll
            for (int a = 0; a < sg_a_count; ++a) {
#pragma unroll
                    for (int b = 0; b < sg_b_count; ++b) {
                        __spirv_Subgroup2DBlockStoreINTEL(sizeof(data_type_c),
                                                        16,
                                                        8,
                                                        1,
                                                        reg_C[a][b],
                                                        pC.get(),
                                                        mat_n * sizeof(data_type_c),
                                                        mat_m,
                                                        mat_n * sizeof(data_type_c),
                                                        store_coord_c);
                        store_coord_c.x += 16;
                    }
                    store_coord_c.x -= sg_b_count * 16;
                    store_coord_c.y += 8;
                }                
#endif
        });
    });
    ev.wait();
    float tolerance = 0.0001;
    int errors = 0;
    float max_diff = 0;
    for (int i = 0; i < mat_m; ++i)
    {
        for (int j = 0; j < mat_n; ++j)
        {
            float gold = 0;
            for (int k = 0; k < mat_k; ++k)
            {
                gold = gold + (float)(memA[i * mat_k + k]) * (float)(memB[j + k * mat_n]);

            }
            float diff = std::abs(memC[i * mat_n + j] - gold);
            if(diff > max_diff) {
                max_diff = diff;
            }
            if (diff > tolerance) {
                errors++;
                if (errors < 10) {
                    std::cout << "[" << i << ", " << j << "]: computed: "
                    << memC[i * mat_n + j] << " gold: "
                    << gold << " diff: " << diff 
                    << std::endl;
                }
            }
        }
    }
    std::cout << "Tolerance: " << tolerance << std::endl;
    std::cout << "Max difference: " << max_diff << std::endl;
    if (errors == 0) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL with " << errors << " errors." << std::endl;
    }
}
