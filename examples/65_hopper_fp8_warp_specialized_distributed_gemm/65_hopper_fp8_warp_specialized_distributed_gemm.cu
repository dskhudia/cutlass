/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Simple Hopper FP8 GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture

    This example demonstrate a simple way to instantiate and run a FP8 GEMM using the new CUTLASS 3.0
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:

    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA) 
    which are more efficient than the Ampere tensor core instructions.

    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large 
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster.

    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).

    4. This example shows all fusions used by vllm fp8 gemm kernels, 

    5. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the 
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can 
    improve performance.

    Examples:

      $ ./65_hopper_fp8_warp_specialized_distributed_gemm --m=2048 --n=2048 --k=2048 --rasterization=N --swizzle=2
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"

// Distributed GEMM headers
#include "cutlass/experimental/distributed/device/dist_gemm_universal_wrapper.hpp"
#include "cutlass/experimental/distributed/kernel/dist_gemm_kernel_wrapper.hpp"
#include "cutlass/experimental/distributed/schedules/dist_gemm_1d_schedules.hpp"

#include "helper.h"
#include "hopper_fp8_commandline.hpp"
#include "scaled_mm_epilogues_c3x.hpp"

// Distributed GEMM helpers
#include "util/benchmark.h"
#include "util/device_copy.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Distributed GEMM configuration
/////////////////////////////////////////////////////////////////////////////////////////////////

// TP size (= number of processors/GPUs)
using TP = _4;


using DistSchedule = cutlass::distributed::schedules::AllGather1D_TilingCD_RotatingA<TP>;
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = cutlass::bfloat16_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_128,_128>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster
using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecialized;
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
                            TileShape, EpilogueTileType, ElementD, ElementD, EpilogueSchedule>;

//using FusionOperation     = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
//    LayoutAux, cutlass::epilogue::thread::ReLU, ElementD, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementC>;

using Epilogue = c3x::ScaledEpilogue<ElementAccumulator, ElementD, EpilogueDescriptor>;
using EVTCompute = typename Epilogue::EVTCompute;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    EVTCompute
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue
>;

// We're going to use the single-device GEMM as reference
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Datatypes for scalars
using ElementScalar = ElementCompute;

// Instantiate Distributed GEMM kernel
using DistGemmKernel = cutlass::distributed::kernel::DistributedGemmKernelWrapper<
    GemmKernel,
    DistSchedule
    >;
using DistGemm = cutlass::distributed::device::DistributedGemmUniversalAdapter<DistGemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

// Reference GEMM tensors; Regular single device gemm.
using HostTensorA = typename cutlass::HostTensor<ElementA, LayoutA>;
using HostTensorB = typename cutlass::HostTensor<ElementB, LayoutB>;
using HostTensorC = typename cutlass::HostTensor<ElementC, LayoutC>;
using HostTensorD = typename cutlass::HostTensor<ElementD, LayoutD>;

HostTensorA tensor_A;
HostTensorB tensor_B;
HostTensorC tensor_C;
HostTensorD tensor_D;
HostTensorD tensor_ref_D;

// DistGEMM tensors (multi-device)
HostTensorA tensor_A_arr[TP{}];
HostTensorB tensor_B_arr[TP{}];
HostTensorD tensor_C_arr[TP{}];
HostTensorD tensor_D_arr[TP{}];

using LayoutScalar = cutlass::layout::PackedVectorLayout;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_alpha;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_beta;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_A;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_B;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_C;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_D;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double tflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double tflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), tflops(tflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed,
  bool is_device_tensor = false) {

  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if (bits_input <= 16) {
    scope_max = 2;
    scope_min = -2;
  }
  else {
    scope_max = 8;
    scope_min = -8;
  }
  if (is_device_tensor) {
      using Real = typename cutlass::RealType<Element>::Type;
      cutlass::reference::device::TensorFillRandomUniform(
              view, seed, static_cast<Real>(scope_max), static_cast<Real>(scope_min), 0);
      cudaDeviceSynchronize();
  } else {

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
  }

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options<RasterOrderOptions> &options) {
  auto problem_shape = cute::make_tuple(options.m, options.n, options.k, options.l);

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);

  initialize_tensor(tensor_A.host_view(), seed + 2022);
  initialize_tensor(tensor_B.host_view(), seed + 2023);
  initialize_tensor(tensor_C.host_view(), seed + 2024);

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();

  // Set up DistGEMM tensors
  auto local_shape_A = DistSchedule::get_local_a_shape(problem_shape);
  auto local_shape_B = DistSchedule::get_local_b_shape(problem_shape);
  auto local_shape_C = DistSchedule::get_local_c_shape(problem_shape);
  auto local_shape_D = DistSchedule::get_local_d_shape(problem_shape);

  auto a_coord_device = cutlass::make_Coord(size(local_shape_A), 1);
  auto b_coord_device = cutlass::make_Coord(size(local_shape_B), 1);
  auto c_coord_device = cutlass::make_Coord(size(local_shape_C), 1);

  int primary_device_idx;
  CUDA_CHECK(cudaGetDevice(&primary_device_idx));

  // Enable any-to-any access
  for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
    int can_access;
    CUDA_CHECK(cudaSetDevice(device_idx));
    for (int peer_idx = 0; peer_idx < TP{}; ++peer_idx) {
      if (peer_idx != device_idx) {
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device_idx, peer_idx));
        if (not can_access) {
          std::cerr << "FAILURE: Device " << device_idx << " can't access device " << peer_idx << "." <<
            std::endl;
          exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaDeviceEnablePeerAccess(peer_idx, 0));
      }
    }

    tensor_A_arr[device_idx].resize(a_coord_device);
    tensor_B_arr[device_idx].resize(b_coord_device);
    tensor_C_arr[device_idx].resize(c_coord_device);
    tensor_D_arr[device_idx].resize(c_coord_device);
  }
  CUDA_CHECK(cudaSetDevice(primary_device_idx));

  if (options.device_scale) {
    scalar_alpha.resize(cutlass::make_Coord(1));
    scalar_beta.resize(cutlass::make_Coord(1));
    scale_A.resize(cutlass::make_Coord(1));
    scale_B.resize(cutlass::make_Coord(1));
    scale_C.resize(cutlass::make_Coord(1));
    scale_D.resize(cutlass::make_Coord(1));

    cutlass::reference::host::TensorFill(scalar_alpha.host_view(), options.alpha);
    cutlass::reference::host::TensorFill(scalar_beta.host_view(), options.beta);
    cutlass::reference::host::TensorFill(scale_A.host_view(), options.scale_a);
    cutlass::reference::host::TensorFill(scale_B.host_view(), options.scale_b);
    cutlass::reference::host::TensorFill(scale_C.host_view(), options.scale_c);
    cutlass::reference::host::TensorFill(scale_D.host_view(), options.scale_d);

    scalar_alpha.sync_device();
    scalar_beta.sync_device();
    scale_A.sync_device();
    scale_B.sync_device();
    scale_C.sync_device();
    scale_D.sync_device();
  }
}

/// Populates a Gemm::Arguments structure from the given commandline options
using GemmArguments = typename Gemm::Arguments;
GemmArguments gemm_args_from_options(const Options<RasterOrderOptions> &options)
{
  
  typename GemmKernel::MainloopArguments mainloop_args{tensor_A.device_data(), stride_A, tensor_B.device_data(),
      stride_B};
  typename GemmKernel::EpilogueArguments epilogue_args{
      Epilogue::prepare_args(
              scale_A.device_data(), scale_B.device_data()),
              tensor_C.device_data(), stride_C,
              tensor_ref_D.device_data(), stride_D};

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    mainloop_args,
    epilogue_args
  };

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8) 
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

using DistGemmArguments = typename DistGemm::Arguments;
DistGemmArguments dist_gemm_args_from_options(
    const Options<RasterOrderOptions> &options,
    int device_idx,
    cudaStream_t stream) {

  auto problem_shape = cute::make_tuple(options.m, options.n, options.k, options.l);

  auto global_A = cute::make_tensor(tensor_A.device_data(),
      cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
  auto global_B = cute::make_tensor(tensor_B.device_data(),
      cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
  auto global_C = cute::make_tensor(tensor_C.device_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_C));

  auto global_A_device_slice = DistSchedule::get_device_slice_A(global_A, device_idx);
  auto global_B_device_slice = DistSchedule::get_device_slice_B(global_B, device_idx);
  auto global_C_device_slice = DistSchedule::get_device_slice_C(global_C, device_idx);

  auto local_shape_A = DistSchedule::get_local_a_shape(problem_shape);
  auto local_shape_B = DistSchedule::get_local_b_shape(problem_shape);
  auto local_shape_C = DistSchedule::get_local_c_shape(problem_shape);
  auto local_shape_D = DistSchedule::get_local_d_shape(problem_shape);

  auto local_stride_A = cutlass::make_cute_packed_stride(StrideA{}, local_shape_A);
  auto local_stride_B = cutlass::make_cute_packed_stride(StrideB{}, local_shape_B);
  auto local_stride_C = cutlass::make_cute_packed_stride(StrideC{}, local_shape_C);
  auto local_stride_D = cutlass::make_cute_packed_stride(StrideD{}, local_shape_D);

  auto local_A = cute::make_tensor(
      tensor_A_arr[device_idx].device_data(),
      make_layout(local_shape_A, local_stride_A));
  auto local_B = cute::make_tensor(
      tensor_B_arr[device_idx].device_data(),
      make_layout(local_shape_B, local_stride_B));
  auto local_C = cute::make_tensor(
      tensor_C_arr[device_idx].device_data(),
      make_layout(local_shape_C, local_stride_C));
  auto local_D = cute::make_tensor(
      tensor_D_arr[device_idx].device_data(),
      make_layout(local_shape_D, local_stride_D));

  // Copy over tensor tiles for the first iteration
  cutlass::device_copy(global_A_device_slice, local_A, stream);
  cutlass::device_copy(global_B_device_slice, local_B, stream);
  cutlass::device_copy(global_C_device_slice, local_C, stream);

  DistGemmArguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,                                       // mode
    problem_shape,                                                                 // problem shape
    {
      reinterpret_cast<const ElementA*>(local_A.data()),
      local_A.stride(),
      reinterpret_cast<const ElementB*>(local_B.data()),
      local_B.stride()
    },                                                                             // mainloop
    {
      Epilogue::prepare_args(scale_A.device_data(), scale_B.device_data()),        //epilogue.thread
      reinterpret_cast<const ElementC*>(local_C.data()),
      local_C.stride(),
      reinterpret_cast<const ElementD*>(local_D.data()),
      local_D.stride(),
    },                                                                             // epilogue
    {},                                                                            // hw_info
    {}                                                                             // scheduler
  };

  return arguments;
}

// Gathers results, moves back to the original full-sized D tensor on the primary device.
void gather_results(const Options<RasterOrderOptions> &options, int device_idx, cudaStream_t stream = nullptr) {

  auto problem_shape = cute::make_tuple(options.m, options.n, options.k, options.l);

  // Global dest
  auto global_D = cute::make_tensor(tensor_D.device_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_D));
  auto global_D_device_slice = DistSchedule::get_device_slice_D(global_D, device_idx);

  // Device_idx local dest
  auto local_shape_D = DistSchedule::get_local_d_shape(problem_shape);
  auto local_stride_D = cutlass::make_cute_packed_stride(StrideD{}, local_shape_D);
  auto local_D = cute::make_tensor(
      tensor_D_arr[device_idx].device_data(),
      make_layout(local_shape_D, local_stride_D)
  );

  // Copy to global dest
  cutlass::device_copy(local_D, global_D_device_slice, stream);
}

bool verify(const Options<RasterOrderOptions> &options) {
  tensor_D.sync_host();
  tensor_ref_D.sync_host();

  bool passed = false;
  if (options.eps == 0.f) {
    passed = cutlass::reference::host::TensorEquals(tensor_ref_D.host_view(), tensor_D.host_view());
  } else {
    double err = cutlass::reference::host::TensorRelativeErrorMetric(
      tensor_D.host_view(),
      tensor_ref_D.host_view());
    passed = err < 1e-5;
  }

  if (options.m <= 64 && options.n <= 64) {
    std::cout << "GEMM output:\n" << tensor_D.host_view() << "\n\n";
    std::cout << "Reference output:\n" << tensor_ref_D.host_view() << "\n\n";
  }

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options<RasterOrderOptions> &options)
{
  int primary_device_idx;
  cudaError_t device_get_result = cudaGetDevice(&primary_device_idx);
  if (device_get_result != cudaSuccess) {
    throw std::runtime_error("cudaGetDevice() failed");
  }

  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  if (num_devices < TP{}) {
      std::cerr << "Distributed GEMM is compiled with TP = " << TP::value << ", but " << 
        "found only " << num_devices << " devices." <<
        std::endl;
      exit(EXIT_FAILURE);
  }


  initialize(options);

  // Reference single-GPU GEMM
  Gemm reference_gemm;
  cutlass::device_memory::allocation<uint8_t> reference_workspace;

  auto reference_arguments = gemm_args_from_options(options);
  size_t reference_workspace_size = Gemm::get_workspace_size(reference_arguments);
  reference_workspace = cutlass::device_memory::allocation<uint8_t>(reference_workspace_size);

  CUTLASS_CHECK(reference_gemm.can_implement(reference_arguments));
  CUTLASS_CHECK(reference_gemm.initialize(reference_arguments, reference_workspace.get()));
  CUTLASS_CHECK(reference_gemm.run());

  using ElementBarrier = typename DistGemm::ElementBarrier;
  using ElementFlag = typename DistGemmKernel::ElementFlag;

  // Set up per-device streams
  cudaStream_t stream_arr[TP{}];

  for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream_arr[device_idx]));
  }

    // Instantiate DistGEMM
  DistGemm dist_gemm_arr[TP{}];  // Distributed GEMM array for multiple devices

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace_arr[TP{}];
  cutlass::device_memory::allocation<uint8_t> exclusive_workspace_arr[TP{}];

  // Cross-device workspace pointer array for gemm.initialize()
  void * workspace_ptr_arr[TP{}];
  void * exclusive_workspace_ptr_arr[TP{}];

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  DistGemmArguments arguments_[TP{}];

  for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));

    arguments_[device_idx] = dist_gemm_args_from_options(options, device_idx, stream_arr[device_idx]);

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = DistGemm::get_workspace_size(arguments_[device_idx]);
    size_t exclusive_workspace_size = DistGemm::get_exclusive_workspace_size();

    workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(workspace_size);
    exclusive_workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(exclusive_workspace_size);

    // Throw workspace pointers into arrays for gemm.initialize()
    workspace_ptr_arr[device_idx] = workspace_arr[device_idx].get();
    exclusive_workspace_ptr_arr[device_idx] = exclusive_workspace_arr[device_idx].get();

    // Zero out exclusive workspace
    cudaMemsetAsync(exclusive_workspace_ptr_arr[device_idx], 0, exclusive_workspace_size, stream_arr[device_idx]);

    cudaDeviceSynchronize();
  }

  for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));

    // Check if the problem size is supported or not
    CUTLASS_CHECK(dist_gemm_arr[device_idx].can_implement(arguments_[device_idx]));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(dist_gemm_arr[device_idx].initialize(
          arguments_,
          workspace_ptr_arr,
          exclusive_workspace_ptr_arr,
          device_idx,
          stream_arr[device_idx],
#ifdef CUTLASS_ENABLE_GDC_FOR_SM90
          /* launch_with_pdl = */ true
#else
          /* launch_with_pdl = */ false
#endif
          ));

    cudaDeviceSynchronize();
  }

  // Correctness / Warmup iteration
  std::cout << std::endl << "  running DistGEMM..." << std::endl;

  for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
  }
  for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
    CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    CUDA_CHECK(cudaGetLastError());
    gather_results(options, device_idx);
  }

  std::cout << "  running DistGEMM finished without runtime errors" << std::endl;

  //// Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;

  result.passed = verify(options);

  std::cout << std::endl << "  Disposition (eps: " << options.eps << "): " << 
    (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
        float elapsed_ms = 0.f;

    // Warmup
    std::cout << "  Warming up for " << options.warmup_iterations << " iterations." << std::endl;
    for (int warmup_iter = 0; warmup_iter < options.warmup_iterations; ++warmup_iter) {
      for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
      }
    }

    for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
      CUDA_CHECK(cudaSetDevice(device_idx));
      CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    }

    CUDA_CHECK(cudaSetDevice(primary_device_idx));

    // Benchmark
    std::cout << "  Profiling for " << options.iterations << " iterations." << std::endl;
    using AtomicBoolean = cuda::atomic<bool>;
    AtomicBoolean* atomic_flag_ptr;
    CUDA_CHECK(cudaHostAlloc(&atomic_flag_ptr, sizeof(AtomicBoolean), cudaHostAllocPortable));
    atomic_flag_ptr->store(false);

    cutlass::DistGpuTimer<TP{}> timer;

    for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
      CUDA_CHECK(cudaSetDevice(device_idx));
      cutlass::delay_kernel<<<1, 1, 0, stream_arr[device_idx]>>>(atomic_flag_ptr);
      CUDA_CHECK(cudaGetLastError());
    }

    for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
      timer.start(device_idx, stream_arr[device_idx]);
    }

    atomic_flag_ptr->store(true);

    for (int profile_iter = 0; profile_iter < options.iterations; ++profile_iter) {
      for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
      }
    }

    for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
      CUDA_CHECK(cudaSetDevice(device_idx));
      timer.stop(device_idx, stream_arr[device_idx]);
    }

    CUDA_CHECK(cudaSetDevice(primary_device_idx));

    for (int device_idx = 0; device_idx < TP{}; ++device_idx) {
      elapsed_ms = max(elapsed_ms, timer.elapsed_millis(device_idx));
    }

    // Compute average runtime and TFLOPs.
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    double avg_runtime_s = (double)(result.avg_runtime_ms / 1000.0);
    // divide by TP to factor for the number of devices
    result.tflops = options.tflops(avg_runtime_s) / TP{};

    auto [local_M, local_N, local_K, local_L] = DistSchedule::get_local_gemm_shape(
        cute::make_tuple(options.m, options.n, options.k, options.l));

    std::cout << std::endl;
    std::cout << "  TP: " << TP::value << std::endl;
    std::cout << "  Problem Size: " << 
      options.m << " x " << 
      options.n << " x " << 
      options.k << " x " << 
      options.l << std::endl;
    std::cout << "  Local GEMM Problem Size: " << 
      local_M << " x " << 
      local_N << " x " << 
      local_K << " x " << 
      local_L<< std::endl;

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  TFLOPS: " << result.tflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA Toolkit 12.5 or newer to run this example
  // and must have compute capability at least 90.
  // Some necessary cuda graph APIs were only introduced in CUDA 12.4.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 4)) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major < 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }
  //
  // Parse options
  //

  Options<RasterOrderOptions> options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  run<Gemm>(options);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
