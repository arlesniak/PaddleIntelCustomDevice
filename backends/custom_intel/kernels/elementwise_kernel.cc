// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"
#include <CL/sycl.hpp>

namespace custom_kernel {

// template <typename T>
// void MultiplyRawKernel(const phi::Context& dev_ctx,
//                        const phi::DenseTensor& x,
//                        const phi::DenseTensor& y,
//                        int axis,
//                        phi::DenseTensor* out) {
//   auto x_dims = x.dims();
//   auto y_dims = y.dims();
//   auto dst_dims = phi::BroadcastDims(axis, x_dims, y_dims);
//   phi::DenseTensor tmp_x, tmp_y;
//   phi::BroadcastTo<T>(dev_ctx, x, dst_dims, axis, &tmp_x);
//   phi::BroadcastTo<T>(dev_ctx, y, dst_dims, axis, &tmp_y);

//   auto x_data = tmp_x.data<T>();
//   auto y_data = tmp_y.data<T>();
//   auto out_data = dev_ctx.template Alloc<T>(out);
//   auto numel = out->numel();
//   for (auto i = 0; i < numel; ++i) {
//     out_data[i] = x_data[i] * y_data[i];
//   }
// }

// template <typename T>
// void MultiplyKernel(const phi::Context& dev_ctx,
//                     const phi::DenseTensor& x,
//                     const phi::DenseTensor& y,
//                     phi::DenseTensor* out) {
//   int axis = -1;
//   MultiplyRawKernel<T>(dev_ctx, x, y, axis, out);
// }

template <typename T>
void MultiplyRawKernelGPU(const phi::Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {

  void* stream = static_cast<void*>(dev_ctx.stream());
  T* out_data = dev_ctx.HostAlloc<T>(out);
  // // std::cout << "out_data=" << out_data << std::endl;
  
  auto ile = out->numel();
  // std::cout << "ile=" << ile << std::endl;
  auto input_x = x.data<T>();
  auto input_y = y.data<T>();
  // std::cout << "y[0]=" << input_y[0] << std::endl;

  auto* q =  static_cast<sycl::queue*>(stream);
  // sycl::queue qu;
  // auto q = &qu;

  // // auto* q =  static_cast<sycl::queue*>(stream);
  T* input_x_gpu = sycl::malloc_device<T>(ile, *q);
  T* input_y_gpu = sycl::malloc_device<T>(ile, *q);
  T* gpu_mem = sycl::malloc_device<T>(ile, *q);

  q->submit([&](sycl::handler& h) {
    q->memcpy(input_x_gpu, input_x, sizeof(T)*ile);
    q->memcpy(input_y_gpu, input_y, sizeof(T)*ile);
  });
  q-> wait();
  q->submit([&](sycl::handler& h) {
    // h.parallel_for(ile, [=,input_x_local=this->input_x, input_y_local=this->input_y, gpu_mem_local=this->gpu_mem](sycl::id<1> i){
    // h.parallel_for(ile, [=,input_x, input_y, gpu_mem](sycl::id<1> i){      
    // h.parallel_for(ile, [=,gpu_mem_local=gpu_mem](sycl::id<1> i){      
    
    // h.parallel_for(ile, [=](sycl::id<1> i){     
    h.parallel_for(ile, [=](auto& i){           
      // ile++;       
        gpu_mem[i] = input_x_gpu[i] * input_y_gpu[i];
        // gpu_mem[i] = input_y_gpu[i];
        // gpu_mem[i] = 0.25f;
        // gpu_mem[i] = input_x[i] * input_y[i];
    });
  });
  q-> wait();

  // q.submit([&](sycl::handler& h) {
  // });
  // q.wait();
  // // sycl::free(gpu_mem, *q);


  q->submit([&](sycl::handler &h){
    h.memcpy(out_data, gpu_mem, sizeof(T)*ile);
  });
  q->wait();
  sycl::free(gpu_mem, *q);

  // auto x_dims = x.dims();
  // auto y_dims = y.dims();
  // auto dst_dims = phi::BroadcastDims(axis, x_dims, y_dims);
  // phi::DenseTensor tmp_x, tmp_y;
  // phi::BroadcastTo<T>(dev_ctx, x, dst_dims, axis, &tmp_x);
  // phi::BroadcastTo<T>(dev_ctx, y, dst_dims, axis, &tmp_y);

  // auto x_data = tmp_x.data<T>();
  // auto y_data = tmp_y.data<T>();
  // auto out_data = dev_ctx.template Alloc<T>(out);
  // auto numel = out->numel();
  // for (auto i = 0; i < numel; ++i) {
  //   out_data[i] = x_data[i] * y_data[i];
  // }
}

template <typename T>
void MultiplyKernelGPU(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  int axis = -1;
  MultiplyRawKernelGPU<T>(dev_ctx, x, y, axis, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(multiply_raw,
                    custom_intel,
                    ALL_LAYOUT,
                    custom_kernel::MultiplyRawKernelGPU,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(multiply,
                    custom_intel,
                    ALL_LAYOUT,
                    custom_kernel::MultiplyKernelGPU,
                    int32_t,
                    int64_t,
                    float,
                    double) {}


// PD_BUILD_PHI_KERNEL(multiply_raw,
//                     custom_intel,
//                     ALL_LAYOUT,
//                     custom_kernel::MultiplyRawKernel,
//                     int32_t,
//                     int64_t,
//                     float,
//                     double) {}

// PD_BUILD_PHI_KERNEL(multiply,
//                     custom_intel,
//                     ALL_LAYOUT,
//                     custom_kernel::MultiplyKernel,
//                     int32_t,
//                     int64_t,
//                     float,
//                     double) {}
