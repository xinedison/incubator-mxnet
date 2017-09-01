/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file convolution.cu
 * \brief
 * \author Bing Xu, Jun Wu
*/

#include "./convolution-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "./cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

#include "./depthwise_convolution-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  // If 1D convolution, use MXNet implementation
  if (param.kernel.ndim() == 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new ConvolutionOp<gpu, DType>(param);
    })
    return op;
  }

  // depth wise conv
  if (param.num_filter == param.num_group &&
      param.layout.value() == mshadow::kNCHW &&
      param.num_filter == (*in_shape)[conv::kData][1] &&
      param.kernel.ndim() == 2 &&
      param.dilate == mshadow::Shape2(1, 1) &&
      dtype == mshadow::kFloat32) {
    op = new DepthwiseConvolutionOp<float>(param, *in_shape, *out_shape);
    return op;
  }

#if MXNET_USE_CUDNN == 1
  // The NVIDIA Pascal architecture was the first to include 16-bit ALUs.
  // Thus, when the framework is compiled with MSHADOW_USE_PASCAL == 1, we
  // perform the convolution calculation in 16-bit when the tensor type is
  // also 16-bit.  For NVIDIA architectures earlier than Pascal (so Maxwell
  // and Kepler), the computation precision is always at least 32-bits.
#if MSHADOW_USE_PASCAL == 1
  // true fp16
  int desired_forward_compute_type = dtype;
  int desired_backward_compute_type = dtype;
#else
  // pseudo fp16
  int desired_forward_compute_type =
    (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;
  int desired_backward_compute_type =
    (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;
#endif  // MSHADOW_USE_PASCAL == 1

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cudnn_off) {
      op = new ConvolutionOp<gpu, DType>(param);
    } else {
      int forward_compute_type = desired_forward_compute_type;
      int backward_compute_type = desired_backward_compute_type;
      bool convolutionIsSupported = CuDNNConvolutionOp<DType>::Supports(param,
                                          forward_compute_type,
                                          backward_compute_type, ctx);

      // If cuDNN can't handle this case with fp16 backprop kernels, try fp32 backprop.
      if (!convolutionIsSupported && backward_compute_type == mshadow::kFloat16) {
        backward_compute_type = mshadow::kFloat32;
        convolutionIsSupported = CuDNNConvolutionOp<DType>::Supports(param,
                                          forward_compute_type,
                                          backward_compute_type, ctx);
      }

      // If cuDNN can't handle this case with fp16 forward kernels, try fp32
      if (!convolutionIsSupported && forward_compute_type == mshadow::kFloat16) {
        forward_compute_type = mshadow::kFloat32;
        convolutionIsSupported = CuDNNConvolutionOp<DType>::Supports(param,
                                          forward_compute_type,
                                          backward_compute_type, ctx);
      }
      if (!convolutionIsSupported) {
        LOG(WARNING) << "This convolution is not supported by cudnn, MXNET convolution is applied.";
        op = new ConvolutionOp<gpu, DType>(param);
      } else {
        if (forward_compute_type != desired_forward_compute_type)
          LOG(WARNING) << "Requested forward compute precision not supported, using fp32.";
        if (backward_compute_type != desired_backward_compute_type)
          LOG(WARNING) << "Requested backward compute precision not supported, using fp32.";
        op = new CuDNNConvolutionOp<DType>(param,
                                         forward_compute_type,
                                         backward_compute_type,
                                         *in_shape, *out_shape, ctx);
      }
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConvolutionOp<gpu, DType>(param);
  })
#endif  // MXNET_USE_CUDNN
  return op;
}

}  // namespace op
}  // namespace mxnet

