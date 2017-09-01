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
 * \file linalg.h
 * \brief Implementation of unified tensor interface for advanced linear algebra functions
 * (specifically BLAS3/LAPACK) from within mxnet.
 */
#ifndef MXNET_OPERATOR_LINALG_IMPL_H_
#define MXNET_OPERATOR_LINALG_IMPL_H_

#include <mxnet/op_attr_types.h>

#include <algorithm>

#include "../common/cuda_utils.h"

// Convenience functions.
inline void linalg_check_batch_size(int A, int B, int C) {
  CHECK_EQ(A, B) << "Inconsistent batch size between arguments to linear algebra operator";
  CHECK_EQ(A, C) << "Inconsistent batch size between arguments to linear algebra operator";
  CHECK_GT(A, 0) << "Zero batch size for arguments to linear algebra operator";
}

//////////////////////////////// GEMM ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "gemm". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is C = gemm(A,B,C), so C is input and output parameter.

template<typename xpu, typename DType>
inline void check_gemm(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                const Tensor<xpu, 2, DType>& C, DType alpha, DType beta, bool tA, bool tB) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ((tA ? A.size(1) : A.size(0)), C.size(0))
    << "Non compatible matrix dimensions between inputs A and C for gemm";
  CHECK_EQ((tB ? B.size(0) : B.size(1)), C.size(1))
    << "Non compatible matrix dimensions between inputs B and C for gemm";
  CHECK_EQ((tA ? A.size(0) : A.size(1)), (tB ? B.size(1) : B.size(0)))
    << "Non compatible matrix dimensions between inputs A and B for gemm";
}

#define LINALG_CPU_GEMM(fname, DType) \
template<> inline \
void linalg_gemm<cpu, DType>(const Tensor<cpu, 2, DType>& A, const Tensor<cpu, 2, DType>& B, \
                             const Tensor<cpu, 2, DType>& C, DType alpha, DType beta, \
                             bool tA, bool tB, Stream<cpu> *s) { \
  check_gemm(A, B, C, alpha, beta, tA, tB); \
  cblas_##fname(CblasRowMajor, (tA ? CblasTrans : CblasNoTrans), (tB ? CblasTrans : CblasNoTrans), \
                C.size(0), C.size(1), (tA ? A.size(0) : A.size(1)), alpha, \
                A.dptr_, A.stride_, B.dptr_, B.stride_, beta, C.dptr_, C.stride_); \
}
LINALG_CPU_GEMM(sgemm, float)
LINALG_CPU_GEMM(dgemm, double)

#define LINALG_CPU_BATCH_GEMM(DType) \
template<> inline \
void linalg_batch_gemm<cpu, DType>(const Tensor<cpu, 3, DType>& A, const Tensor<cpu, 3, DType>& B, \
                                   const Tensor<cpu, 3, DType>& C, DType alpha, DType beta, \
                                   bool tA, bool tB, Stream<cpu> *s) { \
  linalg_check_batch_size(A.size(0), B.size(0), C.size(0)); \
  for (index_t i = 0; i < A.size(0); ++i) { \
    linalg_gemm(A[i], B[i], C[i], alpha, beta, tA, tB); \
  } \
}
LINALG_CPU_BATCH_GEMM(float)
LINALG_CPU_BATCH_GEMM(double)

#ifdef __CUDACC__

template<typename DType>
__global__ void linalgCollectBatchOffsetsGPU(DType *a[], DType* b, int stride, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
      a[i] = b + i * stride;
    }
}

// cublas col-major processing accounted for by switching first two operands

#define LINALG_GPU_GEMM(fname, DType) \
template<> inline \
void linalg_gemm<gpu, DType>(const Tensor<gpu, 2, DType>& A, const Tensor<gpu, 2, DType>& B, \
                             const Tensor<gpu, 2, DType>& C, DType alpha, DType beta, \
                             bool tA, bool tB, Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  check_gemm(A, B, C, alpha, beta, tA, tB); \
  CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s), \
                            (tB ? CUBLAS_OP_T : CUBLAS_OP_N), \
                            (tA ? CUBLAS_OP_T : CUBLAS_OP_N), \
                            C.size(1), C.size(0), (tB ? B.size(1) : B.size(0)), \
                            &alpha, B.dptr_, B.stride_, A.dptr_, A.stride_, \
                            &beta, C.dptr_, C.stride_)) \
}
LINALG_GPU_GEMM(Sgemm, float)
LINALG_GPU_GEMM(Dgemm, double)

// Specialization of linalg_gemm<gpu, DType> for DType=mshadow::half::half_t.
template<> inline
void linalg_gemm<gpu, mshadow::half::half_t>(const Tensor<gpu, 2, mshadow::half::half_t>& A,
                                             const Tensor<gpu, 2, mshadow::half::half_t>& B,
                                             const Tensor<gpu, 2, mshadow::half::half_t>& C,
                                             mshadow::half::half_t alpha,
                                             mshadow::half::half_t beta,
                                             bool tA, bool tB, Stream<gpu> *s) {
  using namespace mxnet;
  using mshadow::gpu;
  CHECK_NOTNULL(s);
  check_gemm(A, B, C, alpha, beta, tA, tB);

#if CUDA_VERSION >= 7050
  auto blas_handle = Stream<gpu>::GetBlasHandle(s);
#if CUDA_VERSION >= 9000
  auto cublas_math_mode = GetEnvAllowTensorCore() ? CUBLAS_TENSOR_OP_MATH
                                                  : CUBLAS_DEFAULT_MATH;
  auto previous_math_mode = SetCublasMathMode(blas_handle, cublas_math_mode);
#endif

  // pseudo-fp16 (fp32 math with fp16 I/O)
  float alpha_f = float(alpha);  // NOLINT(*)
  float beta_f = float(beta);  // NOLINT(*)

  // As of cuda8, cublas adopted the cuda datatype, rather than maintaining its own datatype.
#if CUDA_VERSION >= 8000
  cudaDataType_t half_datatype = CUDA_R_16F;
#else
  cublasDataType_t half_datatype = CUBLAS_DATA_HALF;
#endif
  CUBLAS_CALL(cublasSgemmEx(blas_handle,
                            (tB ? CUBLAS_OP_T : CUBLAS_OP_N),
                            (tA ? CUBLAS_OP_T : CUBLAS_OP_N),
                            C.size(1), C.size(0), (tB ? B.size(1) : B.size(0)),
                            &alpha_f,
                            B.dptr_, half_datatype, B.stride_,
                            A.dptr_, half_datatype, A.stride_,
                            &beta_f,
                            C.dptr_, half_datatype, C.stride_));
#if CUDA_VERSION >= 9000
  SetCublasMathMode(blas_handle, previous_math_mode);
#endif
#else
  LOG(FATAL) << "FP16 gemm requires CUDA version >= 7.5!";
#endif  // CUDA_VERSION >= 7050
}


#define LINALG_GPU_BATCH_GEMM(fname, DType) \
template<> inline \
void linalg_batch_gemm<gpu, DType>(const Tensor<gpu, 3, DType>& A, const Tensor<gpu, 3, DType>& B, \
                                   const Tensor<gpu, 3, DType>& C, DType alpha, DType beta, \
                                   bool tA, bool tB, Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  linalg_check_batch_size(A.size(0), B.size(0), C.size(0)); \
  check_gemm(A[0], B[0], C[0], alpha, beta, tA, tB); \
  Storage::Handle offsetsA, offsetsB, offsetsC; \
  offsetsA = Storage::Get()->Alloc(sizeof(DType*)*A.size(0), Context::GPU()); \
  offsetsB = Storage::Get()->Alloc(sizeof(DType*)*B.size(0), Context::GPU()); \
  offsetsC = Storage::Get()->Alloc(sizeof(DType*)*C.size(0), Context::GPU()); \
  using namespace mshadow::cuda; \
  int ngrid = std::min(kMaxGridNum, \
                       static_cast<int>((A.size(0) + kBaseThreadNum - 1) / kBaseThreadNum)); \
  linalgCollectBatchOffsetsGPU<<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>> \
    (static_cast<DType **>(offsetsA.dptr), A.dptr_, A.size(1)*A.stride_, A.size(0)); \
  linalgCollectBatchOffsetsGPU<<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>> \
    (static_cast<DType **>(offsetsB.dptr), B.dptr_, B.size(1)*B.stride_, B.size(0)); \
  linalgCollectBatchOffsetsGPU<<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>> \
    (static_cast<DType **>(offsetsC.dptr), C.dptr_, C.size(1)*C.stride_, C.size(0)); \
  CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s), \
                            (tB ? CUBLAS_OP_T : CUBLAS_OP_N), \
                            (tA ? CUBLAS_OP_T : CUBLAS_OP_N), \
                            C.size(2), C.size(1), (tB ? B.size(2) : B.size(1)), \
                            &alpha, static_cast<const DType **>(offsetsB.dptr), B.stride_, \
                            static_cast<const DType **>(offsetsA.dptr),  A.stride_, \
                            &beta, static_cast<DType **>(offsetsC.dptr), C.stride_, A.size(0))) \
  Storage::Get()->Free(offsetsA); \
  Storage::Get()->Free(offsetsB); \
  Storage::Get()->Free(offsetsC); \
}
LINALG_GPU_BATCH_GEMM(SgemmBatched, float)
LINALG_GPU_BATCH_GEMM(DgemmBatched, double)

#endif

//////////////////////////////// TRSM ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "trsm". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is B = trsm(A,B), so B is input and output parameter.

template<typename xpu, typename DType>
inline void check_trsm(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                DType alpha, bool rightside, bool lower, bool transpose) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ(A.size(0), A.size(1))
    << "First input of trsm is not a square matrix.";
  CHECK(!rightside || (B.size(1) == A.size(0)))
    << "Non compatible matrix dimensions between inputs A and B for trsm";
  CHECK(rightside || (B.size(0) == A.size(1)))
    << "Non compatible matrix dimensions between inputs A and B for trsm";
}

#define LINALG_CPU_TRSM(fname, DType) \
template<> inline \
void linalg_trsm<cpu, DType>(const Tensor<cpu, 2, DType>& A, const Tensor<cpu, 2, DType>& B, \
                 DType alpha, bool rightside, bool lower, bool transpose, Stream<cpu> *s) { \
  check_trsm(A, B, alpha, rightside, lower, transpose); \
  cblas_##fname(CblasRowMajor, (rightside ? CblasRight : CblasLeft), \
                (lower ? CblasLower : CblasUpper), (transpose ? CblasTrans : CblasNoTrans), \
                CblasNonUnit, B.size(0), B.size(1), alpha, A.dptr_, \
                A.stride_, B.dptr_, B.stride_); \
}
LINALG_CPU_TRSM(strsm, float)
LINALG_CPU_TRSM(dtrsm, double)

#define LINALG_CPU_BATCH_TRSM(DType) \
template<> inline \
void linalg_batch_trsm<cpu, DType>(const Tensor<cpu, 3, DType>& A, const Tensor<cpu, 3, DType>& B, \
                   DType alpha, bool rightside, bool lower, bool transpose, Stream<cpu> *s) { \
  linalg_check_batch_size(A.size(0), B.size(0), B.size(0)); \
  for (index_t i = 0; i < A.size(0); ++i) { \
    linalg_trsm(A[i], B[i], alpha, rightside, lower, transpose); \
  } \
}
LINALG_CPU_BATCH_TRSM(float)
LINALG_CPU_BATCH_TRSM(double)

#ifdef __CUDACC__

// cublas col-major processing accounted for by switching sides and fill mode

#define LINALG_GPU_TRSM(fname, DType) \
template<> inline \
void linalg_trsm<gpu, DType>(const Tensor<gpu, 2, DType>& A, const Tensor<gpu, 2, DType>& B, \
                 DType alpha, bool rightside, bool lower, bool transpose, Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  check_trsm(A, B, alpha, rightside, lower, transpose); \
  CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s), \
                            (rightside ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT), \
                            (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER), \
                            (transpose ? CUBLAS_OP_T : CUBLAS_OP_N), \
                            CUBLAS_DIAG_NON_UNIT, B.size(1), B.size(0), &alpha, \
                            A.dptr_, A.stride_, B.dptr_, B.stride_)); \
}
LINALG_GPU_TRSM(Strsm, float)
LINALG_GPU_TRSM(Dtrsm, double)

#define LINALG_GPU_BATCH_TRSM(fname, DType) \
template<> inline \
void linalg_batch_trsm<gpu, DType>(const Tensor<gpu, 3, DType>& A, const Tensor<gpu, 3, DType>& B, \
                   DType alpha, bool rightside, bool lower, bool transpose, Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  linalg_check_batch_size(A.size(0), B.size(0), B.size(0)); \
  check_trsm(A[0], B[0], alpha, rightside, lower, transpose); \
  Storage::Handle offsetsA, offsetsB; \
  offsetsA = Storage::Get()->Alloc(sizeof(DType*)*A.size(0), Context::GPU()); \
  offsetsB = Storage::Get()->Alloc(sizeof(DType*)*B.size(0), Context::GPU()); \
  using namespace mshadow::cuda; \
  int ngrid = std::min(kMaxGridNum, \
                       static_cast<int>((A.size(0) + kBaseThreadNum - 1) / kBaseThreadNum)); \
  linalgCollectBatchOffsetsGPU<<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>> \
    (static_cast<DType **>(offsetsA.dptr), A.dptr_, A.size(1)*A.stride_, A.size(0)); \
  linalgCollectBatchOffsetsGPU<<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>> \
    (static_cast<DType **>(offsetsB.dptr), B.dptr_, B.size(1)*B.stride_, A.size(0)); \
  CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s), \
                            (rightside ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT), \
                            (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER), \
                            (transpose ? CUBLAS_OP_T : CUBLAS_OP_N), \
                            CUBLAS_DIAG_NON_UNIT, B.size(2), B.size(1), &alpha, \
                            static_cast<const DType **>(offsetsA.dptr), A.stride_, \
                            static_cast<DType **>(offsetsB.dptr), B.stride_, A.size(0))); \
  Storage::Get()->Free(offsetsA); \
  Storage::Get()->Free(offsetsB); \
}
LINALG_GPU_BATCH_TRSM(StrsmBatched, float)
LINALG_GPU_BATCH_TRSM(DtrsmBatched, double)

#endif

/*!
 * \brief Performs gemm, setting alpha and beta as appropriate for `req`.
 *
 * \param A the first operand of the gemm
 * \param B the second operand of the gemm
 * \param C the data to be assigned
 * \tparam tA whether the `A` operand should be transposed first.
 * \tparam tB whether the `B` operand should be transposed first.
 * \tparam s the stream to perform the operation
 * \param req the assignment request
 */
template<typename xpu, typename DType>
inline void linalg_gemm(const Tensor<xpu, 2, DType>& A,
                        const Tensor<xpu, 2, DType>& B,
                        const Tensor<xpu, 2, DType>& C,
                        bool tA, bool tB, Stream<xpu> *s,
                        mxnet::OpReqType req) {
  using namespace mxnet;
  switch (req) {
    case kNullOp:
      break;
    case kWriteTo:
    case kWriteInplace:
      linalg_gemm(A, B, C, DType(1.0), DType(0.0), tA, tB, s);
      break;
    case kAddTo:
      linalg_gemm(A, B, C, DType(1.0), DType(1.0), tA, tB, s);
      break;
    default:
      LOG(FATAL) << "not reached";
  }
}

//////////////////////////////// TRMM ////////////////////////////////////////////

// CPU/GPU-versions of BLAS3 function "trmm". Please refer to the BLAS3-documentation
// for further information about the function and its parameters.
// Note that this is B = trmm(A,B), so B is input and output parameter.

template<typename xpu, typename DType>
inline void check_trmm(const Tensor<xpu, 2, DType>& A, const Tensor<xpu, 2, DType>& B,
                DType alpha, bool rightside, bool lower, bool transpose) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ(A.size(0), A.size(1))
    << "First input of trmm is not a square matrix.";
  CHECK(!rightside || (B.size(1) == A.size(0)))
    << "Non compatible matrix dimensions between inputs A and B for trmm";
  CHECK(rightside || (B.size(0) == A.size(1)))
    << "Non compatible matrix dimensions between inputs A and B for trmm";
}

#define LINALG_CPU_TRMM(fname, DType) \
template<> inline \
void linalg_trmm<cpu, DType>(const Tensor<cpu, 2, DType>& A, const Tensor<cpu, 2, DType>& B, \
                 DType alpha, bool rightside, bool lower, bool transpose, Stream<cpu> *s) { \
  check_trmm(A, B, alpha, rightside, lower, transpose); \
  cblas_##fname(CblasRowMajor, (rightside ? CblasRight : CblasLeft), \
                (lower ? CblasLower : CblasUpper), (transpose ? CblasTrans : CblasNoTrans), \
                CblasNonUnit, B.size(0), B.size(1), alpha, A.dptr_, \
                A.stride_, B.dptr_, B.stride_); \
}
LINALG_CPU_TRMM(strmm, float)
LINALG_CPU_TRMM(dtrmm, double)

#define LINALG_XPU_BATCH_TRMM(xpu, DType) \
template<> inline \
void linalg_batch_trmm<xpu, DType>(const Tensor<xpu, 3, DType>& A, const Tensor<xpu, 3, DType>& B, \
                    DType alpha, bool rightside, bool lower, bool transpose, Stream<xpu> *s) { \
  linalg_check_batch_size(A.size(0), B.size(0), B.size(0)); \
  for (index_t i = 0; i < A.size(0); ++i) { \
    linalg_trmm(A[i], B[i], alpha, rightside, lower, transpose, s); \
  } \
}
LINALG_XPU_BATCH_TRMM(cpu, float)
LINALG_XPU_BATCH_TRMM(cpu, double)

#ifdef __CUDACC__

// cublas col-major processing accounted for by switching sides and fill mode
// doing in-place computation by supplying B as second and third matrix
#define LINALG_GPU_TRMM(fname, DType) \
template<> inline \
void linalg_trmm<gpu, DType>(const Tensor<gpu, 2, DType>& A, const Tensor<gpu, 2, DType>& B, \
                 DType alpha, bool rightside, bool lower, bool transpose, Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  check_trmm(A, B, alpha, rightside, lower, transpose); \
  CUBLAS_CALL(cublas##fname(Stream<gpu>::GetBlasHandle(s), \
                            (rightside ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT), \
                            (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER), \
                            (transpose ? CUBLAS_OP_T : CUBLAS_OP_N), \
                            CUBLAS_DIAG_NON_UNIT, B.size(0), B.size(1), &alpha, \
                            A.dptr_, A.stride_, B.dptr_, B.stride_, \
                            B.dptr_, B.stride_)); \
}
LINALG_GPU_TRMM(Strmm, float)
LINALG_GPU_TRMM(Dtrmm, double)

LINALG_XPU_BATCH_TRMM(gpu, float)
LINALG_XPU_BATCH_TRMM(gpu, double)

#endif

//////////////////////////////// POTRF ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "potrf". Please refer to the LAPACK-documentation
// for further information about the function and its parameters.
// Note that this is A = potrf(A), so A is input and output parameter.

template<typename xpu, typename DType>
inline void check_potrf(const Tensor<xpu, 2, DType>& A, bool lower) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ(A.size(0), A.size(1))
    << "No square matrix as input to potrf.";
}

#define LINALG_CPU_POTRF(fname, DType) \
template<> inline \
void linalg_potrf<cpu, DType>(const Tensor<cpu, 2, DType>& A, bool lower, Stream<cpu> *s) { \
  check_potrf(A, lower); \
  int ret(MXNET_LAPACK_##fname(MXNET_LAPACK_ROW_MAJOR, (lower ? 'L' : 'U'), A.size(0),  \
          A.dptr_ , A.stride_)); \
  CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu."; \
}
LINALG_CPU_POTRF(spotrf, float)
LINALG_CPU_POTRF(dpotrf, double)

#define LINALG_CPU_BATCH_POTRF(DType) \
template<> inline \
void linalg_batch_potrf<cpu, DType>(const Tensor<cpu, 3, DType>& A, bool lower, Stream<cpu> *s) { \
  for (index_t i = 0; i < A.size(0); ++i) { \
    linalg_potrf(A[i], lower); \
  } \
}
LINALG_CPU_BATCH_POTRF(float)
LINALG_CPU_BATCH_POTRF(double)

#if MXNET_USE_CUSOLVER == 1

#define LINALG_GPU_BUFFSIZE_POTRF(fname, DType) \
inline int linalg_potrf_buffsize(const Tensor<gpu, 2, DType>& A, bool lower, Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  int buffsize(0); \
  CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s), \
                                (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER), \
                                 A.size(0), A.dptr_, A.stride_, &buffsize)); \
  return buffsize;  \
}
LINALG_GPU_BUFFSIZE_POTRF(DnSpotrf_bufferSize, float)
LINALG_GPU_BUFFSIZE_POTRF(DnDpotrf_bufferSize, double)

#define LINALG_GPU_POTRF(fname, DType) \
template<> inline \
void linalg_potrf<gpu, DType>(const Tensor<gpu, 2, DType>& A, bool lower, Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  check_potrf(A, lower); \
  int buffsize(linalg_potrf_buffsize(A, lower, s)); \
  Storage::Handle buffer = Storage::Get()->Alloc(sizeof(DType)*buffsize, Context::GPU()); \
  Storage::Handle info = Storage::Get()->Alloc(sizeof(int), Context::GPU()); \
  CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s), \
                (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER), \
                A.size(0), A.dptr_, A.stride_, static_cast<DType *>(buffer.dptr), buffsize, \
                static_cast<int *>(info.dptr))); \
  Storage::Get()->Free(buffer); \
  Storage::Get()->Free(info); \
}
LINALG_GPU_POTRF(DnSpotrf, float)
LINALG_GPU_POTRF(DnDpotrf, double)

#define LINALG_GPU_BATCH_POTRF(fname, DType) \
template<> inline \
void linalg_batch_potrf<gpu, DType>(const Tensor<gpu, 3, DType>& A, bool lower, Stream<gpu> *s) { \
  using namespace mxnet; \
  using mshadow::gpu; \
  CHECK_NOTNULL(s); \
  CHECK_GT(A.size(0), 0); \
  check_potrf(A[0], lower); \
  int buffsize(linalg_potrf_buffsize(A[0], lower, s)); \
  Storage::Handle buffer = Storage::Get()->Alloc(sizeof(DType)*buffsize, Context::GPU()); \
  Storage::Handle info = Storage::Get()->Alloc(sizeof(int), Context::GPU()); \
  for (mshadow::index_t i = 0; i < A.size(0); ++i) { \
    CUSOLVER_CALL(cusolver##fname(Stream<gpu>::GetSolverHandle(s), \
                 (lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER), \
                 A[i].size(0), A[i].dptr_, A[i].stride_, \
                 static_cast<DType *>(buffer.dptr), buffsize, static_cast<int *>(info.dptr))); \
  } \
  Storage::Get()->Free(buffer); \
  Storage::Get()->Free(info); \
}
LINALG_GPU_BATCH_POTRF(DnSpotrf, float)
LINALG_GPU_BATCH_POTRF(DnDpotrf, double)

#endif

//////////////////////////////// POTRI ////////////////////////////////////////////

// CPU/GPU-versions of LAPACK function "potri". Please refer to the LAPACK-documentation
// for further information about the function and its parameters.
// Note that this is A = potri(A), so A is input and output parameter.

template<typename xpu, typename DType>
inline void check_potri(const Tensor<xpu, 2, DType>& A, bool lower) {
  // Any checking that helps user debug potential problems.
  CHECK_EQ(A.size(0), A.size(1)) << "No square matrix as input to potri.";
}

#define LINALG_CPU_POTRI(fname, DType) \
template<> inline \
void linalg_potri<cpu, DType>(const Tensor<cpu, 2, DType>& A, bool lower, Stream<cpu> *s) { \
  check_potri(A, lower); \
  int ret(MXNET_LAPACK_##fname(MXNET_LAPACK_ROW_MAJOR, (lower ? 'L' : 'U'), A.size(0),  \
          A.dptr_ , A.stride_)); \
  CHECK_EQ(ret, 0) << #fname << " failed in lapack on cpu."; \
}
LINALG_CPU_POTRI(spotri, float)
LINALG_CPU_POTRI(dpotri, double)

#define LINALG_CPU_BATCH_POTRI(DType) \
template<> inline \
void linalg_batch_potri<cpu, DType>(const Tensor<cpu, 3, DType>& A, bool lower, Stream<cpu> *s) { \
  for (index_t i = 0; i < A.size(0); ++i) { \
    linalg_potri(A[i], lower); \
  } \
}
LINALG_CPU_BATCH_POTRI(float)
LINALG_CPU_BATCH_POTRI(double)

#ifdef __CUDACC__

// Initializes multiple identity matrices on the same vector.
template<typename DType>
__global__ void linalgInitIdentityGPU(DType *a, int stride, int lda, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
      // index relative to the matrix.
      int index(i % stride);
      a[i] = (index / lda == index % lda ? DType(1.0) : DType(0));
    }
}

// There is no direct support for potri in cuda. We emulate the function by two calls to trsm.
#define LINALG_GPU_POTRI(DType) \
template<> inline \
void linalg_potri<gpu, DType>(const Tensor<gpu, 2, DType>& A, bool lower, Stream<gpu> *s) { \
  using namespace mxnet; \
  CHECK_NOTNULL(s); \
  check_potri(A, lower); \
  Storage::Handle buffer = Storage::Get()->Alloc(sizeof(DType)*A.MSize(), Context::GPU()); \
  using namespace mshadow::cuda; \
  int ngrid = std::min(kMaxGridNum, \
                       static_cast<int>((A.MSize() + kBaseThreadNum - 1) / kBaseThreadNum)); \
  linalgInitIdentityGPU<<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>> \
    (static_cast<DType *>(buffer.dptr), A.MSize(), A.stride_, A.MSize());  \
  Tensor<gpu, 2, DType> B((DType *)buffer.dptr, A.shape_, A.stride_, s); \
  linalg_trsm(A, B, DType(1.0), false, lower, !lower, s); \
  linalg_trsm(A, B, DType(1.0), false, lower, lower, s); \
  Copy(A, B, s); \
  B.dptr_ = 0; \
  Storage::Get()->Free(buffer); \
}
LINALG_GPU_POTRI(float)
LINALG_GPU_POTRI(double)

#define LINALG_GPU_BATCH_POTRI(DType) \
template<> inline \
void linalg_batch_potri<gpu, DType>(const Tensor<gpu, 3, DType>& A, bool lower, Stream<gpu> *s) { \
  using namespace mxnet; \
  CHECK_NOTNULL(s); \
  CHECK_GT(A.size(0), 0); \
  check_potri(A[0], lower); \
  Storage::Handle buffer = Storage::Get()->Alloc(sizeof(DType)*A.MSize(), Context::GPU()); \
  using namespace mshadow::cuda; \
  int ngrid = std::min(kMaxGridNum, \
                       static_cast<int>((A.MSize() + kBaseThreadNum - 1) / kBaseThreadNum)); \
  linalgInitIdentityGPU<<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>> \
    (static_cast<DType *>(buffer.dptr), A.size(1)*A.stride_, A.stride_, A.MSize()); \
  Tensor<gpu, 3, DType> B((DType *)buffer.dptr, A.shape_, A.stride_, s); \
  linalg_batch_trsm(A, B, DType(1.0), false, lower, !lower, s); \
  linalg_batch_trsm(A, B, DType(1.0), false, lower, lower, s); \
  Copy(A, B, s); \
  B.dptr_ = 0; \
  Storage::Get()->Free(buffer); \
}
LINALG_GPU_BATCH_POTRI(float)
LINALG_GPU_BATCH_POTRI(double)

#endif

#endif  // MXNET_OPERATOR_LINALG_IMPL_H_
