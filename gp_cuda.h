#ifndef GP_CUDA_H
#define GP_CUDA_H

#include <cublas_v2.h>
#include <cusolverDn.h>
#include "gp_cuda_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact GP regression -- CUDA (cuBLAS/cuSOLVER), batch training only.
// API mirrors gp.h with a gpcu_ prefix.  All arithmetic is double precision.
//
// Device matrices use column-major layout (cuBLAS/cuSOLVER convention).
// X is stored row-major on device to match the kernel CUDA kernels (x[row*d+k]).
// Training data is copied to device on gpcu_fit(); internal buffers live there.
//
// gpcu_create() takes ownership of the GPCUKernel and frees it on gpcu_destroy().

typedef struct {
    int   dim, n, cap;
    double *d_X;              // (cap x dim) row-major, device
    double *d_y;              // (cap,) device
    double *d_L;              // (n x n) lower Cholesky, column-major, device
    double *d_alpha;          // (n,) device
    double  raw_noise;
    double  dedup_threshold;  // near-duplicate filter; 0 = disabled
    cublasHandle_t      cublas;
    cusolverDnHandle_t  cusolver;
    GPCUKernel         *kernel;  // covariance kernel (owned)
} GPCU;

GPCU *gpcu_create  (int dim, int cap, GPCUKernel *kernel, double noise);
void  gpcu_destroy (GPCU *gp);

double gpcu_get_noise(const GPCU *gp);
void   gpcu_set_noise(GPCU *gp, double v);

// X, y: (n x dim) / (n,) row-major, host double pointers.
// Returns 0, -1 (n > cap), -2 (Cholesky failed).
int  gpcu_fit      (GPCU *gp, const double *X, const double *y, int n, cudaStream_t stream);
int  gpcu_recompute(GPCU *gp, cudaStream_t stream);

// Xs: (m x dim) row-major, host double pointer.  Pass vars=NULL to skip variance.
// Synchronises stream and copies results to host before returning.
void gpcu_predict(const GPCU *gp, const double *Xs,
                  double *means, double *vars, int m, cudaStream_t stream);

// Per-point MLL (always synchronous; uses default stream).
double gpcu_marginal_log_likelihood(const GPCU *gp);

// Gradients of per-point MLL.
void gpcu_mll_grad(const GPCU *gp, double *d_raw_noise, double *kernel_grads,
                   cudaStream_t stream);

int   gpcu_save(const GPCU *gp, const char *path);
GPCU *gpcu_load(const char *path, int extra_cap);

// -- PufferLib / float32 interface --------------------------------------------

// Cast between float32 and float64 device arrays (device-to-device, on stream).
void gpcu_cast_f2d(const float *src, double *dst, int n, cudaStream_t stream);
void gpcu_cast_d2f(const double *src, float *dst,  int n, cudaStream_t stream);

// Fit from device float32 pointers (d_X, d_y already on GPU in float32).
// Casts internally to float64 then calls gpcu_recompute.
int gpcu_fit_f32(GPCU *gp, const float *d_X, const float *d_y, int n,
                 cudaStream_t stream);

// Predict with device float32 in/out.  All compute stays on device; caller
// does not need to sync before reading d_means/d_vars.
void gpcu_predict_f32(const GPCU *gp, const float *d_Xs,
                      float *d_means, float *d_vars,
                      int m, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif // GP_CUDA_H
