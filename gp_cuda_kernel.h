#ifndef GP_CUDA_KERNEL_H
#define GP_CUDA_KERNEL_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA GP covariance kernel interface.
//
// Mirrors gp_kernel.h but operates on device (GPU) arrays.
// d_X, d_K, d_Ks, d_alpha, d_Kinv are device pointers; raw_params lives on
// the host.  build_K/build_Ks/mll_grad accept a cudaStream_t so all GPU work
// is enqueued on the caller's stream.

typedef struct GPCUKernel GPCUKernel;
struct GPCUKernel {
    int     n_params;    // number of unconstrained hyperparameters
    float *raw_params;  // [n_params], host-side unconstrained values
    char    tag[4];      // 4-char type tag used by gpcu_save/gpcu_load

    // Fill d_K (nxn, column-major on device) with k(xi, xj) + sigma_n*delta_ij.
    void (*build_K)(const GPCUKernel *k, const float *d_X, int n, int d,
                    float sigma_n, float *d_K, cudaStream_t stream);

    // Fill d_Ks (nxm, column-major on device) with k(X_train_i, X_test_j).
    void (*build_Ks)(const GPCUKernel *k, const float *d_Xtr, const float *d_Xte,
                     int n, int m, int d, float *d_Ks, cudaStream_t stream);

    // Prior variance k(x, x) for each row of d_X (device, m x d row-major).
    void (*build_kself_batch)(const GPCUKernel *k, const float *d_X, int m, int d,
                               float *d_out, cudaStream_t stream);

    // Prior variance k(x, x) for a single host-side point x (no stream needed).
    float (*k_self)(const GPCUKernel *k, const float *x, int d);

    // Gradient of MLL w.r.t. each raw_params[i].
    // d_alpha: (n,) device, d_Kinv: (nxn, column-major) device.
    // kernel_grads: host array of n_params floats, caller pre-zeroes.
    void (*mll_grad)(const GPCUKernel *k, const float *d_X, int n, int d,
                     cublasHandle_t cublas, cudaStream_t stream,
                     const float *d_alpha, const float *d_Kinv,
                     float *kernel_grads);

    void (*destroy)(GPCUKernel *k);
};

// Built-in: Matern32 + linear kernel (CUDA) with ARD lengthscales.
// Same formula and raw_params layout as gp_kernel.h.
//
// raw_params layout (softplus-constrained):
//   [0..dim-1]  raw_ell_d  ->  ell_d  (per-dimension lengthscale)
//   [dim]       raw_sf     ->  sigma_f
//   [dim+1]     raw_offset ->  offset
//   n_params = dim + 2
#define GPCU_KERNEL_TAG_MATERN32_LINEAR "M32L"

GPCUKernel *gpcu_kernel_matern32_linear(int dim, float lengthscale, float outputscale,
                                        float offset);

float gpcu_kernel_get_lengthscale(const GPCUKernel *k, int d);
float gpcu_kernel_get_outputscale(const GPCUKernel *k);
float gpcu_kernel_get_offset     (const GPCUKernel *k);
void   gpcu_kernel_set_lengthscale(GPCUKernel *k, int d, float v);
void   gpcu_kernel_set_outputscale(GPCUKernel *k, float v);
void   gpcu_kernel_set_offset     (GPCUKernel *k, float v);

#ifdef __cplusplus
}
#endif
#endif // GP_CUDA_KERNEL_H
