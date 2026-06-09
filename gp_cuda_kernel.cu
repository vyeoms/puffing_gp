// gp_cuda_kernel.cu -- Built-in Matern32+Linear GP covariance kernel (CUDA), ARD.
//
//   k(x,x') = sigma_f * (x.x' + offset
//             + (1 + sqrt(3)*r) * exp(-sqrt(3)*r))
//   where r = sqrt(sum_d (x_d - x'_d)^2 / ell_d^2)  (ARD distance)
//
// Device matrices are column-major (cuBLAS convention); X is stored row-major.
//
// __global__ kernel names are prefixed m32lin_ to avoid link conflicts.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gp_cuda_kernel.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        abort(); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
        abort(); \
    } \
} while (0)

#define SP_LB 1e-4
__host__ __device__ __forceinline__ double softplus(double x)      { return (x > 20.0 ? x : log1p(exp(x))) + SP_LB; }
__host__ __device__ __forceinline__ double inv_softplus(double x)  { double v = x - SP_LB; return v > 20.0 ? v : log(expm1(v)); }
__host__ __device__ __forceinline__ double softplus_grad(double x) { return 1.0 / (1.0 + exp(-x)); }

#define SF_IDX(np)  ((np) - 2)
#define OFF_IDX(np) ((np) - 1)

#define GP_BLOCK 16
inline int gp_grid(int n) { return (n + GP_BLOCK - 1) / GP_BLOCK; }

// nxn kernel matrix (column-major output). inv_ells[d] on device.
__global__ void m32lin_k_build_K(
    const double * __restrict__ X,
    double       * __restrict__ K,
    const double * __restrict__ inv_ells,
    int n, int d, double sigma_f, double sigma_n, double offset)
{
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= n) return;
    double dot = 0.0, r2 = 0.0;
    for (int k = 0; k < d; k++) {
        double xr = X[row * d + k], xc = X[col * d + k];
        dot += xr * xc;
        double diff = (xr - xc) * inv_ells[k];
        r2 += diff * diff;
    }
    if (r2 < 0.0) r2 = 0.0;
    double u   = sqrt(3.0) * sqrt(r2);
    double val = sigma_f * (dot + offset + (1.0 + u) * exp(-u));
    if (row == col) val += sigma_n;
    K[col * n + row] = val;
}

// nxm cross-covariance (column-major output). inv_ells[d] on device.
__global__ void m32lin_k_build_Ks(
    const double * __restrict__ Xtr,
    const double * __restrict__ Xte,
    double       * __restrict__ Ks,
    const double * __restrict__ inv_ells,
    int n, int m, int d, double sigma_f, double offset)
{
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= m) return;
    double dot = 0.0, r2 = 0.0;
    for (int k = 0; k < d; k++) {
        double xr = Xtr[row * d + k], xc = Xte[col * d + k];
        dot += xr * xc;
        double diff = (xr - xc) * inv_ells[k];
        r2 += diff * diff;
    }
    if (r2 < 0.0) r2 = 0.0;
    double u = sqrt(3.0) * sqrt(r2);
    Ks[col * n + row] = sigma_f * (dot + offset + (1.0 + u) * exp(-u));
}

// nxn ARD squared-distance matrix (column-major output). inv_ells[d] on device.
__global__ void m32lin_k_build_D_ard(
    const double * __restrict__ X,
    double       * __restrict__ D,
    const double * __restrict__ inv_ells,
    int n, int d)
{
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= n) return;
    double sq = 0.0;
    for (int k = 0; k < d; k++) {
        double diff = (X[row * d + k] - X[col * d + k]) * inv_ells[k];
        sq += diff * diff;
    }
    D[col * n + row] = sq < 0.0 ? 0.0 : sq;
}

// Precompute W = alpha*alpha^T - Kinv (column-major, nxn).
// Used so each per-parameter gradient is a single cublasDdot(W, dK/dparam)
// instead of a dgemv + two ddots.
__global__ void m32lin_k_compute_W(
    const double * __restrict__ alpha,
    const double * __restrict__ Kinv,
    double       * __restrict__ W,
    int n)
{
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= n) return;
    W[col * n + row] = alpha[row] * alpha[col] - Kinv[col * n + row];
}

// dK/d(ell_dd): sigma_f * 3 * (xi_dd - xj_dd)^2 * inv_ell_dd^3 * exp(-sqrt(3)*r).
// D_ard holds the ARD squared distances.
__global__ void m32lin_k_dk_dell_d(
    const double * __restrict__ X,
    const double * __restrict__ D_ard,
    double       * __restrict__ out,
    int n, int d, int dd, double sigma_f, double inv_ell_d3)
{
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= n) return;
    double r2   = D_ard[col * n + row];
    double diff = X[row * d + dd] - X[col * d + dd];
    out[col * n + row] = sigma_f * 3.0 * diff * diff * inv_ell_d3
                       * exp(-sqrt(3.0) * sqrt(r2));
}

__global__ void m32lin_k_fill(double *v, int n, double val)
{
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) v[i] = val;
}

// -- helpers ------------------------------------------------------------------
// Scratch is stream-ordered (cudaMallocAsync/cudaFreeAsync): no device-wide
// sync, and the launchers stay capturable in a CUDA graph.

static double *h2d(const double *h, int n, cudaStream_t stream)
{
    double *d;
    CUDA_CHECK(cudaMallocAsync(&d, (size_t)n * sizeof(double), stream));
    CUDA_CHECK(cudaMemcpyAsync(d, h, (size_t)n * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    return d;
}

static double *device_inv_ells(const GPCUKernel *k, int d, cudaStream_t stream)
{
    double *h = (double *)malloc(d * sizeof(double));
    for (int i = 0; i < d; i++) h[i] = 1.0 / softplus(k->raw_params[i]);
    double *dev = h2d(h, d, stream);
    free(h);
    return dev;
}

// -- launcher functions -------------------------------------------------------

static void m32lin_build_K(const GPCUKernel *k, const double *d_X, int n, int d,
                            double sigma_n, double *d_K, cudaStream_t stream)
{
    int    np      = k->n_params;
    double sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    double offset  = softplus(k->raw_params[OFF_IDX(np)]);
    double *d_inv  = device_inv_ells(k, d, stream);
    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(n), gp_grid(n));
    m32lin_k_build_K<<<grid, block, 0, stream>>>(d_X, d_K, d_inv, n, d, sigma_f, sigma_n, offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_inv, stream));
}

static void m32lin_build_Ks(const GPCUKernel *k, const double *d_Xtr, const double *d_Xte,
                             int n, int m, int d, double *d_Ks, cudaStream_t stream)
{
    int    np      = k->n_params;
    double sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    double offset  = softplus(k->raw_params[OFF_IDX(np)]);
    double *d_inv  = device_inv_ells(k, d, stream);
    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(m), gp_grid(n));
    m32lin_k_build_Ks<<<grid, block, 0, stream>>>(d_Xtr, d_Xte, d_Ks, d_inv, n, m, d, sigma_f, offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_inv, stream));
}

static double m32lin_k_self(const GPCUKernel *k, const double *x, int d)
{
    int    np      = k->n_params;
    double sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    double offset  = softplus(k->raw_params[OFF_IDX(np)]);
    double norm2   = 0.0;
    for (int i = 0; i < d; i++) norm2 += x[i] * x[i];
    return sigma_f * (norm2 + offset + 1.0);
}

// Prior variance k(x,x) = sigma_f*(||x||^2 + offset + 1) for each row of d_X.
__global__ void m32lin_k_kself_batch(
    const double * __restrict__ X, double * __restrict__ out,
    int m, int d, double sigma_f, double offset)
{
    int row = blockIdx.x * 256 + threadIdx.x;
    if (row >= m) return;
    double norm2 = 0.0;
    for (int k = 0; k < d; k++) { double v = X[row * d + k]; norm2 += v * v; }
    out[row] = sigma_f * (norm2 + offset + 1.0);
}

static void m32lin_build_kself_batch(const GPCUKernel *k, const double *d_X,
                                      int m, int d, double *d_out, cudaStream_t stream)
{
    int    np      = k->n_params;
    double sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    double offset  = softplus(k->raw_params[OFF_IDX(np)]);
    m32lin_k_kself_batch<<<(m + 255) / 256, 256, 0, stream>>>(d_X, d_out, m, d, sigma_f, offset);
    CUDA_CHECK(cudaGetLastError());
}

static void m32lin_mll_grad(const GPCUKernel *k, const double *d_X, int n, int d,
                             cublasHandle_t cublas, cudaStream_t stream,
                             const double *d_alpha, const double *d_Kinv,
                             double *kernel_grads)
{
    int    np      = k->n_params;
    double sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    double offset  = softplus(k->raw_params[OFF_IDX(np)]);
    const double one = 1.0, zero = 0.0;

    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    double *d_inv  = device_inv_ells(k, d, stream);
    double *d_D, *d_W, *d_temp;
    CUDA_CHECK(cudaMallocAsync(&d_D,    (size_t)n * n * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_W,    (size_t)n * n * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_temp, (size_t)n * n * sizeof(double), stream));

    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(n), gp_grid(n));

    // Precompute D_ard and W once; each gradient is then a single ddot(W, dK/dparam).
    m32lin_k_build_D_ard<<<grid, block, 0, stream>>>(d_X, d_D, d_inv, n, d);
    CUDA_CHECK(cudaGetLastError());
    m32lin_k_compute_W<<<grid, block, 0, stream>>>(d_alpha, d_Kinv, d_W, n);
    CUDA_CHECK(cudaGetLastError());

    // Per-dim lengthscale gradients
    for (int dd = 0; dd < d; dd++) {
        double ell_d    = softplus(k->raw_params[dd]);
        double inv_ell3 = 1.0 / (ell_d * ell_d * ell_d);
        m32lin_k_dk_dell_d<<<grid, block, 0, stream>>>(d_X, d_D, d_temp, n, d, dd, sigma_f, inv_ell3);
        CUDA_CHECK(cudaGetLastError());
        double dot_val;
        CUBLAS_CHECK(cublasDdot(cublas, n * n, d_W, 1, d_temp, 1, &dot_val));
        kernel_grads[dd] = 0.5 * dot_val * softplus_grad(k->raw_params[dd]);
    }

    // sf gradient: dK/d(sf) = K_no_noise / sigma_f
    m32lin_k_build_K<<<grid, block, 0, stream>>>(d_X, d_temp, d_inv, n, d, sigma_f, 0.0, offset);
    CUDA_CHECK(cudaGetLastError());
    {
        double dot_val;
        CUBLAS_CHECK(cublasDdot(cublas, n * n, d_W, 1, d_temp, 1, &dot_val));
        kernel_grads[SF_IDX(np)] = 0.5 * dot_val / sigma_f
                                 * softplus_grad(k->raw_params[SF_IDX(np)]);
    }

    // offset gradient: dK_ij/d(offset) = sigma_f => g_off = sigma_f * sum_{i,j} W_ij
    {
        double *d_ones, *d_wrow;
        CUDA_CHECK(cudaMallocAsync(&d_ones, (size_t)n * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_wrow, (size_t)n * sizeof(double), stream));
        m32lin_k_fill<<<(n + 255) / 256, 256, 0, stream>>>(d_ones, n, 1.0);
        CUDA_CHECK(cudaGetLastError());
        double w_sum;
        CUBLAS_CHECK(cublasDgemv(cublas, CUBLAS_OP_N, n, n,
                                 &one, d_W, n, d_ones, 1, &zero, d_wrow, 1));
        CUBLAS_CHECK(cublasDdot(cublas, n, d_wrow, 1, d_ones, 1, &w_sum));
        kernel_grads[OFF_IDX(np)] = 0.5 * sigma_f * w_sum
                                  * softplus_grad(k->raw_params[OFF_IDX(np)]);
        CUDA_CHECK(cudaFreeAsync(d_ones, stream));
        CUDA_CHECK(cudaFreeAsync(d_wrow, stream));
    }

    CUDA_CHECK(cudaFreeAsync(d_inv, stream));
    CUDA_CHECK(cudaFreeAsync(d_D, stream));
    CUDA_CHECK(cudaFreeAsync(d_W, stream));
    CUDA_CHECK(cudaFreeAsync(d_temp, stream));
}

static void m32lin_destroy(GPCUKernel *k) { free(k); }

GPCUKernel *gpcu_kernel_matern32_linear(int dim, double lengthscale, double outputscale,
                                        double offset)
{
    int n_params = dim + 2;
    GPCUKernel *k = (GPCUKernel *)malloc(sizeof(GPCUKernel) + (size_t)n_params * sizeof(double));
    k->n_params   = n_params;
    k->raw_params = (double *)(k + 1);
    memcpy(k->tag, GPCU_KERNEL_TAG_MATERN32_LINEAR, 4);
    for (int i = 0; i < dim; i++)
        k->raw_params[i] = inv_softplus(lengthscale);
    k->raw_params[SF_IDX(n_params)]  = inv_softplus(outputscale);
    k->raw_params[OFF_IDX(n_params)] = inv_softplus(offset);
    k->build_K          = m32lin_build_K;
    k->build_Ks         = m32lin_build_Ks;
    k->build_kself_batch = m32lin_build_kself_batch;
    k->k_self           = m32lin_k_self;
    k->mll_grad         = m32lin_mll_grad;
    k->destroy          = m32lin_destroy;
    return k;
}

double gpcu_kernel_get_lengthscale(const GPCUKernel *k, int d) { return softplus(k->raw_params[d]); }
double gpcu_kernel_get_outputscale(const GPCUKernel *k) { return softplus(k->raw_params[SF_IDX(k->n_params)]);  }
double gpcu_kernel_get_offset     (const GPCUKernel *k) { return softplus(k->raw_params[OFF_IDX(k->n_params)]); }
void   gpcu_kernel_set_lengthscale(GPCUKernel *k, int d, double v) { k->raw_params[d] = inv_softplus(v); }
void   gpcu_kernel_set_outputscale(GPCUKernel *k, double v) { k->raw_params[SF_IDX(k->n_params)]  = inv_softplus(v); }
void   gpcu_kernel_set_offset     (GPCUKernel *k, double v) { k->raw_params[OFF_IDX(k->n_params)] = inv_softplus(v); }
