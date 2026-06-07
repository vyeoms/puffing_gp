// gp_cuda.cu -- Exact GP regression, CUDA (cuBLAS/cuSOLVER).
//
// Covariance kernel is pluggable; see gp_cuda_kernel.h for the interface and
// gp_cuda_kernel.cu for the built-in Matern32+Linear implementation.
//
// Device matrix layout:
//   X       (n x d)  row-major  (kernel CUDA kernels address as X[row*d+k])
//   K, L    (n x n)  column-major (cuBLAS/cuSOLVER convention)
//   K_*     (n x m)  column-major
//
// Save format ("GC04"):
//   magic[4], dim(i32), n(i32), raw_noise(f64),
//   kernel_tag[4], n_params(i32), raw_params[n_params](f64),
//   X[n*dim], y[n], L[n*n], alpha[n]

#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "gp_cuda.h"

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

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t _s = (call); \
    if (_s != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__, (int)_s); \
        abort(); \
    } \
} while (0)

#define SP_LB 1e-4
static inline double softplus     (double x)   { return (x > 20.0 ? x : log1p(exp(x))) + SP_LB; }
static inline double inv_softplus (double x)   { double v = x - SP_LB; return v > 20.0 ? v : log(expm1(v)); }
static inline double softplus_grad(double raw) { return 1.0 / (1.0 + exp(-raw)); }

// -- utility device kernels ---------------------------------------------------
// (Covariance kernels live in gp_cuda_kernel.cu)

// Add val to the diagonal of a column-major nxn matrix on device
__global__ void gpcu_k_add_diag(double *A, int n, double val)
{
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) A[(size_t)i * (n + 1)] += val;
}

// Copy diagonal of a column-major nxn matrix to a flat device vector
__global__ void gpcu_k_extract_diag(const double *A, double *d, int n)
{
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) d[i] = A[(size_t)i * (n + 1)];
}

// Per-column squared norms of a column-major nxm matrix
__global__ void gpcu_k_col_sqnorms(const double *V, double *sqnorms, int n, int m)
{
    int j = blockIdx.x * 256 + threadIdx.x;
    if (j >= m) return;
    double sq = 0.0;
    const double *col = V + (size_t)j * n;
    for (int i = 0; i < n; i++) sq += col[i] * col[i];
    sqnorms[j] = sq;
}

// Subtract per-column squared norms from variance vector and clamp to zero
__global__ void gpcu_k_var_subtract(double *vars, const double *sqnorms, int m)
{
    int j = blockIdx.x * 256 + threadIdx.x;
    if (j >= m) return;
    double v = vars[j] - sqnorms[j];
    vars[j] = v > 0.0 ? v : 0.0;
}

// float32 <-> float64 cast (device-to-device)
__global__ void gpcu_k_f2d(const float *src, double *dst, int n)
{
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = (double)src[i];
}
__global__ void gpcu_k_d2f(const double *src, float *dst, int n)
{
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = (float)src[i];
}

void gpcu_cast_f2d(const float *src, double *dst, int n, cudaStream_t stream)
{
    gpcu_k_f2d<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}

void gpcu_cast_d2f(const double *src, float *dst, int n, cudaStream_t stream)
{
    gpcu_k_d2f<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}

// -- host-side near-duplicate filter ------------------------------------------
// Self-contained copy; mirrors gp_filter_near_duplicates in gp.c.

#define KD_LEAF 16

typedef struct { int lo, hi, left, right, split_dim; double split_val; } KDNode;
typedef struct { const double *X; int n, dim, n_nodes; int *idx; KDNode *nodes; } KDTree;

static int          g_kd_split_dim, g_kd_d;
static const double *g_kd_X;

static int kd_cmp_fn(const void *a, const void *b)
{
    double va = g_kd_X[*(const int *)a * g_kd_d + g_kd_split_dim];
    double vb = g_kd_X[*(const int *)b * g_kd_d + g_kd_split_dim];
    return (va > vb) - (va < vb);
}

static void kd_build(KDTree *t, int node, int lo, int hi)
{
    KDNode *nd = &t->nodes[node];
    nd->lo = lo; nd->hi = hi; nd->left = nd->right = -1; nd->split_dim = -1;
    if (hi - lo <= KD_LEAF) return;

    int best = 0; double spread = -1.0;
    for (int k = 0; k < t->dim; k++) {
        double mn = t->X[t->idx[lo] * t->dim + k], mx = mn;
        for (int i = lo + 1; i < hi; i++) {
            double v = t->X[t->idx[i] * t->dim + k];
            if (v < mn) mn = v; if (v > mx) mx = v;
        }
        if (mx - mn > spread) { spread = mx - mn; best = k; }
    }
    int mid = (lo + hi) / 2;
    g_kd_split_dim = best; g_kd_d = t->dim; g_kd_X = t->X;
    qsort(t->idx + lo, hi - lo, sizeof(int), kd_cmp_fn);
    nd->split_dim = best;
    nd->split_val = t->X[t->idx[mid] * t->dim + best];
    int ln = t->n_nodes++, rn = t->n_nodes++;
    nd->left = ln; nd->right = rn;
    kd_build(t, ln, lo, mid);
    kd_build(t, rn, mid, hi);
}

static KDTree kd_create(const double *X, int n, int dim)
{
    KDTree t;
    t.X = X; t.n = n; t.dim = dim; t.n_nodes = 1;
    t.idx   = (int    *)malloc(n * sizeof(int));
    t.nodes = (KDNode *)malloc(2 * n * sizeof(KDNode));
    for (int i = 0; i < n; i++) t.idx[i] = i;
    kd_build(&t, 0, 0, n);
    return t;
}

static void kd_destroy(KDTree *t) { free(t->idx); free(t->nodes); }

static void kd_query_ball(const KDTree *t, int node, const double *q,
                           double r, double r2, int *out, int *cnt)
{
    const KDNode *nd = &t->nodes[node];
    if (nd->split_dim < 0) {
        for (int i = nd->lo; i < nd->hi; i++) {
            int p = t->idx[i]; double sq = 0.0;
            for (int k = 0; k < t->dim; k++) {
                double df = q[k] - t->X[p * t->dim + k]; sq += df * df;
            }
            if (sq <= r2) out[(*cnt)++] = p;
        }
        return;
    }
    double dv = q[nd->split_dim] - nd->split_val;
    if (nd->left  >= 0 && dv <=  r) kd_query_ball(t, nd->left,  q, r, r2, out, cnt);
    if (nd->right >= 0 && dv >= -r) kd_query_ball(t, nd->right, q, r, r2, out, cnt);
}

static int gpcu_filter_near_duplicates(const double *X, int n, int dim,
                                        double threshold, int *kept_indices)
{
    if (n <= 0) return 0;
    if (n == 1) { kept_indices[0] = 0; return 1; }

    KDTree tree   = kd_create(X, n, dim);
    int   *keep   = (int *)malloc(n * sizeof(int));
    int   *nearby = (int *)malloc(n * sizeof(int));
    double r2     = threshold * threshold;
    for (int i = 0; i < n; i++) keep[i] = 1;

    for (int i = n - 1; i >= 0; i--) {
        if (!keep[i]) continue;
        int cnt = 0;
        kd_query_ball(&tree, 0, &X[(size_t)i * dim], threshold, r2, nearby, &cnt);
        for (int j = 0; j < cnt; j++)
            if (nearby[j] != i) keep[nearby[j]] = 0;
    }

    int count = 0;
    for (int i = 0; i < n; i++)
        if (keep[i]) kept_indices[count++] = i;

    free(nearby); free(keep); kd_destroy(&tree);
    return count;
}

// -- hyperparameter access ----------------------------------------------------

double gpcu_get_noise(const GPCU *gp) { return softplus(gp->raw_noise);   }
void   gpcu_set_noise(GPCU *gp, double v) { gp->raw_noise = inv_softplus(v); }

// -- lifecycle ----------------------------------------------------------------

GPCU *gpcu_create(int dim, int cap, GPCUKernel *kernel, double noise)
{
    GPCU *gp  = (GPCU *)calloc(1, sizeof(GPCU));
    gp->dim   = dim;
    gp->cap   = cap;
    gp->kernel = kernel;
    gpcu_set_noise(gp, noise);
    CUDA_CHECK(cudaMalloc(&gp->d_X, (size_t)cap * dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gp->d_y, (size_t)cap       * sizeof(double)));
    CUBLAS_CHECK  (cublasCreate    (&gp->cublas));
    CUSOLVER_CHECK(cusolverDnCreate(&gp->cusolver));
    return gp;
}

void gpcu_destroy(GPCU *gp)
{
    if (!gp) return;
    cudaFree(gp->d_X); cudaFree(gp->d_y);
    cudaFree(gp->d_L); cudaFree(gp->d_alpha);
    cublasDestroy    (gp->cublas);
    cusolverDnDestroy(gp->cusolver);
    if (gp->kernel && gp->kernel->destroy) gp->kernel->destroy(gp->kernel);
    free(gp);
}

// -- internal helpers ---------------------------------------------------------

// Cholesky factorisation of d_K in-place; returns cuSOLVER info (0 = success).
static int run_potrf(GPCU *gp, double *d_K, int n, cudaStream_t stream)
{
    CUSOLVER_CHECK(cusolverDnSetStream(gp->cusolver, stream));
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        gp->cusolver, CUBLAS_FILL_MODE_LOWER, n, d_K, n, &lwork));

    double *d_work; int *d_info;
    CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDpotrf(
        gp->cusolver, CUBLAS_FILL_MODE_LOWER, n, d_K, n, d_work, lwork, d_info));

    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    return h_info;
}

// alpha = K^-1 y via two triangular solves (d_alpha pre-filled with y).
static void solve_alpha(GPCU *gp, const double *d_L, int n, double *d_alpha,
                        cudaStream_t stream)
{
    CUBLAS_CHECK(cublasSetStream(gp->cublas, stream));
    CUBLAS_CHECK(cublasDtrsv(gp->cublas,
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        n, d_L, n, d_alpha, 1));
    CUBLAS_CHECK(cublasDtrsv(gp->cublas,
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
        n, d_L, n, d_alpha, 1));
}

// -- training -----------------------------------------------------------------

int gpcu_recompute(GPCU *gp, cudaStream_t stream)
{
    int n = gp->n;
    if (n == 0) return 0;

    cudaFree(gp->d_L);
    CUDA_CHECK(cudaMalloc(&gp->d_L, (size_t)n * n * sizeof(double)));
    cudaFree(gp->d_alpha);
    CUDA_CHECK(cudaMalloc(&gp->d_alpha, (size_t)n * sizeof(double)));

    gp->kernel->build_K(gp->kernel, gp->d_X, n, gp->dim, gpcu_get_noise(gp), gp->d_L, stream);

    int info = run_potrf(gp, gp->d_L, n, stream);
    if (info != 0) {
        gp->kernel->build_K(gp->kernel, gp->d_X, n, gp->dim, gpcu_get_noise(gp), gp->d_L, stream);
        gpcu_k_add_diag<<<(n + 255) / 256, 256, 0, stream>>>(gp->d_L, n, 1e-8);
        CUDA_CHECK(cudaGetLastError());
        info = run_potrf(gp, gp->d_L, n, stream);
        if (info != 0) return -2;
    }

    CUDA_CHECK(cudaMemcpyAsync(gp->d_alpha, gp->d_y,
                               (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    solve_alpha(gp, gp->d_L, n, gp->d_alpha, stream);
    return 0;
}

int gpcu_fit(GPCU *gp, const double *X, const double *y, int n, cudaStream_t stream)
{
    if (n > gp->cap) return -1;

    if (gp->dedup_threshold > 0.0 && n > 1) {
        int    *idx    = (int *)malloc(n * sizeof(int));
        int     n_kept = gpcu_filter_near_duplicates(X, n, gp->dim,
                                                   gp->dedup_threshold, idx);
        double *X_c    = (double *)malloc((size_t)n_kept * gp->dim * sizeof(double));
        double *y_c    = (double *)malloc((size_t)n_kept * sizeof(double));
        for (int i = 0; i < n_kept; i++) {
            memcpy(&X_c[i * gp->dim], &X[idx[i] * gp->dim],
                   gp->dim * sizeof(double));
            y_c[i] = y[idx[i]];
        }
        free(idx);
        CUDA_CHECK(cudaMemcpyAsync(gp->d_X, X_c,
                                   (size_t)n_kept * gp->dim * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gp->d_y, y_c,
                                   (size_t)n_kept * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        free(X_c); free(y_c);
        gp->n = n_kept;
    } else {
        CUDA_CHECK(cudaMemcpyAsync(gp->d_X, X,
                                   (size_t)n * gp->dim * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gp->d_y, y,
                                   (size_t)n * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        gp->n = n;
    }
    return gpcu_recompute(gp, stream);
}

int gpcu_fit_f32(GPCU *gp, const float *d_X_f, const float *d_y_f,
                 int n, cudaStream_t stream)
{
    if (n > gp->cap) return -1;
    gpcu_k_f2d<<<(n * gp->dim + 255) / 256, 256, 0, stream>>>(d_X_f, gp->d_X, n * gp->dim);
    CUDA_CHECK(cudaGetLastError());
    gpcu_k_f2d<<<(n + 255) / 256, 256, 0, stream>>>(d_y_f, gp->d_y, n);
    CUDA_CHECK(cudaGetLastError());
    gp->n = n;
    return gpcu_recompute(gp, stream);
}

// -- prediction ---------------------------------------------------------------

// Internal: d_Xte already on device (double).  d_means and d_vars are device
// double output buffers (d_vars may be NULL to skip variance).
static void predict_dev(const GPCU *gp, const double *d_Xte,
                        double *d_means, double *d_vars, int m, cudaStream_t stream)
{
    int n = gp->n, d = gp->dim;
    const double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasSetStream(gp->cublas, stream));

    double *d_Ks;
    CUDA_CHECK(cudaMalloc(&d_Ks, (size_t)n * m * sizeof(double)));
    gp->kernel->build_Ks(gp->kernel, gp->d_X, d_Xte, n, m, d, d_Ks, stream);

    // means = Ks^T alpha  (Ks is column-major nxm, CUBLAS_OP_T gives m-vector)
    CUBLAS_CHECK(cublasDgemv(gp->cublas, CUBLAS_OP_T, n, m,
                             &one, d_Ks, n, gp->d_alpha, 1, &zero, d_means, 1));

    if (d_vars) {
        // Prior variances via device kernel (avoids host roundtrip)
        gp->kernel->build_kself_batch(gp->kernel, d_Xte, m, d, d_vars, stream);

        // Posterior: d_vars[j] -= ||L^-1 Ks_j||^2
        CUBLAS_CHECK(cublasDtrsm(gp->cublas,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT, n, m, &one, gp->d_L, n, d_Ks, n));
        double *d_sqnorms;
        CUDA_CHECK(cudaMalloc(&d_sqnorms, (size_t)m * sizeof(double)));
        gpcu_k_col_sqnorms<<<(m + 255) / 256, 256, 0, stream>>>(d_Ks, d_sqnorms, n, m);
        CUDA_CHECK(cudaGetLastError());
        gpcu_k_var_subtract<<<(m + 255) / 256, 256, 0, stream>>>(d_vars, d_sqnorms, m);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(d_sqnorms));
    }

    CUDA_CHECK(cudaFree(d_Ks));
}

void gpcu_predict(const GPCU *gp, const double *Xs, double *means, double *vars,
                  int m, cudaStream_t stream)
{
    int n = gp->n;
    if (n == 0) {
        double sigma_f = gpcu_kernel_get_outputscale(gp->kernel);
        for (int j = 0; j < m; j++) {
            means[j] = 0.0;
            if (vars) vars[j] = sigma_f;
        }
        return;
    }

    double *d_Xte;
    CUDA_CHECK(cudaMalloc(&d_Xte, (size_t)m * gp->dim * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(d_Xte, Xs, (size_t)m * gp->dim * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

    double *d_means, *d_vars = NULL;
    CUDA_CHECK(cudaMalloc(&d_means, (size_t)m * sizeof(double)));
    if (vars) CUDA_CHECK(cudaMalloc(&d_vars, (size_t)m * sizeof(double)));

    predict_dev(gp, d_Xte, d_means, d_vars, m, stream);
    CUDA_CHECK(cudaFree(d_Xte));

    // Sync then copy results to host
    CUDA_CHECK(cudaMemcpy(means, d_means, (size_t)m * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_means));
    if (vars) {
        CUDA_CHECK(cudaMemcpy(vars, d_vars, (size_t)m * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_vars));
    }
}

void gpcu_predict_f32(const GPCU *gp, const float *d_Xs_f,
                      float *d_means_f, float *d_vars_f, int m, cudaStream_t stream)
{
    int n = gp->n;
    if (n == 0) {
        // Fill with prior: zero mean, sigma_f variance
        // (caller responsible for zeroing / using appropriate defaults)
        return;
    }

    // Cast input float32 → float64
    double *d_Xte;
    CUDA_CHECK(cudaMalloc(&d_Xte, (size_t)m * gp->dim * sizeof(double)));
    gpcu_k_f2d<<<(m * gp->dim + 255) / 256, 256, 0, stream>>>(d_Xs_f, d_Xte, m * gp->dim);
    CUDA_CHECK(cudaGetLastError());

    // Compute in float64
    double *d_means_d, *d_vars_d = NULL;
    CUDA_CHECK(cudaMalloc(&d_means_d, (size_t)m * sizeof(double)));
    if (d_vars_f) CUDA_CHECK(cudaMalloc(&d_vars_d, (size_t)m * sizeof(double)));

    predict_dev(gp, d_Xte, d_means_d, d_vars_d, m, stream);
    CUDA_CHECK(cudaFree(d_Xte));

    // Cast outputs float64 → float32 (device-to-device, fully async)
    gpcu_k_d2f<<<(m + 255) / 256, 256, 0, stream>>>(d_means_d, d_means_f, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_means_d));
    if (d_vars_f) {
        gpcu_k_d2f<<<(m + 255) / 256, 256, 0, stream>>>(d_vars_d, d_vars_f, m);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(d_vars_d));
    }
}

// -- likelihood and gradients -------------------------------------------------

double gpcu_marginal_log_likelihood(const GPCU *gp)
{
    int n = gp->n;
    if (n == 0) return 0.0;

    CUBLAS_CHECK(cublasSetStream(gp->cublas, 0));
    double data_fit;
    CUBLAS_CHECK(cublasDdot(gp->cublas, n, gp->d_y, 1, gp->d_alpha, 1, &data_fit));

    // log|L| = sum of log(diag(L)); extract diagonal in one device pass
    double *d_diag;
    CUDA_CHECK(cudaMalloc(&d_diag, (size_t)n * sizeof(double)));
    gpcu_k_extract_diag<<<(n + 255) / 256, 256>>>(gp->d_L, d_diag, n);
    CUDA_CHECK(cudaGetLastError());

    double *h_diag = (double *)malloc(n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_diag, d_diag, (size_t)n * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_diag));

    double log_det = 0.0;
    for (int i = 0; i < n; i++) log_det += log(h_diag[i]);
    free(h_diag);

    return (-0.5 * data_fit - log_det - 0.5 * n * log(2.0 * M_PI)) / n;
}

void gpcu_mll_grad(const GPCU *gp, double *d_raw_noise, double *kernel_grads,
                   cudaStream_t stream)
{
    int n = gp->n;
    if (n == 0) {
        if (d_raw_noise) *d_raw_noise = 0.0;
        if (kernel_grads)
            for (int i = 0; i < gp->kernel->n_params; i++) kernel_grads[i] = 0.0;
        return;
    }

    CUSOLVER_CHECK(cusolverDnSetStream(gp->cusolver, stream));
    CUBLAS_CHECK(cublasSetStream(gp->cublas, stream));

    // K^-1 via dpotrs: K^-1 = L^-T L^-1; solve K X = I using stored L
    double *d_Lc;
    CUDA_CHECK(cudaMalloc(&d_Lc, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(d_Lc, gp->d_L, (size_t)n * n * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));

    double *d_Kinv;
    CUDA_CHECK(cudaMalloc(&d_Kinv, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_Kinv, 0, (size_t)n * n * sizeof(double), stream));
    gpcu_k_add_diag<<<(n + 255) / 256, 256, 0, stream>>>(d_Kinv, n, 1.0);
    CUDA_CHECK(cudaGetLastError());

    int *d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDpotrs(gp->cusolver, CUBLAS_FILL_MODE_LOWER,
                                    n, n, d_Lc, n, d_Kinv, n, d_info));
    CUDA_CHECK(cudaFree(d_Lc));
    CUDA_CHECK(cudaFree(d_info));

    // dK/d(sigma_n) = I  =>  grad = 0.5 * (||alpha||^2 - tr(K^-1))
    // K^-1 is PD so its diagonal is positive; cublasDasum with stride n+1 = tr(K^-1).
    if (d_raw_noise) {
        double term1, term2;
        CUBLAS_CHECK(cublasDdot (gp->cublas, n, gp->d_alpha, 1, gp->d_alpha, 1, &term1));
        CUBLAS_CHECK(cublasDasum(gp->cublas, n, d_Kinv, n + 1, &term2));
        *d_raw_noise = 0.5 * (term1 - term2) * softplus_grad(gp->raw_noise) / n;
    }

    if (kernel_grads) {
        for (int i = 0; i < gp->kernel->n_params; i++) kernel_grads[i] = 0.0;
        gp->kernel->mll_grad(gp->kernel, gp->d_X, n, gp->dim,
                             gp->cublas, stream, gp->d_alpha, d_Kinv, kernel_grads);
        for (int i = 0; i < gp->kernel->n_params; i++) kernel_grads[i] /= n;
    }

    CUDA_CHECK(cudaFree(d_Kinv));
}

// -- kernel registry for gpcu_load --------------------------------------------

typedef struct { const char *tag; GPCUKernel *(*make)(double *, int); } CUKernelFactory;

static GPCUKernel *cu_kf_m32lin(double *rp, int np)
{
    int dim = np - 2;
    GPCUKernel *k = gpcu_kernel_matern32_linear(dim, 1.0, 1.0, 1.0);
    memcpy(k->raw_params, rp, (size_t)np * sizeof(double));
    return k;
}

static const CUKernelFactory cu_kernel_registry[] = {
    { GPCU_KERNEL_TAG_MATERN32_LINEAR, cu_kf_m32lin },
    { NULL, NULL }
};

static GPCUKernel *cu_kernel_from_tag(const char *tag, double *rp, int np)
{
    for (int i = 0; cu_kernel_registry[i].tag; i++)
        if (memcmp(tag, cu_kernel_registry[i].tag, 4) == 0)
            return cu_kernel_registry[i].make(rp, np);
    return NULL;
}

// -- save / load ---------------------------------------------------------------

int gpcu_save(const GPCU *gp, const char *path)
{
    if (!gp->d_L || gp->n == 0) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int n = gp->n, dim = gp->dim, np = gp->kernel->n_params;
    fwrite("GC04",                  1,              4,  f);
    fwrite(&dim,                    sizeof(int),    1,  f);
    fwrite(&n,                      sizeof(int),    1,  f);
    fwrite(&gp->raw_noise,          sizeof(double), 1,  f);
    fwrite(gp->kernel->tag,         1,              4,  f);
    fwrite(&np,                     sizeof(int),    1,  f);
    fwrite(gp->kernel->raw_params,  sizeof(double), np, f);

    double *h_X     = (double *)malloc((size_t)n * dim * sizeof(double));
    double *h_y     = (double *)malloc((size_t)n       * sizeof(double));
    double *h_L     = (double *)malloc((size_t)n * n   * sizeof(double));
    double *h_alpha = (double *)malloc((size_t)n       * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_X,     gp->d_X,     (size_t)n * dim * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y,     gp->d_y,     (size_t)n       * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L,     gp->d_L,     (size_t)n * n   * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_alpha, gp->d_alpha, (size_t)n       * sizeof(double), cudaMemcpyDeviceToHost));

    fwrite(h_X,     sizeof(double), (size_t)n * dim, f);
    fwrite(h_y,     sizeof(double), n,               f);
    fwrite(h_L,     sizeof(double), (size_t)n * n,   f);
    fwrite(h_alpha, sizeof(double), n,               f);

    free(h_X); free(h_y); free(h_L); free(h_alpha);
    fclose(f);
    return 0;
}

GPCU *gpcu_load(const char *path, int extra_cap)
{
    GPCU   *gp      = NULL;
    double *h_X     = NULL, *h_y     = NULL;
    double *h_L     = NULL, *h_alpha = NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

#define RD(ptr, sz, cnt) \
    if (fread(ptr, sz, cnt, f) != (size_t)(cnt)) break

    do {
        char   magic[4], ktag[4];
        int    dim, n, np;
        double rn;

        RD(magic, 1, 4);
        if (memcmp(magic, "GC04", 4) != 0) break;

        RD(&dim,  sizeof(int),    1);
        RD(&n,    sizeof(int),    1);
        RD(&rn,   sizeof(double), 1);
        RD(ktag,  1, 4);
        RD(&np,   sizeof(int),    1);

        double *rp = (double *)malloc((size_t)np * sizeof(double));
        if (!rp) break;
        if (fread(rp, sizeof(double), np, f) != (size_t)np) { free(rp); break; }

        GPCUKernel *k = cu_kernel_from_tag(ktag, rp, np);
        free(rp);
        if (!k) break;

        int cap = n + (extra_cap > 0 ? extra_cap : 0);
        gp = gpcu_create(dim, cap, k, 1.0);
        gp->raw_noise = rn;
        gp->n = n;

        CUDA_CHECK(cudaMalloc(&gp->d_L,     (size_t)n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&gp->d_alpha, (size_t)n     * sizeof(double)));

        h_X     = (double *)malloc((size_t)n * dim * sizeof(double));
        h_y     = (double *)malloc((size_t)n       * sizeof(double));
        h_L     = (double *)malloc((size_t)n * n   * sizeof(double));
        h_alpha = (double *)malloc((size_t)n       * sizeof(double));

        RD(h_X,     sizeof(double), (size_t)n * dim);
        RD(h_y,     sizeof(double), n);
        RD(h_L,     sizeof(double), (size_t)n * n);
        RD(h_alpha, sizeof(double), n);

        CUDA_CHECK(cudaMemcpy(gp->d_X,     h_X,     (size_t)n * dim * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gp->d_y,     h_y,     (size_t)n       * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gp->d_L,     h_L,     (size_t)n * n   * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gp->d_alpha, h_alpha, (size_t)n       * sizeof(double), cudaMemcpyHostToDevice));

        free(h_X); free(h_y); free(h_L); free(h_alpha);
        fclose(f);
        return gp;
    } while (0);

#undef RD
    free(h_X); free(h_y); free(h_L); free(h_alpha);
    if (gp) gpcu_destroy(gp);
    fclose(f);
    return NULL;
}
