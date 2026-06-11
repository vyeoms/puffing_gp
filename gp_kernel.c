// gp_kernel.c -- Built-in Matern32+Linear GP covariance kernel (CPU), ARD.
//
//   k(x,x') = sigma_f * (x.x' + offset
//             + (1 + sqrt(3)*r) * exp(-sqrt(3)*r))
//   where r = sqrt(sum_d (x_d - x'_d)^2 / ell_d^2)  (ARD distance)
//
// dK/d(ell_d)   = sigma_f * 3 * (xi_d-xj_d)^2/ell_d^3 * exp(-sqrt(3)*r)
// dK/d(sigma_f) = x.x' + offset + (1 + sqrt(3)*r)*exp(-sqrt(3)*r)
// dK/d(offset)  = sigma_f  (constant for all i,j)
//
// BLAS: dgemm, ddot

#include "gp_kernel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#define SP_LB 1e-4
static inline float softplus     (float x) { return (x > 20.0 ? x : log1p(exp(x))) + SP_LB; }
static inline float inv_softplus (float x) { float v = x - SP_LB; return v > 20.0 ? v : log(expm1(v)); }
static inline float softplus_grad(float x) { return 1.0 / (1.0 + exp(-x)); }

// Index helpers -- n_params = dim+2, so SF = n_params-2, OFF = n_params-1.
#define SF_IDX(np)  ((np) - 2)
#define OFF_IDX(np) ((np) - 1)

static void matern32lin_build_K(const GPKernel *k, const float *X, int n, int d,
                            float sigma_n, float *K)
{
    int    np      = k->n_params;
    float sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    float offset  = softplus(k->raw_params[OFF_IDX(np)]);

    // Precompute inv_ell[d] and build scaled X.
    float *inv_ell = (float *)malloc(d * sizeof(float));
    for (int dd = 0; dd < d; dd++)
        inv_ell[dd] = 1.0 / softplus(k->raw_params[dd]);

    float *Xs = (float *)malloc((size_t)n * d * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int dd = 0; dd < d; dd++)
            Xs[i * d + dd] = X[i * d + dd] * inv_ell[dd];
    free(inv_ell);

    // Lower triangles only (K is symmetric): DOT = X*X^T in K, DOT_s = Xs*Xs^T
    cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, d, 1.0, X, d, 0.0, K, n);
    float *DOT_s = (float *)malloc((size_t)n * n * sizeof(float));
    cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, d, 1.0, Xs, d, 0.0, DOT_s, n);
    free(Xs);

    // Fill lower triangle, mirror to upper
    for (int i = 0; i < n; i++) {
        float  ns_i = DOT_s[(size_t)i * (n + 1)];
        float *Ki   = K + (size_t)i * n;
        float *Ds_i = DOT_s + (size_t)i * n;
        for (int j = 0; j < i; j++) {
            float r2 = ns_i + DOT_s[(size_t)j * (n + 1)] - 2.0 * Ds_i[j];
            if (r2 < 0.0) r2 = 0.0;
            float u = sqrt(3.0) * sqrt(r2);
            float v = sigma_f * (Ki[j] + offset + (1.0 + u) * exp(-u));
            Ki[j] = v;
            K[(size_t)j * n + i] = v;
        }
        // diagonal: r = 0, Matern32(0) = 1
        Ki[i] = sigma_f * (Ki[i] + offset + 1.0) + sigma_n;
    }
    free(DOT_s);
}

static void matern32lin_build_Ks(const GPKernel *k, const float *Xtr, const float *Xte,
                              int n, int m, int d, float *Ks)
{
    int    np      = k->n_params;
    float sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    float offset  = softplus(k->raw_params[OFF_IDX(np)]);

    float *inv_ell = (float *)malloc(d * sizeof(float));
    for (int dd = 0; dd < d; dd++)
        inv_ell[dd] = 1.0 / softplus(k->raw_params[dd]);

    float *Xtrs = (float *)malloc((size_t)n * d * sizeof(float));
    float *Xtes = (float *)malloc((size_t)m * d * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int dd = 0; dd < d; dd++)
            Xtrs[i * d + dd] = Xtr[i * d + dd] * inv_ell[dd];
    for (int j = 0; j < m; j++)
        for (int dd = 0; dd < d; dd++)
            Xtes[j * d + dd] = Xte[j * d + dd] * inv_ell[dd];
    free(inv_ell);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, m, d, 1.0, Xtr, d, Xte, d, 0.0, Ks, m);

    float *DOT_s  = (float *)malloc((size_t)n * m * sizeof(float));
    float *ntr_s  = (float *)malloc(n * sizeof(float));
    float *nte_s  = (float *)malloc(m * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, m, d, 1.0, Xtrs, d, Xtes, d, 0.0, DOT_s, m);
    for (int i = 0; i < n; i++)
        ntr_s[i] = cblas_sdot(d, &Xtrs[i * d], 1, &Xtrs[i * d], 1);
    for (int j = 0; j < m; j++)
        nte_s[j] = cblas_sdot(d, &Xtes[j * d], 1, &Xtes[j * d], 1);
    free(Xtrs);
    free(Xtes);

    for (int i = 0; i < n; i++) {
        float  ns_i = ntr_s[i];
        float *Ki   = Ks + i * m;
        float *Ks_i = DOT_s + i * m;
        for (int j = 0; j < m; j++) {
            float dot = Ki[j];
            float r2  = ns_i + nte_s[j] - 2.0 * Ks_i[j];
            if (r2 < 0.0) r2 = 0.0;
            float u   = sqrt(3.0) * sqrt(r2);
            Ki[j] = sigma_f * (dot + offset + (1.0 + u) * exp(-u));
        }
    }
    free(DOT_s);
    free(ntr_s);
    free(nte_s);
}

static float matern32lin_k_self(const GPKernel *k, const float *x, int d)
{
    int    np      = k->n_params;
    float sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    float offset  = softplus(k->raw_params[OFF_IDX(np)]);
    float norm2   = cblas_sdot(d, x, 1, x, 1);
    // r=0 when xi==xj, so Matern32(0)=1
    return sigma_f * (norm2 + offset + 1.0);
}

static void matern32lin_mll_grad(const GPKernel *k, const float *X, int n, int d,
                             const float *alpha, const float *Kinv,
                             float *kernel_grads)
{
    int    np      = k->n_params;
    float sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    float offset  = softplus(k->raw_params[OFF_IDX(np)]);

    float *inv_ells = (float *)malloc(d * sizeof(float));
    float *inv_ell3 = (float *)malloc(d * sizeof(float));
    for (int dd = 0; dd < d; dd++) {
        float ell   = softplus(k->raw_params[dd]);
        inv_ells[dd] = 1.0 / ell;
        inv_ell3[dd] = inv_ells[dd] * inv_ells[dd] * inv_ells[dd];
    }

    // DOT lower triangle: DOT[i*n+j] = xi.xj (for sf gradient term)
    float *DOT = (float *)malloc((size_t)n * n * sizeof(float));
    cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, d, 1.0, X, d, 0.0, DOT, n);

    float *g_ell = (float *)calloc(d, sizeof(float));
    float  g_sf  = 0.0;
    float *tmp   = (float *)malloc(d * sizeof(float));

    // dK terms are symmetric: visit each off-diagonal pair once with weight 2.
    for (int i = 0; i < n; i++) {
        // diagonal: r = 0, only sf gradient contributes
        float aa_ii = alpha[i] * alpha[i] - Kinv[(size_t)i * (n + 1)];
        g_sf += aa_ii * (DOT[(size_t)i * (n + 1)] + offset + 1.0);

        for (int j = 0; j < i; j++) {
            float r2 = 0.0;
            // compute ARD distance and store per-dim diffs
            for (int dd = 0; dd < d; dd++) {
                float diff = X[i * d + dd] - X[j * d + dd];
                tmp[dd] = diff;
                float sd = diff * inv_ells[dd];
                r2 += sd * sd;
            }
            float u       = sqrt(3.0) * sqrt(r2);
            float e_ij    = exp(-u);
            float aa_kinv = 2.0 * (alpha[i] * alpha[j] - Kinv[(size_t)i * n + j]);
            g_sf += aa_kinv * (DOT[(size_t)i * n + j] + offset + (1.0 + u) * e_ij);
            float coeff = aa_kinv * sigma_f * 3.0 * e_ij;
            for (int dd = 0; dd < d; dd++)
                g_ell[dd] += coeff * tmp[dd] * tmp[dd] * inv_ell3[dd];
        }
    }
    free(DOT);
    free(tmp);

    // dK/d(offset) = sigma_f for every (i,j)
    float alpha_sum = 0.0, kinv_sum = 0.0;
    for (int i = 0; i < n; i++) alpha_sum += alpha[i];
    for (int i = 0; i < n * n; i++) kinv_sum += Kinv[i];
    float g_off = sigma_f * (alpha_sum * alpha_sum - kinv_sum);

    for (int dd = 0; dd < d; dd++)
        kernel_grads[dd] = 0.5 * g_ell[dd] * softplus_grad(k->raw_params[dd]);
    free(g_ell);
    free(inv_ells);
    free(inv_ell3);
    kernel_grads[SF_IDX(np)]  = 0.5 * g_sf  * softplus_grad(k->raw_params[SF_IDX(np)]);
    kernel_grads[OFF_IDX(np)] = 0.5 * g_off * softplus_grad(k->raw_params[OFF_IDX(np)]);
}

static void matern32lin_destroy(GPKernel *k) { free(k); }

GPKernel *gp_kernel_matern32_linear(int dim, float lengthscale, float outputscale,
                                    float offset)
{
    int n_params = dim + 2;
    GPKernel *k  = (GPKernel *)malloc(sizeof(GPKernel) + (size_t)n_params * sizeof(float));
    k->n_params   = n_params;
    k->raw_params = (float *)(k + 1);
    memcpy(k->tag, GP_KERNEL_TAG_MATERN32_LINEAR, 4);
    for (int i = 0; i < dim; i++)
        k->raw_params[i] = inv_softplus(lengthscale);
    k->raw_params[SF_IDX(n_params)]  = inv_softplus(outputscale);
    k->raw_params[OFF_IDX(n_params)] = inv_softplus(offset);
    k->build_K   = matern32lin_build_K;
    k->build_Ks  = matern32lin_build_Ks;
    k->k_self    = matern32lin_k_self;
    k->mll_grad  = matern32lin_mll_grad;
    k->destroy   = matern32lin_destroy;
    return k;
}

float gp_kernel_get_lengthscale(const GPKernel *k, int d) { return softplus(k->raw_params[d]); }
float gp_kernel_get_outputscale(const GPKernel *k) { return softplus(k->raw_params[SF_IDX(k->n_params)]);  }
float gp_kernel_get_offset     (const GPKernel *k) { return softplus(k->raw_params[OFF_IDX(k->n_params)]); }
void   gp_kernel_set_lengthscale(GPKernel *k, int d, float v) { k->raw_params[d] = inv_softplus(v); }
void   gp_kernel_set_outputscale(GPKernel *k, float v) { k->raw_params[SF_IDX(k->n_params)]  = inv_softplus(v); }
void   gp_kernel_set_offset     (GPKernel *k, float v) { k->raw_params[OFF_IDX(k->n_params)] = inv_softplus(v); }
