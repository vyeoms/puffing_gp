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
static inline double softplus     (double x) { return (x > 20.0 ? x : log1p(exp(x))) + SP_LB; }
static inline double inv_softplus (double x) { double v = x - SP_LB; return v > 20.0 ? v : log(expm1(v)); }
static inline double softplus_grad(double x) { return 1.0 / (1.0 + exp(-x)); }

// Index helpers — n_params = dim+2, so SF = n_params-2, OFF = n_params-1.
#define SF_IDX(np)  ((np) - 2)
#define OFF_IDX(np) ((np) - 1)

static void m32lin_build_K(const GPKernel *k, const double *X, int n, int d,
                            double sigma_n, double *K)
{
    int    np      = k->n_params;
    double sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    double offset  = softplus(k->raw_params[OFF_IDX(np)]);

    // Precompute inv_ell[d] and build scaled X.
    double *inv_ell = (double *)malloc(d * sizeof(double));
    for (int dd = 0; dd < d; dd++)
        inv_ell[dd] = 1.0 / softplus(k->raw_params[dd]);

    double *Xs = (double *)malloc((size_t)n * d * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int dd = 0; dd < d; dd++)
            Xs[i * d + dd] = X[i * d + dd] * inv_ell[dd];
    free(inv_ell);

    // DOT[i*n+j] = xi.xj  (for linear kernel term)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, n, d, 1.0, X, d, X, d, 0.0, K, n);

    // DOT_s and scaled norms for ARD distance computation
    double *DOT_s  = (double *)malloc((size_t)n * n * sizeof(double));
    double *norms_s = (double *)malloc(n * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, n, d, 1.0, Xs, d, Xs, d, 0.0, DOT_s, n);
    for (int i = 0; i < n; i++)
        norms_s[i] = DOT_s[i * (n + 1)];  // diagonal of Xs*Xs^T
    free(Xs);

    for (int i = 0; i < n; i++) {
        double  ns_i = norms_s[i];
        double *Ki   = K + i * n;
        double *Ks_i = DOT_s + i * n;
        for (int j = 0; j < n; j++) {
            double dot = Ki[j];
            double r2  = ns_i + norms_s[j] - 2.0 * Ks_i[j];
            if (r2 < 0.0) r2 = 0.0;
            double u   = sqrt(3.0) * sqrt(r2);
            Ki[j] = sigma_f * (dot + offset + (1.0 + u) * exp(-u));
        }
        Ki[i] += sigma_n;
    }
    free(DOT_s);
    free(norms_s);
}

static void m32lin_build_Ks(const GPKernel *k, const double *Xtr, const double *Xte,
                              int n, int m, int d, double *Ks)
{
    int    np      = k->n_params;
    double sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    double offset  = softplus(k->raw_params[OFF_IDX(np)]);

    double *inv_ell = (double *)malloc(d * sizeof(double));
    for (int dd = 0; dd < d; dd++)
        inv_ell[dd] = 1.0 / softplus(k->raw_params[dd]);

    double *Xtrs = (double *)malloc((size_t)n * d * sizeof(double));
    double *Xtes = (double *)malloc((size_t)m * d * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int dd = 0; dd < d; dd++)
            Xtrs[i * d + dd] = Xtr[i * d + dd] * inv_ell[dd];
    for (int j = 0; j < m; j++)
        for (int dd = 0; dd < d; dd++)
            Xtes[j * d + dd] = Xte[j * d + dd] * inv_ell[dd];
    free(inv_ell);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, m, d, 1.0, Xtr, d, Xte, d, 0.0, Ks, m);

    double *DOT_s  = (double *)malloc((size_t)n * m * sizeof(double));
    double *ntr_s  = (double *)malloc(n * sizeof(double));
    double *nte_s  = (double *)malloc(m * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, m, d, 1.0, Xtrs, d, Xtes, d, 0.0, DOT_s, m);
    for (int i = 0; i < n; i++)
        ntr_s[i] = cblas_ddot(d, &Xtrs[i * d], 1, &Xtrs[i * d], 1);
    for (int j = 0; j < m; j++)
        nte_s[j] = cblas_ddot(d, &Xtes[j * d], 1, &Xtes[j * d], 1);
    free(Xtrs);
    free(Xtes);

    for (int i = 0; i < n; i++) {
        double  ns_i = ntr_s[i];
        double *Ki   = Ks + i * m;
        double *Ks_i = DOT_s + i * m;
        for (int j = 0; j < m; j++) {
            double dot = Ki[j];
            double r2  = ns_i + nte_s[j] - 2.0 * Ks_i[j];
            if (r2 < 0.0) r2 = 0.0;
            double u   = sqrt(3.0) * sqrt(r2);
            Ki[j] = sigma_f * (dot + offset + (1.0 + u) * exp(-u));
        }
    }
    free(DOT_s);
    free(ntr_s);
    free(nte_s);
}

static double m32lin_k_self(const GPKernel *k, const double *x, int d)
{
    int    np      = k->n_params;
    double sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    double offset  = softplus(k->raw_params[OFF_IDX(np)]);
    double norm2   = cblas_ddot(d, x, 1, x, 1);
    // r=0 when xi==xj, so Matern32(0)=1
    return sigma_f * (norm2 + offset + 1.0);
}

static void m32lin_mll_grad(const GPKernel *k, const double *X, int n, int d,
                             const double *alpha, const double *Kinv,
                             double *kernel_grads)
{
    int    np      = k->n_params;
    double sigma_f = softplus(k->raw_params[SF_IDX(np)]);
    double offset  = softplus(k->raw_params[OFF_IDX(np)]);

    double *inv_ells = (double *)malloc(d * sizeof(double));
    double *inv_ell3 = (double *)malloc(d * sizeof(double));
    for (int dd = 0; dd < d; dd++) {
        double ell   = softplus(k->raw_params[dd]);
        inv_ells[dd] = 1.0 / ell;
        inv_ell3[dd] = inv_ells[dd] * inv_ells[dd] * inv_ells[dd];
    }

    // DOT[i*n+j] = xi.xj (for sf gradient term)
    double *DOT = (double *)malloc((size_t)n * n * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, n, d, 1.0, X, d, X, d, 0.0, DOT, n);

    double *g_ell = (double *)calloc(d, sizeof(double));
    double  g_sf  = 0.0;
    double *tmp   = (double *)malloc(d * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double dot    = DOT[i * n + j];
            double r2     = 0.0;
            // compute ARD distance and store per-dim diffs
            for (int dd = 0; dd < d; dd++) {
                double diff = X[i * d + dd] - X[j * d + dd];
                tmp[dd] = diff;
                double sd = diff * inv_ells[dd];
                r2 += sd * sd;
            }
            if (r2 < 0.0) r2 = 0.0;
            double u       = sqrt(3.0) * sqrt(r2);
            double e_ij    = exp(-u);
            double aa_kinv = alpha[i] * alpha[j] - Kinv[i * n + j];
            g_sf += aa_kinv * (dot + offset + (1.0 + u) * e_ij);
            double coeff = aa_kinv * sigma_f * 3.0 * e_ij;
            for (int dd = 0; dd < d; dd++)
                g_ell[dd] += coeff * tmp[dd] * tmp[dd] * inv_ell3[dd];
        }
    }
    free(DOT);
    free(tmp);

    // dK/d(offset) = sigma_f for every (i,j)
    double alpha_sum = 0.0, kinv_sum = 0.0;
    for (int i = 0; i < n; i++) alpha_sum += alpha[i];
    for (int i = 0; i < n * n; i++) kinv_sum += Kinv[i];
    double g_off = sigma_f * (alpha_sum * alpha_sum - kinv_sum);

    for (int dd = 0; dd < d; dd++)
        kernel_grads[dd] = 0.5 * g_ell[dd] * softplus_grad(k->raw_params[dd]);
    free(g_ell);
    free(inv_ells);
    free(inv_ell3);
    kernel_grads[SF_IDX(np)]  = 0.5 * g_sf  * softplus_grad(k->raw_params[SF_IDX(np)]);
    kernel_grads[OFF_IDX(np)] = 0.5 * g_off * softplus_grad(k->raw_params[OFF_IDX(np)]);
}

static void m32lin_destroy(GPKernel *k) { free(k); }

GPKernel *gp_kernel_matern32_linear(int dim, double lengthscale, double outputscale,
                                    double offset)
{
    int n_params = dim + 2;
    GPKernel *k  = (GPKernel *)malloc(sizeof(GPKernel) + (size_t)n_params * sizeof(double));
    k->n_params   = n_params;
    k->raw_params = (double *)(k + 1);
    memcpy(k->tag, GP_KERNEL_TAG_MATERN32_LINEAR, 4);
    for (int i = 0; i < dim; i++)
        k->raw_params[i] = inv_softplus(lengthscale);
    k->raw_params[SF_IDX(n_params)]  = inv_softplus(outputscale);
    k->raw_params[OFF_IDX(n_params)] = inv_softplus(offset);
    k->build_K   = m32lin_build_K;
    k->build_Ks  = m32lin_build_Ks;
    k->k_self    = m32lin_k_self;
    k->mll_grad  = m32lin_mll_grad;
    k->destroy   = m32lin_destroy;
    return k;
}

double gp_kernel_get_lengthscale(const GPKernel *k, int d) { return softplus(k->raw_params[d]); }
double gp_kernel_get_outputscale(const GPKernel *k) { return softplus(k->raw_params[SF_IDX(k->n_params)]);  }
double gp_kernel_get_offset     (const GPKernel *k) { return softplus(k->raw_params[OFF_IDX(k->n_params)]); }
void   gp_kernel_set_lengthscale(GPKernel *k, int d, double v) { k->raw_params[d] = inv_softplus(v); }
void   gp_kernel_set_outputscale(GPKernel *k, double v) { k->raw_params[SF_IDX(k->n_params)]  = inv_softplus(v); }
void   gp_kernel_set_offset     (GPKernel *k, double v) { k->raw_params[OFF_IDX(k->n_params)] = inv_softplus(v); }
