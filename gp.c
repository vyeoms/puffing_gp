// gp.c -- Exact GP regression, CPU (CBLAS/LAPACK), batch training only.
//
// Covariance kernel is pluggable; see gp_kernel.h for the interface and
// gp_kernel.c for the built-in Matern32+Linear implementation.
//
// BLAS/LAPACK:  dpotrf, dpotri, dtrsv, dtrsm, dgemv
//
// Save format ("GP04"):
//   magic[4], dim(i32), n(i32), raw_noise(f64),
//   kernel_tag[4], n_params(i32), raw_params[n_params](f64),
//   X[n*dim], y[n], L[n*n], alpha[n]

#include "gp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cblas.h>
#include <lapacke.h>

#define SP_LB 1e-4
static inline float softplus     (float x)   { return (x > 20.0 ? x : log1p(exp(x))) + SP_LB; }
static inline float inv_softplus (float x)   { float v = x - SP_LB; return v > 20.0 ? v : log(expm1(v)); }
static inline float softplus_grad(float raw) { return 1.0 / (1.0 + exp(-raw)); }

float gp_get_noise(const GP *gp) { return softplus(gp->raw_noise);   }
void   gp_set_noise(GP *gp, float v) { gp->raw_noise = inv_softplus(v); }

GP *gp_create(int dim, int cap, GPKernel *kernel, float noise)
{
    GP *gp     = (GP *)calloc(1, sizeof(GP));
    gp->dim    = dim;
    gp->cap    = cap;
    gp->X      = (float *)calloc((size_t)cap * dim, sizeof(float));
    gp->y      = (float *)calloc(cap, sizeof(float));
    gp->L      = (float *)malloc((size_t)cap * cap * sizeof(float));
    gp->alpha  = (float *)malloc((size_t)cap * sizeof(float));
    gp->kernel = kernel;
    gp_set_noise(gp, noise);
    return gp;
}

void gp_destroy(GP *gp)
{
    if (!gp) return;
    free(gp->X); free(gp->y); free(gp->L); free(gp->alpha);
    if (gp->kernel && gp->kernel->destroy) gp->kernel->destroy(gp->kernel);
    free(gp);
}

int gp_recompute(GP *gp)
{
    int n = gp->n;
    if (n == 0) return 0;

    float  sigma_n = gp_get_noise(gp);
    float *K       = gp->L;  // factor in place; L/alpha are cap-sized buffers
    gp->kernel->build_K(gp->kernel, gp->X, n, gp->dim, sigma_n, K);

    lapack_int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', n, K, n);
    if (info != 0) {
        gp->kernel->build_K(gp->kernel, gp->X, n, gp->dim, sigma_n, K);
        for (int i = 0; i < n; i++) K[i * n + i] += 1e-8;
        info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', n, K, n);
        if (info != 0) return -2;
    }
    // dpotrf writes only the lower triangle; upper is ignored by dtrsv(CblasLower).
    memcpy(gp->alpha, gp->y, n * sizeof(float));
    // alpha = L^-T L^-1 y  (two triangular solves)
    cblas_strsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                n, gp->L, n, gp->alpha, 1);
    cblas_strsv(CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit,
                n, gp->L, n, gp->alpha, 1);
    return 0;
}

int gp_fit(GP *gp, const float *X, const float *y, int n)
{
    if (n > gp->cap) return -1;

    if (gp->dedup_threshold > 0.0 && n > 1) {
        int *idx    = (int *)malloc(n * sizeof(int));
        int  n_kept = gp_filter_near_duplicates(X, n, gp->dim,
                                                gp->dedup_threshold, idx);
        for (int i = 0; i < n_kept; i++) {
            memcpy(&gp->X[i * gp->dim], &X[idx[i] * gp->dim],
                   gp->dim * sizeof(float));
            gp->y[i] = y[idx[i]];
        }
        free(idx);
        gp->n = n_kept;
    } else {
        memcpy(gp->X, X, (size_t)n * gp->dim * sizeof(float));
        memcpy(gp->y, y, (size_t)n * sizeof(float));
        gp->n = n;
    }
    return gp_recompute(gp);
}

void gp_predict(const GP *gp, const float *Xs, float *means, float *vars, int m)
{
    int n = gp->n;
    if (n == 0) {
        // prior: zero mean, k(x*, x*) variance
        for (int j = 0; j < m; j++) {
            means[j] = 0.0;
            if (vars) vars[j] = gp->kernel->k_self(gp->kernel, &Xs[j * gp->dim], gp->dim);
        }
        return;
    }

    float *Ks = (float *)malloc((size_t)n * m * sizeof(float));
    gp->kernel->build_Ks(gp->kernel, gp->X, Xs, n, m, gp->dim, Ks);

    // means = Ks^T alpha
    cblas_sgemv(CblasRowMajor, CblasTrans,
                n, m, 1.0, Ks, m, gp->alpha, 1, 0.0, means, 1);

    if (vars) {
        // prior variances k(x*, x*)
        for (int j = 0; j < m; j++)
            vars[j] = gp->kernel->k_self(gp->kernel, &Xs[j * gp->dim], gp->dim);

        // posterior: var[j] = k(x*,x*) - ||L^-1 Ks_j||^2
        cblas_strsm(CblasRowMajor, CblasLeft, CblasLower,
                    CblasNoTrans, CblasNonUnit,
                    n, m, 1.0, gp->L, n, Ks, m);
        for (int j = 0; j < m; j++) {
            float v = vars[j] - cblas_sdot(n, Ks + j, m, Ks + j, m);
            vars[j] = v > 0.0 ? v : 0.0;
        }
    }
    free(Ks);
}

float gp_marginal_log_likelihood(const GP *gp)
{
    int n = gp->n;
    if (n == 0) return 0.0;
    float log_det = 0.0;
    for (int i = 0; i < n; i++) log_det += log(gp->L[i * n + i]);
    return (-0.5 * cblas_sdot(n, gp->y, 1, gp->alpha, 1)
            - log_det
            - 0.5 * n * log(2.0 * M_PI)) / n;
}

void gp_mll_grad(const GP *gp, float *d_raw_noise, float *kernel_grads)
{
    int n = gp->n;
    if (n == 0) {
        if (d_raw_noise) *d_raw_noise = 0.0;
        if (kernel_grads)
            for (int i = 0; i < gp->kernel->n_params; i++) kernel_grads[i] = 0.0;
        return;
    }

    // K^-1 via dpotri on a copy of L; dpotri fills only the lower triangle
    float *Kinv = (float *)malloc((size_t)n * n * sizeof(float));
    memcpy(Kinv, gp->L, (size_t)n * n * sizeof(float));
    LAPACKE_spotri(LAPACK_ROW_MAJOR, 'L', n, Kinv, n);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            Kinv[i * n + j] = Kinv[j * n + i];

    // Noise gradient: dK/d(sigma_n) = I, so tr((alpha*alpha^T - Kinv)*I) = ||alpha||^2 - tr(Kinv)
    if (d_raw_noise) {
        float g_n = 0.0;
        for (int i = 0; i < n; i++)
            g_n += gp->alpha[i] * gp->alpha[i] - Kinv[i * n + i];
        *d_raw_noise = 0.5 * g_n * softplus_grad(gp->raw_noise) / n;
    }

    if (kernel_grads) {
        for (int i = 0; i < gp->kernel->n_params; i++) kernel_grads[i] = 0.0;
        gp->kernel->mll_grad(gp->kernel, gp->X, n, gp->dim,
                             gp->alpha, Kinv, kernel_grads);
        for (int i = 0; i < gp->kernel->n_params; i++) kernel_grads[i] /= n;
    }

    free(Kinv);
}

// --- kd-tree for gp_filter_near_duplicates ---

#define KD_LEAF 16

typedef struct { int lo, hi, left, right, split_dim; float split_val; } KDNode;
typedef struct { const float *X; int n, dim, n_nodes; int *idx; KDNode *nodes; } KDTree;

static int          g_kd_split_dim, g_kd_d;
static const float *g_kd_X;

static int kd_cmp_fn(const void *a, const void *b)
{
    float va = g_kd_X[*(const int *)a * g_kd_d + g_kd_split_dim];
    float vb = g_kd_X[*(const int *)b * g_kd_d + g_kd_split_dim];
    return (va > vb) - (va < vb);
}

static void kd_build(KDTree *t, int node, int lo, int hi)
{
    KDNode *nd = &t->nodes[node];
    nd->lo = lo; nd->hi = hi; nd->left = nd->right = -1; nd->split_dim = -1;
    if (hi - lo <= KD_LEAF) return;

    int best = 0; float spread = -1.0;
    for (int k = 0; k < t->dim; k++) {
        float mn = t->X[t->idx[lo] * t->dim + k], mx = mn;
        for (int i = lo + 1; i < hi; i++) {
            float v = t->X[t->idx[i] * t->dim + k];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
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

static KDTree kd_create(const float *X, int n, int dim)
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

static void kd_query_ball(const KDTree *t, int node, const float *q,
                           float r, float r2, int *out, int *cnt)
{
    const KDNode *nd = &t->nodes[node];
    if (nd->split_dim < 0) {
        for (int i = nd->lo; i < nd->hi; i++) {
            int p = t->idx[i]; float sq = 0.0;
            for (int k = 0; k < t->dim; k++) {
                float df = q[k] - t->X[p * t->dim + k]; sq += df * df;
            }
            if (sq <= r2) out[(*cnt)++] = p;
        }
        return;
    }
    float dv = q[nd->split_dim] - nd->split_val;
    if (nd->left  >= 0 && dv <=  r) kd_query_ball(t, nd->left,  q, r, r2, out, cnt);
    if (nd->right >= 0 && dv >= -r) kd_query_ball(t, nd->right, q, r, r2, out, cnt);
}

int gp_filter_near_duplicates(const float *X, int n, int dim,
                               float threshold, int *kept_indices)
{
    if (n <= 0) return 0;
    if (n == 1) { kept_indices[0] = 0; return 1; }

    KDTree tree   = kd_create(X, n, dim);
    int   *keep   = (int *)malloc(n * sizeof(int));
    int   *nearby = (int *)malloc(n * sizeof(int));
    float r2     = threshold * threshold;
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

// -- kernel registry for gp_load ----------------------------------------------

typedef struct { const char *tag; GPKernel *(*make)(float *, int); } KernelFactory;

static GPKernel *kf_m32lin(float *rp, int np)
{
    int dim = np - 2;
    GPKernel *k = gp_kernel_matern32_linear(dim, 1.0, 1.0, 1.0);
    memcpy(k->raw_params, rp, (size_t)np * sizeof(float));
    return k;
}

static const KernelFactory cpu_kernel_registry[] = {
    { GP_KERNEL_TAG_MATERN32_LINEAR, kf_m32lin },
    { NULL, NULL }
};

static GPKernel *cpu_kernel_from_tag(const char *tag, float *rp, int np)
{
    for (int i = 0; cpu_kernel_registry[i].tag; i++)
        if (memcmp(tag, cpu_kernel_registry[i].tag, 4) == 0)
            return cpu_kernel_registry[i].make(rp, np);
    return NULL;
}

// -- save / load ---------------------------------------------------------------

int gp_save(const GP *gp, const char *path)
{
    if (gp->n == 0) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int32_t dim = gp->dim, n = gp->n, np = gp->kernel->n_params;
    fwrite("GP04",                  1,              4,  f);
    fwrite(&dim,                    sizeof dim,     1,  f);
    fwrite(&n,                      sizeof n,       1,  f);
    fwrite(&gp->raw_noise,          sizeof(float), 1,  f);
    fwrite(gp->kernel->tag,         1,              4,  f);
    fwrite(&np,                     sizeof np,      1,  f);
    fwrite(gp->kernel->raw_params,  sizeof(float), np, f);
    fwrite(gp->X,     sizeof(float), (size_t)n * dim, f);
    fwrite(gp->y,     sizeof(float), n,               f);
    fwrite(gp->L,     sizeof(float), (size_t)n * n,   f);
    fwrite(gp->alpha, sizeof(float), n,               f);

    fclose(f);
    return 0;
}

GP *gp_load(const char *path, int extra_cap)
{
    GP   *gp = NULL;
    FILE *f  = fopen(path, "rb");
    if (!f) return NULL;

#define RD(ptr, sz, cnt) \
    if (fread(ptr, sz, cnt, f) != (size_t)(cnt)) break

    do {
        char    magic[4], ktag[4];
        int32_t dim, n, np;
        float  rn;

        RD(magic, 1, 4);
        if (memcmp(magic, "GP04", 4) != 0) break;

        RD(&dim,  sizeof dim, 1);
        RD(&n,    sizeof n,   1);
        RD(&rn,   sizeof(float), 1);
        RD(ktag,  1, 4);
        RD(&np,   sizeof np, 1);

        float *rp = (float *)malloc((size_t)np * sizeof(float));
        if (!rp) break;
        if (fread(rp, sizeof(float), np, f) != (size_t)np) { free(rp); break; }

        GPKernel *k = cpu_kernel_from_tag(ktag, rp, np);
        free(rp);
        if (!k) break;

        int cap = n + (extra_cap > 0 ? extra_cap : 0);
        gp = gp_create(dim, cap, k, 1.0);
        gp->raw_noise = rn;
        gp->n         = n;

        RD(gp->X,     sizeof(float), (size_t)n * dim);
        RD(gp->y,     sizeof(float), n);
        RD(gp->L,     sizeof(float), (size_t)n * n);
        RD(gp->alpha, sizeof(float), n);

        fclose(f);
        return gp;
    } while (0);

#undef RD
    if (gp) gp_destroy(gp);
    fclose(f);
    return NULL;
}
