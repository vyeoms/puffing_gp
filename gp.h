#ifndef GP_H
#define GP_H

#include <stdint.h>
#include "gp_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact GP regression -- CPU (CBLAS/LAPACK), batch training only.
//
// The covariance kernel is a pluggable GPKernel (see gp_kernel.h).
// Noise is held separately in raw_noise (softplus-constrained like kernel params).
// gp_create() takes ownership of the kernel and frees it on gp_destroy().
//
// Hyperparameter convention (GPyTorch-compatible):
//   constrained = softplus(raw) + 1e-6
//   raw         = inv_softplus(constrained)
// Read/write raw_noise and kernel->raw_params[] directly for gradient-based
// optimisation; use get/set functions for constrained (positive) values.

typedef struct {
    int        dim, n, cap;
    double    *X;               // (cap x dim) row-major
    double    *y;               // (cap,)
    double    *L;               // (cap x cap) buffer; leading n x n is lower Cholesky of K_y, row-major
    double    *alpha;           // (cap,) buffer; leading n is K_y^-1 * y
    double     raw_noise;
    double     dedup_threshold; // near-duplicate filter; 0 = disabled
    GPKernel  *kernel;          // covariance kernel (owned by this struct)
} GP;

// Takes ownership of kernel; caller must not free it separately.
GP   *gp_create  (int dim, int cap, GPKernel *kernel, double noise);
void  gp_destroy (GP *gp);

double gp_get_noise(const GP *gp);
void   gp_set_noise(GP *gp, double v);

// Returns 0, -1 (n > cap), or -2 (Cholesky failed).
int  gp_fit      (GP *gp, const double *X, const double *y, int n);
int  gp_recompute(GP *gp);

// Xs: (m x dim) row-major.  Pass vars=NULL to skip variance.
void gp_predict(const GP *gp, const double *Xs,
                double *means, double *vars, int m);

// Per-point MLL: (-0.5*y^T*K^-1*y - log|L| - 0.5*n*log(2*pi)) / n
double gp_marginal_log_likelihood(const GP *gp);

// Gradients of per-point MLL w.r.t. unconstrained hyperparameters; O(n^3).
//   d_raw_noise:  gradient for gp->raw_noise (pass NULL to skip)
//   kernel_grads: array of gp->kernel->n_params doubles (pass NULL to skip)
void gp_mll_grad(const GP *gp, double *d_raw_noise, double *kernel_grads);

int  gp_save(const GP *gp, const char *path);
GP  *gp_load(const char *path, int extra_cap);

// Remove near-duplicate points from a candidate training set before fitting.
// Iterates newest-to-oldest: the most recently added point in each cluster
// of points within `threshold` Euclidean distance survives.
//
// X:             (n x dim) row-major candidate inputs
// kept_indices:  caller-allocated array of at least n ints
// Returns:       number of points kept (written to kept_indices[0..count-1])
int gp_filter_near_duplicates(const double *X, int n, int dim,
                               double threshold, int *kept_indices);

#ifdef __cplusplus
}
#endif
#endif // GP_H
