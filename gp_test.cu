// gp_test.cu -- Cross-validate CPU (gp.c) vs CUDA (gp_cuda.cu).
//
// Prints max absolute error in means, variances, and MLL between the two
// backends.  A passing run should show errors < 1e-10.
// Also prints MLL gradients side-by-side.
//
// Build:  make gp_test
// Run:    ./gp_test [n [dim]]

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "gp.h"
#include "gp_kernel.h"
#include "gp_cuda.h"
#include "gp_cuda_kernel.h"

// Deterministic data generation via LCG
static void make_data(double *X, double *y, int n, int d, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n * d; i++) {
        s = s * 1664525u + 1013904223u;
        X[i] = ((double)(s >> 1) / (double)(1u << 31)) * 10.0 - 5.0;
    }
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        double noise = ((double)(s >> 1) / (double)(1u << 31)) * 0.2 - 0.1;
        y[i] = sin(X[i * d]) + noise;
    }
}

static void make_test(double *Xs, int m, int d, unsigned seed)
{
    unsigned s = seed + 9999;
    for (int i = 0; i < m * d; i++) {
        s = s * 1664525u + 1013904223u;
        Xs[i] = ((double)(s >> 1) / (double)(1u << 31)) * 12.0 - 6.0;
    }
}

static double maxabsdiff(const double *a, const double *b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv)
{
    int n   = argc > 1 ? atoi(argv[1]) : 500;
    int dim = argc > 2 ? atoi(argv[2]) : 3;
    int m   = 200;

    double lengthscale = 1.0, outputscale = 1.0, noise = 0.01, offset = 1.0;

    printf("n=%d  dim=%d  m=%d  ell=%.2f  sigma_f=%.2f  sigma_n=%.4f\n\n",
           n, dim, m, lengthscale, outputscale, noise);

    double *X  = (double *)malloc((size_t)n * dim * sizeof(double));
    double *y  = (double *)malloc(n * sizeof(double));
    double *Xs = (double *)malloc((size_t)m * dim * sizeof(double));
    make_data(X, y, n, dim, 42);
    make_test(Xs, m, dim, 99);

    // CPU GP
    GP *gp = gp_create(dim, n,
                       gp_kernel_matern32_linear(dim, lengthscale, outputscale, offset),
                       noise);
    gp_fit(gp, X, y, n);

    double *cpu_means = (double *)malloc(m * sizeof(double));
    double *cpu_vars  = (double *)malloc(m * sizeof(double));
    gp_predict(gp, Xs, cpu_means, cpu_vars, m);
    double cpu_mll = gp_marginal_log_likelihood(gp);

    // kernel grads: [0..dim-1]=ell, [dim]=sf, [dim+1]=offset
    int np = dim + 2;
    double cpu_d_noise;
    double *cpu_kgrads = (double *)malloc(np * sizeof(double));
    gp_mll_grad(gp, &cpu_d_noise, cpu_kgrads);

    // CUDA GP
    GPCU *gpcu = gpcu_create(dim, n,
                             gpcu_kernel_matern32_linear(dim, lengthscale, outputscale, offset),
                             noise);
    gpcu_fit(gpcu, X, y, n, 0);

    double *gpu_means = (double *)malloc(m * sizeof(double));
    double *gpu_vars  = (double *)malloc(m * sizeof(double));
    gpcu_predict(gpcu, Xs, gpu_means, gpu_vars, m, 0);
    double gpu_mll = gpcu_marginal_log_likelihood(gpcu);

    double gpu_d_noise;
    double *gpu_kgrads = (double *)malloc(np * sizeof(double));
    gpcu_mll_grad(gpcu, &gpu_d_noise, gpu_kgrads, 0);

    // Comparison
    printf("Prediction errors (CPU vs CUDA):\n");
    printf("  |mean_err|_inf  = %.3e\n", maxabsdiff(cpu_means, gpu_means, m));
    printf("  |var_err|_inf   = %.3e\n", maxabsdiff(cpu_vars,  gpu_vars,  m));
    printf("  |MLL_err|       = %.3e\n", fabs(cpu_mll - gpu_mll));

    printf("\nMarginal log likelihood:\n");
    printf("  CPU  MLL = %.6f\n", cpu_mll);
    printf("  CUDA MLL = %.6f\n", gpu_mll);

    printf("\nGradients (should match CPU vs CUDA):\n");
    for (int k = 0; k < dim; k++) {
        char name[32];
        snprintf(name, sizeof name, "d_raw_lengthscale[%d]", k);
        printf("  %-28s  CPU = %+.6e   CUDA = %+.6e   |err| = %.3e\n",
               name, cpu_kgrads[k], gpu_kgrads[k],
               fabs(cpu_kgrads[k] - gpu_kgrads[k]));
    }
    printf("  %-28s  CPU = %+.6e   CUDA = %+.6e   |err| = %.3e\n",
           "d_raw_outputscale", cpu_kgrads[np - 2], gpu_kgrads[np - 2],
           fabs(cpu_kgrads[np - 2] - gpu_kgrads[np - 2]));
    printf("  %-28s  CPU = %+.6e   CUDA = %+.6e   |err| = %.3e\n",
           "d_raw_noise",       cpu_d_noise,   gpu_d_noise,
           fabs(cpu_d_noise    - gpu_d_noise));
    printf("  %-28s  CPU = %+.6e   CUDA = %+.6e   |err| = %.3e\n",
           "d_raw_offset",      cpu_kgrads[np - 1], gpu_kgrads[np - 1],
           fabs(cpu_kgrads[np - 1] - gpu_kgrads[np - 1]));

    // Save/load round-trip -- CPU
    gp_save(gp, "/tmp/gp_test.bin");
    GP *gp2 = gp_load("/tmp/gp_test.bin", 0);
    double *rt_means = (double *)malloc(m * sizeof(double));
    double *rt_vars  = (double *)malloc(m * sizeof(double));
    gp_predict(gp2, Xs, rt_means, rt_vars, m);
    printf("\nSave/load round-trip (CPU):\n");
    printf("  |mean_err|_inf  = %.3e\n", maxabsdiff(cpu_means, rt_means, m));
    printf("  |var_err|_inf   = %.3e\n", maxabsdiff(cpu_vars,  rt_vars,  m));
    gp_destroy(gp2); free(rt_means); free(rt_vars);

    // Save/load round-trip -- CUDA
    gpcu_save(gpcu, "/tmp/gpcu_test.bin");
    GPCU *gpcu2 = gpcu_load("/tmp/gpcu_test.bin", 0);
    double *rt2_means = (double *)malloc(m * sizeof(double));
    double *rt2_vars  = (double *)malloc(m * sizeof(double));
    gpcu_predict(gpcu2, Xs, rt2_means, rt2_vars, m, 0);
    printf("\nSave/load round-trip (CUDA):\n");
    printf("  |mean_err|_inf  = %.3e\n", maxabsdiff(gpu_means, rt2_means, m));
    printf("  |var_err|_inf   = %.3e\n", maxabsdiff(gpu_vars,  rt2_vars,  m));
    gpcu_destroy(gpcu2); free(rt2_means); free(rt2_vars);

    gp_destroy(gp);
    gpcu_destroy(gpcu);
    free(X); free(y); free(Xs);
    free(cpu_means); free(cpu_vars); free(cpu_kgrads);
    free(gpu_means); free(gpu_vars); free(gpu_kgrads);
    return 0;
}
