#ifndef GP_KERNEL_H
#define GP_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

// CPU GP covariance kernel interface.
//
// A kernel owns its hyperparameters in raw_params[] (unconstrained values)
// and provides function pointers to build covariance matrices and compute
// MLL gradients.  Noise is NOT part of the kernel; the GP struct holds it.
//
// To define a custom kernel: allocate a GPKernel, fill n_params/raw_params/tag
// and assign the five function pointers.  Pass it to gp_create(), which takes
// ownership and calls k->destroy(k) on gp_destroy().

typedef struct GPKernel GPKernel;
struct GPKernel {
    int     n_params;    // number of unconstrained hyperparameters
    double *raw_params;  // [n_params], unconstrained values
    char    tag[4];      // 4-char type tag used by gp_save/gp_load ("M32L", etc.)

    // Build nxn covariance matrix K (row-major). Adds sigma_n to diagonal.
    void (*build_K)(const GPKernel *k, const double *X, int n, int d,
                    double sigma_n, double *K);

    // Build nxm cross-covariance Ks (row-major): k(X_train_i, X_test_j).
    void (*build_Ks)(const GPKernel *k, const double *Xtr, const double *Xte,
                     int n, int m, int d, double *Ks);

    // Prior variance k(x, x) for a single point x (length d).
    double (*k_self)(const GPKernel *k, const double *x, int d);

    // Gradient of MLL w.r.t. each raw_params[i]:
    //   kernel_grads[i] = 0.5 * softplus_grad(raw_params[i])
    //                         * tr((alpha*alpha^T - Kinv) * dK/d(raw_params[i]))
    // Caller pre-zeroes kernel_grads[0..n_params-1] before calling.
    // alpha: (n,), Kinv: (nxn row-major, symmetric).
    void (*mll_grad)(const GPKernel *k, const double *X, int n, int d,
                     const double *alpha, const double *Kinv, double *kernel_grads);

    void (*destroy)(GPKernel *k);
};

// Built-in: Matern32 + linear (dot-product) kernel with ARD lengthscales.
//
//   k(x,x') = sigma_f * (x.x' + offset
//             + (1 + sqrt(3)*r) * exp(-sqrt(3)*r))
//   where r = sqrt(sum_d (x_d - x'_d)^2 / ell_d^2)  (ARD distance)
//   = ScaleKernel(AdditiveKernel(PolynomialKernel(power=1), Matern32(ARD)))
//
// raw_params layout (all softplus-constrained: constrained = softplus(raw) + 1e-4):
//   [0..dim-1]  raw_ell_d   ->  ell_d  (per-dimension lengthscale)
//   [dim]       raw_sf      ->  sigma_f (output scale)
//   [dim+1]     raw_offset  ->  offset
//   n_params = dim + 2
#define GP_KERNEL_TAG_MATERN32_LINEAR "M32L"

GPKernel *gp_kernel_matern32_linear(int dim, double lengthscale, double outputscale,
                                    double offset);

// Named accessors for the Matern32+Linear kernel (softplus-constrained).
double gp_kernel_get_lengthscale(const GPKernel *k, int d);
double gp_kernel_get_outputscale(const GPKernel *k);
double gp_kernel_get_offset     (const GPKernel *k);
void   gp_kernel_set_lengthscale(GPKernel *k, int d, double v);
void   gp_kernel_set_outputscale(GPKernel *k, double v);
void   gp_kernel_set_offset     (GPKernel *k, double v);

#ifdef __cplusplus
}
#endif
#endif // GP_KERNEL_H
