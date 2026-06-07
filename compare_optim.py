"""
compare_optim.py -- mini-batch hyperparameter optimization: gp_torch vs GPyTorch.

Runs Adam on the same batches from the same starting point.  Expected result:
MLL values and hyperparameter trajectories match closely step-for-step.

Known parametrization differences (do not indicate a bug):
  - GPyTorch GaussianLikelihood noise: softplus(raw) + 1e-4  (GreaterThan(1e-4))
    Both implementations now use SP_LB=1e-4, so raw_noise gradients agree to ~1e-9.
  - Kernel params agree to ~1e-5 (floating-point rounding differences).
  - For ARD comparison: GPyTorch MaternKernel uses ard_num_dims=1 here,
    matching our single-dim ARD lengthscale for the 1-D test case.

Usage: python3 compare_optim.py
"""

import math, sys
import numpy as np
import torch
import gpytorch
from gpytorch.kernels import MaternKernel, PolynomialKernel, AdditiveKernel, ScaleKernel

sys.path.insert(0, ".")
from gp_torch import GP

N     = 300
BATCH = 40
STEPS = 60
LR    = 0.05
SEED  = 42


def make_data(n, seed):
    rng = np.random.RandomState(seed)
    X   = rng.uniform(-5, 5, (n, 1)).astype(np.float64)
    y   = (np.sin(X[:, 0]) + 0.1 * rng.randn(n)).astype(np.float64)
    return X, y


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lik):
        super().__init__(tx, ty, lik)
        self.mean_module  = gpytorch.means.ZeroMean()
        matern = MaternKernel(nu=1.5, ard_num_dims=1)
        linear = PolynomialKernel(power=1)
        self.covar_module = ScaleKernel(AdditiveKernel(linear, matern))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def main():
    X, y   = make_data(N, SEED)
    Xt, yt = torch.from_numpy(X).double(), torch.from_numpy(y).double()

    rng     = np.random.RandomState(SEED + 1)
    batches = [rng.choice(N, BATCH, replace=False) for _ in range(STEPS)]

    # shared initial constrained hyperparameters
    ell0, sf0, noise0, off0 = 1.0, 1.0, 0.01, 1.0

    # -- our model ---------------------------------------------------------
    gp = GP(dim=1, capacity=BATCH,
            lengthscale=ell0, outputscale=sf0, noise=noise0, offset=off0,
            use_cuda=False)
    opt_ours = torch.optim.Adam(gp.parameters(), lr=LR)
    gp.train()

    # -- GPyTorch model ----------------------------------------------------
    lik = gpytorch.likelihoods.GaussianLikelihood()
    gpt = ExactGP(Xt, yt, lik)
    gpt.double()
    # Match initial constrained hyperparameters
    gpt.covar_module.base_kernel.kernels[1].lengthscale = ell0
    gpt.covar_module.outputscale                         = sf0
    lik.noise                                            = noise0
    gpt.covar_module.base_kernel.kernels[0].raw_offset.data.fill_(
        math.log(math.expm1(off0)))

    mll_fn  = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gpt)
    opt_gpt = torch.optim.Adam(gpt.parameters(), lr=LR)
    gpt.train(); lik.train()

    # -- step-by-step comparison -------------------------------------------
    print(f"\n{'step':>4}  {'MLL (ours)':>11}  {'MLL (gpt)':>11}  {'|ΔMLL|':>8}"
          f"  {'ell_ours':>9}  {'ell_gpt':>9}  {'noise_ours':>11}  {'noise_gpt':>11}")
    print("-" * 102)

    for step, idx in enumerate(batches):
        X_b, y_b = Xt[idx], yt[idx]

        # ours: fit loads batch + Cholesky; mll(recompute=False) skips a second one
        gp.fit(X_b.numpy(), y_b.numpy())
        opt_ours.zero_grad()
        loss_ours = -gp.mll(recompute=False)
        loss_ours.backward()
        mll_ours = -loss_ours.item()
        opt_ours.step()

        # gpytorch: update stored training data to the batch, then compute MLL
        gpt.set_train_data(inputs=X_b, targets=y_b, strict=False)
        opt_gpt.zero_grad()
        with gpytorch.settings.max_cholesky_size(BATCH + 1):
            loss_gpt = -mll_fn(gpt(X_b), y_b)
        loss_gpt.backward()
        mll_gpt = -loss_gpt.item()
        opt_gpt.step()

        if step % 10 == 0 or step == STEPS - 1:
            ell_ours = float(gp.lengthscale[0])
            ell_g    = gpt.covar_module.base_kernel.kernels[1].lengthscale.flatten()[0].item()
            print(f"{step:>4}  {mll_ours:>11.5f}  {mll_gpt:>11.5f}  "
                  f"{abs(mll_ours - mll_gpt):>8.2e}"
                  f"  {ell_ours:>9.4f}  {ell_g:>9.4f}"
                  f"  {gp.noise:>11.6f}  {lik.noise.item():>11.6f}")

    # -- final hyperparameter comparison -----------------------------------
    ell_ours = float(gp.lengthscale[0])
    ell_gpt  = gpt.covar_module.base_kernel.kernels[1].lengthscale.flatten()[0].item()
    ours_vals = [ell_ours, gp.outputscale, gp.noise, gp.offset]
    gpt_vals  = [
        ell_gpt,
        gpt.covar_module.outputscale.item(),
        lik.noise.item(),
        gpt.covar_module.base_kernel.kernels[0].offset.item(),
    ]
    print()
    print("Final hyperparameters:")
    for label, ov, gv in zip(["lengthscale", "outputscale", "noise", "offset"],
                              ours_vals, gpt_vals):
        print(f"  {label:12s}  ours={ov:.6f}  gpt={gv:.6f}  |Δ|={abs(ov - gv):.2e}")

    # -- gradient agreement at a fixed point (sanity check) ----------------
    print()
    print("Gradient agreement check (fresh models, same batch, same params):")
    idx0 = batches[0]
    X_b0, y_b0 = Xt[idx0], yt[idx0]

    # fresh ours
    gp2 = GP(dim=1, capacity=BATCH,
             lengthscale=ell0, outputscale=sf0, noise=noise0, offset=off0,
             use_cuda=False)
    gp2.train()
    gp2.fit(X_b0.numpy(), y_b0.numpy())
    (-gp2.mll(recompute=False)).backward()
    # raw_lengthscale is (1,); flatten all param grads into a list
    grads_ours = []
    for p in gp2.parameters():
        grads_ours.extend(p.grad.detach().flatten().tolist())

    # fresh gpytorch
    lik2 = gpytorch.likelihoods.GaussianLikelihood()
    gpt2 = ExactGP(Xt, yt, lik2)
    gpt2.double()
    gpt2.covar_module.base_kernel.kernels[1].lengthscale = ell0
    gpt2.covar_module.outputscale                         = sf0
    lik2.noise                                            = noise0
    gpt2.covar_module.base_kernel.kernels[0].raw_offset.data.fill_(
        math.log(math.expm1(off0)))
    gpt2.train(); lik2.train()
    mll_fn2 = gpytorch.mlls.ExactMarginalLogLikelihood(lik2, gpt2)
    gpt2.set_train_data(inputs=X_b0, targets=y_b0, strict=False)
    with gpytorch.settings.max_cholesky_size(BATCH + 1):
        (-mll_fn2(gpt2(X_b0), y_b0)).backward()

    # collect gpytorch gradients in the same param order as ours
    gpt_grads = [
        gpt2.covar_module.base_kernel.kernels[1].raw_lengthscale.grad.flatten()[0].item(),
        gpt2.covar_module.raw_outputscale.grad.item(),
        lik2.noise_covar.raw_noise.grad.item(),
        gpt2.covar_module.base_kernel.kernels[0].raw_offset.grad.item(),
    ]

    labels = ["raw_ell_0", "raw_outputscale", "raw_noise", "raw_offset"]
    for label, og, gg in zip(labels, grads_ours, gpt_grads):
        print(f"  {label:16s}  ours={og:+.6f}  gpt={gg:+.6f}  |Δ|={abs(og - gg):.2e}")


if __name__ == "__main__":
    main()
