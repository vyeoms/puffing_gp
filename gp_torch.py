"""
gp_torch.py -- PyTorch-compatible GP wrapper (puffing_gpcu / puffing_gp backend).

Drop-in replacement for GPyTorch exact GP training with any torch.optim optimizer:

    from gp_torch import GP

    model = GP(dim=5, capacity=n, lengthscale=1.0, outputscale=1.0, noise=0.01)
    model.fit(X, y)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for step in range(200):
        optimizer.zero_grad()
        loss = -model.mll()
        loss.backward()
        optimizer.step()

    model.eval()
    means, vars_ = model.predict(Xs)

Raw (unconstrained) parameters are nn.Parameters.  ARD lengthscales are stored
in a (dim,) Parameter vector.  Constrained: constrained = softplus(raw) + 1e-4.

model.eval() triggers a final sync + Cholesky recompute so predict() is fast.
"""

import numpy as np
import torch
import torch.nn as nn


class _MLLFunction(torch.autograd.Function):
    """Scalar MLL value with analytical gradients injected into autograd."""

    @staticmethod
    def forward(ctx, params, backend):
        # params wires all nn.Parameters into the graph; MLL is read from backend.
        ctx.backend = backend
        return params.new_tensor(backend.log_marginal_likelihood)

    @staticmethod
    def backward(ctx, grad_output):
        # mll_grad() returns flat (ell_0,...,ell_{dim-1}, sf, noise, offset)
        grads = torch.tensor(ctx.backend.mll_grad(),
                             dtype=grad_output.dtype, device=grad_output.device)
        return grad_output * grads, None


class GP(nn.Module):
    """
    Exact GP regression backed by puffing_gpcu (CUDA) or puffing_gp (CPU).

    Parameters
    ----------
    dim, capacity    : input dimension and max training points
    lengthscale,
    outputscale,
    noise, offset    : initial constrained hyperparameter values
    use_cuda         : prefer CUDA backend when available (default True)
    """

    def __init__(self, dim, capacity,
                 lengthscale=1.0, outputscale=1.0, noise=1e-2, offset=1.0,
                 use_cuda=True):
        super().__init__()
        self._backend = _make_backend(dim, capacity,
                                      lengthscale, outputscale, noise, offset,
                                      use_cuda)
        # raw_lengthscale is (dim,) for ARD; others are scalars
        self.raw_lengthscale = nn.Parameter(
            torch.tensor(np.asarray(self._backend.raw_lengthscale), dtype=torch.float64))
        self.raw_outputscale = nn.Parameter(
            torch.tensor(self._backend.raw_outputscale, dtype=torch.float64))
        self.raw_noise = nn.Parameter(
            torch.tensor(self._backend.raw_noise, dtype=torch.float64))
        self.raw_offset = nn.Parameter(
            torch.tensor(self._backend.raw_offset, dtype=torch.float64))

    # -- constrained read-only properties ----------------------------------

    @property
    def lengthscale(self):
        return np.asarray(self._backend.lengthscale)  # shape (dim,)

    @property
    def outputscale(self): return self._backend.outputscale

    @property
    def noise(self): return self._backend.noise

    @property
    def offset(self): return self._backend.offset

    @property
    def log_marginal_likelihood(self): return self._backend.log_marginal_likelihood

    @property
    def lengthscale_range(self):
        ells = self.lengthscale
        return float(np.min(ells)), float(np.max(ells))

    @property
    def n(self):        return self._backend.n

    @property
    def dim(self):      return self._backend.dim

    @property
    def capacity(self): return self._backend.capacity

    # -- core API ----------------------------------------------------------

    def fit(self, X, y):
        """Copy training data to the backend and compute the initial Cholesky."""
        self._sync()
        self._backend.fit(_np64(X), _np64(y))

    def recompute(self):
        """Sync params to backend and re-factorize without reloading data."""
        self._sync()
        self._backend.recompute()

    def mll(self, recompute=True):
        """
        Marginal log-likelihood as a scalar tensor with autograd support.

        Syncs raw params to the backend, optionally recomputes the Cholesky,
        then returns a tensor whose .backward() fills .grad on all raw_*
        Parameters.

        Pass recompute=False when fit() was just called and the Cholesky is
        already fresh (avoids a redundant O(n^3) factorisation).
        """
        self._sync()
        if recompute:
            self._backend.recompute()
        # cat produces a flat (dim+3,) vector that wires all parameters into
        # the graph; backward splits grad back to each parameter via torch.cat.
        params = torch.cat([self.raw_lengthscale,
                            self.raw_outputscale.unsqueeze(0),
                            self.raw_noise.unsqueeze(0),
                            self.raw_offset.unsqueeze(0)])
        return _MLLFunction.apply(params, self._backend)

    def predict(self, Xs):
        """
        Returns (means, vars_) as float64 tensors of shape (m,).

        Call model.eval() once after the training loop (does final sync +
        recompute) before predicting.
        """
        self._sync()
        means, vars_ = self._backend.predict(_np64(Xs))
        return torch.from_numpy(means), torch.from_numpy(vars_)

    def eval(self):
        """Switch to eval mode, syncing params and recomputing the Cholesky."""
        result = super().eval()
        if hasattr(self, '_backend'):
            self._sync()
            self._backend.recompute()
        return result

    def save(self, path):
        self._sync()
        self._backend.save(path)

    @classmethod
    def load(cls, path, extra_cap=0, use_cuda=True):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj._backend = _load_backend(path, extra_cap, use_cuda)
        obj.raw_lengthscale = nn.Parameter(
            torch.tensor(np.asarray(obj._backend.raw_lengthscale), dtype=torch.float64))
        obj.raw_outputscale = nn.Parameter(
            torch.tensor(obj._backend.raw_outputscale, dtype=torch.float64))
        obj.raw_noise = nn.Parameter(
            torch.tensor(obj._backend.raw_noise, dtype=torch.float64))
        obj.raw_offset = nn.Parameter(
            torch.tensor(obj._backend.raw_offset, dtype=torch.float64))
        return obj

    def __repr__(self):
        ells = self.lengthscale
        if len(ells) == 1:
            ell_s = f"{ells[0]:.3g}"
        else:
            ell_s = f"[{np.min(ells):.3g}..{np.max(ells):.3g}]"
        return (f"<GP dim={self.dim} n={self.n} cap={self.capacity} "
                f"ell={ell_s} sf={self.outputscale:.3g} "
                f"noise={self.noise:.3g}>")

    # -- internal ----------------------------------------------------------

    def _sync(self):
        self._backend.raw_lengthscale = self.raw_lengthscale.detach().cpu().numpy()
        self._backend.raw_outputscale = self.raw_outputscale.item()
        self._backend.raw_noise       = self.raw_noise.item()
        self._backend.raw_offset      = self.raw_offset.item()


# -- helpers ---------------------------------------------------------------

def _np64(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(x, dtype=np.float64)


def _make_backend(dim, capacity, lengthscale, outputscale, noise, offset, use_cuda):
    if use_cuda:
        try:
            from puffing_gpcu import GP as _GPCU
            return _GPCU(dim=dim, capacity=capacity, lengthscale=lengthscale,
                         outputscale=outputscale, noise=noise, offset=offset)
        except ImportError:
            pass
    from puffing_gp import GP as _GP
    return _GP(dim=dim, capacity=capacity, lengthscale=lengthscale,
               outputscale=outputscale, noise=noise, offset=offset)


def _load_backend(path, extra_cap, use_cuda):
    if use_cuda:
        try:
            from puffing_gpcu import GP as _GPCU
            return _GPCU.load(path, extra_cap)
        except ImportError:
            pass
    from puffing_gp import GP as _GP
    return _GP.load(path, extra_cap)
