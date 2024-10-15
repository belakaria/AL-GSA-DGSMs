from typing import Any, MutableMapping, Optional, Tuple

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels.rbf_kernel_grad import RBFKernelGrad
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor

class ConstantMeanPartialObsGrad(ConstantMean):
    """A mean function for use with partial gradient observations.

    This follows gpytorch.means.constant_mean_grad and sets the prior mean for
    derivative observations to 0, though unlike that function it allows for
    partial observation of derivatives.

    The final column of input should be an index that is 0 if the observation
    is of f, or i if it is of df/dxi.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = torch.zeros(input.shape[:-1], dtype=input.dtype)
        idx = input[..., -1].to(dtype=torch.long) > 0
        mean[~idx] = self.constant
        return mean


def _get_deriv_idx(x):
    assert len(x.shape) == 2
    deriv_idx = {0: [x[0, -1].item()]}
    last_uniq = 0
    for i in range(1, x.shape[0]):
        if torch.equal(x[i, :-1], x[last_uniq, :-1]):
            deriv_idx[last_uniq].append(x[i, -1].item())
        else:
            last_uniq = int(i)
            deriv_idx[last_uniq] = [x[i, -1].item()]
    return deriv_idx


def _get_deriv_idx_batch(x):
    xtil = x
    for _ in range(len(x.shape) - 2):
        xtil = xtil[0]
    return _get_deriv_idx(xtil)


def _deriv_idx_to_slice(deriv_idx, d):
    p = []
    for i, j in enumerate(deriv_idx.keys()):
        for k in deriv_idx[j]:
            p.append(i * (d + 1) + k)
    return p


class RBFKernelPartialObsGrad(RBFKernelGrad):
    """An RBF kernel over observations of f, and partial/non-overlapping
    observations of the gradient of f.

    gpytorch.kernels.rbf_kernel_grad assumes a block structure where every
    partial derivative is observed at the same set of points at which x is
    observed. This generalizes that by allowing f and any subset of the
    derivatives of f to be observed at different sets of points.

    The final column of x1 and x2 needs to be an index that identifies what is
    observed at that point. It should be 0 if this observation is of f, and i
    if it is of df/dxi.

    IMPORTANT: If x1 and x2 have a batch dimension, this assumes that the deriv
    layout in each batch is the same.
    """

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params: Any
    ) -> torch.Tensor:
        # Compute which elements to send for grad computation
        deriv_idx1 = _get_deriv_idx_batch(x1)
        deriv_idx2 = _get_deriv_idx_batch(x2)
        xtil1 = x1[..., list(deriv_idx1.keys()), :-1]
        xtil2 = x2[..., list(deriv_idx2.keys()), :-1]
        K = super().forward(xtil1, xtil2, diag=diag, **params)
        # Extract the correct set of points
        d = x1.shape[-1] - 1
        p1 = _deriv_idx_to_slice(deriv_idx1, d)
        p2 = _deriv_idx_to_slice(deriv_idx2, d)

        if not diag:
            return K[..., p1, :][..., p2]
        else:
            return K[..., p1]

    def num_outputs_per_input(self, x1: torch.Tensor, x2: torch.Tensor) -> int:
        return 1


def get_model(
    X: Tensor,
    Y: Tensor,
    bounds: Tensor,
    state_dict: Optional[MutableMapping[str, Tensor]] = None,
) -> GPyTorchModel:
    """
    Get a fitted model with derivative kernel.

    Args:
        X (m x d): Training data X.
        Y (m x 1): Training data Y.
        bounds (2 x d): Problem bounds.
        statedict: State dict to use (otherwise model is fit).

    Returns: fitted model.
    """
    m, d = X.shape
    X_aug = torch.cat((X, torch.zeros(m, dtype=X.dtype, device=X.device).unsqueeze(1)), dim=1)
    bounds_aug = torch.cat((bounds, torch.tensor([[0], [1]], dtype=X.dtype, device=X.device)), dim=1)
    Y_var = 1e-10 * torch.ones_like(Y)

    base_kernel = RBFKernelPartialObsGrad(
        ard_num_dims=d,
        lengthscale_prior=GammaPrior(3.0, 6.0),
    )
    covar_module = ScaleKernel(base_kernel, outputscale_prior=GammaPrior(2.0, 0.15))
    input_transform=Normalize(d=(d + 1), bounds=bounds_aug)
    model = FixedNoiseGP(
        train_X=X_aug,
        train_Y=Y,
        train_Yvar=Y_var,
        covar_module=covar_module,
        mean_module=ConstantMeanPartialObsGrad(),
        input_transform=input_transform,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = fit_gpytorch_model(mll)
    return model


def untransform_derivs(
    model: GPyTorchModel,
    df: Optional[Tensor] = None,
    var_df: Optional[Tensor] = None,
    cov_f_df: Optional[Tensor] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Untransform derivative predictions.

    Correctly adjusts for input and output transforms on the model in the mean and/or
    variance and/or covariance of the derivative predictions. Covariance is betwen
    derivative and f.

    We have:
    x_til = (x - l) / (u - l)
    f_til = (f - mu) / sigma

    The model computed df_til/dx_til, and then did the transform (via Standardize)
    Z = df_til/dx_til * sigma + mu

    The correct transform is
    df/dx = df_til/dx_til * (sigma / (u - l))

    And so to get to the correct transform, we do
    df/dx = (Z - mu) / (u - l)

    Args:
        model: Model that produced the derivative predictions.
        mean ((b) x m x d): Derivative predictive mean.
        var ((b) x m x d): Derivative predictive variance.

    Returns: In-place transformed mean and var.
    """
    x_scale = model.input_transform.coefficient[0, :-1]
    mu = model.outcome_transform.means[0, 0]
    if df is not None:
        df = (df - mu) / x_scale
    if var_df is not None:
        var_df = var_df / (x_scale**2)
    if cov_f_df is not None:
        cov_f_df = cov_f_df / x_scale
    return df, var_df, cov_f_df
