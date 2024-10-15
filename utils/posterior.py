from dataclasses import dataclass
from typing import Tuple

import torch
from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor

from utils.model import untransform_derivs

@dataclass
class PosteriorMoments:
    """Container for moments of the GP posterior.

    Attributes:
        f (b x m): Mean for f(x)
        var_f (b x m): Variance for f(x)
        df (b x m x d): Mean for each df/dx_i
        var_df (b x m x d): Variance for each df/dx_i
    """

    f: Tensor
    var_f: Tensor
    df: Tensor
    var_df: Tensor


def posterior_at_xstar_xq(
    model: GPyTorchModel,
    Xstar: Tensor,
    Xq: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Evaluate the joint posterior of the d derivatives at the points in Xq, with the f at
    the single point Xstar.

    Assumes that model will have Standardize and Normalize botorch transforms, and
    applies the correct untransform.

    Args:
        model: The model to evaluate. Must have RBFKernelPartialObsGrad kernel.
        Xstar: (b x 1 x d) tensor.
        Xq: (b x m x d) tensor.

    Returns:
        Mu_s: (b x 1) mean at Xstar.
        Sigma2_s: (b x 1) variance at Xstar.
        Mu_q: (b x m x (d+1)) mean at Xq for f and all derivatives of f.
        Sigma2_q: (b x m x (d+1)) variance at Xq for f and all derivatives of f.
        Sigma_sq: (b x m x (d+1)) covariance between Xstar and each point in Xq, for f and
            each derivative.
    """
    # Evaluate posterior and extract needed components
    b, m, d = Xq.shape
    # Add in derivative indicators
    Xstar_aug = (
        torch.cat(
            (Xstar, torch.zeros(torch.Size([b, 1, 1]), dtype=Xstar.dtype)), dim=-1
        )
        .unsqueeze(1)
        .expand(-1, m, -1, -1)
    )  # (b x m x 1 x d+1)
    deriv_indic = (
        torch.arange(0, d + 1, dtype=Xq.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand([b, m, -1, -1])
    )  # (b x m x d+1 x 1)
    Xq_aug = torch.cat(
        (Xq.unsqueeze(2).expand(-1, -1, d + 1, -1), deriv_indic), dim=-1
    )  # (b x m x d+1 x d+1)

    # Put them together
    Xext = torch.cat((Xstar_aug, Xq_aug), dim=-2)

    posterior = model.posterior(Xext)

    Mu_s = posterior.mean[:, 0, 0, :]
    Mu_q = posterior.mean[:, :, 1:, 0]
    Cov = posterior.distribution.covariance_matrix  # (b x m x d+2 x d+2)
    Sigma2_s = Cov[:, 0, 0, 0].unsqueeze(-1)
    Sigma2_q = torch.diagonal(Cov[..., 1:, 1:], dim1=-1, dim2=-2)

    Sigma_sq = Cov[..., 1:, 0]
    Mu_q[..., 1:], Sigma2_q[..., 1:], Sigma_sq[..., 1:] = untransform_derivs(
        model=model,
        df=Mu_q[..., 1:],
        var_df=Sigma2_q[..., 1:],
        cov_f_df=Sigma_sq[..., 1:],
    )
    return Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq


def lookahead_deriv_posterior(
    model: GPyTorchModel, Xstar: Tensor, Xq: Tensor, obs_var: float
) -> Tuple[PosteriorMoments, PosteriorMoments]:
    """
    Evaluate the look-ahead posterior of f and all its derivatives at Xq, given an
    observation at Xstar.

    Args:
        model: The model to evaluate. Must have RBFKernelPartialObsGrad kernel.
        Xstar: (b x 1 x d) tensor.
        Xq: (b x m x d) tensor.
        obs_var: The variance of the observation.

    Returns:
        post_n: Means and variances at Xq for f and each df/dx_i, under current posterior
        post_np1: Means and variances at Xq conditioned on observation at Xstar
    """
    Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq = posterior_at_xstar_xq(
        model=model, Xstar=Xstar, Xq=Xq
    )
    # Use Gaussian conditioning formula for look-ahead variance
    b, m, d = Xq.shape
    var_q = Sigma2_q - Sigma_sq**2 / (  # pyre-ignore
        obs_var + Sigma2_s.unsqueeze(-1).expand(-1, m, d + 1)
    )
    # First prepare the distributions without look-ahead
    post_n = PosteriorMoments(
        f=Mu_q[..., 0],
        var_f=Sigma2_q[..., 0],
        df=Mu_q[..., 1:],
        var_df=Sigma2_q[..., 1:],
    )
    # Use plug-in estimate for the look-ahead mean
    post_np1 = PosteriorMoments(
        f=Mu_q[..., 0],
        var_f=var_q[..., 0],
        df=Mu_q[..., 1:],
        var_df=var_q[..., 1:],
    )
    return post_n, post_np1

