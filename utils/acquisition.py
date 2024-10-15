import math
from typing import Any, Callable

import mpmath
import numpy as np
import scipy
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor, distributions

from utils.posterior import lookahead_deriv_posterior, PosteriorMoments


class LookaheadAcquisitionFunction(AcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        obs_var: float,
        **kwargs: Any,
    ) -> None:
        """
        A localized look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            obs_var: The variance of the look-ahead observation.
        """
        super().__init__(model=model)
        self.obs_var = obs_var

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        """Compute acquisition value given posterior moments.

        Args:
            post_n: PosteriorMoments under the current GP posterior.
            post_np1: PosteriorMoments under the look-ahead posterior, given X.

        Returns: (b) tensor of acquisition values.
        """
        raise NotImplementedError


class LocalLookaheadAcquisitionFunction(LookaheadAcquisitionFunction):
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate acquisition function at X.

        Args:
            X: (b x 1 x d) point at which to evalaute acquisition function.

        Returns: (b) tensor of acquisition values.
        """
        post_n, post_np1 = lookahead_deriv_posterior(
            model=self.model,
            Xstar=X,
            Xq=X,
            obs_var=self.obs_var,
        )  # Return shape here has m=1
        return self._compute_acqf(post_n, post_np1)


class GlobalLookaheadAcquisitionFunction(LookaheadAcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        obs_var: float,
        bounds: Tensor,
        query_set_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        A global look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            obs_var: The variance of the look-ahead observation.
            bounds (2 x d): Bounds for generating query set.
            query_set_size: Size of Xq to generate.
        """
        super().__init__(model=model, obs_var=obs_var)
        Xq = draw_sobol_samples(bounds=bounds, n=query_set_size, q=1).squeeze(-2)
        self.register_buffer("Xq", Xq)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate acquisition function at X.

        Args:
            X: (b x 1 x d) point at which to evalaute acquisition function.

        Returns: (b) tensor of acquisition values.
        """
        Xq_batch = self.Xq.expand(X.shape[0], *self.Xq.shape)
        post_n, post_np1 = lookahead_deriv_posterior(
            model=self.model,
            Xstar=X,
            Xq=Xq_batch,
            obs_var=self.obs_var,
        )
        return self._compute_acqf(post_n, post_np1)


class MaxVariance(LocalLookaheadAcquisitionFunction):
    """Variance of posterior of f"""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return post_n.var_f.mean(dim=-1)


def normal_entropy(variance: Tensor) -> Tensor:
    """Entropy of a normal RV."""
    return 0.5 * torch.log(2 * math.pi * variance) + 0.5


class InformationGain(LocalLookaheadAcquisitionFunction):
    """Information gain about f"""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (normal_entropy(post_n.var_f) - normal_entropy(post_np1.var_f)).mean(
            dim=-1
        )


class GlobalInformationGain(GlobalLookaheadAcquisitionFunction):
    """Information gain about f"""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (normal_entropy(post_n.var_f) - normal_entropy(post_np1.var_f)).mean(
            dim=-1
        )


class DerivVarianceTrace(LocalLookaheadAcquisitionFunction):
    """Trace of the derivative covariance matrix."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return post_n.var_df.sum(-1).mean(-1)

class DerivVarianceTraceReduction(LocalLookaheadAcquisitionFunction):
    """Trace of the derivative covariance matrix."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return ((post_n.var_df - post_np1.var_df).sum(dim=-1).mean(dim=-1))



class DerivSumInformationGain(LocalLookaheadAcquisitionFunction):
    """Sum of information gains for each derivative."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            (normal_entropy(post_n.var_df) - normal_entropy(post_np1.var_df))
            .sum(dim=-1)
            .mean(dim=-1)
        )


class GlobalDerivSumInformationGain(GlobalLookaheadAcquisitionFunction):
    """Sum of information gains for each derivative."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            (normal_entropy(post_n.var_df) - normal_entropy(post_np1.var_df))
            .sum(dim=-1)
            .mean(dim=-1)
        )


def variance_normal_sq(mean: Tensor, variance: Tensor) -> Tensor:
    """Variance of the square of a normal RV.

    Take X ~ N(mu, sigma^2). Then,
    Var(X^2) = 2 sigma^4 + 4 sigma^2 mu^2.
    """
    return 2 * variance**2 + 4 * variance * mean**2


class DerivSquareVarianceTrace(LocalLookaheadAcquisitionFunction):
    """Sum of variances of square of each derivative."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            variance_normal_sq(mean=post_n.df, variance=post_n.var_df)
            .sum(dim=-1)
            .mean(dim=-1)
        )


class DerivSquareVarianceTraceReduction(LocalLookaheadAcquisitionFunction):
    """Reduction in the sum of variances of square of each derivative."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            (
                variance_normal_sq(mean=post_n.df, variance=post_n.var_df)
                - variance_normal_sq(mean=post_np1.df, variance=post_np1.var_df)
            )
            .sum(dim=-1)
            .mean(dim=-1)
        )


class GlobalDerivSquareVarianceTraceReduction(GlobalLookaheadAcquisitionFunction):
    """Reduction in the sum of variances of square of each derivative."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            (
                variance_normal_sq(mean=post_n.df, variance=post_n.var_df)
                - variance_normal_sq(mean=post_np1.df, variance=post_np1.var_df)
            )
            .sum(dim=-1)
            .mean(dim=-1)
        )


def h_2F2_term(x: np.ndarray) -> np.ndarray:
    x_reshaped = x.reshape(
        -1,
    )
    y = np.empty(len(x_reshaped))
    for i, x_i in enumerate(x_reshaped):
        y[i] = float(mpmath.hyp2f2(1, 1, 1.5, 2, -x_i)) 
    y_old = y.reshape(-1, 1).reshape(x.shape)
    return y_old


class h_chisq_fn(torch.autograd.Function):
    """
    For computing entropy of a noncentral ChiSq RV.

    From:
    Moser (2020) Expected Logarithm and Negative Integer Moments of a
    Noncentral Chi-Square-Distributed Random Variable. Entropy 22:1048

    This implements Eq. 45, the h_n function for n=1.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        x = input.numpy()
        if len(input.shape) == 0:
            # Support scalar evaluation
            x = np.array([x])
        euler_gamma = 0.577215664901532
        res = -euler_gamma - 2 * np.log(2) + 2 * x * h_2F2_term(x)
        if len(input.shape) == 0:
            res = res[0]
        return torch.tensor(res, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        x = input.numpy()
        if len(input.shape) == 0:
            # Support scalar evaluation
            x = np.array([x])
        new_x = x.reshape(
            -1,
        )
        res = np.empty(len(new_x))

        indx_pos = new_x > 0
        if sum(indx_pos) > 0:
            sqrt_x = np.sqrt(new_x[indx_pos])
            erfi_x = scipy.special.erfi(sqrt_x)
            erfi_x[erfi_x == np.inf] = 10**5
            res[indx_pos] = (
                scipy.special.gamma(0.5) / sqrt_x * np.exp(-new_x[indx_pos]) * erfi_x
            )
        res[~indx_pos] = 2.0
        if len(input.shape) == 0:
            res = res[0]
        grad_output_new = grad_output.reshape(
            -1,
        )
        grad_input = grad_output_new * torch.tensor(res, dtype=input.dtype)
        grad_input = grad_input.reshape(-1, 1).reshape(x.shape)
        return grad_input


h_chisq = h_chisq_fn.apply

def ncx2_exp_log(lbda):
    """
    E[log(Z)] for Z noncentral chi-squared with param lbda
    Eq. 54 of Moser (2020) above.
    """
    return torch.log(torch.tensor(2.)) + h_chisq(lbda / 2)

def ncx2_entropy_approx1(lbda):
    """
    An approximation to the entropy of a noncentral chi-squared RV, using the Normal
    approximation to a noncentral chisquare of:
    Abdel-Aty (1954) Approximate formulae for the percentage points and the probability
    integral of the non-central chi-square distribution. Biometrika 41(3/4):538-540.

    Specifically uses the "first approximation" provided in the paper.
    """
    f = 1 + lbda ** 2 / (1 + 2 * lbda)
    h_Y = normal_entropy(variance=(2 / (9 * f)))
    C = torch.log(torch.tensor(3)) + torch.log(1 + lbda) / 3.
    ElogZ = ncx2_exp_log(lbda)
    return h_Y + C + (2 / 3.) * ElogZ

def ncx2_entropy_approx2(lbda):
    """
    An approximation to the entropy of a noncentral chi-squared RV with 1 dof,
    using the asymptotic normal distribution:
    (Z - (1 + lbda)) / sqrt(2 * (1 + 2*lbda)) ~ N(0, 1) as lbda -> inf.
    """
    h_Y = normal_entropy(variance=torch.tensor(1.0))
    C = 0.5 * torch.log(torch.tensor(2)) + 0.5 * torch.log(1 + 2 * lbda)
    return h_Y + C

def ncx2_entropy_approx3(lbda):
    """
    An approximation to the entropy of a noncentral chi-squared RV using
    the normal entropy upperbound.
    """
    variance = 2 * (1 + 2 * lbda)
    return normal_entropy(variance=variance)

def normal_squared_entropy(
    mean: Tensor,
    variance: Tensor,
    ncx2_approx_entropy: Callable[[Tensor], Tensor],
) -> Tensor:
    """Entropy of the square of a normal R.V."""
    lbda = (mean ** 2) / variance
    h1 = ncx2_approx_entropy(lbda)
    return h1 + torch.log(variance)

class DerivSquareSumInformationGain(LocalLookaheadAcquisitionFunction):
    """Sum of information gains for square of each derivative."""
    def __init__(
        self,
        model: GPyTorchModel,
        obs_var: float,
        ncx2_approx_entropy: Callable[[Tensor], Tensor] = ncx2_entropy_approx1,
        **kwargs: Any,
    ) -> None:
        """
        A localized look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            obs_var: The variance of the look-ahead observation.
            ncx2_approx_entropy: A function that approximates the entropy of a noncentral
                chi-squared RV.
        """
        super().__init__(model=model,obs_var=obs_var)
        self.ncx2_entropy = ncx2_approx_entropy

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            (
                normal_squared_entropy(
                    mean=post_n.df,
                    variance=post_n.var_df,
                    ncx2_approx_entropy=self.ncx2_entropy,
                )
                - normal_squared_entropy(
                    mean=post_np1.df,
                    variance=post_np1.var_df,
                    ncx2_approx_entropy=self.ncx2_entropy,
                )
            )
            .sum(dim=-1)
            .mean(dim=-1)
        )


class GlobalDerivSquareSumInformationGain(GlobalLookaheadAcquisitionFunction):
    """Sum of information gains for square of each derivative."""
    def __init__(
        self,
        model: GPyTorchModel,
        obs_var: float,
        bounds: Tensor,
        query_set_size: int = 100,
        ncx2_approx_entropy: Callable[[Tensor], Tensor] = ncx2_entropy_approx1,
    ) -> None:
        super().__init__(model=model,bounds=bounds,obs_var=obs_var,query_set_size=query_set_size)
        self.ncx2_entropy = ncx2_approx_entropy

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            (
                normal_squared_entropy(
                    mean=post_n.df,
                    variance=post_n.var_df,
                    ncx2_approx_entropy=self.ncx2_entropy,
                )
                - normal_squared_entropy(
                    mean=post_np1.df,
                    variance=post_np1.var_df,
                    ncx2_approx_entropy=self.ncx2_entropy,
                )
            )
            .sum(dim=-1)
            .mean(dim=-1)
        )

def mean_normal_abs(mean: Tensor, variance: Tensor) -> Tensor:
    """Mean of the absolute of a normal RV.

    Take X ~ N(mu, sigma^2). Then,
    mu(|X|) = sigma*sqrt(2/pi)*exp(-mu^2/2sigma^2)+mu*(1-2*cdf(-mu/sigma)).
    """
    dist = distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    std = torch.sqrt(variance)
    term1 = torch.sqrt(torch.tensor(2 / torch.pi)) * std * torch.exp(-0.5 * (mean ** 2) / variance)
    term2 = mean * (1 - 2 * dist.cdf(-mean / std))
    return term1 + term2

def variance_normal_abs(mean: Tensor, variance: Tensor) -> Tensor:
    """Variance of the absolute of a normal RV.

    Take X ~ N(mu, sigma^2). Then,
    Var(|X|) = mu^2 + sigma^2 - mu_abs^2.
    """
    abs_mean = mean_normal_abs(mean, variance)
    return mean ** 2 + variance - abs_mean ** 2


class DerivAbsVarianceTrace(LocalLookaheadAcquisitionFunction):
    """Sum of variances of absolute of each derivative."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            variance_normal_abs(mean=post_n.df, variance=post_n.var_df)
            .sum(dim=-1)
            .mean(dim=-1)
        )


class DerivAbsVarianceTraceReduction(LocalLookaheadAcquisitionFunction):
    """Reduction in the sum of variances of absolute of each derivative."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            (
                variance_normal_abs(mean=post_n.df, variance=post_n.var_df)
                - variance_normal_abs(mean=post_np1.df, variance=post_np1.var_df)
            )
            .sum(dim=-1)
            .mean(dim=-1)
        )


class GlobalDerivAbsVarianceTraceReduction(GlobalLookaheadAcquisitionFunction):
    """Reduction in the sum of variances of absolute of each derivative."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            (
                variance_normal_abs(mean=post_n.df, variance=post_n.var_df)
                - variance_normal_abs(mean=post_np1.df, variance=post_np1.var_df)
            )
            .sum(dim=-1)
            .mean(dim=-1)
        )
class GlobalDerivVarianceTraceReduction(GlobalLookaheadAcquisitionFunction):
    """Trace of the derivative covariance matrix."""

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return ((post_n.var_df - post_np1.var_df).sum(dim=-1).mean(dim=-1))

def foldnorm_entropy_approx1(mean, variance):
    w = math.pi * (mean_normal_abs(mean, variance) ** 2 - mean ** 2) / (2 * variance)
    return normal_entropy(variance) - w * torch.log(torch.tensor(2))

def foldnorm_entropy_approx2(mean: Tensor, variance: Tensor, terms: int = 3) -> Tensor:
    """Entropy of the absolute of a normal R.V.
    Using the Taylor approximation of
    Tsagris et al. (2014) "On the Folded Normal Distribution", Mathematics 2:12-28
    Eq. 40
    This is numerically unstable for sigma small relative to mu and should not be used.
    """
    dist = distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    std = torch.sqrt(variance)
    def func(n):
        n_t = torch.tensor(n)
        factor = ((-1) ** (n_t+1))/n_t
        an = (-2 * n_t * mean)/variance
        exp_part = torch.exp(((mean -2 * n_t *mean) ** 2 - mean ** 2)/(2 * variance))
        cdf1 = (1 - dist.cdf((-mean + an)/std))
        cdf2 = (1 - dist.cdf((mean + an) / std))
        res = factor * exp_part * (cdf1 + cdf2)
        return res
    abs_mean = mean_normal_abs(mean, variance)
    entropy_t = (
        torch.log(torch.sqrt(2 * torch.pi * variance)) + 0.5 + (mean ** 2 - mean * abs_mean) / variance
    )
    for i in range(1, terms + 1):
        entropy_t -= func(i)
    return entropy_t

def foldnorm_entropy_approx3(mean: Tensor, variance: Tensor) -> Tensor:
    """Entropy of folded normal, approximated via the normal upper bound.
    """
    fn_variance = variance_normal_abs(mean=mean, variance=variance)
    return normal_entropy(variance=fn_variance)

class DerivAbsSumInformationGain(LocalLookaheadAcquisitionFunction):
    """Sum of information gains for absolute value of each derivative."""
    def __init__(
        self,
        model: GPyTorchModel,
        obs_var: float,
        foldnorm_approx_entropy: Callable[[Tensor], Tensor] = foldnorm_entropy_approx1,
        **kwargs: Any,
    ) -> None:
        """
        A localized look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            obs_var: The variance of the look-ahead observation.
            foldnorm_approx_entropy: A function that approximates the entropy of a folded normal RV.
        """
        super().__init__(model=model,obs_var=obs_var)
        self.foldnorm_entropy = foldnorm_approx_entropy

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            (
                self.foldnorm_entropy(
                    mean=post_n.df,
                    variance=post_n.var_df,
                )
                - self.foldnorm_entropy(
                    mean=post_np1.df,
                    variance=post_np1.var_df,
                )
            )
            .sum(dim=-1)
            .mean(dim=-1)
        )


class GlobalDerivAbsSumInformationGain(GlobalLookaheadAcquisitionFunction):
    """Sum of information gains for absolute of each derivative."""
    def __init__(
        self,
        model: GPyTorchModel,
        obs_var: float,
        bounds: Tensor,
        query_set_size: int = 100,
        foldnorm_approx_entropy: Callable[[Tensor], Tensor] = foldnorm_entropy_approx1,
    ) -> None:
        super().__init__(model=model,bounds=bounds,obs_var=obs_var,query_set_size=query_set_size)
        self.foldnorm_entropy = foldnorm_approx_entropy

    def _compute_acqf(
        self,
        post_n: PosteriorMoments,
        post_np1: PosteriorMoments,
    ) -> Tensor:
        return (
            (
                self.foldnorm_entropy(
                    mean=post_n.df,
                    variance=post_n.var_df,
                )
                - self.foldnorm_entropy(
                    mean=post_np1.df,
                    variance=post_np1.var_df,
                )
            )
            .sum(dim=-1)
            .mean(dim=-1)
        )
