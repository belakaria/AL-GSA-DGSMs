from typing import Dict, List, Tuple

import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor
from utils.acquisition import (
    DerivAbsVarianceTrace,
    DerivVarianceTrace,
    GlobalDerivAbsVarianceTraceReduction,
    GlobalDerivSquareSumInformationGain,
    GlobalDerivSquareVarianceTraceReduction,
    GlobalDerivSumInformationGain,
    GlobalInformationGain,
    DerivAbsVarianceTraceReduction,
    DerivSquareSumInformationGain,
    DerivSquareVarianceTrace,
    DerivSquareVarianceTraceReduction,
    DerivSumInformationGain,
    InformationGain,
    MaxVariance,
    DerivVarianceTraceReduction,
    GlobalDerivVarianceTraceReduction,
    DerivAbsSumInformationGain,
    GlobalDerivAbsSumInformationGain,
    ncx2_entropy_approx1,
    ncx2_entropy_approx2,
    ncx2_entropy_approx3,
    foldnorm_entropy_approx1,
    foldnorm_entropy_approx2,
    foldnorm_entropy_approx3,
)
from functools import partial

from utils.model import untransform_derivs
from utils.test_functions import (
    A1,
    a_function,
    B1,
    b_function,
    C1,
    C2,
    Gsobol,
    Ishigami,
    Morris,
    MorrisReduced,
)
from botorch.test_functions.synthetic import Hartmann, Branin
from botorch.test_functions.multi_objective import CarSideImpact, VehicleSafety

def compute_dgsms(
    model: GPyTorchModel,
    bounds: Tensor,
    nsamp: int = 10000,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes DGSM measures given a model with RBFKernelPartialObsGrad

    Args:
        model: Model.
        bounds (2 x d): Bounds.
        nsamp: Number of QMC samples for averaging over derivative measures.

    Returns:
        dgsm_raw (d): Average of gradient across space.
        dgsm_abs (d): Average of absolute value of gradient.
        dgsm_sq (d): Average of square of gradient.
    """
    d = bounds.shape[1]
    Xtest = draw_sobol_samples(bounds=bounds, n=nsamp, q=1)
    deriv_indic = (
        torch.arange(0, d + 1).unsqueeze(0).unsqueeze(-1).expand(nsamp, d + 1, 1)
    )
    Xtest_aug = torch.cat((Xtest.expand(-1, d + 1, -1), deriv_indic), dim=2)
    with torch.no_grad():
        deriv_posterior = model.posterior(Xtest_aug)
        deriv_posterior_mean = deriv_posterior.mean.squeeze(-1)  # (nsamp x d)
        deriv_mean = deriv_posterior_mean[..., 1:]
        deriv_mean, _, _ = untransform_derivs(model=model, df=deriv_mean)
        dgsm_raw = deriv_mean.mean(dim=0)
        dgsm_abs = deriv_mean.abs().mean(dim=0)
        dgsm_sq = deriv_mean.pow(2).mean(dim=0)
        return dgsm_raw, dgsm_abs, dgsm_sq


def update_dgsms(
    dgsm_dict: Dict[str, List[List[float]]],
    model: GPyTorchModel,
    bounds: Tensor,
    nsamp: int = 10000,
) -> Dict[str, List[List[float]]]:
    """
    Update a dict of DGSM results with keys "raw", "abs", and "square" to
    append results from model.

    Args:
        dgsm_dict: DGSM result as described.
        model: Model.
        bounds (2 x d): Bounds.
        nsamp: Number of QMC samples for averaging over derivative measures.

    Returns: dgsm_dict updated with results from model.
    """
    dgsm_raw, dgsm_abs, dgsm_sq = compute_dgsms(
        model=model,
        bounds=bounds,
        nsamp=nsamp,
    )
    dgsm_dict["raw"].append(dgsm_raw.tolist())
    dgsm_dict["abs"].append(dgsm_abs.tolist())
    dgsm_dict["square"].append(dgsm_sq.tolist())
    return dgsm_dict

def create_evaluator_class(superclass, index):
    """
    Creates a dynamic class inheriting from the given superclass and adds
    evaluation functionality to selectively return the output based on an index.

    Parameters:
    - superclass: The class from which to inherit.
    - index: The index of the output to return from the evaluate method.

    Returns:
    - A new class with the specified superclass and a custom evaluate method.
    """

    class Evaluator(superclass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.index = index

        def evaluate_true(self, X):
            """
            Evaluate and return the value of a specific output based on the given index.

            Parameters:
            - X : Input to the evaluation function.

            Returns:
            - The value of the evaluated output at the specified index.
            """
            # Assuming the superclass has a method named `evaluate_true` that returns
            # a tensor or a list with all outputs.
            all_outputs = super().evaluate_true(X)
            # Select and return the specific output value based on self.index
            return all_outputs[:, self.index]

    return Evaluator()

def select_problem(function_name, **kwargs):
    if function_name == "Ishigami1":
        return Ishigami(b=0.1).to(**kwargs)
    elif function_name == "Ishigami2":
        return Ishigami(b=0.05).to(**kwargs)
    elif function_name == "Gsobol6":
        return Gsobol(dim=6).to(**kwargs)
    elif function_name == "Gsobol8":
        return Gsobol(dim=8).to(**kwargs)
    elif function_name == "Gsobol10":
        return Gsobol(dim=10).to(**kwargs)
    elif function_name == "Gsobol15":
        return Gsobol(dim=15).to(**kwargs)
    elif function_name == "Morris":
        return Morris().to(**kwargs)
    elif function_name == "MorrisReduced":
        return MorrisReduced().to(**kwargs)
    elif function_name == "a_function":
        return a_function().to(**kwargs)
    elif function_name == "b_function":
        return b_function().to(**kwargs)
    elif function_name == "A1":
        return A1().to(**kwargs)
    elif function_name == "B1":
        return B1().to(**kwargs)
    elif function_name == "C1":
        return C1().to(**kwargs)
    elif function_name == "C2":
        return C2().to(**kwargs)
    elif function_name == "Hartmann6":
        return Hartmann(dim=6).to(**kwargs)
    elif function_name == "Hartmann3":
        return Hartmann(dim=3).to(**kwargs)
    elif function_name == "Hartmann4":
        return Hartmann(dim=4).to(**kwargs)
    elif function_name == "Branin":
        return Branin().to(**kwargs)
    elif function_name == "CarSideImpactWeight":
        return create_evaluator_class(index=0,superclass=CarSideImpact)
    elif function_name == "VehicleSafetyWeight":
        return create_evaluator_class(index=0,superclass=VehicleSafety)
    elif function_name == "VehicleSafetyAcceleration":
        return create_evaluator_class(index=1,superclass=VehicleSafety)
    else:
        raise ValueError("Problem not defined")


def select_method(method_name):
    if method_name == "MaxVariance":
        return MaxVariance
    elif method_name == "InformationGain":
        return InformationGain
    elif method_name == "GlobalInformationGain":
        return GlobalInformationGain
    elif method_name == "DerivVarianceTrace":
        return DerivVarianceTrace
    elif method_name == "DerivVarianceTraceReduction":
        return DerivVarianceTraceReduction
    elif method_name == "DerivSumInformationGain":
        return DerivSumInformationGain
    elif method_name == "GlobalDerivSumInformationGain":
        return GlobalDerivSumInformationGain
    elif method_name == "DerivSquareVarianceTrace":
        return DerivSquareVarianceTrace
    elif method_name == "DerivSquareVarianceTraceReduction":
        return DerivSquareVarianceTraceReduction
    elif method_name == "GlobalDerivSquareVarianceTraceReduction":
        return GlobalDerivSquareVarianceTraceReduction
    elif method_name == "GlobalDerivVarianceTraceReduction":
        return GlobalDerivVarianceTraceReduction
    elif method_name == "DerivAbsVarianceTrace":
        return DerivAbsVarianceTrace
    elif method_name == "DerivAbsVarianceTraceReduction":
        return DerivAbsVarianceTraceReduction
    elif method_name == "GlobalDerivAbsVarianceTraceReduction":
        return GlobalDerivAbsVarianceTraceReduction
    elif method_name == "DerivSquareSumInformationGain1":# we called this New before
        return partial(DerivSquareSumInformationGain,ncx2_approx_entropy=ncx2_entropy_approx1)
    elif method_name == "DerivSquareSumInformationGain2":
        return partial(DerivSquareSumInformationGain,ncx2_approx_entropy=ncx2_entropy_approx2)
    elif method_name == "DerivSquareSumInformationGain3":
        return partial(DerivSquareSumInformationGain,ncx2_approx_entropy=ncx2_entropy_approx3)
    elif method_name == "GlobalDerivSquareSumInformationGain1":
        return partial(GlobalDerivSquareSumInformationGain,ncx2_approx_entropy=ncx2_entropy_approx1)
    elif method_name == "GlobalDerivSquareSumInformationGain2":
        return partial(GlobalDerivSquareSumInformationGain,ncx2_approx_entropy=ncx2_entropy_approx2)
    elif method_name == "GlobalDerivSquareSumInformationGain3":
        return partial(GlobalDerivSquareSumInformationGain,ncx2_approx_entropy=ncx2_entropy_approx3)
    elif method_name == "DerivAbsSumInformationGain1":
        return partial(DerivAbsSumInformationGain,foldnorm_approx_entropy=foldnorm_entropy_approx1)
    elif method_name == "DerivAbsSumInformationGain2":
        return partial(DerivAbsSumInformationGain,foldnorm_approx_entropy=foldnorm_entropy_approx2)
    elif method_name == "DerivAbsSumInformationGain3":
        return partial(DerivAbsSumInformationGain,foldnorm_approx_entropy=foldnorm_entropy_approx3)
    elif method_name == "GlobalDerivAbsSumInformationGain1":
        return partial(GlobalDerivAbsSumInformationGain,foldnorm_approx_entropy=foldnorm_entropy_approx1)
    elif method_name == "GlobalDerivAbsSumInformationGain2":
        return partial(GlobalDerivAbsSumInformationGain,foldnorm_approx_entropy=foldnorm_entropy_approx2)
    elif method_name == "GlobalDerivAbsSumInformationGain3":
        return partial(GlobalDerivAbsSumInformationGain,foldnorm_approx_entropy=foldnorm_entropy_approx3)
    elif method_name == "Sobol" or "activegp" in method_name:
        return None
    else:
        raise ValueError("Method not defined")
