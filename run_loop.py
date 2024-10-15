import sys
import time
import warnings
from typing import Dict, List, Optional, MutableMapping

import torch
from torch import Tensor
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.utils.warnings import NumericalWarning

from utils.model import get_model
from utils.util import select_method, select_problem, update_dgsms
from scipy.optimize import minimize as scipyminimize
import numpy as np
import subprocess


warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_loop(
    function_name: str,
    method_name: str,
    n_init: Optional[int] = 20,
    num_al_iter: Optional[int] = 280,
    seed: Optional[int] = 1000,
) -> Dict[str, List[List[float]]]:
    """
    run the active learning loop or the sobol grid sequentially

    Args:
        model: The model to evaluate. Must have RBFKernelPartialObsGrad kernel.
        Xstar: (b x 1 x d) tensor.
        Xq: (b x m x d) tensor.
        function_name: the test function name,
        method_name: the algorithm that will be run,
        n_init: number of initial points,
        num_al_iter: number of evaluations,
        seed: random seed
    """
    # Get problem
    torch.manual_seed(seed)
    dtype = torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem = select_problem(
        function_name,
        dtype=dtype,
        device=device,
    )

    # Make initial data
    X = draw_sobol_samples(bounds=problem.bounds, n=n_init, q=1).squeeze(-2)
    Y = problem(X).unsqueeze(-1)
    obs_var = 1e-9 * Y.max().item()
    print("Initial input", X)
    print("Initial output", Y)
    # Fit initial model
    gp = get_model(
        X=X,
        Y=Y,
        bounds=problem.bounds,
    )
    # Compute DGSMs
    dgsm_dict = {"raw": [], "abs": [], "square": [], "opt_time": [], "X": [], "Y": []}
    dgsm_dict = update_dgsms(
        dgsm_dict=dgsm_dict,
        model=gp,
        bounds=problem.bounds,
    )
    print("dgsm raw", dgsm_dict["raw"][-1])
    print("dgsm abs", dgsm_dict["abs"][-1])
    print("dgsm square", dgsm_dict["square"][-1])
    # Prepare for acquisition
    AcqClass = select_method(method_name)
    if method_name == "Sobol":
        SobolX = draw_sobol_samples(bounds=problem.bounds, n=num_al_iter, q=1).squeeze(
            -2
        )

    for i in range(num_al_iter):
        start_time = time.time()
        if AcqClass is not None:
            Acqf = AcqClass(
                model=gp,
                obs_var=obs_var,
                bounds=problem.bounds,
            )
            new_x, _ = optimize_acqf(
                acq_function=Acqf,
                bounds=problem.bounds,
                q=1,
                num_restarts=20,
                raw_samples=512,
            )
        elif method_name == "Sobol":
            # Do Sobol
            new_x = SobolX[i, :].unsqueeze(0)
        elif "activegp" in method_name:
            # Write out necessary info
            np.savetxt('X.txt', X.numpy())
            np.savetxt('Y.txt', Y.numpy())
            np.savetxt('lower.txt', problem.bounds[0, :].numpy())
            np.savetxt('upper.txt', problem.bounds[1, :].numpy())
            # Run activegp
            subprocess.run(["Rscript", "activegp_script.R", method_name.split(':')[1]])
            # Load in new point
            new_x = torch.tensor(np.loadtxt('x_opt.txt'), dtype=dtype, device=device).unsqueeze(0)
            # Overwrite the file, so we throw an error if it isn't written in the future
            with open('x_opt.txt', 'w') as f:
                f.write('')
        opt_time = time.time() - start_time
        print(f"Completed iteration {i} ({opt_time})")
        # evaluate the new point
        new_y = problem(new_x.detach()).unsqueeze(-1)
        print("new_x", new_x)
        print("new_y", new_y)
        # update data
        X = torch.cat((X, new_x), dim=0)
        Y = torch.cat((Y, new_y), dim=0)
        # reinitialize the model
        gp = get_model(
            X=X,
            Y=Y,
            bounds=problem.bounds,
            state_dict=gp.state_dict(),
        )
        dgsm_dict = update_dgsms(
            dgsm_dict=dgsm_dict,
            model=gp,
            bounds=problem.bounds,
        )
        print("dgsm raw", dgsm_dict["raw"][-1])
        print("dgsm abs", dgsm_dict["abs"][-1])
        print("dgsm square", dgsm_dict["square"][-1])
        dgsm_dict["opt_time"].append([opt_time])
    dgsm_dict["X"].append([X])
    dgsm_dict["Y"].append([Y])
    return dgsm_dict
