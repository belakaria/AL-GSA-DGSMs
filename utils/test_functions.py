import math
from typing import List, Optional, Tuple

import torch

from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor
import numpy as np

class Ishigami(SyntheticTestFunction):
    r"""Ishigami test function.

    three-dimensional function (usually evaluated on `[-pi, pi]^3`):

        f(x) = sin(x_1) + a sin(x_2)^2 + b x_3^4 sin(x_1)

    Here `a` and `b` are constants where a=7 and b=0.1 or b=0.05
    Proposed to test sensitivity analysis methods because it exhibits strong
    nonlinearity and nonmonotonicity and a peculiar dependence on x_3.
    """

    def __init__(
        self,
        b: float = 0.1,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            b: the b constant, should be 0.1 or 0.05.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negative the objective.
        """
        self._optimizers = None
        if b not in (0.1, 0.05):
            raise ValueError("b parameter should be 0.1 or 0.05")
        self.dim = 3
        if b == 0.1:
            self.si = [0.3138, 0.4424, 0]
            self.si_t = [0.558, 0.442, 0.244]
            self.s_ij = [0, 0.244, 0]
            self.dgsm_gradient = [0.0005, 0.0005, 0.0005]
            self.dgsm_gradient_abs = [1.88, 4.45, 1.98]
            self.dgsm_gradient_square = [7.7, 24.5, 11]
        elif b == 0.05:
            self.si = [0.218, 0.687, 0]
            self.si_t = [0.3131, 0.6868, 0.095]
            self.s_ij = [0, 0.094, 0]
            self.dgsm_gradient = [0.0004, 0.0004, 0.0004]
            self.dgsm_gradient_abs = [1.26, 4.45, 0.98]
            self.dgsm_gradient_square = [2.8, 24.5, 2.75]
        self._bounds = [(-math.pi, math.pi) for _ in range(self.dim)]
        self.b = b
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def compute_dgsm(self, X: Tensor) -> Tuple[List[float], List[float], List[float]]:
        r"""Compute derivative global sensitivity measures.

        This function can be called separately to estimate the dgsm measure
        The exact global integrals of these values are already added under
        as attributes dgsm_gradient, dgsm_gradient_abs, and dgsm_gradient_square.

        Args:
            X: Set of points at which to compute derivative measures.

        Returns: The average gradient, absolute gradient, and square gradients.
        """
        dx_1 = torch.cos(X[..., 0]) * (1 + self.b * (X[..., 2] ** 4))
        dx_2 = 14 * torch.cos(X[..., 1]) * torch.sin(X[..., 1])
        dx_3 = self.b * 4 * (X[..., 2] ** 3) * torch.sin(X[..., 0])
        gradient_measure = [
            torch.mean(dx_1).item(),
            torch.mean(dx_1).item(),
            torch.mean(dx_1).item(),
        ]
        gradient_absolute_measure = [
            torch.mean(torch.abs(dx_1)).item(),
            torch.mean(torch.abs(dx_2)).item(),
            torch.mean(torch.abs(dx_3)).item(),
        ]
        gradient_square_measure = [
            torch.mean(torch.pow(dx_1, 2)).item(),
            torch.mean(torch.pow(dx_2, 2)).item(),
            torch.mean(torch.pow(dx_3, 2)).item(),
        ]
        return gradient_measure, gradient_absolute_measure, gradient_square_measure

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        t = (
            torch.sin(X[..., 0])
            + 7 * (torch.sin(X[..., 1]) ** 2)
            + self.b * (X[..., 2] ** 4) * torch.sin(X[..., 0])
        )
        return t


class Morris(SyntheticTestFunction):
    r"""Morris test function.

    20-dimensional function (usually evaluated on `[0, 1]^20`):

        f(x) = sum_{i=1}\^20 beta_i w_i + sum_{i<j}\^20 beta_ij w_i w_j
        + sum_{i<j<l}\^20 beta_ijl w_i w_j w_l + 5w_1 w_2 w_3 w_4

    Proposed to test sensitivity analysis methods
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
        """
        self._optimizers = None
        self.dim = 20
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self.si = [
            0.005,
            0.008,
            0.017,
            0.009,
            0.016,
            0,
            0.069,
            0.1,
            0.15,
            0.1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        self.dgsm_gradient = [
            -47.6840,
            -47.6723,
            -9.2987,
            -47.6605,
            -9.4831,
            -47.6790,
            -9.4401,
            -28.8511,
            -26.1642,
            -28.8462,
            -68.1501,
            -66.8496,
            -68.1661,
            -66.8513,
            -68.1640,
            -66.8495,
            -68.1679,
            -66.8555,
            -68.1426,
            -66.8441,
        ]
        self.dgsm_gradient_abs = [
            135.1225,
            135.1225,
            117.4992,
            135.0901,
            117.5788,
            118.0753,
            77.4511,
            84.4297,
            84.5565,
            84.4296,
            91.7510,
            91.2040,
            91.7662,
            91.2047,
            91.7688,
            91.2029,
            91.7723,
            91.1978,
            91.7529,
            91.1923,
        ]
        self.dgsm_gradient_square = [
            36629.7414,
            36627.3141,
            103987.9600,
            36615.9760,
            103889.2989,
            25468.0737,
            44102.8125,
            13611.9649,
            13501.9957,
            13614.3125,
            17458.1122,
            17251.0162,
            17464.5750,
            17253.2673,
            17464.2558,
            17250.9887,
            17464.3090,
            17252.5585,
            17457.3987,
            17248.9220,
        ]
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        W = []
        t1 = 0
        t2 = 0
        t3 = 0
        for i in range(self.dim):
            if i in [2, 4, 6]:
                wi = 2 * (1.1 * X[..., i] / (X[..., i] + 0.1) - 0.5)
            else:
                wi = 2 * (X[..., i] - 0.5)
            W.append(wi)
            if i < 10:
                betai = 20
            else:
                betai = (-1) ** (i + 1)
            t1 = t1 + betai * wi
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                if i < 6 or j < 6:
                    beta_ij = -15
                else:
                    beta_ij = (-1) ** (i + j + 2)
                t2 = t2 + beta_ij * W[i] * W[j]
                for k in range(j + 1, self.dim):
                    if i < 5 or j < 5 or k < 5:
                        beta_ijk = -10
                    else:
                        beta_ijk = 0
                    t3 = t3 + beta_ijk * W[i] * W[j] * W[k]
        t4 = 5 * W[0] * W[1] * W[2] * W[3]
        return t1 + t2 + t3 + t4


class Gsobol(SyntheticTestFunction):
    r"""Gsobol test function.

    d-dimensional function (usually evaluated on `[0, 1]^d`):

        f(x) = Prod_{i=1}\^{d} ((\|4x_i-2\|+a_i)/(1+a_i)), a_i >=0

    common combinations of dimension and a vector:

        dim=8, a= [0, 1, 4.5, 9, 99, 99, 99, 99]
        dim=6, a=[0, 0.5, 3, 9, 99, 99]
        dim = 15, a= [1, 2, 5, 10, 20, 50, 100, 500, 1000, ..., 1000]

    Proposed to test sensitivity analysis methods
    First order Sobol indices have closed form expression S_i=V_i/V with :

        V_i= 1/(3(1+a_i)\^2)
        V= Prod_{i=1}\^{d} (1+V_i) - 1

    """

    def __init__(
        self,
        dim: int,
        a: List = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            dim: Dimensionality of the problem. If 6, 8, or 15, will use standard a.
            a: a parameter, unless dim is 6, 8, or 15.
            noise_std: Standard deviation of observation noise.
            negate: Return negatie of function.
        """
        self._optimizers = None
        self.dim = dim
        self._bounds = [(0, 1) for _ in range(self.dim)]
        if dim == 6:
            self.a = [0, 0.5, 3, 9, 99, 99]
            self.dgsm_gradient = [
                1.6639e-03,
                -1.6564e-03,
                -5.6448e-05,
                -1.8646e-04,
                1.0199e-06,
                1.5404e-05,
            ]
            self.dgsm_gradient_abs = [3.9996, 2.6663, 0.9998, 0.3999, 0.0400, 0.0400]
            self.dgsm_gradient_square = [
                1.8814e01,
                9.7107e00,
                1.5355e00,
                2.4996e-01,
                2.5077e-03,
                2.5077e-03,
            ]
        elif dim == 8:
            self.a = [0, 1, 4.5, 9, 99, 99, 99, 99]
            self.dgsm_gradient = [
                -2.0735e-03,
                8.7157e-04,
                3.2215e-04,
                -1.7128e-04,
                2.9619e-06,
                -8.7956e-06,
                1.2616e-05,
                5.6680e-06,
            ]
            self.dgsm_gradient_abs = [
                3.9992,
                1.9999,
                0.7272,
                0.3999,
                0.0400,
                0.0400,
                0.0400,
                0.0400,
            ]
            self.dgsm_gradient_square = [
                1.7578e01,
                5.4097e00,
                7.6645e-01,
                2.3358e-01,
                2.3436e-03,
                2.3436e-03,
                2.3436e-03,
                2.3436e-03,
            ]
        elif dim == 10:
            self.a = [0, 0, 6.52, 6.52, 6.52, 6.52, 6.52, 6.52, 6.52, 6.52]
            self.dgsm_gradient = [
                -1.6346e-05,
                9.3331e-04,
                2.2981e-04,
                -1.2880e-04,
                -5.1617e-05,
                -1.8003e-04,
                1.0380e-05,
                1.3431e-05,
                2.5867e-04,
                5.7872e-04,
            ]
            self.dgsm_gradient_abs = [
                3.9987,
                3.9993,
                0.5318,
                0.5317,
                0.5318,
                0.5318,
                0.5318,
                0.5318,
                0.5318,
                0.5317,
            ]
            self.dgsm_gradient_square = [
                22.3485,
                22.3561,
                0.5241,
                0.5240,
                0.5241,
                0.5240,
                0.5241,
                0.5241,
                0.5241,
                0.5240,
            ]
        elif dim == 15:
            self.a = [
                1,
                2,
                5,
                10,
                20,
                50,
                100,
                500,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
            ]
            self.dgsm_gradient = [
                -3.3269e-04,
                -7.5893e-05,
                3.6695e-04,
                -2.9651e-05,
                4.5118e-05,
                2.7741e-05,
                1.6353e-05,
                -2.0110e-06,
                -1.0852e-08,
                2.3127e-07,
                2.0614e-07,
                -2.0125e-06,
                9.9512e-07,
                1.2871e-06,
                2.5418e-06,
            ]
            self.dgsm_gradient_abs = [
                1.9997,
                1.3332,
                0.6665,
                0.3636,
                0.1904,
                0.0784,
                0.0396,
                0.0080,
                0.0040,
                0.0040,
                0.0040,
                0.0040,
                0.0040,
                0.0040,
                0.0040,
            ]
            self.dgsm_gradient_square = [
                4.2009e00,
                1.9506e00,
                5.0098e-01,
                1.5003e-01,
                4.1247e-02,
                6.9977e-03,
                1.7844e-03,
                7.2523e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
                1.8167e-05,
            ]
        else:
            self.a = a
        self.optimal_sobol_indicies()
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def optimal_sobol_indicies(self):
        vi = []
        dim = self.dim
        for i in range(dim):
            vi.append(1 / (3 * ((1 + self.a[i]) ** 2)))
        self.vi = Tensor(vi)
        self.V = torch.prod((1 + self.vi)) - 1
        self.si = self.vi / self.V
        si_t = []
        for i in range(dim):
            si_t.append(
                (
                    self.vi[i]
                    * torch.prod(self.vi[:i] + 1)
                    * torch.prod(self.vi[i + 1 :] + 1)
                )
                / self.V
            )
        self.si_t = Tensor(si_t)

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        t = 1
        dim = self.dim
        for i in range(dim):
            t = t * (torch.abs(4 * X[..., i] - 2) + self.a[i]) / (1 + self.a[i])
        return t


class MorrisReduced(SyntheticTestFunction):
    r"""Morris Reduced test function.

    4-dimensional function (usually evaluated on `[0, 1]^4`):

        f(x) = sum_{i=1}\^4 beta_i x_i + sum_{i<=j}\^4 beta_ij x_i x_j
        + sum_{i<=j<=l}\^4 beta_ijl x_i x_j x_l

    Proposed to test sensitivity analysis methods
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
        """
        self._optimizers = None
        self.dim = 4
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self.dgsm_gradient = [90.0640, 71.0459, 41.4663, 20.8226]
        self.dgsm_gradient_abs = [90.0640, 71.0459, 41.4663, 20.8226]
        self.dgsm_gradient_square = [9077.6625, 5880.7383, 2019.7125, 566.9906]
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        t1 = 0
        t2 = 0
        t3 = 0
        bi = [0.05, 0.59, 10, 0.21]
        bij = [
            [0, 80, 60, 40],
            [0, 30, 0.73, 0.18],
            [0, 0, 0.64, 0.93],
            [0, 0, 0, 0.06],
        ]
        bij4 = [[0, 10, 0.98, 0.19], [0, 0, 0.49, 50], [0, 0, 0, 1], [0, 0, 0, 0]]
        for i in range(self.dim):
            t1 = t1 + bi[i] * X[..., i]
            for j in range(i, self.dim):
                t2 = t2 + bij[i][j] * X[..., i] * X[..., j]
                for k in range(j, self.dim):
                    if k == self.dim - 1:
                        t3 = t3 + bij4[i][j]
        return t1 + t2 + t3


class a_function(SyntheticTestFunction):
    r""" "a"-function test function.

    6-dimensional function (usually evaluated on `[0, 1]^6`):

        f(x) = cos([1,x1,x5,x3]phi) + sin([1,x4,x2,x6]gamma)

    Proposed to test sensitivity analysis methods
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
        """
        self._optimizers = None
        self.dim = 6
        self._bounds = [(-1, 1) for _ in range(self.dim)]
        self.dgsm_gradient = [-0.4360, 0.5207, 0.3964, 0.4686, 0.4360, -0.5728]
        self.dgsm_gradient_abs = [0.7002, 0.6572, 0.6365, 0.5915, 0.7002, 0.7229]
        self.dgsm_gradient_square = [0.6060, 0.5245, 0.5008, 0.4248, 0.6060, 0.6346]
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        phi = torch.tensor([-0.8, -1.1, 1.1, 1]).to(device=X.device, dtype=X.dtype)
        gamma = torch.tensor([0.5, 0.9, 1, -1.1]).to(device=X.device, dtype=X.dtype)
        y1 = torch.cat(
            [
                torch.ones(X.shape[0]).unsqueeze(1),
                X.index_select(1, torch.tensor([0, 4, 2])),
            ],
            dim=1,
        )
        y2 = torch.cat(
            [
                torch.ones(X.shape[0]).unsqueeze(1),
                X.index_select(1, torch.tensor([3, 1, 5])),
            ],
            dim=1,
        )
        v1 = torch.matmul(y1, phi)
        v2 = torch.matmul(y2, gamma)
        t = torch.cos(v1) + torch.sin(v2)
        return t


class b_function(SyntheticTestFunction):
    r""" "b"-function test function.

    6-dimensional function (usually evaluated on `[0, 1]^6`):

        f(x) = cos([1,x1,x5,x3]phi) + sin([1,x4,x2,x6]gamma) + [1,x3,x4]lamda

    Proposed to test sensitivity analysis methods
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
        """
        self._optimizers = None
        self.dim = 6
        self._bounds = [(-1, 1) for _ in range(self.dim)]
        self.dgsm_gradient = [-0.4360, 0.5207, 0.7464, -0.1313, 0.4360, -0.5728]
        self.dgsm_gradient_abs = [0.7002, 0.6572, 0.8363, 0.4668, 0.7002, 0.7229]
        self.dgsm_gradient_square = [0.6060, 0.5245, 0.9121, 0.3370, 0.6060, 0.6346]
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        phi = torch.tensor([-0.8, -1.1, 1.1, 1]).to(device=X.device, dtype=X.dtype)
        gamma = torch.tensor([0.5, 0.9, 1, -1.1]).to(device=X.device, dtype=X.dtype)
        lamda = torch.tensor([0.5, 0.35, -0.6]).to(device=X.device, dtype=X.dtype)
        y1 = torch.cat(
            [
                torch.ones(X.shape[0]).unsqueeze(1),
                X.index_select(1, torch.tensor([0, 4, 2])),
            ],
            dim=1,
        )
        y2 = torch.cat(
            [
                torch.ones(X.shape[0]).unsqueeze(1),
                X.index_select(1, torch.tensor([3, 1, 5])),
            ],
            dim=1,
        )
        y3 = torch.cat(
            [
                torch.ones(X.shape[0]).unsqueeze(1),
                X.index_select(1, torch.tensor([2, 3])),
            ],
            dim=1,
        )
        v1 = torch.matmul(y1, phi)
        v2 = torch.matmul(y2, gamma)
        v3 = torch.matmul(y3, lamda)
        t = torch.cos(v1) + torch.sin(v2) + torch.pow(v3, 2)
        return t


class A1(SyntheticTestFunction):
    r"""A1 test function.

    n-dimensional function (usually evaluated on `[0, 1]^10`):

        f(x) = sum_{i=1}^{i=n} (-1)^i \Pi_{j=1}^{j=i} x_j

    Proposed to test sensitivity analysis methods
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
        """
        self._optimizers = None
        self.dim = 10
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self.dgsm_gradient = [
            -0.6659,
            0.3341,
            -0.1660,
            0.0840,
            -0.0410,
            0.0215,
            -0.0098,
            0.0059,
            -0.0020,
            0.0020,
        ]
        self.dgsm_gradient_abs = [
            0.6659,
            0.3341,
            0.1660,
            0.0840,
            0.0410,
            0.0215,
            0.0098,
            0.0059,
            0.0020,
            0.0020,
        ]
        self.dgsm_gradient_square = [
            4.9910e-01,
            1.6727e-01,
            5.5177e-02,
            1.8771e-02,
            6.0270e-03,
            2.1637e-03,
            6.2308e-04,
            2.8052e-04,
            5.1118e-05,
            5.0931e-05,
        ]
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        total = 0
        for i in range(1, self.dim + 1):
            sign = (-1) ** i
            total = total + sign * torch.prod(X[..., :i], dim=1)
        return total


class B1(SyntheticTestFunction):
    r"""B1 test function.

    n-dimensional function (usually evaluated on `[0, 1]^10`):

        f(x) =  \Pi_{i=1}^{i=n} (n-x_i)/(n-0.5)

    Proposed to test sensitivity analysis methods
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
        """
        self._optimizers = None
        self.dim = 10
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self.dgsm_gradient = [
            -0.1053,
            -0.1053,
            -0.1053,
            -0.1053,
            -0.1053,
            -0.1053,
            -0.1053,
            -0.1053,
            -0.1053,
            -0.1053,
        ]
        self.dgsm_gradient_abs = [
            0.1053,
            0.1053,
            0.1053,
            0.1053,
            0.1053,
            0.1053,
            0.1053,
            0.1053,
            0.1053,
            0.1053,
        ]
        self.dgsm_gradient_square = [
            0.0112,
            0.0112,
            0.0112,
            0.0112,
            0.0112,
            0.0112,
            0.0112,
            0.0112,
            0.0112,
            0.0112,
        ]
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        X_ = (self.dim - X) / (self.dim - 0.5)
        total = torch.prod(X_, dim=1)
        return total


class C1(SyntheticTestFunction):
    r"""C1 test function.

    n-dimensional function (usually evaluated on `[0, 1]^10`):

        f(x) =  2^n\Pi_{i=1}^{i=n} x_i

    Proposed to test sensitivity analysis methods
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
        """
        self._optimizers = None
        self.dim = 10
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self.dgsm_gradient = [
            2.0035,
            2.0027,
            2.0021,
            2.0030,
            2.0046,
            2.0022,
            2.0053,
            2.0045,
            2.0025,
            2.0049,
        ]
        self.dgsm_gradient_abs = [
            2.0035,
            2.0027,
            2.0021,
            2.0030,
            2.0046,
            2.0022,
            2.0053,
            2.0045,
            2.0025,
            2.0049,
        ]
        self.dgsm_gradient_square = [
            53.3507,
            53.1658,
            53.3660,
            53.5822,
            53.5270,
            53.4191,
            53.5050,
            53.3956,
            53.3644,
            53.4045,
        ]
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        X_ = torch.prod(X, dim=1)
        total = (2**self.dim) * X_
        return total


class C2(SyntheticTestFunction):
    r"""C2 test function.

    n-dimensional function (usually evaluated on `[0, 1]^10`):

        f(x) =  \Pi_{i=1}^{i=n} |4 x_i-2|

    Proposed to test sensitivity analysis methods
    """

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
        """
        self._optimizers = None
        self.dim = 10
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self.dgsm_gradient = [
            0.0019,
            -0.0007,
            0.0014,
            -0.0057,
            0.0025,
            -0.0037,
            -0.0008,
            -0.0007,
            0.0087,
            0.0041,
        ]
        self.dgsm_gradient_abs = [
            3.9924,
            3.9882,
            3.9985,
            3.9907,
            3.9912,
            3.9929,
            3.9959,
            3.9923,
            3.9911,
            3.9923,
        ]
        self.dgsm_gradient_square = [
            212.6942,
            211.1638,
            212.9353,
            211.9558,
            211.0787,
            212.3983,
            212.5103,
            212.2815,
            211.3165,
            212.1272,
        ]
        super().__init__(noise_std=noise_std, negate=negate)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        X_ = torch.abs(4 * X - 2)
        total = torch.prod(X_, dim=1)
        return total
