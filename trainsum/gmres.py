# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Any, Callable, Optional
from dataclasses import dataclass
from copy import deepcopy
import time
from math import sqrt

from .backend import ArrayLike, Device, DType, namespace_of_arrays, size, to_device
from .utils import check_pos, check_non_neg
from .matrixleastsquares import MatrixLeastSquares
from .lstsqsolver import LstsqSolver


class GMRESData[T: ArrayLike]:
    device: Device
    dtype: DType
    subspace: int
    guess: T
    basis: T

    def __init__(self, subspace: int, guess: T) -> None:
        self.device = guess.device
        self.dtype = guess.dtype
        self.subspace = min(subspace, size(guess))
        self.guess = deepcopy(guess)
        xp = namespace_of_arrays(guess)
        self.basis = xp.zeros(
            (self.subspace + 1, *guess.shape), device=self.device, dtype=self.dtype
        )


@dataclass
class GMRESResult[T: ArrayLike]:
    #: Solution of the linear system.
    array: T
    #: Time taken to compute the solution.
    time: float
    #: Convergence history of the residuals at each step.
    residuals: list[float]


@dataclass
class GMRES:
    """
    GMRES linear solver.
    """

    #: Number of times the subspace is created and solved
    nsteps: int = 10

    #: Size of the Arnoldi basis
    subspace: int = 10

    #: Minimum residual, after which the algorithm is stopped
    eps: float = 1e-8

    #: Least squares solver for solving the matrix in the Arnoldi basis
    solver: MatrixLeastSquares = LstsqSolver()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("nsteps", "subspace"):
            check_pos(name, value)
        elif name == "eps":
            check_non_neg(name, value)
        super().__setattr__(name, value)

    def __call__[T: ArrayLike](
        self, mat: Callable[[T], T], rhs: T, guess: Optional[T] = None, /
    ) -> GMRESResult[T]:
        """
        Solve the linear system with mat being a linear operator and rhs being the right hand side of
        the equation. The initial guess can be provided, otherwise it will be initialized to zero.
        """
        xp = namespace_of_arrays(rhs)
        if guess is None:
            guess = xp.zeros_like(rhs)

        self._check_input(rhs, guess)

        stamp = time.time()
        residuals: list[float] = []
        data = GMRESData[T](self.subspace, guess)

        for _ in range(self.nsteps):
            residual = self._arnoldi_cycle(mat, rhs, data)
            residuals.append(residual)
            if residuals[-1] <= self.eps:
                break

        return GMRESResult(
            array=data.guess, time=time.time() - stamp, residuals=residuals
        )

    def _arnoldi_cycle[T: ArrayLike](
        self, mat: Callable[[T], T], rhs: T, data: GMRESData[T]
    ) -> float:
        xp = namespace_of_arrays(rhs)
        subspace, guess, basis, dtype = (
            data.subspace,
            data.guess,
            data.basis,
            data.dtype,
        )

        # Work arrays
        hess = xp.zeros((subspace + 1, subspace), dtype=dtype)
        cs = xp.zeros(subspace, dtype=dtype)
        sn = xp.zeros(subspace, dtype=dtype)
        g = xp.zeros(subspace + 1, dtype=dtype)

        # Initialize with current residual
        r = rhs - mat(guess)
        res_norm = float(abs(xp.sqrt(xp.sum(xp.conj(r) * r))))
        if res_norm <= self.eps:
            return res_norm

        basis[0, :] = r / res_norm
        g[0] = res_norm

        idx = 0
        for i in range(subspace):
            # Modified Gram-Schmidt to build Arnoldi basis
            self._gram_schmidt(mat, data, hess, i)

            # Apply all previous Givens rotations to new column i
            for j in range(i):
                tmp = cs[j] * hess[j, i] + sn[j] * hess[j + 1, i]
                hess[j + 1, i] = -xp.conj(sn[j]) * hess[j, i] + cs[j] * hess[j + 1, i]
                hess[j, i] = tmp

            # Compute new Givens rotation to zero out hess[i+1, i]
            cs[i], sn[i], hess[i, i] = self._givens(hess[i, i], hess[i + 1, i])
            hess[i + 1, i] = 0.0

            # Apply new rotation to g (g[i+1] is 0 before this)
            g[i + 1] = -xp.conj(sn[i]) * g[i]
            g[i] = cs[i] * g[i]

            # Residual norm is simply |g[i+1]| — no solve needed!
            idx += 1
            res_norm = float(abs(g[idx]))

            if res_norm <= self.eps:
                break

        # Solve the upper triangular system ONCE at the end
        if idx == 0:
            return res_norm
        y = self.solver(hess[:idx, :idx], g[:idx])
        y = to_device(y, data.device)
        guess += xp.tensordot(y, basis[:idx, :], axes=[[0], [0]])
        return res_norm

    def _gram_schmidt[T: ArrayLike](
        self, mat: Callable[[T], T], data: GMRESData[T], hess: T, idx: int
    ) -> None:
        xp = namespace_of_arrays(data.guess)
        basis = data.basis
        vec = mat(basis[idx, :])
        for j in range(idx + 1):
            hess[j, idx] = xp.sum(xp.conj(basis[j, :]) * vec)
            vec = vec - hess[j, idx] * basis[j, :]
        norm = float(abs(xp.sqrt(xp.sum(xp.conj(vec) * vec))))
        hess[idx + 1, idx] = norm
        if norm > 0.0:
            basis[idx + 1, :] = vec / norm

    def _givens(self, a, b):
        """
        Compute complex Givens rotation (c, s, r) such that:
            [[c,       s     ]   [a]   [r]
             [-conj(s), c     ]] [b] = [0]

        c is real and non-negative. Works for both real and complex inputs.
        """
        xp = namespace_of_arrays(b)
        abs_a = abs(a)
        abs_b = abs(b)
        if abs_b == 0.0:
            return 1.0, 0.0, a
        if abs_a == 0.0:
            return 0.0, 1.0, b
        denom = sqrt(abs_a**2 + abs_b**2)
        c = abs_a / denom
        phase_a = a / abs_a
        s = phase_a * xp.conj(b) / denom
        r = phase_a * denom
        return c, s, r

    def _check_input(self, rhs: ArrayLike, guess: ArrayLike) -> None:
        if guess.shape != rhs.shape:
            raise ValueError("x0 and b must have the same shape.")
        if guess.device != rhs.device:
            raise ValueError("x0 and b must be on the same device.")
        if guess.dtype != rhs.dtype:
            raise ValueError("x0 and b must have the same dtype.")
