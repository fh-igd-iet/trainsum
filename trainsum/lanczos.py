# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Any, Sequence, Callable
from dataclasses import dataclass
from copy import deepcopy
import time


from .matrixeigenvaluedecomposition import MatrixEigenvalueDecomposition
from .eighsolver import EigHSolver
from .backend import DType, Device, ArrayLike, namespace_of_arrays, size, shape

class LanczosData[T: ArrayLike]:
    device: Device
    dtype: DType
    subspace: int
    basis: T
    offset: int
    eigvec: T
    eigval: float

    def __init__(self, subspace: int, guess: T, states: Sequence[T]) -> None:
        self._check_input(guess, states)
        self.device = guess.device
        self.dtype = guess.dtype
        self.subspace = min(subspace, size(guess)-len(states))

        xp = namespace_of_arrays(guess)
        self.basis = xp.zeros((self.subspace+len(states), *guess.shape), 
                              device=guess.device, dtype=guess.dtype)
        self.offset = len(states)
        for i, state in enumerate(states):
            self.basis[i,:] = state
        self.eigvec = deepcopy(guess)

    def _check_input(self, guess: T, states: Sequence[T]) -> None:
        if not all(guess.device == state.device for state in states):
            raise ValueError("All inputs must be on the same device")
        if not all(guess.dtype == state.dtype for state in states):
            raise ValueError("All inputs must have the same dtype")

@dataclass(kw_only=True)
class LanczosResult[T: ArrayLike]:
    #: Eigenvector
    array: T
    #: Eigenvalue
    value: float
    #: Time taken to compute the eigenvector and eigenvalue.
    time: float
    #: Convergence history of the eigenvalues at each step.
    eps: list[float]

@dataclass
class Lanczos:
    """
    Lanczos eigenvalue solver.
    """

    #: Number of times the subspace is created and solved
    nsteps: int = 1

    #: Size of the Lanczos basis
    subspace: int = 3

    #: Minimum difference between eigenvalues of consecutive steps, after which the algorithm is stopped
    eps: float = 1e-8

    #: Eigenvalue solver for diagonalizing the matrix in the Lanczos basis
    solver: MatrixEigenvalueDecomposition = EigHSolver()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("eps", "nsteps", "subspace") and value <= 0:
            raise ValueError(f"{name} must be a positive number above zero.")
        super().__setattr__(name, value)

    def __call__[T: ArrayLike](
            self,
            ham: Callable[[T], T],
            guess: T,
            states: Sequence[T] = [], /) -> LanczosResult[T]:
        """
        Solve the eigenvalue problem for some linear map with an initial guess and a set of orthogonal states.
        """
        xp = namespace_of_arrays(guess)
        self._check_input(guess, states)

        if min(self.subspace, size(guess)-len(states)) < len(states):
            return LanczosResult(array=guess,
                                 value=float(xp.sum(guess*ham(guess))),
                                 time=0.0,
                                 eps=[0.0])


        stamp = time.time()

        data = LanczosData(self.subspace, guess, states)
        data.eigvec = deepcopy(guess)
        for i in range(data.offset):
            self._gram_schmidt(data.basis[i,:], data.basis[:i,:])
            data.basis[i,:] /= xp.sqrt(xp.sum(data.basis[i,:]*data.basis[i,:]))

        eps = [float(xp.sum(data.eigvec*ham(data.eigvec)))]
        for _ in range(self.nsteps):
            self._step(ham, data)
            eps.append(data.eigval)
            if abs(eps[-1]-eps[-2]) < self.eps:
                break

        return LanczosResult(array=data.eigvec,
                             value=float(xp.sum(data.eigvec*ham(data.eigvec))),
                             time=time.time() - stamp,
                             eps=eps)
        
    def _step[T: ArrayLike](
            self,
            mat: Callable[[T], T],
            data: LanczosData[T]) -> None:

        xp = namespace_of_arrays(data.basis)
        vecs, guess, subspace, off = data.basis, data.eigvec, data.subspace, data.offset

        a = xp.zeros(subspace, dtype=data.dtype)
        b = xp.zeros(subspace, dtype=data.dtype)
        sub_mat = xp.zeros((data.subspace, data.subspace))

        vecs[off] = guess
        for j in range(off):
            val = xp.sum(vecs[off,:] * vecs[j,:])
            vecs[off,:] = vecs[off,:] - val * vecs[j,:]
        vecs[off] /= xp.sqrt(xp.sum(vecs[off]*vecs[off]))

        for i in range(off+1, off+subspace):
            vecs[i,:] = mat(vecs[i-1,:])
            if i > off+1:
                vecs[i,:] = vecs[i,:] - b[i-off-2] * vecs[i-2,:]
            a_val = xp.sum(vecs[i-1,:] * vecs[i,:])
            a[i-off-1] = a_val
            vecs[i,:] = vecs[i,:] - a_val * vecs[i-1,:]
            self._gram_schmidt(vecs[i,:], vecs[:i,:])

            b_val = xp.sqrt(xp.sum(vecs[i,:] * vecs[i,:]))
            vecs[i,:] = vecs[i,:] / b_val
            b[i-off-1] = b_val
        a[-1] = xp.sum(vecs[-1,:] * mat(vecs[-1,:]))

        sub_mat[0,0] = a[0]
        sub_mat[0,1] = b[0]
        for i in range(1, data.subspace-1):
            sub_mat[i,i-1] = b[i-1]
            sub_mat[i,i] = a[i]
            sub_mat[i,i+1] = b[i]
        sub_mat[-1,-1] = a[-1]
        sub_mat[-1,-2] = b[-2]

        eigvals, eigvecs = self.solver(sub_mat)
        data.eigval = float(eigvals[0])
        data.eigvec = xp.tensordot(eigvecs[:,0], vecs[off:,...], axes=([0],[0]))

    def _gram_schmidt[T: ArrayLike](self, vec: T, basis: T) -> None:
        xp = namespace_of_arrays(vec)
        for i in range(shape(basis)[0]):
            val = xp.sum(vec * basis[i,:])
            vec -= val * basis[i,:]

    def _check_input[T: ArrayLike](self, guess: T, states: Sequence[T]) -> None:
        if any(orth.shape != guess.shape for orth in states):
            raise ValueError("Orthogonal states must have the same dimensions as the initial state")
