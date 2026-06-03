# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence, overload, Generator

from .backend import ArrayLike, ArrayNamespace
from .digit import Digit
from .dimension import Dimension
from .trainshape import trainshape, TrainShape
from .trainbase import TrainBase
from .full import full
from .lstsqsolver import LstsqSolver
from .integerequation import IntegerEquation

@overload
def binary_train[T: ArrayLike](
        xp: ArrayNamespace[T],
        shape: TrainShape,
        eqs: Sequence[IntegerEquation],
        check_linear_dependencies: bool = False,
        /) -> TrainBase[T]: ...
@overload
def binary_train[T: ArrayLike](
        xp: ArrayNamespace[T],
        dim: Dimension,
        eqs: Sequence[IntegerEquation],
        check_linear_dependencies: bool = False,
        /) -> TrainBase[T]: ...
@overload
def binary_train[T: ArrayLike](
        xp: ArrayNamespace[T],
        dims: tuple[Dimension, ...],
        eqs: Sequence[IntegerEquation],
        check_linear_dependencies: bool = False,
        /) -> TrainBase[T]: ...
def binary_train[T: ArrayLike](
        xp: ArrayNamespace[T],
        dims: Dimension | tuple[Dimension, ...] | TrainShape,
        eqs: Sequence[IntegerEquation],
        check_linear_dependencies: bool = False,
        ) -> TrainBase[T]:

    if isinstance(dims, Dimension):
        dims = (dims,)

    if not any(eq.has_solution() for eq in eqs):
        shape = trainshape(*dims) if not isinstance(dims, TrainShape) else dims
        return full(xp, shape, 0.0)
    
    if isinstance(dims, TrainShape):
        shape = dims
        dims = tuple(shape.dims)
        eqs = list(eqs)
        cores = []
        for dgts in shape.digits[:-1]:
            idxs = get_bases(dims, dgts)
            core, eqs = factor_eqs(xp, eqs, idxs,
                                   check_linear_dependencies)
            cores.append(core)
        cores.append(vector(xp, eqs))

    else:
        eqs = list(eqs)
        cores = []
        digits = []
        dim_pos = [0 for _ in dims]
        while any(dim_pos[i] < len(dims[i])-1 for i in range(len(dims))):
            min_eqs = int(10**10)
            tmp = None
            for idxs in iter_positions(dims, dim_pos):
                tmp_core, tmp_eqs = factor_eqs(xp, eqs, idxs,
                                               check_linear_dependencies)
                if len(tmp_eqs) < min_eqs and len(tmp_eqs) > 0:
                    min_eqs = len(tmp_eqs)
                    tmp = (tmp_core, tmp_eqs, idxs)
            if tmp is None:
                raise RuntimeError("Failed to factor equations")

            cores.append(tmp[0])
            eqs = tmp[1]

            dgts = []
            for i in range(len(dim_pos)):
                if len(tmp[2][i]) != 0:
                    dgts.append(dims[i][dim_pos[i]])
                    dim_pos[i] += 1
            digits.append(dgts)

        dgts = []
        for i in range(len(dim_pos)):
            if dim_pos[i] < len(dims[i]):
                dgts.append(dims[i][dim_pos[i]])
                dim_pos[i] += 1
        digits.append(dgts)
        cores.append(vector(xp, eqs))

        shape = trainshape(*dims, digits=digits)

    if cores[0].shape[0] != 1:
        tmp = xp.ones(cores[0].shape[0])
        cores[0] = xp.tensordot(cores[0], tmp, axes=([0], [0]))[xp.newaxis, ...]

    train = TrainBase(shape, cores)
    return train

def factor_eqs[T: ArrayLike, S: IntegerEquation](
    xp: ArrayNamespace[T],
    eqs: Sequence[S],
    idxs: Sequence[Sequence[int]],
    check_linear_dependencies: bool = False,
) -> tuple[T, Sequence[S]]:
    if len(eqs) == 0:
        raise ValueError("No matrices provided")

    dims = eqs[0].dims
    if any(dims != m.dims for m in eqs):
        raise ValueError("All matrices must have the same shape")

    shape = get_shape(dims, idxs)
    solver = LstsqSolver()

    eq_dict, neqs, coeffs = {}, [], []
    for eq in eqs:
        cpos, ceqs = eq.factorize(*idxs) # current
        idx = 0
        coeffs.append([])
        for eq, pos in zip(ceqs, cpos):
            if eq in eq_dict:
                idx = eq_dict[eq]
                val = 1.0
                coeffs[-1].append((pos, idx, val))
            elif len(eq_dict) == 0:
                idx = len(eq_dict)
                eq_dict[eq] = idx
                neqs.append(eq)
                val = 1.0
                coeffs[-1].append((pos, idx, val))
            elif not check_linear_dependencies:
                idx = len(eq_dict)
                eq_dict[eq] = idx
                neqs.append(eq)
                val = 1.0
                coeffs[-1].append((pos, idx, val))
            else:
                tmp = [*neqs, eq]
                mat = xp.asarray([[neq.inner(neq2) for neq2 in tmp] for neq in tmp])
                m = mat[:,:-1]
                rhs = mat[:,-1]
                idx += 1

                alpha = solver(m, rhs)
                residual_norm = xp.sum((m @ alpha - rhs)**2)
                if residual_norm < 1e-10 and xp.sum(xp.abs(alpha)) > 1e-10:
                    for eq, val in zip(neqs, alpha):
                        if abs(val) > 1e-10:
                            coeffs[-1].append((pos, eq_dict[eq], val))
                else:
                    idx = len(eq_dict)
                    eq_dict[eq] = idx
                    neqs.append(eq)
                    val = 1.0
                    coeffs[-1].append((pos, idx, val))

    core = xp.zeros((len(eqs), *shape, len(eq_dict)))
    for i, eq_coeffs in enumerate(coeffs):
        for pos, idx, val in eq_coeffs:
            core[i, *pos, idx] += val

    return core, [eq for eq in eq_dict]

def vector[T: ArrayLike, S: IntegerEquation](
    xp: ArrayNamespace[T], eqs: Sequence[S]
) -> T:
    shape = eqs[0].tensor_shape()
    core = xp.zeros((len(eqs), *shape, 1))
    for i, eq in enumerate(eqs):
        core[i,...,0] = eq.to_tensor(xp)
    cut = [slice(None) if fac != 1 else 0 for fac in shape]
    return core[:,*cut,:]

def get_bases(dims: Sequence[Dimension], digits: Sequence[Digit]) -> Sequence[Sequence[int]]:
    idxs = [[] for _ in range(len(dims))]
    dmap = {dim.idf: i for i, dim in enumerate(dims)}
    for digit in digits:
        didx = dmap[digit.idf]
        idxs[didx].append(dims[didx].index(digit))
    return idxs

def get_shape(dims: Sequence[Dimension], idxs: Sequence[Sequence[int]]) -> Sequence[int]:
    facs = []
    for dim, idx in zip(dims, idxs):
        if len(idx) != 0:
            facs.extend(dim[i].base for i in idx)
    return facs

from itertools import product
def iter_positions(dims: Sequence[Dimension], dim_pos: Sequence[int]) -> Generator[tuple[tuple[int,...], ...]]:
    tmp = []
    for dim, idx in zip(dims, dim_pos):
        if idx < len(dim):
            tmp.append(((idx,), tuple()))
        else:
            tmp.append((tuple(),))
    for idxs in product(*tmp):
        if all(len(idx) == 0 for idx in idxs):
            continue
        yield idxs
