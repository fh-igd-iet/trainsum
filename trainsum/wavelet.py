# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from math import prod
from typing import Sequence

from .backend import ArrayLike, ArrayNamespace
from .dimension import Dimension
from .trainbase import TrainBase
from .trainshape import TrainShape
from .trainshape import trainshape
from .einsumcontraction import EinsumContraction
from .exactcontractor import ExactContractor
from .einsumequation import EinsumEquation
from .exactaddition import ExactAddition
from .binarytrain import binary_train
from .linearintegerequation import LinearIntegerEquation


def dwt[T: ArrayLike](
    xp: ArrayNamespace[T],
    dim: Dimension,
    coeffs: Sequence[float],
) -> TrainBase:
    """Construct a wavelet-like tensor in the given dimension with the given coefficients."""
    if len(coeffs) > dim.size():
        raise ValueError("Number of coefficients cannot exceed the size of the dimension.")
    if dim[0].base != 2:
        raise ValueError("Only dimensions with a leading base of 2 are supported.")
    dims = (
        Dimension([d.base for d in dim[1:]]),
        Dimension([d.base for d in dim]),
        Dimension([len(coeffs)]),
    )

    coeffs_ar1 = xp.asarray(coeffs)[xp.newaxis, :, xp.newaxis]
    coeff_train1 = TrainBase(trainshape(dims[-1]), [coeffs_ar1])

    coeffs_ar2 = coeffs_ar1[:, ::-1, :]
    coeffs_ar2[0, 1::2, 0] *= -1
    coeff_train2 = TrainBase(trainshape(dims[-1]), [coeffs_ar2])

    shape = trainshape(*dims, mode="interleaved_rear")
    eq1 = LinearIntegerEquation(dims, coeffs=(2, -1, 1), rhs=0)
    eq2 = LinearIntegerEquation(dims, coeffs=(2, -1, 1), rhs=dims[1].size())
    bin_train = binary_train(xp, dims, [eq1, eq2])
    #print(bin_train.shape)

    eq = EinsumEquation("ijk,k->ij", bin_train.shape, coeff_train1.shape)
    contr = EinsumContraction(eq)
    expr = ExactContractor(contr)
    res1 = expr(bin_train, coeff_train1)
    res2 = expr(bin_train, coeff_train2)

    tmp1 = xp.asarray([1.0, 0.0])[xp.newaxis,:,xp.newaxis]
    tmp2 = xp.asarray([0.0, 1.0])[xp.newaxis,:,xp.newaxis]

    ndim0, ndim1 = dim, Dimension([d.base for d in dim])
    digits = [(ndim0[0],)]
    pos0, pos1 = 0, 0
    dim0, dim1 = bin_train.shape.dims[:2]
    bin_dgts = bin_train.shape.digits
    for i in range(len(bin_train.shape)):
        dgts = []
        if pos0 < len(dim0) and dim0[pos0] in bin_dgts[i]:
            dgts.append(ndim0[pos0+1])
            pos0 += 1
        if pos1 < len(dim1) and dim1[pos1] in bin_dgts[i]:
            dgts.append(ndim1[pos1])
            pos1 += 1
        digits.append(tuple(dgts))
    shape = TrainShape((ndim0, ndim1), digits)
    res1 = TrainBase(shape, [tmp1, *res1.data])
    res2 = TrainBase(shape, [tmp2, *res2.data])

    add = ExactAddition()
    res = add(res1, res2)

    return res

def idwt[T: ArrayLike](
    xp: ArrayNamespace[T],
    dim: Dimension,
    coeffs: Sequence[float],
) -> TrainBase:
    base = dwt(xp, dim, coeffs)
    shape = TrainShape(base.shape.dims[::-1], base.shape.digits)
    return TrainBase(shape, base.data, copy_data=False)


"""
def wavelet[T: ArrayLike](
    xp: ArrayNamespace[T],
    dim: Dimension,
    coeffs: Sequence[float],
    decomp: MatrixDecomposition = SVDecomposition(max_rank=8),
) -> TrainBase:
    n_input = dim.size()
    n_coeffs = len(coeffs)
    assert n_input >= n_coeffs

    strat = SweepingStrategy(ncores=2, nsweeps=1)

    permutated_shifts = [permutated_shift(xp, dim, i) for i in range(n_coeffs)]

    coefficients = [
        get_ith_coffencients_tensor(xp, dim, coeffs, i, decomp) for i in range(n_coeffs)
    ]

    expr = EinsumExpression[T](
        "ab,a->ab", permutated_shifts[0].shape, coefficients[0].shape, method="exact"
    )

    X = tuple(expr(P, C) for P, C in zip(permutated_shifts, coefficients))

    _add = DecompositionAddition(strat, decomp)
    guess = _add(*X)  # type: ignore

    var_add = VariationalAddition(guess.shape, decomp, strat)
    guess = var_add(guess, *X)  # type: ignore
    return guess


def permutated_shift[T: ArrayLike](
    xp: ArrayNamespace[T], dim: Dimension, shift_by: int
) -> TrainBase:
    dim_2 = Dimension([*[d.base for d in dim[1:]], dim[0].base])

    shape = TrainShape([dim, dim_2], [digits for digits in zip(dim, dim_2)])
    decomp = TensorDecomposition(QRDecomposition[T]())

    P = shift(xp, dim, shift_by // 2, circular=True)
    data = []
    tmp = P.data[0]

    if shift_by % 2 == 1:
        tmp = xp.flip(tmp, axis=2)

    for i in range(len(P.data) - 1):
        supercore = xp.tensordot(tmp, P.data[i + 1], axes=([3], [0]))
        supercore = xp.permute_dims(supercore, (0, 1, 4, 3, 2, 5))
        res = decomp.left(supercore, 3)
        data.append(res.left)
        tmp = res.right
    data.append(tmp)

    return TrainBase(shape, data)


def coefficients_tensor[T: ArrayLike](
    xp: ArrayNamespace[T],
    dim: Dimension,
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    shift_by: int,
    decomp: MatrixDecomposition,
) -> TrainBase:
    strat = SweepingStrategy(ncores=2, nsweeps=1)

    shape = TrainShape(dim, [(d,) for d in dim])  # [[*dim]] = [[d1, d2, ...]]

    n_input = dim.size()
    n_half = n_input // 2

    # shift forward and backward to cut and put in correct position
    # [ 0  |  n_half - shift_by  |  h_half  |  n_input - shift_by]
    # T1 = [c1 | 0 | 0 | 0], T3 = [0 | 0 | c3 | 0]
    # T2 = [0 | c2 | 0 | 0], T4 = [0 | 0 | 0 | c4]

    T1 = matmul(
        full(xp, shape, c1),
        matmul(shift(xp, dim, n_half + shift_by), shift(xp, dim, -n_half - shift_by)),
    )
    T3 = matmul(
        full(xp, shape, c3),
        matmul(shift(xp, dim, n_half + shift_by), shift(xp, dim, -shift_by)),
    )

    Ts = [T1, T3]

    if shift_by > 0:
        T2 = matmul(
            matmul(full(xp, shape, c2), shift(xp, dim, n_input - shift_by)),
            shift(xp, dim, -n_half),
        )
        T4 = matmul(full(xp, shape, c4), shift(xp, dim, n_input - shift_by))

        Ts.append(T2)
        Ts.append(T4)

    _add = DecompositionAddition[T, SVDecompositionResult](strat, decomp)
    guess = _add(*Ts)
    # var_add = VariationalAddition(guess.shape, decomp, strat)
    # guess = var_add(guess, *Ts)
    return guess


def get_ith_coffencients_tensor[T: ArrayLike](
    xp: ArrayNamespace[T],
    dim: Dimension,
    coeffs: Sequence[float],
    i: int,
    decomp: MatrixDecomposition,
) -> TrainBase:
    n_coeffs = len(coeffs)

    is_odd = i % 2 == 1
    odd = 1 - 2 * is_odd  # multiply by -1 if odd

    ix = i + 2 * (not (is_odd))
    iy = ix - 1
    # ix: 0 -> 2, 1 -> 1, 2 -> 4, 3 -> 3, 4 -> 6, 5 -> 5, ...
    # iy: 0 -> 1, 1 -> 0, 2 -> 3, 3 -> 2, 4 -> 5, 5 -> 4, ...

    c1 = coeffs[i]
    c2 = coeffs[iy]
    c3 = coeffs[-ix] * -odd
    c4 = coeffs[n_coeffs - i - 1] * odd

    return coefficients_tensor(xp, dim, c1, c2, c3, c4, i // 2, decomp)
"""
