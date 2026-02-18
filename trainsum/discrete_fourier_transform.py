# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from copy import deepcopy

from .backend import ArrayNamespace, ArrayLike
from .matrixdecomposition import MatrixDecomposition
from .dimension import Dimension
from .trainshape import TrainShape
from .trainbase import TrainBase

from .sweepingstrategy import SweepingStrategy
from .svdecomposition import SVDecomposition
from .einsumexpression import EinsumExpression
from .domain import Domain
from .uniformgrid import UniformGrid
from .polyval import polyval
from .conj import conj
from .utils import namespace_of_trains

def qft[T: ArrayLike](
        xp: ArrayNamespace[T],
        dim: Dimension,
        decomp: MatrixDecomposition = SVDecomposition(max_rank=16)
        ) -> TrainBase:

    layers = [fft_layer(xp, dim, i) for i in range(len(dim))]

    strat = SweepingStrategy(ncores=2)
    ref_shape = deepcopy(layers[0].shape)
    ref_shape.ranks = 16
    layer_shape = deepcopy(layers[0].shape)
    layer_shape.ranks = 2
    expr = EinsumExpression(
        "ab,ab->ab", ref_shape, layer_shape,
        method="decomposition",
        decomposition=decomp,
        strategy=strat)

    res = layers[0]
    for l in layers[1:]:
        res = expr(res, l)
        if not isinstance(res, TrainBase):
            raise ValueError("Expected a TrainBase as result here.")
    res.data[0][...] /= xp.sqrt(xp.asarray(dim.size()))
    return res

def iqft[T: ArrayLike](
        xp: ArrayNamespace[T],
        dim: Dimension,
        decomp: MatrixDecomposition = SVDecomposition(max_rank=16)
        ) -> TrainBase:
    op = conj(qft(xp, dim, decomp))
    shape = TrainShape(op.shape.dims[::-1], op.shape.digits)
    return TrainBase(shape, op.data, copy_data=False)

def fft_layer[T: ArrayLike](
        xp: ArrayNamespace[T],
        dim: Dimension,
        idx: int
        ) -> TrainBase[T]:
    info = xp.__array_namespace_info__()
    ctype = info.dtypes()["complex128"]
    factor = -2*xp.pi*1.0j/dim.size()
    ref_xi = dim[idx].factor
    rank = dim[idx].base

    dims = (Dimension([d.base for d in dim]),
            Dimension([d.base for d in dim[::-1]]))

    data = []
    for i, (row_digit, col_digit) in enumerate(zip(dims[0], reversed(dims[1]))):
        tmp = xp.zeros((rank, row_digit.base, col_digit.base, rank), dtype=ctype)
        if i == idx:
            for j in range(row_digit.base):
                for k in range(col_digit.base):
                    row_fac = row_digit.factor * j
                    col_fac = col_digit.factor * k
                    tmp[j,j,k,j] = xp.exp(xp.asarray(factor*row_fac*col_fac))
        else:
            for j in range(row_digit.base):
                for k in range(col_digit.base):
                    for l in range(rank):
                        row_fac = ref_xi * l
                        col_fac = col_digit.factor * k
                        tmp[l,j,k,l] = xp.exp(xp.asarray(factor*row_fac*col_fac))
        data.append(tmp)
    tmp = xp.ones(rank, dtype=ctype)
    data[0] = xp.tensordot(data[0], tmp, axes=([0],[0]))
    data[0] = xp.expand_dims(data[0], axis=0)
    data[-1] = xp.tensordot(tmp, data[-1], axes=([0],[3]))
    data[-1] = xp.expand_dims(data[-1], axis=3)

    shape = TrainShape(dims[::-1], [digits for digits in zip(dims[0], reversed(dims[1]))])
    return TrainBase(shape, data)

def qftshift[T: ArrayLike](train: TrainBase[T], axis: int=0) -> TrainBase[T]:
    if 0 < axis < len(train.shape.dims):
        raise ValueError("Axis out of range.")
    dim = train.shape.dims[axis]
    xp = namespace_of_trains(train)
    if dim[0].base != 2:
        raise NotImplementedError("QFT shift is only implemented for dimensions with leading bit 2.")
    res = deepcopy(train)
    for i, digits in enumerate(res.shape.digits):
        if dim[0] in digits:
            idx = digits.index(dim[0])
            res.data[i][...] = xp.flip(res.data[i], axis=idx+1)
    return res

def iqftshift[T: ArrayLike](train: TrainBase[T], axis: int=0) -> TrainBase[T]:
    return qftshift(train, axis)

def qftfreq[T: ArrayLike](
        xp: ArrayNamespace[T],
        dim: Dimension,
        d: float
        ) -> TrainBase[T]:
    if dim.size() % 2 == 0: # even
        domain = Domain(-dim.size()/2, dim.size()/2 - 1)
    else: # odd
        domain = Domain(-(dim.size()-1) / 2, -(dim.size()-1) / 2)
    grid = UniformGrid(dim, domain)
    freqs = polyval(xp, grid, [1.0, 0.0], 0.0)
    freqs = qftshift(freqs)
    freqs.data[0][...] /= dim.size()*d
    return freqs
