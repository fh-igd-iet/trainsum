# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from math import comb

from .backend import ArrayNamespace, ArrayLike
from .uniformgrid import UniformGrid
from .trainshape import TrainShape
from .trainbase import TrainBase
from .full import full

def polyval[T: ArrayLike](xp: ArrayNamespace[T], grid: UniformGrid, coeffs: Sequence[float], offset: float) -> TrainBase[T]:
    if grid.ndims != 1:
        raise ValueError("Polynomial function only supports 1D uniform grid")
    dim = grid.dims[0]
    shape = TrainShape(dim, [(d,) for d in dim])

    domain = grid.domains[0]
    spacing = grid.spacings[0]
    off = (domain.lower-offset)/(len(dim))
    order = len(coeffs)

    if len(coeffs) == 1:
        return full(xp, shape, next(iter(coeffs)))
    
    cores = []
    for digit in dim:
        xi = spacing*digit.factor
        data = xp.zeros((order, digit.base, order))
        for j in range(order):
            for k in range(digit.base):
                row_data = xp.asarray(_get_row(order-j, k*xi+off))
                data[j,k,j:] = row_data
        cores.append(data)
        
    tmp = xp.reshape(xp.asarray(coeffs), (1, order))
    cores[0] = xp.tensordot(tmp, cores[0], axes=([1], [0]))
    tmp = xp.reshape(xp.asarray([*[0.0]*(order-1), 1]), (order, 1))
    idx = len(cores[-1].shape)-1
    cores[-1] = xp.tensordot(cores[-1], tmp, axes=([idx], [0]))
    return TrainBase(shape, cores, copy_data=False)

def _get_row(num: int, val: float) -> list[float]:
    row = [0.0]*num
    row[0] = 1
    for i in range(1, num):
        row[i] = comb(num-1, num-i-1) * val**i
    return row
