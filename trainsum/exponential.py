# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from math import exp as mexp

from .backend import ArrayNamespace, ArrayLike
from .uniformgrid import UniformGrid
from .trainbase import TrainBase
from .trainshape import TrainShape

def exp[T: ArrayLike](xp: ArrayNamespace[T], grid: UniformGrid, factor: float, offset: float) -> TrainBase[T]:
    if len(grid.dims) != 1:
        raise ValueError("Exponential function only supports 1D uniform grid")
    dim = grid.dims[0]
    shape = TrainShape(dim, [(d,) for d in dim])
    domain = grid.domains[0]
    spacing = grid.spacings[0]
    
    data = []
    for digit in dim:
        xi = spacing*digit.factor
        tmp = xp.zeros((1, digit.base, 1))
        for j in range(digit.base):
            tmp[:,j,:] = mexp(factor * j*xi)
        data.append(tmp)
    data[0] *= mexp(factor * (domain.lower - offset))
    return TrainBase(shape, data, copy_data=False)
