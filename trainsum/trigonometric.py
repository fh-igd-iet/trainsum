# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from math import cos as mcos
from math import sin as msin
from math import pi

from .backend import ArrayNamespace, ArrayLike
from .uniformgrid import UniformGrid
from .trainshape import TrainShape
from .trainbase import TrainBase
from .full import full

def sin[T: ArrayLike](xp: ArrayNamespace[T], grid: UniformGrid, factor: float, offset: float) -> TrainBase: 
    train = cos(xp, grid, factor, offset+pi/2/abs(factor))
    if factor < 0.0:
        train.set_data(0, -1*train.data[0])
    return train

def cos[T: ArrayLike](xp: ArrayNamespace[T], grid: UniformGrid, factor: float, offset: float) -> TrainBase:
    _check_grid(grid)
    shape = TrainShape(grid.dims[0], [(d,) for d in grid.dims[0]])

    if factor == 0.0:
        return full(xp, shape, 1.0)

    dim = grid.dims[0]
    domain = grid.domains[0]
    spacing = grid.spacings[0]
    off = (domain.lower-offset)/(len(dim))
    
    cores = []
    for digit in dim:
        xi = spacing*digit.factor
        data = xp.zeros((2, digit.base, 2))
        for j in range(digit.base):
            theta = factor * (j*xi + off)
            rot = xp.asarray(_givens_rotation(theta))
            data[:,j,:] = rot
        cores.append(data)

    tmp = xp.ones((1, 2))
    cores[0] = xp.tensordot(tmp, cores[0],
                            axes=([1], [0]))

    tmp = xp.ones((2, 1))
    idx = len(cores[-1].shape)-1
    cores[-1] = xp.tensordot(cores[-1], tmp,
                             axes=([idx], [0]))

    cores[0] = 0.5 * cores[0]
    return TrainBase(shape, cores, copy_data=False)

def _givens_rotation(theta: float) -> list[list[float]]:
    return [[mcos(theta), -msin(theta)],
            [msin(theta),  mcos(theta)]]

def _check_grid(grid: UniformGrid) -> None:
    if grid.ndims != 1:
        raise ValueError("Only 1D uniform grid possible")
