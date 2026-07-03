# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from math import ceil, prod

from .backend import ArrayNamespace, ArrayLike
from .dimension import Dimension
from .trainshape import trainshape
from .trainbase import TrainBase

from .integerequations import IntegerEquations
from .linearintegerequation import LinearIntegerEquation
from .rangeintegerequation import RangeIntegerEquation
from .moduleintegerequation import ModuloIntegerEquation
from .binarytrain import binary_train


def slice_vector[T: ArrayLike](
    xp: ArrayNamespace[T], dim: Dimension, slc: int | slice, /
) -> TrainBase[T]:

    dims = (dim,)
    if isinstance(slc, int):
        start = slc
        stop = slc + 1
        step = 1
    else:
        start = int(slc.start) if slc.start is not None else 0
        stop = int(slc.stop) if slc.stop is not None else dim.size()
        step = int(slc.step) if slc.step is not None else 1

    eqs = []
    eqs.append(RangeIntegerEquation(dims, (start,), (stop,)))
    if step > 1:
        eqs.append(ModuloIntegerEquation(dims, (step,), start%step))
    eq = IntegerEquations(dims, eqs)

    shape = trainshape(dim)
    bin_train = binary_train(xp, shape, [eq])
    return bin_train


def slice_operator[T: ArrayLike](
    xp: ArrayNamespace[T], dim: Dimension, slc: slice, /
) -> TrainBase[T]:
    start = int(slc.start) if slc.start is not None else 0
    stop = int(slc.stop) if slc.stop is not None else dim.size()
    step = int(slc.step) if slc.step is not None else 1
    if stop - start == 1:
        raise ValueError("Slice operator requires at least two elements in the slice.")

    size = ceil((stop - start) / step)
    ndim = Dimension(size)
    ndim = Dimension([d.base for d in ndim[::-1]])
    dims = (ndim, dim)

    eq = LinearIntegerEquation(dims, coeffs=(-step, 1), rhs=start)
    bin_train = binary_train(xp, dims, [eq])
    return bin_train

def shape_middle_aligned(dim1: Dimension, dim2: Dimension):
    digits = []
    idx = 0
    for i in range(len(dim2)):
        if dim2.size() // prod(d.base for d in dim2[:i+1]) < dim1.size():
            if idx < len(dim1):
                digits.append([dim1[idx], dim2[i]])
                idx += 1
                continue
        digits.append([dim2[i]])
    return trainshape(dim1, dim2, digits=digits)
