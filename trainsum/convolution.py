# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .binarytrain import binary_train
from .dimension import Dimension
from .einsum import einsum
from .linearintegerequation import LinearIntegerEquation
from .trainbase import TrainBase
from .trainshape import trainshape
from .utils import namespace_of_trains


def convolution_operator[T: ArrayLike](
    dim1: Dimension, dim2: Dimension
) -> tuple[tuple[Dimension, Dimension, Dimension], LinearIntegerEquation]:
    out_dim = Dimension(dim1.size() + dim2.size() - 1)
    dims = (out_dim, dim1, dim2)
    eq = LinearIntegerEquation(dims, coeffs=(1, -1, -1), rhs=0)
    return dims, eq


def convolve[T: ArrayLike](train1: TrainBase[T], train2: TrainBase[T]) -> TrainBase[T]:
    if len(train1.shape.dims) != 1 or len(train2.shape.dims) != 1:
        raise ValueError("Convolution is only defined here for one-dimensional tensor trains.")

    dims, eq = convolution_operator(train1.shape.dims[0], train2.shape.dims[0])
    xp = namespace_of_trains(train1, train2)
    shape = trainshape(*dims, mode="interleaved_rear")
    op = binary_train(xp, shape, [eq])
    res = einsum("ijk,j,k->i", op, train1, train2)
    if not isinstance(res, TrainBase):
        raise RuntimeError("Unexpected result type from einsum.")
    return res
