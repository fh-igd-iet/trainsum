# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence, overload

from .backend import ArrayLike, ArrayNamespace
from .dimension import Dimension
from .trainshape import trainshape, TrainShape
from .trainbase import TrainBase

from .linearintegerequation import LinearIntegerEquation
from .binarytrain import binary_train

@overload
def shift[T: ArrayLike](
        xp: ArrayNamespace[T],
        dim: TrainShape,
        shift: int | Sequence[int], /) -> TrainBase[T]: ...
@overload
def shift[T: ArrayLike](
        xp: ArrayNamespace[T],
        dim: Dimension,
        shift: int | Sequence[int], /) -> TrainBase[T]: ...
@overload
def shift[T: ArrayLike](
        xp: ArrayNamespace[T],
        dims: tuple[Dimension, Dimension],
        shift: int | Sequence[int], /) -> TrainBase[T]: ...
def shift[T: ArrayLike](
        xp: ArrayNamespace[T],
        dim: Dimension | tuple[Dimension, Dimension] | TrainShape,
        shift: int | Sequence[int], /
        ) -> TrainBase[T]:
    if isinstance(dim, TrainShape):
        shape = dim
        dim = shape.dims[0], shape.dims[1]
    elif isinstance(dim, Dimension):
        dim = dim, Dimension([d.base for d in dim])
        shape = trainshape(*dim, mode="interleaved_rear")
    else:
        shape = trainshape(*dim, mode="interleaved_rear")

    if isinstance(shift, int):
        shift = (shift,)

    eqs = []
    for shift_val in shift:
        eqs.append(LinearIntegerEquation(dim, coeffs=(-1,1), rhs=shift_val))
    bin_train = binary_train(xp, shape, eqs)
    return bin_train
