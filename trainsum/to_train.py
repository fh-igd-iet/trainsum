# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike, namespace_of_arrays, shape
from .trainshape import TrainShape
from .trainbase import TrainBase

def to_train[T: ArrayLike](tshape: TrainShape, data: T) -> TrainBase[T]:
    xp = namespace_of_arrays(data)
    if len(tshape.dims) != len(shape(data)):
        raise ValueError("Data dimensionality does not match the provided TrainShape.")
    if any(dim.size() != s for dim, s in zip(tshape.dims, shape(data))):
        raise ValueError("Data shape does not match the provided TrainShape.")
    ref_digits = []
    for dgts in tshape.digits:
        ref_digits.extend(dgts)
    tshape = TrainShape(tshape.dims, [ref_digits])

    digits = [d for dim in tshape.dims for d in dim]
    ref_digits = sum((list(digits) for digits in tshape.digits), start=[])
    perm = [digits.index(d) for d in ref_digits]

    data = xp.reshape(data, [d.base for dim in tshape.dims for d in dim])
    data = xp.permute_dims(data, perm)
    return TrainBase(tshape, [data[xp.newaxis,...,xp.newaxis]])
