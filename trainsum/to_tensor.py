# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike, namespace_of_arrays
from .trainshape import TrainShape
from .trainbase import TrainBase
from .utils import namespace_of_trains

def to_tensor[T: ArrayLike](train: TrainBase[T]) -> T:
    xp = namespace_of_trains(train)
    data = xp.ones(1, device=train.device, dtype=train.dtype)
    for tdata in train.data:
        idx = len(data.shape)-1
        data = xp.tensordot(data, tdata, axes=([idx], [0]))
    data = xp.reshape(data, data.shape[:-1])
    data = digit_back_permutation(train.shape, data)
    return data

def digit_back_permutation[T: ArrayLike](tshape: TrainShape, data: T) -> T:
    xp = namespace_of_arrays(data)
    flat = sum((list(tshape.middle(i)) for i in range(len(tshape))), start=[])
    if tuple(flat) != tuple(data.shape):
        raise ValueError("Data shape does not match train cores")
    digits = [d for dim in tshape.dims for d in dim]
    ref_digits = sum((list(digits) for digits in tshape.digits), start=[])
    perm = [ref_digits.index(d) for d in digits]

    data = xp.reshape(data, [digit.base for digits in tshape.digits for digit in digits])
    new_shape = [dim.size() for dim in tshape.dims]
    data = xp.permute_dims(data, perm)
    return xp.reshape(data, new_shape)
