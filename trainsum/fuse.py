# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .dimension import Dimension
from .trainshape import TrainShape, change_dims
from .trainbase import TrainBase

def fuse[T: ArrayLike](*trains: TrainBase[T]) -> TrainBase[T]:
    """Concatenates trains forming their outer product"""
    if len(trains) < 2:
        raise ValueError("fuse() requires at least two trains")

    all_dims = []
    all_digits = []
    all_data = []

    for train in trains:
        # copy shape and dimensions
        dims = [Dimension([d.base for d in dim]) for dim in train.shape.dims]
        shape = change_dims(train.shape, dims)

        all_dims.extend(dims)
        all_digits.extend(shape.digits)
        all_data.extend(train.data)

    shape = TrainShape(all_dims, all_digits)
    return TrainBase(shape, all_data)
