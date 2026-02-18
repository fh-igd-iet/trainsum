# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .dimension import Dimension
from .trainshape import TrainShape, change_dims
from .trainbase import TrainBase

def fuse[T: ArrayLike](train1: TrainBase[T], train2: TrainBase[T]) -> TrainBase[T]:
    dims1 = [Dimension([d.base for d in dim]) for dim in train1.shape.dims]
    shape1 = change_dims(train1.shape, dims1)

    dims2 = [Dimension([d.base for d in dim]) for dim in train2.shape.dims]
    shape2 = change_dims(train2.shape, dims2)

    dims = [*dims1, *dims2]
    digits = [*shape1.digits, *shape2.digits]

    shape = TrainShape(dims, digits)
    data = [*train1.data, *train2.data]
    return TrainBase(shape, data)
