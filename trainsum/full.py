# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .trainshape import TrainShape
from .trainbase import TrainBase
from .backend import ArrayNamespace, ArrayLike, ArrayNamespace

def full[T: ArrayLike](xp: ArrayNamespace[T], shape: TrainShape, value: float) -> TrainBase[T]:
    data = []
    for i in range(len(shape)):
        data.append(xp.ones((1, *shape.middle(i), 1)))
    data[0] = value * data[0]
    return TrainBase(shape, data, copy_data=False)
