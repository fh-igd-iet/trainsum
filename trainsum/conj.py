# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .trainbase import TrainBase
from .utils import namespace_of_trains

def conj[T: ArrayLike](train: TrainBase[T]) -> TrainBase[T]:
    xp = namespace_of_trains(train)
    data_conj = [xp.conj(d) for d in train.data]
    return TrainBase(train.shape, data_conj)
