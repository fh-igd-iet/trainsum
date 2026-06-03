# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .trainbase import TrainBase
from .fuse import fuse

# alias for fuse
def outer[T: ArrayLike](*trains: TrainBase[T]) -> TrainBase[T]:
    return fuse(*trains)
