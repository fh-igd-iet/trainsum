# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Any, Sequence
from abc import abstractmethod
from dataclasses import dataclass

from .backend import ArrayLike
from .dimension import Dimension

@dataclass(frozen=True, init=False)
class Grid:

    ndims: int
    dims: Sequence[Dimension]

    def index_dtype(self) -> Any: ...

    @abstractmethod
    def to_coords[T: ArrayLike](self, idxs: T) -> T: ...
    @abstractmethod
    def to_idxs[T: ArrayLike](self, coords: T) -> T: ...
