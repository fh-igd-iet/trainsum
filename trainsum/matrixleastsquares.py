# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Protocol
from .backend import ArrayLike

class MatrixLeastSquares(Protocol):
    """Protocol for a matrix least squares solver."""
    
    def __call__(self, A: ArrayLike, B: ArrayLike, /) -> ArrayLike:
        """Solve the least squares problem for the given matrices A and B, returning the solution X that minimizes ||AX - B||."""
        ...


