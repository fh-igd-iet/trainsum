# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Protocol
from .backend import ArrayLike

class MatrixEigenvalueDecomposition(Protocol):
    """Protocol for a matrix eigenvalue decomposition."""
    
    def __call__(self, mat: ArrayLike, /) -> tuple[ArrayLike, ArrayLike]:
        """
        Decompose a matrix into its eigenvalues and eigenvectors.
        """
        ...

