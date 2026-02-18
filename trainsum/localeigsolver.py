# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Protocol, Sequence, Callable
from .backend import ArrayLike

class LocalEigSolverResult[T: ArrayLike](Protocol):
    """Protocol for the result of a local eigenvalue solver."""

    #: The eigenvector corresponding to the computed eigenvalue.
    array: T

    #: The computed eigenvalue.
    value: float

class LocalEigSolver[T: LocalEigSolverResult](Protocol):
    """Protocol for a local eigenvalue solver."""

    def __call__[S: ArrayLike](self,
                 mat: Callable[[S], S],
                 guess: S,
                 states: Sequence[S] = [], /
                 ) -> T:
        """
        Solve an eigenvalue problem for a linear map with an initial guess and optional states to orthogonalize against.
        """
        ...
