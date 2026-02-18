# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Protocol, Callable, Optional
from .backend import ArrayLike

class LocalLinSolverResult[T: ArrayLike](Protocol):
    """Protocol for the result of a local linear solver."""

    #: The solution array.
    array: T

    #: The residuals of the linear solver.
    residuals: list[float]

class LocalLinSolver[T: LocalLinSolverResult](Protocol):
    """Protocol for a local linear solver."""

    def __call__[S: ArrayLike](
        self,
        mat: Callable[[S], S],
        rhs: S,
        guess: Optional[S] = None, /
        ) -> T:
        """
        Solve a linear problem for a linear map with a right-hand side and an optional initial guess.
        """
        ...
