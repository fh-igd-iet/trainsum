# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Protocol
from dataclasses import dataclass
from .backend import ArrayLike

@dataclass(kw_only=True)
class MatrixDecompositionResult[T: ArrayLike]:
    #: Left matrix of the decomposition.
    left: T
    #: Right matrix of the decomposition.
    right: T

class MatrixDecomposition[T: ArrayLike, S: MatrixDecompositionResult](Protocol):

    def right(self, mat: T) -> S: ...
    """Decompose input matrix into some left matrix and a right orthonormal matrix"""

    def right_shape(self, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]: ...
    """Given the shape of an input matrix, return the shape of the output matrices"""

    def left(self, mat: T) -> S: ...
    """Decompose input matrix into a left orthonormal matrix and some right matrix"""

    def left_shape(self, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]: ...
    """Given the shape of an input matrix, return the shape of the output matrices"""
