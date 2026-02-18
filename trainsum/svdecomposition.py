# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Any
from dataclasses import dataclass
from .backend import ArrayLike, namespace_of_arrays
from .matrixdecomposition import MatrixDecompositionResult
from .utils import check_non_neg, check_pos

@dataclass(kw_only=True)
class SVDecompositionResult[T: ArrayLike](MatrixDecompositionResult[T]):
    #: Left matrix of the decomposition.
    left: T
    #: Right matrix of the decomposition.
    right: T
    #: Singular values.
    singular_values: T

@dataclass(kw_only=True)
class SVDecomposition[T: ArrayLike]:
    """
    Singular value decomposition. The number of singular values to keep is determined by max_rank
    and cutoff. All singular values below cutoff are discarded, and at most max_rank singular values are kept.
    """

    max_rank: int = 50
    cutoff: float = 0.0

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "max_rank" and value is not None:
            check_pos(name, value)
        elif name == "cutoff":
            check_non_neg(name, value)
        super().__setattr__(name, value)

    def right(self, mat: T) -> SVDecompositionResult[T]:
        """Calculate :math:`U \\Sigma V^H` and return :math:`U \\Sigma` and :math:`V^H`."""
        xp = namespace_of_arrays(mat)
        u, s, vh = self._svd(mat)
        u = u * s[xp.newaxis,:]
        return SVDecompositionResult(left=u, right=vh, singular_values=s)

    def left(self, mat: T) -> SVDecompositionResult[T]:
        """Calculate :math:`U \\Sigma V^H` and return :math:`U` and :math:`\\Sigma V^H`."""
        xp = namespace_of_arrays(mat)
        u, s, vh = self._svd(mat)
        vh = s[:,xp.newaxis] * vh
        return SVDecompositionResult(left=u, right=vh, singular_values=s)

    def _svd(self, mat: T) -> tuple[T, T, T]:
        xp = namespace_of_arrays(mat)
        if not hasattr(xp, "linalg"):
            raise NotImplementedError("Linalg extension missing on this backend, implement your own SVDecomposition!.")
        u, s, vh = xp.linalg.svd(mat, full_matrices=False)
        numel = max(1, min(int(xp.sum(s > self.cutoff)), self.max_rank))
        return u[:,:numel], s[:numel], vh[:numel,:]

    def left_shape(self, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Calculate the shape of the left function."""
        m, n = shape
        k = min(m, n, self.max_rank)
        return (m, k), (k, n)

    def right_shape(self, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Calculate the shape of the right function."""
        m, n = shape
        k = min(m, n, self.max_rank)
        return (m, k), (k, n)

    def __repr__(self) -> str:
        return f"SVDecomposition(max_rank={self.max_rank}, cutoff={self.cutoff})"
