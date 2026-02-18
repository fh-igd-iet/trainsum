# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .backend import namespace_of_arrays
from .matrixdecomposition import MatrixDecompositionResult

class QRDecomposition[T: ArrayLike]:
    """
    QR decomposition. Decomposes a matrix into an orthonormal matrix and an upper triangular matrix.
    """

    def right(self, mat: T) -> MatrixDecompositionResult[T]:
        """Calculate :math:`M=QR` and return :math:`Q` and :math:`R`."""
        q, r = self._qr(mat.T)
        #return r.T, q.T
        return MatrixDecompositionResult(left=r.T, right=q.T)

    def left(self, mat: T) -> MatrixDecompositionResult[T]:
        """Calculate :math:`M=LQ` and return :math:`L` and :math:`Q`."""
        q, r = self._qr(mat)
        return MatrixDecompositionResult(left=q, right=r)

    def _qr(self, mat: T) -> tuple[T, T]:
        xp = namespace_of_arrays(mat)
        if not hasattr(xp, "linalg"):
            raise NotImplementedError("Linalg extension missing on this backend, implement your own QRDecomposition!.")
        q, r = xp.linalg.qr(mat, mode='reduced')
        return q, r

    def left_shape(self, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Calculate the shape of the left function."""
        m, n = shape
        k = min(m, n)
        return (m, k), (k, n)

    def right_shape(self, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Calculate the shape of the right function."""
        m, n = shape
        k = min(m, n)
        return (m, k), (k, n)

    def __repr__(self) -> str:
        return f"QRDecomposition()"
