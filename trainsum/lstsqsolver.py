# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike, namespace_of_arrays
import array_api_compat

class LstsqSolver:
    def __call__[T: ArrayLike](self, A: T, b: T) -> T:
        if array_api_compat.is_cupy_array(A):
            import cupy as cp
            return cp.linalg.lstsq(A, b)[0] # type: ignore

        xp = namespace_of_arrays(A, b)
        if not (hasattr(xp, "linalg")) or not (hasattr(xp.linalg, "lstsq")):
            raise NotImplementedError(
                f"Method linalg.lstsq is not implemented for backend {xp}.")
        return xp.linalg.lstsq(A, b)[0] # type: ignore
