# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike, namespace_of_arrays

class EigHSolver:
    def __call__[T: ArrayLike](self, mat: T) -> tuple[T, T]:
        xp = namespace_of_arrays(mat)
        if not hasattr(xp, "linalg"):
            raise NotImplementedError(
                f"Extension linalg is missing from namespace {xp}.")
        vals, vecs = xp.linalg.eigh(mat)
        return vals, vecs
