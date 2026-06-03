# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import product
import pulp as pl

from .backend import ArrayNamespace, ArrayLike
from .dimension import Dimension

PulpStandardSolver = pl.COIN_CMD
PulpSolver = pl.GLPK_CMD | pl.COIN_CMD | pl.COPT_CMD | pl.SCIP_CMD | pl.CHOCO_CMD | pl.CPLEX_CMD | pl.FSCIP_CMD | pl.HiGHS_CMD | pl.MIPCL_CMD | pl.GUROBI_CMD | pl.XPRESS_CMD | pl.PULP_CBC_CMD

class IntegerEquation(ABC):

    @property
    def solver(self) -> PulpSolver: ...
    @solver.setter
    def solver(self, value: PulpSolver) -> None:...

    @property
    def dims(self) -> tuple[Dimension, ...]: ...

    @property
    def evaluated(self) -> tuple[tuple[None | int, ...], ...]: ...
    @evaluated.setter
    def evaluated(self, value: tuple[tuple[None | int, ...], ...]): ...

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def has_solution(self) -> bool: ...

    @abstractmethod
    def to_tensor[T: ArrayLike](self, xp: ArrayNamespace[T]) -> T: ...

    @abstractmethod
    def inner(self, other: IntegerEquation) -> float: ...

    def tensor_shape(self) -> tuple[int, ...]:
        shape = []
        for dim, evaled in zip(self.dims, self.evaluated):
            for i, val in enumerate(evaled):
                if val is None:
                    shape.append(dim[i].base)
        return tuple(shape)

    def factorize(
            self, *idxs: Sequence[int],
    ) -> tuple[Sequence[Sequence[int]], Sequence[IntegerEquation]]:
        if len(idxs) != self.ndim:
            raise ValueError("Number of indices must match the number of dimensions.")
        for idx in idxs:
            if isinstance(idx, tuple) and len(idx) != len(set(idx)):
                raise ValueError("Indices in a tuple must be unique.")

        facs = []
        dim_idxs = []
        digit_idxs = []
        for i, idx, dim, ev in zip(range(len(idxs)), idxs, self.dims, self.evaluated):
            if len(idx) == 0:
                dim_idxs.append(None)
                digit_idxs.append(None)
                facs.append(1)
                continue

            for j in idx:
                dim_idxs.append(i)
                digit_idxs.append(j)
                if ev[j] is None:
                    facs.append(dim[j].base)
                else:
                    facs.append(1)
        if sum(facs) == self.ndim:
            raise ValueError("All indices have already been factorized.")

        pos = []
        eqs = []
        for fac_vals in product(*[range(fac) for fac in facs]):
            #print(fac_vals, flush=True)
            neq = deepcopy(self)
            evaled = list(list(ev) for ev in neq.evaluated)
            for dim_idx, digit_idx, val in zip(dim_idxs, digit_idxs, fac_vals):
                if dim_idx is not None:
                    evaled[dim_idx][digit_idx] = val
            neq.evaluated = tuple(tuple(ev) for ev in evaled)

            p = [val for fac, val in zip(facs, fac_vals) if fac > 1]
            if neq.has_solution():
                eqs.append(neq)
                pos.append(p)
        return pos, eqs
