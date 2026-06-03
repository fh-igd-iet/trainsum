# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from .backend import ArrayNamespace, ArrayLike, default_ftype
from .dimension import Dimension
from .integerequation import IntegerEquation

class IntegerEquations(IntegerEquation):

    _dims: tuple[Dimension, ...]
    _evaluated: tuple[tuple[None | int, ...], ...]
    _eqs: tuple[IntegerEquation, ...]

    @property
    def dims(self) -> tuple[Dimension, ...]:
        return self._dims
    @property
    def evaluated(self) -> tuple[tuple[None | int, ...], ...]:
        return self._evaluated
    @evaluated.setter
    def evaluated(self, value: tuple[tuple[None | int, ...], ...]):
        self._evaluated = value
        for eq in self._eqs:
            eq.evaluated = value

    @property
    def eqs(self) -> tuple[IntegerEquation, ...]:
        return self._eqs

    def __init__(self, dims: Sequence[Dimension], eqs: Sequence[IntegerEquation]):
        self._dims = tuple(dims)
        tmp = []
        for dim in self.dims:
            tmp.append(tuple(None for _ in range(len(dim))))
        self._evaluated = tuple(tuple(ev) for ev in tmp)
        self._eqs = tuple(eqs)

    def __hash__(self) -> int:
        return hash(tuple([hash(eq) for eq in self.eqs]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntegerEquations):
            return False
        return all(eq1 == eq2 for eq1, eq2 in zip(self.eqs, other.eqs))

    def has_solution(self) -> bool:
        return all(eq.has_solution() for eq in self.eqs)

    def to_tensor[T: ArrayLike](self, xp: ArrayNamespace[T]) -> T:
        tensor = xp.ones(self.tensor_shape())
        for eq in self.eqs:
            tensor *= eq.to_tensor(xp)
        tensor = xp.astype(tensor > 0, default_ftype(xp))
        return tensor

    def inner(self, other: IntegerEquation) -> float:
        if self == other:
            return 1.0
        return 0.0
