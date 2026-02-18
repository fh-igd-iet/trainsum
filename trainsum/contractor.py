# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Callable, Sequence, Literal
from types import NoneType
from copy import deepcopy
import opt_einsum as oe
#from opt_einsum.typing import OptimizeKind
from .backend import ArrayLike, shape

OptimizeKind = Literal["optimal", "dp", "greedy", "random-greedy", "random-greedy-128", "branch-all", "branch-2", "auto", "auto-hq"]
DEFAULT_OPTIMIZER: OptimizeKind = "greedy"

class ArrayContractor[T: ArrayLike]:

    _ops: list[NoneType | ArrayLike]
    _shapes: Sequence[Sequence[int]]
    _opt: OptimizeKind
    _idxs: Sequence[int]
    _expr: Callable[..., ArrayLike]

    def __init__(
            self,
            eq: str,
            *ops: Sequence[int] | T,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> None:
        self._shapes = []
        self._ops = []
        self._idxs = []
        for i, op in enumerate(ops):
            if isinstance(op, Sequence):
                self._shapes.append(op)
                self._ops.append(None)
                self._idxs.append(i)
            else:
                self._shapes.append(shape(op))
                self._ops.append(deepcopy(op))
        if len(self._idxs) == 0:
            res = oe.contract(eq, *self._ops, optimize=optimizer)
            self._expr = lambda: res
        else:
            self._expr = oe.contract_expression(
                eq,
                *self._shapes,
                optimize=optimizer)
            #print(str(self._expr).split("\n")[0])

    def __call__(self, *ops: T) -> T:
        if len(ops) != len(self._idxs):
            raise ValueError("Number of provided operands does not match the number required operands.")
        if len(self._idxs) == 0:
            return self._expr() # type: ignore
        for idx, op in zip(self._idxs, ops):
            self._ops[idx] = op
        return self._expr(*self._ops) # type: ignore
