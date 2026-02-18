# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from dataclasses import dataclass

from .backend import ArrayLike
from .trainshape import TrainShape
from .trainbase import TrainBase
from .utils import get_shapes
from .einsumequation import EinsumEquation
from .einsumcontraction import EinsumContraction
from .evaluation import Evaluation
from .contractor import OptimizeKind

@dataclass(frozen=True, init=False)
class EvaluationExpression:

    equation: str
    optimizer: OptimizeKind
    chunk_size: int
    _exprs: list[Evaluation]
    _idxs: list[list[int]]
    _slices: list[list[int]]

    def __init__(
            self,
            equation: str,
            *operands: TrainShape | TrainBase,
            optimizer: OptimizeKind = "greedy",
            chunk_size: int = 1024) -> None:
        object.__setattr__(self, "equation", equation)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "chunk_size", chunk_size)
        shapes = get_shapes(*operands)
        org_eq = EinsumEquation(equation, *shapes)
        tmp_idxs, eqs = break_up_equation(org_eq)
        exprs = []
        op_idxs = []
        slices = []
        for idxs, eq in zip(tmp_idxs, eqs):
            contr = EinsumContraction(eq)
            expr = Evaluation(contr, optimizer=optimizer, chunk_size=chunk_size)
            expr.calc_expressions(*[operands[i] for i in idxs])
            idxs = [i for i in idxs if isinstance(operands[i], TrainShape)]
            slc = [org_eq.result.index(char) for char in contr.equation.result]
            slices.append(slc)
            exprs.append(expr)
            op_idxs.append(idxs)
        object.__setattr__(self, "_exprs", exprs)
        object.__setattr__(self, "_idxs", op_idxs)
        object.__setattr__(self, "_slices", slices)

    def __call__[T: ArrayLike](
            self,
            idxs: T,
            *operands: TrainBase[T]) -> T:
        res = self._exprs[0](idxs[self._slices[0], ...], *[operands[i] for i in self._idxs[0]])
        for i, (op_idxs, expr) in enumerate(zip(self._idxs[1:], self._exprs[1:]), start=1):
            res *= expr(idxs[self._slices[i],...], *[operands[i] for i in op_idxs])
        return res

def break_up_equation(eq: EinsumEquation) -> tuple[list[list[int]], Sequence[EinsumEquation]]:
    res_set = set(eq.result)
    op_set = list(eq.operands)
    op_idxs = list(range(len(op_set)))
    eqs = []
    eq_idxs = []
    while len(op_set) != 0:
        idxs = [len(op_set)-1]
        ref = op_set.pop()
        out = [ref]
        for i, op in enumerate(op_set):
            ovlp = set(ref) & set(op)
            if len(ovlp) > 0 and len(ovlp&res_set) != len(ovlp):
                op_set.pop(i)
                op_idxs.pop(i)
                idxs.append(i)
                out.append(op)
        res = "".join(char for char in eq.result if any(char in op for op in out))
        eq_str = ",".join(out) + f"->{res}"

        eqs.append(EinsumEquation(eq_str, *[eq.shapes[i] for i in idxs]))
        eq_idxs.append(idxs)
    return eq_idxs, eqs
