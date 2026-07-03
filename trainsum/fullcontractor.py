# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from types import NoneType
from copy import deepcopy
from typing import Sequence

from .trainshape import TrainShape
from .trainbase import TrainBase
from .einsumequation import EinsumEquation
from .einsumcontraction import EinsumContraction
from .contractor import ArrayContractor, OptimizeKind
from .utils import check_operand_shapes, get_shapes, namespace_of_trains, shape_map
from .contractorinput import ContractorInput

class FullContractor:
    optimizer: OptimizeKind
    _contr: EinsumContraction
    _inp: NoneType | ContractorInput = None
    _expr: NoneType | Sequence[ArrayContractor] = None

    def __init__(
        self, contr: EinsumContraction, optimizer: OptimizeKind = "greedy"
    ) -> None:
        if contr.result_shape is not None:
            raise ValueError("FullContractor can only be used for full contractions.")

        self.optimizer = deepcopy(optimizer)
        self._contr = self._get_contr(contr)

    def __call__(self, *ops: TrainBase, expr: bool = False) -> float | complex:
        shapes = get_shapes(*ops)
        if expr or self._inp is None:
            self.calc_expressions(*shapes)
        else:
            self._inp.check_operands(*ops)
        return self._contract(*ops)

    def calc_expressions(self, *ops: TrainShape | TrainBase) -> None:
        check_operand_shapes(self._contr.operand_shapes, get_shapes(*ops))
        self._inp = ContractorInput(*ops)
        self._expr = self._expression(*ops)

    def _expression(self, *shapes: TrainShape | TrainBase) -> Sequence[ArrayContractor]:
        exprs = []
        for lcontr in self._contr:
            eq = f"{lcontr.result.left}," + ",".join(str(op) for op in lcontr.operands) + f"->{lcontr.result.right}"
            _, tns = lcontr.get_constants(*shapes)
            smap = shape_map(shapes, lcontr)
            tmp_shape = tuple(smap[char] for char in lcontr.result.left)
            expr = ArrayContractor(eq, tmp_shape, *tns, optimizer=self.optimizer)
            exprs.append(expr)
        return exprs

    def _contract(self, *ops: TrainBase) -> float | complex:
        if self._expr is None or self._inp is None:
            raise RuntimeError("Expression or input cannot be None here.")
        xp = namespace_of_trains(*ops)
        tmp = xp.ones(1)
        for lcontr, expr in zip(self._contr, self._expr):
            tns = lcontr.get_data(*ops, idx_map=self._inp.idx_map)
            tmp = expr(tmp, *tns)

        if xp.isdtype(tmp.dtype, "complex floating"):
            return complex(tmp[0])
        return float(tmp[0])

    def _get_contr(self, contr: EinsumContraction) -> EinsumContraction:
        res = "".join(set(char for op in contr.equation.operands for char in op))
        eq = ",".join(str(op) for op in contr.equation.operands) + f"->{res}"
        neq = EinsumEquation(eq, *contr.operand_shapes)
        return EinsumContraction(neq)
