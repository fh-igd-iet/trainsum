# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from types import NoneType
from copy import deepcopy

from .trainshape import TrainShape
from .trainbase import TrainBase
from .einsumcontraction import EinsumContraction
from .contractor import ArrayContractor, OptimizeKind
from .utils import check_operand_shapes, get_shapes
from .contractorinput import ContractorInput

class FullContractor:

    optimizer: OptimizeKind
    _contr: EinsumContraction
    _inp: NoneType | ContractorInput = None
    _expr: NoneType | ArrayContractor = None

    def __init__(self,
                 contr: EinsumContraction, 
                 optimizer: OptimizeKind = "greedy") -> None:
        if contr.result_shape is not None:
            raise ValueError("FullContractor can only be used for full contractions.")

        self._contr = deepcopy(contr)
        self.optimizer = deepcopy(optimizer)

    def __call__(self, *ops: TrainBase, expr: bool = False) -> float:
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

    def _expression(self, *shapes: TrainShape | TrainBase) -> ArrayContractor:
        lcontr = self._contr[0]
        eq = ",".join(str(op) for op in lcontr.operands) + f"->{lcontr.result}"
        _, tns = lcontr.get_constants(*shapes)
        expr = ArrayContractor(eq, *tns, optimizer=self.optimizer)
        return expr

    def _contract(self, *ops: TrainBase) -> float:
        if self._expr is None or self._inp is None:
            raise RuntimeError("Expression or input cannot be None here.")
        tns = self._contr[0].get_data(*ops, idx_map=self._inp.idx_map)
        res = self._expr(*tns)
        return float(res[0,0])
