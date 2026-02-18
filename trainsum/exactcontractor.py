# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from types import NoneType
from copy import deepcopy
from math import prod

from .backend import ArrayLike
from .trainshape import TrainShape
from .trainbase import Normalization, TrainBase
from .einsumcontraction import EinsumContraction
from .contractor import ArrayContractor, OptimizeKind
from .contractorinput import ContractorInput
from .utils import check_operand_shapes, get_shapes

class ExactContractor:

    optimizer: OptimizeKind
    _contr: EinsumContraction
    _inp: NoneType | ContractorInput = None
    _exprs: NoneType | Sequence[ArrayContractor] = None

    def __init__(self,
                 contr: EinsumContraction,
                 optimizer: OptimizeKind = "greedy") -> None:
        if contr.result_shape is None:
            raise ValueError("ExactContractor requires a result shape.")

        self.optimizer = deepcopy(optimizer)
        self._contr = deepcopy(contr)

    def __call__[T: ArrayLike](
            self,
            *ops: TrainBase[T],
            expr: bool = False) -> TrainBase[T]:
        #get_device_dtype(ops)
        shapes = get_shapes(*ops)
        if expr or self._inp is None or self._exprs is None:
            self.calc_expressions(*shapes)
        else:
            self._inp.check_operands(*ops)
        return self._contract(*ops, expr=expr)

    def calc_expressions(self, *ops: TrainShape | TrainBase) -> None:
        check_operand_shapes(self._contr.operand_shapes, get_shapes(*ops))
        self._inp = ContractorInput(*ops)
        self._exprs = [self._expression(i, *ops) for i in range(len(self._contr))]

    def _expression(self, idx: int, *shapes: TrainShape | TrainBase) -> ArrayContractor:
        lcontr = self._contr[idx]
        eq = ",".join(str(op) for op in lcontr.operands) + f"->{lcontr.result}"
        tns = lcontr.get_shapes(*shapes)
        _, tns = lcontr.get_constants(*shapes)
        expr = ArrayContractor(eq, *tns, optimizer=self.optimizer)
        return expr

    def _contract(self, *ops: TrainBase, expr: bool = False) -> TrainBase:
        if self._contr.full_result_shape is None:
            raise ValueError("Contraction needs a result.")
        elif self._inp is None or self._exprs is None:
            raise RuntimeError("Expression or input cannot be None here.")

        xp, _, _ = self._inp.infos(*ops)

        core_data = []
        for i, contr in enumerate(self._contr):
            if expr:
                tns = contr.get_data(*ops)
                data = self._expression(i, *ops)(*tns)
            else:
                tns = contr.get_data(*ops, idx_map=self._inp.idx_map)
                data = self._exprs[i](*tns)

            dleft = len(contr.result.left)
            dright = len(contr.result.right)
            left_shape = prod(data.shape[:dleft])
            right_shape = prod(data.shape[-dright:])
            middle_shape = data.shape[dleft:-dright]
            data = xp.reshape(data, (left_shape, *middle_shape, right_shape))
            core_data.append(data)
        norm = [Normalization.NONE] * len(core_data)
        return TrainBase(self._contr.full_result_shape, core_data,
                         norm=norm, copy_data=False)
