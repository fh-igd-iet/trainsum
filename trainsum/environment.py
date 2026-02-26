# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Generator, Sequence
from types import NoneType
from copy import deepcopy
from dataclasses import dataclass

from .backend import ArrayLike
from .localrange import LocalRange
from .trainshape import TrainShape
from .trainbase import TrainBase
from .einsumcontraction import EinsumContraction
from .contractor import ArrayContractor, OptimizeKind, DEFAULT_OPTIMIZER
from .contractorinput import ContractorInput
from .utils import check_operand_shapes, shape_map, get_shapes

@dataclass(kw_only=True, frozen=True)
class EnvironmentData[T: ArrayLike]:
    left: T
    right: T

class Environment:

    optimizer: OptimizeKind
    _contr: EinsumContraction
    _inp: NoneType | ContractorInput = None
    _to_right_exprs: Sequence[ArrayContractor] = []
    _to_left_exprs: Sequence[ArrayContractor] = []

    def __init__(self, 
                 contr: EinsumContraction,
                 optimizer: OptimizeKind=DEFAULT_OPTIMIZER) -> None:
        if contr.result_shape is not None or contr.full_result_shape is not None:
            raise ValueError("InnerGenerator requires an EinsumContraction without result.")

        self.optimizer = deepcopy(optimizer)
        self._contr = deepcopy(contr)

    def __call__[T: ArrayLike](self, *ops: TrainBase[T], expr: bool = False) -> Generator[EnvironmentData[T], LocalRange]:
        gen = self._gen(*ops, expr=expr)
        next(gen) # warm up
        return gen

    def calc_expressions(self, *ops: TrainShape | TrainBase) -> None:
        check_operand_shapes(self._contr.operand_shapes, get_shapes(*ops))
        self._inp = ContractorInput(*ops)
        self._to_right_exprs = [self._to_right_expression(i, *ops) for i in range(len(self._contr))]
        self._to_left_exprs = [self._to_left_expression(i, *ops) for i in range(len(self._contr))]

    # ------------------------------------------------------------------------
    # Contraction generators

    def _gen(self, *ops: TrainBase, expr: bool = False) -> Generator[EnvironmentData, LocalRange]:
        shapes = get_shapes(*ops)
        if self._inp is None:
            self.calc_expressions(*shapes)
        else:
            self._inp.check_operands(*ops)

        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")
        xp, device, dtype = self._inp.infos(*ops)

        data = EnvironmentData(left=xp.zeros(1), right=xp.zeros(1))
        cache = self._init_cache(xp, device, dtype)
        while True:
            lrange = yield data
            idxs = list(range(lrange.begin, lrange.end))
            for i in idxs:
                cache[i+1] = None

            start = False
            for i in range(idxs[0]):
                start = start or cache[i+1] is None
                if start:
                    cache[i+1] = self._contract_to_right(i, cache, ops, expr=expr)

            start = False
            for i in range(len(self._contr)-1, idxs[-1], -1):
                start = start or cache[i+1] is None
                if start:
                    cache[i+1] = self._contract_to_left(i, cache, ops, expr=expr)

            left, right = cache[idxs[0]], cache[idxs[-1]+2]
            if left is None or right is None:
                raise RuntimeError("Environment data could not be computed.")
            data = EnvironmentData(left=left,
                                   right=right)

    def _init_cache(self, xp, device, dtype) -> list[NoneType | ArrayLike]:
        first_ndim = len(self._contr[0].result.left)
        first = xp.ones([1]*first_ndim, dtype=dtype, device=device)

        last_ndim = len(self._contr[-1].result.right)
        last = xp.ones([1]*last_ndim, dtype=dtype, device=device)

        return [first, *[None]*len(self._contr), last]

    def _contract_to_right(self,
                           idx: int,
                           cache: Sequence[NoneType | ArrayLike],
                           ops: Sequence[TrainBase],
                           expr: bool=False) -> ArrayLike:
        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")
        if not expr:
            tns = self._contr[idx].get_data(*ops, idx_map=self._inp.idx_map)
            return self._to_right_exprs[idx](cache[idx], *tns)
        else:
            tns = self._contr[idx].get_data(*ops, idx_map=self._inp.idx_map)
            to_right_expr = self._to_right_expression(idx, *[op.shape for op in ops])
            return to_right_expr(cache[idx], *tns)

    def _contract_to_left(self,
                          idx: int,
                          cache: Sequence[NoneType | ArrayLike],
                          ops: Sequence[TrainBase],
                          expr: bool=False) -> ArrayLike:
        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")
        if not expr:
            tns = self._contr[idx].get_data(*ops, idx_map=self._inp.idx_map)
            return self._to_left_exprs[idx](*tns, cache[idx+2])
        else:
            tns = self._contr[idx].get_data(*ops, idx_map=self._inp.idx_map)
            to_left_expr = self._to_left_expression(idx, *[op.shape for op in ops])
            return to_left_expr(*tns, cache[idx+2])

    # ------------------------------------------------------------------------
    # Expression builders

    def _to_right_expression(
            self,
            idx: int,
            *ops: TrainShape | TrainBase
            ) -> ArrayContractor:
        lcontr = self._contr[idx]

        left = lcontr.result.left
        right = lcontr.result.right
        eq = f"{left}," + ",".join(str(op) for op in lcontr.operands) + f"->{right}"

        smap = shape_map(ops, lcontr)
        tmp = tuple([smap[char] for char in left])

        const, tns = lcontr.get_constants(*ops)
        const = [val+1 for val in const]
        return ArrayContractor(eq, tmp, *tns, optimizer=self.optimizer)

    def _to_left_expression(
            self, 
            idx: int,
            *ops: TrainShape | TrainBase
            ) -> ArrayContractor:
        lcontr = self._contr[idx]

        left = lcontr.result.left
        right = lcontr.result.right
        eq = ",".join(str(op) for op in lcontr.operands) + f",{right}->{left}"

        smap = shape_map(ops, lcontr)
        tmp = tuple([smap[char] for char in right])

        _, tns = lcontr.get_constants(*ops)
        return ArrayContractor(eq, *tns, tmp, optimizer=self.optimizer)
