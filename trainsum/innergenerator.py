# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Generator, Sequence
from types import NoneType
from copy import deepcopy

from .backend import ArrayLike
from .localrange import LocalRange
from .trainshape import TrainShape
from .trainbase import TrainBase
from .einsumcontraction import EinsumContraction
from .operandstring import OperandString
from .localcontraction import LocalContraction
from .environment import Environment
from .sweepingstrategy import SweepingStrategy
from .contractor import ArrayContractor, OptimizeKind, DEFAULT_OPTIMIZER
from .contractorinput import ContractorInput
from .utils import shape_map
from .generatorcallabletype import GeneratorCallableType
from .contains import contains

class InnerGenerator:

    optimizer: OptimizeKind
    _contr: EinsumContraction
    _env: Environment
    _inp: NoneType | ContractorInput
    _target: int
    _cmap: Sequence[int]
    _exprs: dict[tuple[LocalRange, GeneratorCallableType], tuple[LocalContraction, ArrayContractor]]

    def __init__(self,
                 contr: EinsumContraction,
                 target: int,
                 optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> None:

        if contr.result_shape is not None or contr.full_result_shape is not None:
            raise ValueError("InnerGenerator requires an EinsumContraction without result.")
        elif target < 0 or target >= len(contr.operand_shapes):
            raise ValueError("Target index is out of bounds.")

        self.optimizer = deepcopy(optimizer)

        self._target = target
        self._contr = deepcopy(contr)
        ref = self._contr.operand_shapes[self._target]
        res = contains(self._contr.operation_shape, ref)
        self._cmap = res.core_idxs
        self._env = Environment(contr)
        self._exprs = {}
        self._inp = None

    def __call__[T: ArrayLike](
            self,
            *operands: TrainBase[T],
            expr: bool = False) -> Generator[T, tuple[LocalRange, GeneratorCallableType]]:
        gen = self._gen(*operands, calc_expr=expr)
        next(gen) # warm up
        return gen

    def calc_expressions(self,
                         strategy: SweepingStrategy,
                         *ops: TrainShape | TrainBase) -> None:
        self._inp = ContractorInput(*ops)
        ref = ops[self._target]
        ref = ref if isinstance(ref, TrainShape) else ref.shape

        self._exprs.clear()
        for _, lrange in strategy(ref):
            if lrange in self._exprs:
                continue
            self._expression(lrange, GeneratorCallableType.FULL, *ops)
            self._expression(lrange, GeneratorCallableType.LEFT, *ops)
            self._expression(lrange, GeneratorCallableType.RIGHT, *ops)
        self._env.calc_expressions(*ops)

    # ------------------------------------------------------------------------
    # Contraction generators

    def _gen[T: ArrayLike](
            self,
            *ops: TrainBase[T],
            calc_expr: bool = False) -> Generator[T, tuple[LocalRange, GeneratorCallableType]]:
        shapes = [op.shape for op in ops]
        if self._inp is None:
            self._inp = ContractorInput(*shapes)
        self._inp.check_operands(*ops)

        self._env.optimizer = self.optimizer
        env_gen = self._env(*ops, expr=calc_expr)
        data = None
        try:
            while True:
                lrange, gtype = yield data # type: ignore
                trange = self._transform_range(lrange)
                env_data = env_gen.send(trange)
                if calc_expr:
                    self._expression(lrange, gtype, *shapes)
                    loc, expr = self._exprs[lrange, gtype]
                    tns = loc.get_data(*ops, idx_map=self._inp.idx_map)
                else:
                    if (lrange, gtype) not in self._exprs:
                        self._expression(lrange, gtype, *shapes)
                    loc, expr = self._exprs[lrange, gtype]
                    tns = loc.get_data(*ops, idx_map=self._inp.idx_map)

                if gtype == GeneratorCallableType.FULL:
                    data = expr(env_data.left, *tns, env_data.right)
                elif gtype == GeneratorCallableType.LEFT:
                    data = expr(env_data.left, *tns)
                else:  # gtype == GeneratorCallableType.RIGHT
                    data = expr(*tns, env_data.right)

        except GeneratorExit:
            env_gen.close()

    # ------------------------------------------------------------------------
    # Expression builders

    def _expression(
            self,
            lrange: LocalRange,
            gtype: GeneratorCallableType,
            *ops: TrainShape | TrainBase
            ) -> None:
        trange = self._transform_range(lrange)

        left = self._contr[trange.begin].result.left
        right = self._contr[trange.end-1].result.right

        res_str = OperandString()
        lefts, rights, idxs = [], [], []

        tcontr = LocalContraction()
        for lcontr in self._contr[trange.begin:trange.end]:
            zipped = zip(lcontr.train_idxs, lcontr.core_idxs, lcontr.operands)
            for train_idx, core_idx, op_str in zipped:
                flag = lrange.begin <= core_idx < lrange.end
                if train_idx == self._target and flag:
                    res_str.middle += op_str.middle
                    lefts.append(op_str.left)
                    rights.append(op_str.right)
                    idxs.append(core_idx)
                else:
                    tcontr.add_operand(train_idx, core_idx, op_str)

        _, lefts = zip(*sorted(zip(idxs, lefts)))
        _, rights = zip(*sorted(zip(idxs, rights)))
        res_str.left = lefts[0]
        res_str.right = rights[-1]

        smap = shape_map(ops, *self._contr[trange.begin:trange.end])
        left_shape = [smap[char] for char in left]
        right_shape = [smap[char] for char in right]
        _, tns = tcontr.get_constants(*ops)

        if gtype == GeneratorCallableType.FULL:
            res_str.left  = lefts[0]
            res_str.right = rights[-1]
            eq = f"{left}," + ",".join(str(op) for op in tcontr.operands) + f",{right}->{res_str}"
            tns = [left_shape, *tns, right_shape]
        elif gtype == GeneratorCallableType.LEFT:
            res_str.left  = lefts[0]
            res_str.right = right[1:]
            eq = f"{left}," + ",".join(str(op) for op in tcontr.operands) + f"->{res_str}"
            tns = [left_shape, *tns]
        else:  # gtype == GeneratorCallableType.RIGHT
            res_str.left  = left[1:]
            res_str.right = rights[-1]
            eq = ",".join(str(op) for op in tcontr.operands) + f",{right}->{res_str}"
            tns = [*tns, right_shape]

        tcontr.set_result(res_str)
        contr = ArrayContractor(eq, *tns, optimizer=self.optimizer)
        self._exprs[lrange, gtype] = tcontr, contr

    def _transform_range(self, lrange: LocalRange) -> LocalRange:
        begin, end = self._cmap[lrange.begin], self._cmap[lrange.end-1]+1
        return LocalRange(begin=begin, end=end)
