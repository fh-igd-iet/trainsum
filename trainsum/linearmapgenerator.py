# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Callable, Generator
from copy import deepcopy

from .backend import ArrayLike
from .localrange import LocalRange
from .trainshape import TrainShape, change_dims
from .trainbase import TrainBase
from .einsumequation import EinsumEquation
from .einsumcontraction import EinsumContraction
from .operandstring import OperandString
from .localcontraction import LocalContraction
from .environment import Environment
from .contractor import ArrayContractor, OptimizeKind, DEFAULT_OPTIMIZER
from .sweepingstrategy import SweepingStrategy
from .utils import shape_map, get_shapes, check_operand_shapes
from .contractorinput import ContractorInput
from .generatorcallabletype import GeneratorCallableType
from .contains import contains

class LinearMapGenerator[T: ArrayLike]:

    @property
    def result_shape(self) -> TrainShape:
        return self._ref_shape

    optimizer: OptimizeKind
    _contr: EinsumContraction
    _target: int
    _inp: None | ContractorInput = None
    _ref_shape: TrainShape
    _exprs: dict[tuple[LocalRange, GeneratorCallableType], tuple[LocalContraction, ArrayContractor]]

    def __init__(
            self,
            contr: EinsumContraction,
            target: int,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> None:

        self.optimizer = deepcopy(optimizer)
        self._contr = self._get_contr(contr, target)
        self._env = Environment(self._contr)
        self._target = target
        self._ref_shape = self._contr.operand_shapes[0]

        res = contains(self._contr.operation_shape, self._contr.operand_shapes[0])
        self._lcmap = res.core_idxs
        res = contains(self._contr.operation_shape, self._contr.operand_shapes[-1])
        self._rcmap = res.core_idxs
        if self._lcmap != self._rcmap:
            raise NotImplementedError("Only linear maps, where the input digits reside on the "\
                                      "same cores as the output digits are currently supported.")
        self._exprs = {}

    def __call__(
            self,
            *ops: TrainBase[T],
            expr: bool = False
            ) -> Generator[Callable[[T], T], tuple[LocalRange, GeneratorCallableType]]:
        gen = self._gen(*ops, calc_expr=expr)
        next(gen)  # warm up
        return gen

    def calc_expressions(self, strat: SweepingStrategy, *ops: TrainShape | TrainBase) -> None:
        ops_ = list(ops)
        guess = ops_.pop(self._target)
        if not isinstance(guess, TrainShape):
            guess = guess.shape
        ops_ = [guess, *ops_ , guess]
        shapes = get_shapes(*ops_)
        check_operand_shapes(self._contr.operand_shapes, shapes)

        self._inp = ContractorInput(*ops_)
        self._exprs = {}
        for _, lrange in strat(guess):
            if lrange not in self._exprs:
                self._expression(lrange, GeneratorCallableType.FULL, *ops_)
                self._expression(lrange, GeneratorCallableType.LEFT, *ops_)
                self._expression(lrange, GeneratorCallableType.RIGHT, *ops_)
        self._env.calc_expressions(*ops_)
    
    
    # ------------------------------------------------------------------------
    # Contraction generators

    def _gen(
            self,
            *ops: TrainBase[T],
            calc_expr: bool = False
            ) -> Generator[Callable[[T], T], tuple[LocalRange, GeneratorCallableType]]:
        ops_ = list(ops)
        guess = ops_.pop(self._target)
        ops_ = [guess, *ops_ , guess]
        shapes = get_shapes(*ops_)
        if self._inp is None:
            self._inp = ContractorInput(*shapes)
        self._inp.check_operands(*ops_)

        env_gen = self._env(*ops_)
        func = lambda x: x
        try:
            while True:
                lrange, mtype = yield func
                if calc_expr or lrange not in self._exprs:
                    self._expression(lrange, mtype, *shapes)
                env_data = env_gen.send(lrange)
                tcontr, expr = self._exprs[lrange, mtype]
                tns = tcontr.get_data(*ops_, idx_map=self._inp.idx_map)
                if mtype == GeneratorCallableType.FULL:
                    func = lambda x: expr(env_data.left, *tns, x, env_data.right)
                elif mtype == GeneratorCallableType.LEFT:
                    func = lambda x: expr(env_data.left, *tns, x)
                else:  # mtype == GeneratorCallableType.RIGHT:
                    func = lambda x: expr(*tns, x, env_data.right)
        finally:
            env_gen.close()

    # ------------------------------------------------------------------------
    # Expression builders

    def _expression(self, lrange: LocalRange, mtype: GeneratorCallableType, *ops: TrainShape | TrainBase) -> None:
        shapes = get_shapes(*ops)
        trange = self._transform_range(lrange)

        left = self._contr[trange.begin].result.left
        right = self._contr[trange.end-1].result.right

        inp_str = OperandString(left=left[-1], right=right[-1])
        res_str = OperandString()
        tcontr = LocalContraction()
        for lcontr in self._contr[trange.begin:trange.end]:
            zipped = zip(lcontr.train_idxs, lcontr.core_idxs, lcontr.operands)
            for train_idx, core_idx, op_str in zipped:
                flag = lrange.begin <= core_idx < lrange.end
                if train_idx == 0 and flag:
                    res_str.middle += op_str.middle
                elif train_idx == len(self._contr.operand_shapes)-1 and flag:
                    inp_str.middle += op_str.middle
                else:
                    tcontr.add_operand(train_idx, core_idx, op_str)

        smap = shape_map(shapes, *self._contr[trange.begin:trange.end])
        left_shape = [smap[char] for char in left]
        right_shape = [smap[char] for char in right]
        inp_shape = [smap[char] for char in str(inp_str)]
        _, tns = tcontr.get_constants(*ops)

        if mtype == GeneratorCallableType.FULL:
            res_str.left  = left[0]
            res_str.right = right[0]
            eq = f"{left}," + ",".join(str(op) for op in tcontr.operands) + f",{inp_str},{right}->{res_str}"
            tns = [left_shape, *tns, inp_shape, right_shape]
        elif mtype == GeneratorCallableType.LEFT:
            res_str.left  = left[0]
            res_str.right = right[1:]
            eq = f"{left}," + ",".join(str(op) for op in tcontr.operands) + f",{inp_str}->{res_str}"
            tns = [left_shape, *tns, inp_shape]
        else:  # mtype == GeneratorCallableType.RIGHT
            res_str.left  = left[1:]
            res_str.right = right[0]
            eq = f",".join(str(op) for op in tcontr.operands) + f",{inp_str},{right}->{res_str}"
            tns = [*tns, inp_shape, right_shape]

        #tcontr.set_result(res_str)
        expr = ArrayContractor(eq, *tns, optimizer=self.optimizer)
        self._exprs[lrange, mtype] = tcontr, expr

    def _transform_range(self, lrange: LocalRange) -> LocalRange:
        lbegin, lend = self._lcmap[lrange.begin], self._lcmap[lrange.end-1]+1
        rbegin, rend = self._rcmap[lrange.begin], self._rcmap[lrange.end-1]+1
        begin = min(lbegin, rbegin)
        end = max(lend, rend)
        return LocalRange(begin=begin, end=end)

    def _get_contr(self, contr: EinsumContraction, target: int) -> EinsumContraction:
        ops = [*contr.operand_shapes]
        ref = ops.pop(target)
        ops = [ref, *ops, ref]
        op_strs = [str(op) for op in contr.equation.operands]
        ref_op_str = op_strs.pop(target)
        eq = ",".join([contr.equation.result, *op_strs, ref_op_str]) + "->"
        eq = EinsumEquation(eq, *ops)

        dim_map = {}
        for in_shape, out_shape in zip(ops, eq.shapes):
            for in_dim, out_dim in zip(in_shape.dims, out_shape.dims):
                dim_map[in_dim] = out_dim
        new_op_shape_dims = [dim_map[dim] for dim in contr.operation_shape.dims]
        op_shape = change_dims(contr.operation_shape, new_op_shape_dims)
        return EinsumContraction(eq, op_shape=op_shape)

    #def _get_ops(self, *ops: TrainShape)
