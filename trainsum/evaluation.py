# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from math import prod

from .localcontraction import LocalContraction
from .backend import ArrayLike, shape_size, size, namespace_of_arrays
from .dimension import Dimension
from .digit import Digit, Digits
from .trainshape import TrainShape
from .trainbase import TrainBase
from .einsumcontraction import EinsumContraction, get_symbol_generator
from .contractorinput import ContractorInput
from .utils import check_operand_shapes, get_shapes, shape_map
from .coreslicing import CoreSlicing
from .contractor import ArrayContractor, OptimizeKind

class Evaluation:

    optimizer: OptimizeKind
    chunk_size: int
    _contr: EinsumContraction
    _inp: None | ContractorInput
    _tmp_str: str
    _exprs: Sequence[ArrayContractor]
    _core_slicing: Sequence[CoreSlicing]
    _res_digits: Sequence[Digits]

    def __init__(
            self,
            contr: EinsumContraction,
            optimizer: OptimizeKind = "greedy",
            chunk_size: int = 1024) -> None:
        self._contr = contr
        self.optimizer = optimizer
        self.chunk_size = chunk_size
        self._tmp_str = next(get_symbol_generator(contr))

        self._inp = None
        self._exprs = []
        self._core_slicing = []
        self._res_digits = []

    def __call__[T: ArrayLike](
            self,
            idxs: T,
            *ops: TrainBase[T],
            expr: bool = False) -> T:
        #idxs_ = {dim: idxs[char] for char, dim in zip(self._contr.equation.result, self._contr.equation.result_dims)}
        idxs_ = {dim: idxs[i] for i, dim in enumerate(self._contr.equation.result_dims)}
        self._check_idxs(idxs_)

        shapes = get_shapes(*ops)
        if expr or self._inp is None:
            self.calc_expressions(*shapes)
        else:
            self._inp.check_operands(*ops)

        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")
        xp, device, dtype = self._inp.infos(*ops)
        dmap = _digit_map(idxs_)

        shape = next(iter(idxs_.values())).shape
        length = shape_size(shape)
        res = xp.empty(length, device=device, dtype=dtype)
        for i in range(length // self.chunk_size + 1):
            start = i * self.chunk_size
            stop = min((i+1) * self.chunk_size, length)
            if start >= stop:
                break
            chunk_idxs = {digit: vals[start:stop] for digit, vals in dmap.items()}
            res[start:stop] = self._contract(chunk_idxs, *ops)

        return xp.reshape(res, shape)

    def calc_expressions(self, *operands: TrainShape | TrainBase) -> None:
        shapes = get_shapes(*operands)
        check_operand_shapes(self._contr.operand_shapes, shapes)

        self._const_ops = [op for op in operands if not isinstance(op, TrainShape)]
        self._inp = ContractorInput(*operands)

        self._exprs = []
        self._core_slicing = []
        self._res_digits = []
        for lcontr in self._contr:
            self._exprs.append(self._expression(lcontr, *shapes))
            self._core_slicing.append(CoreSlicing(lcontr, *operands))
            self._res_digits.append(_result_digits(lcontr, *self._contr.operand_shapes))

    # ------------------------------------------------------------------------
    # Contraction generators

    def _contract[T: ArrayLike](
            self,
            dmap: dict[Digit, T],
            *ops: TrainBase[T]) -> T:
        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")
        xp, device, dtype = self._inp.infos(*ops)
        length = size(next(iter(dmap.values())))

        res = xp.ones((length, *[1]*len(self._contr[0].result.left)),
                      device=device, dtype=dtype)
        tmp = xp.empty(length, device=device, dtype=dtype)
        for expr, slc, digits in zip(self._exprs, self._core_slicing, self._res_digits):
            new_res = xp.empty((length, *slc.get_right_shape(*ops)),
                               device=device, dtype=dtype)
            tmp = _get_ids(dmap, digits, tmp)
            for i in range(prod(d.base for d in digits)):
                mask = xp.where(tmp == i, True, False)
                if size(mask) == 0:
                    continue
                mats = slc.get_matrices(i, *ops)
                new_res[mask,...] = expr(res[mask], *mats)
            res = new_res

        return res[:,0]

    # ------------------------------------------------------------------------
    # Expression builders

    def _expression(self, lcontr: LocalContraction, *shapes: TrainShape) -> ArrayContractor:
        tmp = f"{self._tmp_str}{lcontr.result.left}"
        res = f"{self._tmp_str}{lcontr.result.right}"
        ops = []
        tns = []
        for op, tn in zip(lcontr.operands, lcontr.get_shapes(*shapes)):
            new_mid = ""
            new_shape = [tn[0]]
            for i, char in enumerate(op.middle):
                if char in lcontr.result.middle:
                    continue
                new_mid += char
                new_shape.append(tn[i+1])
            new_shape.append(tn[-1])
            ops.append(f"{op.left}{new_mid}{op.right}")
            tns.append(new_shape)

        eq = f"{tmp}," + ",".join(op for op in ops) + f"->{res}"

        smap = shape_map(shapes, lcontr)
        smap[self._tmp_str] = self.chunk_size
        tmp_shape = [smap[char] for char in tmp]

        return ArrayContractor(eq, tmp_shape, *tns,
                               optimizer=self.optimizer)

    def _check_idxs[T: ArrayLike](self, idxs: dict[Dimension, T]) -> None:
        if not all(dim in self._contr.equation.result_dims for dim in idxs.keys()):
            raise ValueError("All dimensions in idxs must be in the result dimensions of the equation.")
        if len(set(idxs[dim].shape for dim in idxs.keys())) != 1:
            raise ValueError("All index arrays must have the same .")

def _digit_map[T: ArrayLike](idxs: dict[Dimension, T]) -> dict[Digit, T]:
    xp = namespace_of_arrays(next(iter(idxs.values())))
    dmap = {}
    for dim, didxs in idxs.items():
        vals = dim.to_digits(didxs)
        for i, digit in enumerate(dim):
            length = size(vals[i])
            dmap[digit] = xp.reshape(vals[i,...], (length,))
    return dmap

def _get_ids[T: ArrayLike](dmap: dict[Digit, T], digits: Digits, x: T) -> T:
    ncombs = prod(digit.base for digit in digits)
    x[...] = 0
    for digit in digits:
        ncombs //= digit.base
        x += dmap[digit]*ncombs
    return x

def _result_digits(lcontr: LocalContraction, *ops: TrainShape) -> Digits:
    dmap = {}
    for op_str, tidx, cidx in zip(lcontr.operands, lcontr.train_idxs, lcontr.core_idxs):
        digits = ops[tidx].digits[cidx]
        for char, digit in zip(op_str.middle, digits):
            if digit not in dmap:
                dmap[char] = digit
    return [dmap[char] for char in lcontr.result.middle]
