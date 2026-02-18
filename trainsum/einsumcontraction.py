# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Optional, Sequence, Generator
from types import NoneType
from dataclasses import dataclass
from copy import deepcopy

from .sequenceof import SequenceOf
from .digit import Digit
from .trainshape import TrainShape, change_dims
from .utils import symbol_generator
from .einsumequation import EinsumEquation
from .operationspace import einsum_operation_shape, einsum_result_shape
from .operandstring import OperandString
from .localcontraction import LocalContraction
from .contains import contains

@dataclass(frozen=True)
class EinsumContraction(SequenceOf[LocalContraction]):
    equation: EinsumEquation
    operation_shape: TrainShape
    operand_shapes: Sequence[TrainShape]
    full_result_shape: NoneType | TrainShape
    result_shape: NoneType | TrainShape

    def __init__(self,
                 eq: EinsumEquation,
                 op_shape: Optional[TrainShape] = None,
                 result: Optional[TrainShape] = None) -> None:
        if op_shape is None:
            op_shape = einsum_operation_shape(eq)
        full_res_shape = einsum_result_shape(eq, op_shape)
        op_shape, full_res_shape, res_shape = _get_result(op_shape, full_res_shape, result)

        trains = []
        core_maps = []
        reverse = []
        for train in eq.shapes:
            res = contains(op_shape, train)
            rev, cmap = res.reverse, res.core_idxs
            if len(cmap) == 0:
                raise ValueError("Train cores do not align with operation space.")
            trains.append(train.reverse() if rev else deepcopy(train))
            core_maps.append(cmap)
            reverse.append(rev)

        sgen = symbol_generator()
        digit_sym_map = _digit_symbol_map(eq.operands, trains, sgen)
        train_syms = [_train_symbols(op, digit_sym_map, sgen) for op in trains]

        locs = []
        is_none = full_res_shape is None
        first_left, last_right = next(sgen), next(sgen)
        for i in range(len(op_shape)):
            lcontr = LocalContraction()
            res_str = OperandString()
            perm = []
            zipped = zip(core_maps, trains, train_syms, reverse)
            for train_idx, (cmap, train, syms, rev) in enumerate(zipped):
                idxs = [j for j, idx in enumerate(cmap) if i == idx]
                for j, idx in enumerate(idxs):

                    sym = syms[idx]
                    if i == 0 and idx == 0:
                        sym.left = first_left
                    if i == len(op_shape)-1 and idx == len(train)-1:
                        sym.right = last_right

                    lcontr.add_operand(train_idx,
                                       idx if not rev else len(train)-1-idx,
                                       sym.reverse() if rev else sym)
                    
                    if not is_none:
                        for digit in train.digits[idx]:
                            if digit in full_res_shape.digits[i] and digit_sym_map[digit] not in res_str.middle:
                                res_str.middle += digit_sym_map[digit]
                                perm.append(_digit_index(res_shape, digit))

                    if j == 0 and idx != 0:
                        res_str.left += sym.left
                    if j == len(idxs)-1 and idx != len(train)-1:
                        res_str.right += sym.right

            if i == 0:
                res_str.left = first_left
            if i == len(op_shape)-1:
                res_str.right = last_right

            if not is_none and not full_res_shape.digits[i] is None:
                tmp = deepcopy(perm)
                tmp.sort()
                perm = [tmp.index(p) for p in perm]
                tmp = {idx: i for i, idx in enumerate(perm)}
                res_str.middle = "".join([res_str.middle[tmp[j]] for j in range(len(perm))])

            lcontr.set_result(res_str)
            locs.append(lcontr)
            #print(",".join(str(op) for op in lcontr.operands) + "->" + str(lcontr.result))
            #print("operand idxs:", [val for val in lcontr.train_idxs])
            #print("core idx    :", [val for val in lcontr.core_idxs])
            #print("result digits:", [str(d) for d in lcontr.core])
            #print("-----------------")

        object.__setattr__(self, "equation", eq)
        object.__setattr__(self, "operation_shape", op_shape)
        object.__setattr__(self, "operand_shapes", eq.shapes)
        object.__setattr__(self, "result_shape", res_shape)
        object.__setattr__(self, "full_result_shape", full_res_shape)
        super().__init__(locs)

def get_symbol_generator(einsum_contr: Sequence[LocalContraction]) -> Generator[str]:
    """Get a symbol generator that avoids conflicts with existing symbols in the contraction."""
    char_set = set()
    for lcontr in einsum_contr:
        for op in lcontr.operands:
            char_set.update(str(op))
        char_set.update(str(lcontr.result))
    sym_gen = symbol_generator()
    while True:
        char = next(sym_gen)
        if char in char_set:
            continue
        yield char

def _train_symbols(cores: TrainShape,
                  digit_sym_map: dict[Digit, str],
                  sym_gen: Generator[str]) -> list[OperandString]:
    """Get the symbol representation of a trainshape based on its digits"""
    res = []
    left = next(sym_gen)
    for core in cores.digits:
        op_str = OperandString(left=left, right=next(sym_gen))
        for digit in core:
            op_str.middle += digit_sym_map[digit]
        left = op_str.right
        res.append(op_str)
    return res

def _digit_symbol_map(op_strs: Sequence[str], ops: Sequence[TrainShape], symbol_gen: Generator[str]) -> dict[Digit, str]:
    """Create a mapping of digits to unique symbols"""
    digit_map = dict[Digit, str]()
    tmp_map = {}
    for op_str, op in zip(op_strs, ops):
        for op_char, dim in zip(op_str, op.dims):
            if op_char not in tmp_map:
                tmp_map[op_char] = [next(symbol_gen) for _ in range(len(dim))]
            dim_syms = tmp_map[op_char]
            for digit in dim:
                digit_map[digit] = dim_syms[digit.idx]
    return digit_map

def _get_result(
        op_shape: TrainShape,
        result: NoneType | TrainShape,
        guess_result: NoneType | TrainShape
        ) -> tuple[TrainShape, NoneType | TrainShape, NoneType | TrainShape]:
    """Validate and return the contraction result shape."""
    if result is None and guess_result is None:
        return op_shape, None, None
    elif result is None and guess_result is not None:
        raise ValueError("Cannot provide a guess result to full contraction.")
    elif result is not None and guess_result is None:
        return op_shape, result, deepcopy(result)
    elif result is not None and guess_result is not None:
        guess_result = change_dims(guess_result, result.dims)
        res = contains(result, guess_result)
        rev, cmap = res.reverse, res.core_idxs
        if len(cmap) == 0:
            raise ValueError("Result shape does not align with contraction result.")
        if rev:
            result = result.reverse()
            op_shape = op_shape.reverse()
        return op_shape, result, deepcopy(guess_result)
    raise RuntimeError("Unreachable code in get_result.")

def _digit_index(shape: TrainShape, digit: Digit) -> int:
    """Get the index of a digit in a TrainShape."""
    idx = 0
    for core in shape.digits:
        for d in core:
            if d == digit:
                return idx
            idx += 1
    raise ValueError("Digit not found in TrainShape.")
