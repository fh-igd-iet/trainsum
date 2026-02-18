# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence, Generator
from types import NoneType
from copy import deepcopy

from .digit import Digits
from .trainshape import TrainShape
from .einsumequation import EinsumEquation

def einsum_operation_shape(eq: EinsumEquation) -> TrainShape:
    space = operation_space(eq)
    dims = list(set(dim for op in eq.shapes for dim in op.dims))
    operation_shape = TrainShape(dims, space)
    return operation_shape

def einsum_result_shape(eq: EinsumEquation, op_shape: TrainShape) -> NoneType | TrainShape:
    space = []
    for core in op_shape.digits:
        res_digits = [digit for digit in core if any(digit in dim for dim in eq.result_dims)]
        space.append(res_digits)

    if all(len(digits) != 0 for digits in space):
        return TrainShape(eq.result_dims, space)
    elif any(len(digits) != 0 for digits in space):
        raise ValueError("Inconsistent operation shape: some cores have relevant digits while others do not.")
    return None

def operation_space(eq: EinsumEquation) -> Sequence[Digits]:
    space = eq.shapes[0].digits
    for i in iter_ops(eq, 0):
        op = eq.shapes[i].digits
        tmps = [fuse_operands(space, op),
                fuse_operands(op, space),
                fuse_operands(space, list(reversed(op))),
                fuse_operands(op, list(reversed(space))),
                fuse_operands(list(reversed(op)), space),
                fuse_operands(list(reversed(space)), op),
                fuse_operands(list(reversed(space)), list(reversed(op))),
                fuse_operands(list(reversed(op)), list(reversed(space))),
                ]
        sums = [size_of_space(eq, tmp)-len(tmp) for tmp in tmps]
        min_idx = sums.index(min(sums))
        space = tmps[min_idx]

    space = prune_space(eq, space)
    #for i in range(len(space)):
    #    space[i] = sorted(space[i], key=lambda d: (d.idf, d.idx))
    return space


def prune_space(eq: EinsumEquation, space: Sequence[Digits]) -> list[Digits]:
    """Prune the operation space to only include digits relevant to the equation result."""
    pruned_space = []
    digit_set = set(digit for dim in eq.result_dims for digit in dim)
    tmp = []
    for digits in space:
        if not any(digit in digit_set for digit in digits):
            tmp = fuse_digits(tmp, digits)
        else:
            pruned_space.append(fuse_digits(tmp, digits))
            tmp = []
    if len(tmp) > 0 and len(pruned_space) > 0:
        pruned_space[-1] = fuse_digits(pruned_space[-1], tmp)
    elif len(tmp) > 0:
        pruned_space.append(tmp)
    return pruned_space

def size_of_space(eq: EinsumEquation, space: Sequence[Digits]) -> int:
    """Calculate the size of the operation space based on the equation result."""
    out_sum = 0
    for digits in space:
        prod_dim = 1
        for digit in digits:
            if any(digit in dim for dim in eq.result_dims):
                prod_dim *= digit.base
        out_sum += prod_dim if prod_dim != 1 else 0
    return out_sum

def fuse_operands(ref: Sequence[Digits],
                  op: Sequence[Digits]) -> list[Digits]:
    """Contract the operands according to matching digits."""

    tmp1 = set(digit for digits in ref for digit in digits)
    tmp2 = set(digit for digits in op for digit in digits)
    shared_digits = tmp1 & tmp2 #intersection

    op_idx = 0
    res = list[Digits]()
    for i, ref_digits in enumerate(ref):
        in_res = [share_any_digits(res[i], ref_digits) for i in range(len(res))]
        if len(res) > 0 and any(in_res):
            res.append(ref_digits)
            idx = in_res.index(True)
            lres = len(res)
            for _ in range(lres-idx-1):
                res[idx] = fuse_digits(res[idx], res.pop())
        elif any(digit in shared_digits for digit in ref_digits):
            tmp = ref_digits
            while op_idx < len(op) and not share_any_digits(tmp, op[op_idx]):
                tmp = fuse_digits(tmp, op[op_idx])
                op_idx += 1
            while op_idx < len(op) and share_any_digits(tmp, op[op_idx]):
                tmp = fuse_digits(tmp, op[op_idx])
                op_idx += 1
            res.append(tmp)
        elif 0 < op_idx < len(op):
            tmp = ref_digits
            while op_idx < len(op) and not share_any_digits(tmp, op[op_idx]):
                tmp = fuse_digits(tmp, op[op_idx])
                op_idx += 1
            res.append(tmp)
        else:
            res.append(ref_digits)

    for i in range(op_idx, len(op)):
        res.append(list(op[i]))

    return res

def iter_ops(eq: EinsumEquation, start: int) -> Generator[int]:
    """Generate the indices of operands in the order they should be processed."""
    op_strs = list(deepcopy(op) for op in eq.operands)
    cur_op_str = op_strs.pop(start)
    while len(op_strs) > 0:
        op_str_idx = next_operand_idx(cur_op_str, op_strs)
        op_str = op_strs.pop(op_str_idx)
        cur_op_str += "".join(char for char in op_str if char not in cur_op_str)
        yield eq.operands.index(op_str)

def next_operand_idx(cur_op: str, op_strs: Sequence[str]) -> int:
    """Find the index of the next operand that shares dimensions with the current operand."""
    for i, op in enumerate(op_strs):
        if any(char in cur_op for char in op):
            return i
    raise ValueError("No connected operand found.")

def fuse_digits(set1: Digits, set2: Digits) -> Digits:
    """Fuse two sets of digits, avoiding duplicates based on similarity and mapping."""
    res = list(set1)
    for digit2 in set2:
        if not any(digit1 == digit2 for digit1 in set1):
            res.append(digit2)
    return res

def share_any_digits(set1: Digits, set2: Digits) -> bool:
    """Check if two sets of digits share any digits based on similarity and mapping."""
    for digit1 in set1:
        for digit2 in set2:
            if digit1 == digit2:
                return True
    return False

def share_all_digits(set1: Digits, set2: Digits) -> bool:
    """Check if two sets of digits share any digits based on similarity and mapping."""
    for digit1 in set1:
        for digit2 in set2:
            if digit1 != digit2:
                return False
    return True
