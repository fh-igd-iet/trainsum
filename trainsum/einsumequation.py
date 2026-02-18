# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from copy import deepcopy
from dataclasses import dataclass

from .dimension import Dimension
from .trainshape import TrainShape, change_dims

@dataclass(frozen=True)
class EinsumEquation:
    result: str
    operands: Sequence[str]

    result_dims: Sequence[Dimension]
    shapes: Sequence[TrainShape]

    def __init__(self, eq: str, *ops: TrainShape) -> None:
        res_str = eq.split("->")[1]
        op_strs = eq.split("->")[0].split(",")
        _check_equation(op_strs, res_str)
        _check_dimensions(op_strs, ops)
        dim_map = _dimension_mapping(op_strs, *ops)
        ops = _einsum_operands(op_strs, ops, dim_map)
        object.__setattr__(self, "result", res_str)
        object.__setattr__(self, "operands", op_strs)
        object.__setattr__(self, "result_dims", [dim_map[char] for char in res_str])
        object.__setattr__(self, "shapes", ops)

    def __str__(self) -> str:
        return ",".join(self.operands) + f"->{self.result}"

def _dimension_mapping(op_strs: Sequence[str], *ops: TrainShape) -> dict[str, Dimension]:
    """Create a mapping from einsum equation characters to dimensions of the operands."""
    dims = {}
    for op_str, op in zip(op_strs, ops):
        for op_char, dim in zip(op_str, op.dims):
            if op_char in dims:
                continue
            dims[op_char] = Dimension([d.base for d in dim])
    return dims

def _einsum_operands(op_strs: Sequence[str], 
                     ops: Sequence[TrainShape],
                     dim_mapping: dict[str, Dimension]) -> tuple[TrainShape]:
    """Create einsum operands with correct dimensions based on the dimension mapping."""
    nops = []
    for op_str, op in zip(op_strs, ops):
        op_dims = [dim_mapping[char] for char in op_str]
        nop = change_dims(op, op_dims)
        nops.append(nop)
    return tuple(nops)

def _check_equation(ops: Sequence[str], res: str) -> None:
    """Check the validity of an einsum equation."""
    if len(ops) == 0:
        raise ValueError("An einsum equation cannot have zero oeprands.")

    op_list = [deepcopy(op) for op in ops]
    ref = set(op_list.pop(0))
    while len(op_list) > 0:
        len_ops = len(op_list)
        for i, op in enumerate(op_list):
            if any(char in ref for char in op):
                ref.update(op_list.pop(i))
                break
        if len_ops == len(op_list):
            raise ValueError("Disjoint einsum graphs are not allowed.")

    if any(char not in ''.join(ops) for char in res):
        raise ValueError("Result dimensions must appear in at least one operand.")

    for op in [*ops, res]:
        if len(set(op)) != len(op):
            raise ValueError(f"Duplicate dimensions in operands are not allowed, {op}")


def _check_dimensions(op_strs: Sequence[str], ops: Sequence[TrainShape]) -> None:
    """Check if the operand dimensions are compatible with the einsum equation and with each other."""
    if len(op_strs) != len(ops):
        raise ValueError(f"Number of provided operands in the equation "\
                         f"({len(op_strs)}) and in the function call "\
                         f"({len(ops)}) do not match")
    dim_map = {}
    for i, (str_op, op) in enumerate(zip(op_strs, ops)):
        if len(str_op) != len(op.dims):
            raise ValueError(f"Operand {i} have a different number of dimensions.")
        for op_char, dim in zip(str_op, op.dims):
            if op_char in dim_map and dim != dim_map[op_char]:
                raise ValueError(f"Dimension mismatch in dimension {op_char}.")
            else:
                dim_map[op_char] = dim

