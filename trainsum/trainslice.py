# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .trainbase import TrainBase
from .slice import slice_operator, slice_vector
from .utils import namespace_of_trains, symbol_generator
from .einsum import einsum

def trainslice[T: ArrayLike](
    train: TrainBase[T],
    cut: tuple[int | slice, ...],
) -> float | complex | TrainBase[T]:
    if len(cut) != len(train.shape.dims):
        raise ValueError("The number of slices must match the number of dimensions.")

    xp = namespace_of_trains(train)
    sgen = symbol_generator()

    ops = []
    op_strs = []
    inp_str = ""
    res_str = ""
    for dim, slc in zip(train.shape.dims, cut):
        inp_str += next(sgen)
        if isinstance(slc, int) or get_num(slc, dim.size()) == 1:
            op = slice_vector(xp, dim, slc)
            ops.append(op)
            op_strs.append(inp_str[-1])
        else:
            op = slice_operator(xp, dim, slc)
            ops.append(op)
            op_strs.append(next(sgen)+inp_str[-1])
            res_str += op_strs[-1][0]
    eq = ",".join(op_strs) + f",{inp_str}->{res_str}"
    res = einsum(eq, *ops, train)
    return res

def get_num(slc: slice, dim: int) -> int:
    start = slc.start if slc.start is not None else 0
    stop = slc.stop if slc.stop is not None else dim
    step = slc.step if slc.step is not None else 1
    return (stop - start + step - 1) // step
