# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from trainsum.exactaddition import ExactAddition
from .backend import ArrayLike
from .trainbase import TrainBase
from .slice import slice_operator, slice_vector

from .utils import namespace_of_trains, symbol_generator
from .einsum import einsum
from .add import add
from .full import full


def assign[T: ArrayLike](
    train: TrainBase[T], cut: tuple[int | slice, ...], value: TrainBase[T]
) -> TrainBase[T]:
    if len(cut) != len(train.shape.dims):
        raise ValueError("The number of slices must match the number of dimensions.")
    exact_add = ExactAddition()

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
            op_strs.append(inp_str[-1]+next(sgen))
            res_str += op_strs[-1][1]

    tmp = full(xp, train.shape, 1.0)
    slc_ops = [slice_vector(xp, dim, slc) for dim, slc in zip(train.shape.dims, cut)]

    eq = ",".join(char for char in inp_str) + f",{inp_str}->{inp_str}"
    slc = einsum(eq, *slc_ops, tmp, result_shape=train.shape)
    if not isinstance(slc, TrainBase):
        raise ValueError("The result of slicing is not a tensor train.")
    slc.data[0][...] *= -1.0
    slc = exact_add(tmp, slc)

    eq = f"{inp_str},{inp_str}->{inp_str}"
    org = einsum(eq, slc, train)
    eq = ",".join(op_strs) + f",{res_str},{inp_str}->{res_str}"
    new = einsum(eq, *ops, tmp, value, result_shape=train.shape)
    if not isinstance(org, TrainBase) or not isinstance(new, TrainBase):
        raise ValueError("The result of slicing is not a tensor train.")
    res = add(org, new)
    return res

def get_num(slc: slice, dim: int) -> int:
    start = slc.start if slc.start is not None else 0
    stop = slc.stop if slc.stop is not None else dim
    step = slc.step if slc.step is not None else 1
    return (stop - start + step - 1) // step
