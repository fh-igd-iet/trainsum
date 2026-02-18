# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence, Generator, Any
from itertools import product
import opt_einsum as oe

from .backend import get_index_dtype, namespace_of_arrays, ArrayLike, ArrayNamespace, device, size
from .trainshape import TrainShape
from .trainbase import TrainBase
from .localcontraction import LocalContraction

def trains_match(*trains: TrainBase) -> None:
    ref = trains[0]
    for train in trains[1:]:
        if not ref.shape == train.shape:
            raise ValueError("Trains do not match in shape, device or dtype")

def get_device_dtype(ops: Sequence[TrainBase]) -> tuple[Any, Any]:
    if len(ops) == 0:
        raise ValueError("No operands provided")
    device, dtype = ops[0].device, ops[0].dtype
    if not all(op.dtype == dtype for op in ops[1:]):
        raise ValueError("All operand dtypes must match")
    if not all(op.device == device for op in ops[1:]):
        raise ValueError("All operand devices must match")
    return device, dtype

def check_operand_shapes(ref_shapes: Sequence[TrainShape], shapes: Sequence[TrainShape]) -> None:
    if len(ref_shapes) != len(shapes):
        raise ValueError("Number of operand shapes do not match contraction")
    for ref_op, op in zip(ref_shapes, shapes):
        if ref_op != op:
            raise ValueError("Operand shapes do not match contraction")

def shape_map(shapes: Sequence[TrainShape | TrainBase], 
              *lcontrs: LocalContraction) -> dict[str, int]:
    smap: dict[str, int] = {}
    for lcontr in lcontrs:
        tns = lcontr.get_shapes(*shapes)
        for op, tn in zip(lcontr.operands, tns):
            smap.update({char: size for char, size in zip(str(op), tn)})
    return smap

def namespace_of_trains[T: ArrayLike](*trains: TrainBase[T]) -> ArrayNamespace[T]:
    return namespace_of_arrays(*(train.data[0] for train in trains))

def get_shapes(*operands: TrainShape | TrainBase) -> Sequence[TrainShape]:
    return [op.shape if isinstance(op, TrainBase) else op for op in operands]

#------------------------------------------------------------------------------------
#typing stuff

def block_tensor[T: ArrayLike](*tns: T) -> T:
    if not all(tn.ndim == tns[0].ndim for tn in tns):
        raise ValueError("All matrices must have the same number of dimensions")
    xp = namespace_of_arrays(*tns)
    index_dtype = get_index_dtype(xp)
    dev, dtype = device(tns[0]), tns[0].dtype

    dims = [xp.asarray(tn.shape, dtype=index_dtype) for tn in tns]
    shape = tuple(int(sum((dim[i] for dim in dims[1:]), start=dims[0][i])) for i in range(size(dims[0])))
    block = xp.zeros(shape, device=dev, dtype=dtype)

    off = xp.zeros(size(dims[0]), dtype=index_dtype)
    for dim, tn in zip(dims,tns):
        lower, upper = off, off + dim
        cut = tuple(slice(lower[i], upper[i]) for i in range(size(lower)))
        block[cut] = tn
        off += dim
    return block

def check_pos(msg: str, value: int | float):
    if value <= 0:
        raise ValueError(f"{msg} must be above zero, got {value}")

def check_non_neg(msg: str, value: int | float):
    if value < 0:
        raise ValueError(f"{msg} must be a positive, got {value}")

def symbol_generator() -> Generator[str]:
    idx = 0
    while True:
        yield oe.get_symbol(idx)
        idx += 1

def sequence_product(seq: Sequence[int]) -> Generator[Sequence[int]]:
    ranges = [range(s) for s in seq]
    for idxs in product(*ranges):
        yield idxs
