# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from copy import deepcopy
from math import prod
import numpy as np

from .backend import get_index_dtype, ArrayLike, namespace_of_arrays
from .localcontraction import LocalContraction
from .trainshape import TrainShape
from .trainbase import TrainBase

class CoreSlicing[T: ArrayLike]:

    _lcontr: LocalContraction
    _idx_map: dict[int, int]
    _mat_map: dict[tuple[int, int], Sequence[int] | Sequence[slice] | T]
    _bases: Sequence[int]

    def __init__(self, lcontr: LocalContraction, *ops: TrainShape | TrainBase[T]) -> None:
        if len(lcontr.result.middle) == 0:
            raise ValueError("Local contraction without result are and invalid input.")
        self._lcontr = deepcopy(lcontr)
        self._idx_map = _index_map(*ops)
        self._mat_map = _matrix_map(lcontr, *ops)

    def get_matrices(self, comb: int, *ops: TrainBase[T]) -> Sequence[T]:
        mats = []
        for i, (op_idx, core_idx) in enumerate(zip(self._lcontr.train_idxs, self._lcontr.core_idxs)):
            if op_idx in self._idx_map:
                idx = self._idx_map[op_idx]
                cut = self._mat_map[comb, i]
                mats.append(ops[idx].data[core_idx][:,*cut,:])
            else:
                mats.append(self._mat_map[comb, i])
        return mats

    def get_right_shape(self, *ops: TrainBase) -> Sequence[int]:
        shape = {}
        for i, (op, op_idx, core_idx) in enumerate(zip(self._lcontr.operands, self._lcontr.train_idxs, self._lcontr.core_idxs)):
            if op_idx in self._idx_map:
                idx = self._idx_map[op_idx]
                shape[op.right] = ops[idx].data[core_idx].shape[-1]
                shape[op.left] = ops[idx].data[core_idx].shape[0]
            else:
                shape[op.right] = self._mat_map[0, i].shape[-1] # type: ignore
                shape[op.left] = self._mat_map[0, i].shape[0]  # type: ignore
        return [shape[char] for char in self._lcontr.result.right]

def _index_map(*ops: TrainShape | TrainBase) -> dict[int, int]:
    idx_map = {}
    for i in range(len(ops)):
        if not isinstance(ops[i], TrainBase):
            idx_map[i] = len(idx_map)
    return idx_map

def _matrix_map[T: ArrayLike](
        lcontr: LocalContraction,
        *ops: TrainShape | TrainBase[T]
        ) -> dict[tuple[int, int], Sequence[int] | Sequence[slice] | T]:
    rshape = result_shape(lcontr, *ops)
    smap = _slice_map(lcontr, rshape)
    const, tns = lcontr.get_constants(*ops)
    cuts = {}
    for i in range(prod(rshape)):
        for j, (tn, op_str) in enumerate(zip(tns, lcontr.operands)):
            cut = tuple(smap[char][i] if char in smap else slice(None) for char in op_str.middle)
            if j in const: # tn is not a shape
                xp = namespace_of_arrays(tn) # type: ignore
                cuts[i,j] = xp.asarray(tn[:,*cut,:])# type: ignore
            else:
                cuts[i,j] = cut
    return cuts
 
def _slice_map(
        lcontr: LocalContraction,
        rshape: Sequence[int],
        ) -> dict[str, Sequence[slice] | Sequence[int]]:
    tmp = np.zeros((len(rshape), *rshape), dtype=np.uint64)
    ncombs = prod(rshape)
    for i, base in enumerate(rshape):
        ncombs //= base
        for j in range(base):
            cut = (i,) + (slice(None),)*i + (j,) + (slice(None),)*(len(rshape)-i-1)
            tmp[cut] = j
    tmp = np.reshape(tmp, (len(rshape), prod(rshape)))

    smap = {}
    for i, char in enumerate(lcontr.result.middle):
        if char not in smap:
            smap[char] = [int(val) for val in tmp[i,:]]
    return smap

def result_shape(lcontr, *ops: TrainShape | TrainBase) -> Sequence[int]:
    smap = {}
    for op_str, shape in zip(lcontr.operands, lcontr.get_shapes(*ops)):
        for i, char in enumerate(op_str.middle):
            if char not in smap:
                smap[char] = shape[i+1]
    return [smap[char] for char in lcontr.result.middle]
