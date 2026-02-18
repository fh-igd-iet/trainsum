# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from types import NoneType
from dataclasses import dataclass
from collections import deque
from copy import deepcopy

from .backend import ArrayLike, shape
from .digit import Digits
from .direction import Direction
from .trainshape import TrainShape
from .trainbase import TrainBase
from .matrixdecomposition import MatrixDecompositionResult
from .tensordecomposition import TensorDecomposition
from .utils import get_device_dtype, namespace_of_trains, block_tensor, sequence_product

@dataclass(kw_only=True)
class CutResult[T: ArrayLike]:
    digits: Digits
    data: T

class AdditionCore[T: ArrayLike]:

    @property
    def data(self) -> T:
        if self._data is None:
            raise RuntimeError("AdditionCore is empty, cannot return data.")
        return self._data

    _data: NoneType | T
    _direc: Direction
    _digit_queue: deque[Digits]
    _shape: TrainShape

    def __init__(self, shape: TrainShape, direction: Direction) -> None:
        self._shape = deepcopy(shape)
        self._direc = direction
        self._digit_queue = deque()
        self._data = None
    
    def add(self, idx: int, *trains: TrainBase[T]) -> None:
        #if not all(self._shape[idx].similar(train.shape[idx]) for train in trains):
        #    raise ValueError("Train shapes do not match for addition at index {idx}.")
        digits = self._shape.digits[idx]
        if self._direc == Direction.TO_RIGHT:
            self._digit_queue.append(digits)
        else: # Direction.TO_LEFT
            self._digit_queue.appendleft(digits)

        if self._data is None:
            self._data = self._add_first(idx, *trains)
        elif self._direc == Direction.TO_RIGHT:
            self._data = self._add_right(idx, *trains)
        else: # Direction.TO_LEFT
            self._data = self._add_left(idx, *trains)

    def cut[S: MatrixDecompositionResult](self, decomp: TensorDecomposition[T, S]) -> CutResult[T]:
        if self._data is None:
            raise RuntimeError("Cannot cut when data is None.")
        first_fit = all(d1==d2 for d1, d2 in zip(self._digit_queue[0], self._shape.digits[0]))
        last_fit  = all(d1==d2 for d1, d2 in zip(self._digit_queue[-1], self._shape.digits[-1]))
        if len(self._digit_queue) == 1 and (first_fit or last_fit):
            data = self._data
            self._data = None
            digits = self._digit_queue.pop()
        elif self._direc == Direction.TO_RIGHT:
            digits = self._digit_queue.popleft()
            num = 1+len(digits)
            res = decomp.left(self._data, num)
            self._data = res.right
            data = res.left
        else: # Direction.TO_LEFT
            digits = self._digit_queue.pop()
            num = 1+len(digits)
            res = decomp.right(self._data, -num)
            self._data = res.left
            data = res.right
        return CutResult(digits=digits, data=data)

    def _add_first(self, idx: int, *trains: TrainBase[T]) -> T:
        xp = namespace_of_trains(*trains)
        device, dtype = get_device_dtype(trains)

        tns = [train.data[idx] for train in trains]

        left = 0 if idx == 0 else slice(None)
        left_size = 1 if idx == 0 else sum(shape(tn)[0] for tn in tns)
        right = 0 if idx == len(trains[0].shape)-1 else slice(None)
        right_size = 1 if idx == len(trains[0].shape)-1 else sum(shape(tn)[-1] for tn in tns)
        middle = self._shape.middle(idx)

        data = xp.zeros((left_size, *middle, right_size),
                        device=device, dtype=dtype)
        for idxs in sequence_product(self._shape.middle(idx)):
            cut = (left, *idxs, right)
            data[cut] = block_tensor(*[tn[cut] for tn in tns])

        return data

    def _add_left(self, idx: int, *trains: TrainBase[T]) -> T:
        if self._data is None:
            raise RuntimeError("Cannot add left when data is None.")

        xp = namespace_of_trains(*trains)
        device, dtype = get_device_dtype(trains)
        is_begin = (idx == 0)

        tns = [train.data[idx] for train in trains]
        shapes = [shape(tn) for tn in tns]
        dshape = shape(self._data)

        left = 1 if is_begin else sum(sh[0] for sh in shapes)
        middle = self._shape.middle(idx)
        if sum(sh[-1] for sh in shapes) != dshape[0]:
            raise ValueError("Left ranks do not match for addition.")

        data = xp.zeros((left, *middle, *dshape[1:]),
                        device=device, dtype=dtype)

        off0, off1 = 0, 0
        idx = len(shapes[0]) - 1
        for sh, tn in zip(shapes, tns):
            cut0 = slice(off0, off0+sh[-1])
            cut1 = slice(None) if is_begin else slice(off1, off1+sh[0])
            data[cut1,...] += xp.tensordot(tn, self._data[cut0,...], axes=([idx],[0]))
            off0 += sh[-1]
            off1 += sh[0]

        return data

    def _add_right(self, idx: int, *trains: TrainBase[T]) -> T:
        if self._data is None:
            raise RuntimeError("Cannot add right when data is None.")

        xp = namespace_of_trains(*trains)
        device, dtype = get_device_dtype(trains)
        is_end = (idx == len(self._shape)-1)

        tns = [train.data[idx] for train in trains]
        shapes = [shape(tn) for tn in tns]
        dshape = shape(self._data)

        middle = self._shape.middle(idx)
        right = 1 if is_end else sum(shape[-1] for shape in shapes)
        if sum(sh[0] for sh in shapes) != dshape[-1]:
            raise ValueError("Right ranks do not match for addition.")

        data = xp.zeros((*dshape[:-1], *middle, right),
                        device=device, dtype=dtype)

        off0, off1 = 0, 0
        idx = len(dshape) - 1
        for sh, tn in zip(shapes, tns):
            cut0 = slice(off0, off0+sh[0])
            cut1 = slice(None) if is_end else slice(off1, off1+sh[-1])
            data[...,cut1] += xp.tensordot(self._data[...,cut0], tn, axes=([idx],[0]))
            off0 += sh[0]
            off1 += sh[-1]

        return data
