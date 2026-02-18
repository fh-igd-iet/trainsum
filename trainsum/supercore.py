# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from types import NoneType
from dataclasses import dataclass
from copy import deepcopy
from collections import deque

from .backend import ArrayLike, namespace_of_arrays, size
from .digit import Digits
from .normalization import Normalization
from .matrixdecomposition import MatrixDecompositionResult
from .tensordecomposition import TensorDecomposition

@dataclass(kw_only=True)
class CutResult[T: ArrayLike, S: MatrixDecompositionResult]:
    shape: Digits
    norm: Normalization
    data: T
    decomp_result: NoneType | S = None

class SuperCore[T: ArrayLike]:

    _digit_queue: deque[Digits]
    _data: NoneType | T
    norm: Normalization

    @property
    def data(self) -> T:
        if self._data is None:
            raise ValueError("Supercore has no data yet.")
        return self._data
    @data.setter
    def data(self, value: T) -> None:
        if self._data is None:
            raise ValueError("Cannot set data without adding cores.")
        if value.shape[1:-1] != self._data.shape[1:-1]:
            raise ValueError(f"New data shape does not match existing shape.")
        if size(value) == 0:
            raise ValueError("Cannot set data with a size of zero.")
        self._data = value
        self.norm = Normalization.NONE

    @property
    def shapes(self) -> Sequence[Digits]:
        return self._digit_queue

    #-------------------------------------------------------------------------
    #constructor & methods

    def __init__(self) -> None:
        self._digit_queue = deque()
        self._norm = Normalization.NONE
        self._data = None

    def set_data(self, cores: Sequence[Digits], data: T, norm: Normalization = Normalization.NONE) -> None:
        if len(cores) == 0:
            raise ValueError("Cannot set data with empty core sequence.")
        if size(data) == 0:
            raise ValueError("Cannot set data with a size of zero.")
        core_middle = []
        for core in cores:
            core_middle.extend([d.base for d in core])
        if tuple(data.shape[1:-1]) != tuple(core_middle):
            raise ValueError(f"Shape of cores and data shape do not match: ")
        self._digit_queue = deque(deepcopy(core) for core in cores)
        self._data = data
        self._norm = norm

    def add_right(self, digits: Digits, data: T) -> None:
        if self._data is None:
            self._add_empty(digits, data)
            return
        if tuple(d.base for d in digits) != tuple(data.shape[1:-1]):
            raise ValueError(f"Core shape and data shape do not match: ")
        if self._data.shape[-1] != data.shape[0]:
            raise ValueError(f"Added core has invalid left rank")

        xp = namespace_of_arrays(data)
        self._data = xp.tensordot(self._data, data, axes=([-1], [0]))
        self._digit_queue.append(deepcopy(digits))
        self._norm = Normalization.NONE
        
    def add_left(self, digits: Digits, data: T) -> None:
        if self._data is None:
            self._add_empty(digits, data)
            return
        if tuple(d.base for d in digits) != tuple(data.shape[1:-1]):
            raise ValueError(f"Core shape and data shape do not match: ")
        if self._data.shape[0] != data.shape[-1]:
            raise ValueError(f"Added core has invalid right rank")

        xp = namespace_of_arrays(data)
        self._data = xp.tensordot(data, self._data, axes=([-1], [0]))
        self._digit_queue.appendleft(deepcopy(digits))
        self._norm = Normalization.NONE

    def _add_empty(self, digits: Digits, data: T) -> None:
        self._digit_queue.append(deepcopy(digits))
        self._data = data

    def cut_right[S: MatrixDecompositionResult](
            self,
            decomp: TensorDecomposition[T, S]
            ) -> CutResult[T, S]:
        if self._data is None or len(self._digit_queue) == 0:
            raise ValueError("Supercore has no data to cut.")
        if len(self._digit_queue) == 1:
            return self._clear(decomp)

        digits = self._digit_queue.pop()
        split = len(digits)+1
        res = decomp.right(self._data, -split)
        self._data, data = res.left, res.right
        self._norm = Normalization.NONE
        return CutResult(shape=digits, norm=Normalization.RIGHT,
                         data=data, decomp_result=res)

    def cut_left[S: MatrixDecompositionResult](
            self,
            decomp: TensorDecomposition[T, S]
            ) -> CutResult[T, S]:
        if self._data is None or len(self._digit_queue) == 0:
            raise ValueError("Supercore has no data to cut.")
        if len(self._digit_queue) == 1:
            return self._clear(decomp)

        digits = self._digit_queue.popleft()
        split = len(digits)+1
        res = decomp.left(self._data, split)
        data, self._data = res.left, res.right
        self._norm = Normalization.NONE
        return CutResult(shape=digits, norm=Normalization.LEFT,
                         data=data, decomp_result=res)

    def _clear[S: MatrixDecompositionResult](
            self,
            _: TensorDecomposition[T, S]
            ) -> CutResult[T, S]:
        if self._data is None:
            raise ValueError("Supercore has no data to clear.")
        if len(self._digit_queue) != 1:
            raise Exception("Supercore has more than one core, cannot clear.")
        core = self._digit_queue.pop()
        #core.left = self._data.shape[0]
        #core.right = self._data.shape[-1]
        data = self._data
        norm = self._norm
        self._data = None
        self._norm = Normalization.NONE
        #return core, data, norm
        return CutResult(shape=core, norm=norm, data=data)
