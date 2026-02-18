# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from math import prod

from .backend import DType, Device, get_index_dtype, ArrayLike, namespace_of_arrays, device
from .digit import Digit
from .dimension import Dimension
from .valueset import ValueSet
from .sequenceof import SequenceOf

class FunctionalCore[T: ArrayLike](SequenceOf[Digit]):
    _left: ValueSet
    _middle: ValueSet
    _right: ValueSet

    @property
    def left(self) -> ValueSet[T]:
        return self._left

    @property
    def middle(self) -> ValueSet[T]:
        return self._middle

    @property
    def right(self) -> ValueSet[T]:
        return self._right

    @property
    def dtype(self) -> DType:
        return self._middle.dtype

    @property
    def device(self) -> Device:
        return self._middle.device

    def __init__(
            self,
            dims: Sequence[Dimension],
            digits: Sequence[Digit],
            left_storage: T,
            right_storage: T) -> None:
        if device(left_storage) != device(right_storage):
            raise ValueError("Left and right storage must be on the same device")

        self._left = ValueSet(left_storage)
        self._right = ValueSet(right_storage)

        xp = namespace_of_arrays(left_storage, right_storage)
        tmp = xp.empty((len(dims), prod(digit.base for digit in digits)),
                       dtype=get_index_dtype(xp),
                       device=device(left_storage))
        self._middle = self._get_middle(dims, digits, tmp)
        super().__init__(digits)

    def _get_middle(self, dims: Sequence[Dimension], digits: Sequence[Digit], storage: T) -> ValueSet[T]:
        xp = namespace_of_arrays(storage)
        int_type = get_index_dtype(xp)
        middle = ValueSet(storage)
        dim_map = {dim.idf: i for i, dim in enumerate(dims)}
        idxs = xp.zeros((len(dims), *[digit.base for digit in digits]), dtype=int_type)
        for i, digit in enumerate(digits):
            dim_idx = dim_map[digit.idf]
            tmp = xp.zeros((len(dims), digit.base), dtype=int_type)
            tmp[dim_idx,:] = xp.asarray([i*digit.factor for i in range(digit.base)], dtype=int_type)
            view = [len(dims),*[1]*i, -1,*[1]*(len(digits)-i-1)]
            idxs += xp.reshape(tmp, view)
        idxs = xp.reshape(idxs, (idxs.shape[0], prod(idxs.shape[1:])))
        middle.add(idxs)
        return middle
