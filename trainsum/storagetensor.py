# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from types import EllipsisType

from .backend import Device, DType, ArrayLike, shape

class StorageTensor[T: ArrayLike]:

    _data: T
    _shape: list[int]
    _cut: list[slice]
    _view: T

    def __init__(self, storage: T) -> None:
        self._data = storage
        self._shape = [0]*len(storage.shape)
        self._cut = [slice(0,0)]*len(storage.shape)
        self._view = self._data[*self._cut]

    def __getitem__(self, idx: int | slice | EllipsisType | T | Sequence[int | slice | EllipsisType | T], /) -> T:
        return self._view[idx]

    def __setitem__(self, idx: int | slice | EllipsisType | T | Sequence[int | slice | EllipsisType | T], value: float | T) -> None:
        self._view[idx] = value

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    @property
    def max_shape(self) -> Sequence[int]:
        return shape(self._data)

    @property
    def dtype(self) -> DType:
        return self._data.dtype

    @property
    def device(self) -> Device:
        return self._data.device

    def add_to_dim(self, dim: int, size: int) -> None:
        size = self._shape[dim]+size
        if size > shape(self._data)[dim]:
            raise ValueError("Exceeding maximum size")
        self._shape[dim] = size
        self._cut[dim] = slice(0, size)
        self._view = self._data[*self._cut]
